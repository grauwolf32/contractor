package auditcontroller

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"maps"
	"slices"
	"sort"
	"strconv"

	"github.com/grauwolf32/contractor/internal/agentskills"
	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditservice"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/config/auditprofile"
	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/runtimesettings"
	"github.com/grauwolf32/contractor/internal/contracts/scan"
	"github.com/grauwolf32/contractor/internal/runservice"
)

const (
	// Task sets use zip.Store, so their archive contains every nested task byte.
	// Reserve the maximum package manifest plus more than the complete ZIP header
	// footprint for the Server maximum of 64 short, generated member paths.
	maximumTaskSetZIPStructureBytes = 64 << 10
	maximumBatchedTaskPayloadBytes  = int64(
		auditdomain.MaximumArchiveBytes - auditdomain.MaximumManifestBytes - maximumTaskSetZIPStructureBytes,
	)
)

type RoleOutputLookup interface {
	GetArtifactLink(context.Context, string, string) (auditstore.ArtifactLink, error)
}

type PreparationOutputLookup interface {
	GetPreparationOutput(context.Context, string, string, string) (auditstore.AcceptedPreparationOutput, error)
}

type PinnedSubmissionBuilder struct {
	artifacts    ArtifactAccess
	outputs      RoleOutputLookup
	roundResults RoundResultsStore
	claim        auditstore.ControllerClaim
}

func NewPinnedSubmissionBuilder(
	access ArtifactAccess, outputLookups ...RoleOutputLookup,
) (*PinnedSubmissionBuilder, error) {
	if access == nil {
		return nil, fmt.Errorf("Audit submission Artifact access is required")
	}
	if len(outputLookups) > 1 || len(outputLookups) == 1 && outputLookups[0] == nil {
		return nil, fmt.Errorf("Audit submission output lookup is invalid")
	}
	result := &PinnedSubmissionBuilder{artifacts: access}
	if len(outputLookups) == 1 {
		result.outputs = outputLookups[0]
	}
	return result, nil
}

func (b *PinnedSubmissionBuilder) WithClaim(claim auditstore.ControllerClaim) SubmissionBuilder {
	bound := *b
	bound.artifacts = b.artifacts.WithClaim(claim)
	bound.claim = claim
	return &bound
}

func (b *PinnedSubmissionBuilder) PrepareRole(
	ctx context.Context,
	snapshot auditstore.ReconcileSnapshot,
	workflowRole string,
	attempt int,
) (PreparedSubmission, error) {
	audit := snapshot.Audit
	if attempt < 1 || b.outputs == nil {
		return PreparedSubmission{}, invalidSubmission("Audit role attempt or output lookup is invalid")
	}
	profile, err := auditprofile.DecodeResolvedAuditProfileSnapshot(audit.ProfileSnapshot)
	if err != nil || profile.Ref.Name != audit.Profile.Name || profile.Ref.Version != audit.Profile.Version ||
		profile.Ref.Digest != audit.Profile.Digest {
		return PreparedSubmission{}, invalidSubmission("pinned AuditProfile cannot be decoded")
	}
	binding, exists := profile.Workflows[workflowRole]
	if !exists || binding.Kind != auditprofile.AuditWorkflowPrepare && binding.Kind != auditprofile.AuditWorkflowDiscovery && binding.Kind != auditprofile.AuditWorkflowAssessment {
		return PreparedSubmission{}, invalidSubmission("Audit role binding is invalid")
	}
	var roundID *string
	roundScope := ""
	if binding.Kind == auditprofile.AuditWorkflowPrepare {
		if audit.Phase != auditdomain.AuditPhasePreparing || snapshot.Round != nil || audit.CurrentRoundID != nil || attempt > binding.MaxRunAttempts {
			return PreparedSubmission{}, invalidSubmission("preparation requires a bounded attempt before the first Round")
		}
	} else {
		if snapshot.Round == nil || audit.CurrentRoundID == nil || snapshot.Round.RoundID != *audit.CurrentRoundID {
			return PreparedSubmission{}, invalidSubmission("Audit role is outside an immutable current Round")
		}
		roundScope = snapshot.Round.RoundID
		roundID = &roundScope
	}
	baseline, err := auditservice.DecodeBaseline(audit.BaselineSnapshot)
	if err != nil {
		return PreparedSubmission{}, invalidSubmission("pinned Audit baseline cannot be decoded")
	}
	manifest := auditdomain.ExecutionManifest{Schema: auditdomain.ExecutionManifestSchema, Items: []auditdomain.ExecutionItem{}}
	encodedManifest, err := auditdomain.EncodeExecutionManifest(manifest)
	if err != nil {
		return PreparedSubmission{}, invalidSubmission("empty role execution manifest is invalid")
	}
	manifestDigest := auditdomain.DigestBytes(encodedManifest)
	manifestArtifact, err := b.artifacts.PutImmutableProject(
		ctx, audit.ProjectID,
		contracts.ArtifactRef{
			Namespace: auditdomain.ArtifactNamespace(audit.AuditID),
			Name:      "role-manifest-" + stringsDigest(manifestDigest),
		},
		artifacts.Payload{MediaType: "application/json", Data: encodedManifest},
	)
	if err != nil {
		return PreparedSubmission{}, err
	}
	runInputs, err := b.resolveRoleInputs(ctx, snapshot, workflowRole, binding, baseline, manifestArtifact)
	if err != nil {
		return PreparedSubmission{}, err
	}
	if err := b.validateRulepackInputs(ctx, audit, binding, runInputs); err != nil {
		return PreparedSubmission{}, err
	}
	auditInputs := auditInputSlots(binding, nil)
	parameters, err := resolveRoleParameters(binding, baseline.Scope)
	if err != nil {
		return PreparedSubmission{}, err
	}
	skills, err := selectSkills(binding.Workflow, baseline.Skills)
	if err != nil {
		return PreparedSubmission{}, err
	}
	kind := auditstore.ExecutionRole(binding.Kind)
	attemptText := strconv.Itoa(attempt)
	executionID := auditdomain.DeterministicID(
		"audit-role-execution", audit.AuditID, roundScope, string(kind), workflowRole, attemptText,
	)
	submissionKey := auditdomain.DeterministicID("audit-role-submission", executionID, manifestDigest)
	requestDigest, err := submissionDigest(struct {
		Labels                  map[string]string                    `json:"labels"`
		RequiredPlacementLabels []string                             `json:"requiredPlacementLabels"`
		Schema                  string                               `json:"schema"`
		AuditID                 string                               `json:"auditId"`
		RoundID                 *string                              `json:"roundId,omitempty"`
		ExecutionID             string                               `json:"executionId"`
		Kind                    auditstore.ExecutionRole             `json:"kind"`
		WorkflowRole            string                               `json:"workflowRole"`
		Attempt                 int                                  `json:"attempt"`
		ProfileDigest           string                               `json:"profileDigest"`
		Manifest                auditstore.ExactArtifact             `json:"manifest"`
		Workflow                resolved.ResolvedWorkflow            `json:"workflow"`
		Parameters              map[string]string                    `json:"parameters"`
		Inputs                  map[string]auditstore.ExactArtifact  `json:"inputs"`
		AuditInputs             map[string]bool                      `json:"auditInputs"`
		RuntimeConfig           any                                  `json:"runtimeConfig"`
		Skills                  []agentskills.RunSkillSnapshot       `json:"skills"`
		ProjectTarget           *runtimesettings.HTTPOriginTargetRef `json:"projectTarget,omitempty"`
	}{
		Labels:                  baseline.Labels.Clone(),
		RequiredPlacementLabels: append([]string{}, baseline.RequiredPlacementLabels...),
		Schema:                  "contractor.audit.role-submission.v1", AuditID: audit.AuditID,
		RoundID: roundID, ExecutionID: executionID, Kind: kind,
		WorkflowRole: workflowRole, Attempt: attempt, ProfileDigest: audit.Profile.Digest,
		Manifest: manifestArtifact, Workflow: binding.Workflow, Parameters: parameters,
		Inputs: runInputs, AuditInputs: auditInputs, RuntimeConfig: baseline.RuntimeConfig, Skills: skills,
		ProjectTarget: baseline.ProjectHTTPTarget,
	})
	if err != nil {
		return PreparedSubmission{}, err
	}
	var preparation *auditstore.PreparationSnapshot
	if kind == auditstore.ExecutionPrepare {
		preparation = &auditstore.PreparationSnapshot{Inputs: maps.Clone(runInputs), Parameters: maps.Clone(parameters)}
		for name, input := range preparation.Inputs {
			preparation.Inputs[name] = input.Clone()
		}
	}
	roundResults := make(map[string]auditstore.ExactArtifact)
	for slot, mapping := range binding.Inputs {
		if mapping.Source == auditprofile.AuditInputFromRoundResults {
			roundResults[slot] = runInputs[slot].Clone()
		}
	}
	var materializationID string
	var materialize runservice.AuditInputMaterializer
	if len(roundResults) > 0 {
		materializationID, materialize, err = auditservice.RoundResultsInputMaterializer(roundResults)
		if err != nil {
			return PreparedSubmission{}, invalidSubmission("Round results materializer authority is invalid")
		}
	}
	return PreparedSubmission{
		Intent: auditstore.CreateExecutionIntentParams{
			ExecutionID: executionID, RoundID: roundID, Role: kind, Preparation: preparation,
			WorkflowRole: workflowRole, LaneClass: resolved.WorkflowLaneClass(binding.Workflow),
			RoleAttempt: &attempt, Manifest: manifestArtifact, SubmissionKey: submissionKey,
			RequestDigest: requestDigest, Members: []auditstore.ExecutionMemberIntent{},
		},
		Run: runservice.AuditCreateParams{
			AuditInputs:  auditInputs,
			RoundResults: roundResults, InputMaterializationID: materializationID, MaterializeInputs: materialize,
			RequiredPlacementLabels: append([]string{}, baseline.RequiredPlacementLabels...),
			ExecutionID:             executionID, Workflow: binding.Workflow,
			RuntimeConfig: baseline.RuntimeConfig, Skills: skills,
			ProjectHTTPTarget: baseline.ProjectHTTPTarget,
			Parameters:        parameters, Inputs: runInputs,
			ExecutionManifest: manifestArtifact, RequestDigest: requestDigest,
		},
	}, nil
}

func (b *PinnedSubmissionBuilder) resolveRoleInputs(
	ctx context.Context,
	snapshot auditstore.ReconcileSnapshot,
	workflowRole string,
	binding auditprofile.ResolvedAuditWorkflowBinding,
	baseline auditservice.BaselineSnapshot,
	executionManifest auditstore.ExactArtifact,
) (map[string]auditstore.ExactArtifact, error) {
	result := make(map[string]auditstore.ExactArtifact)
	for _, slot := range slices.Sorted(maps.Keys(binding.Inputs)) {
		mapping := binding.Inputs[slot]
		switch mapping.Source {
		case auditprofile.AuditInputFromAudit:
			input, exists := baseline.Inputs[mapping.Name]
			if !exists {
				if binding.Workflow.Inputs[slot].Required {
					return nil, invalidSubmission("required baseline input is missing")
				}
				continue
			}
			result[slot] = input.Clone()
		case auditprofile.AuditInputFromExecutionManifest:
			result[slot] = executionManifest.Clone()
		case auditprofile.AuditInputFromPreparation:
			output, err := b.preparationOutput(ctx, snapshot.Audit.AuditID, mapping)
			if err != nil {
				return nil, err
			}
			result[slot] = output.Clone()
		case auditprofile.AuditInputFromRetainedOutput:
			link, err := b.outputs.GetArtifactLink(
				ctx, snapshot.Audit.AuditID,
				auditstore.RoleOutputLogicalKey(snapshot.Round.Ordinal, mapping.Role, mapping.Name),
			)
			if err != nil {
				return nil, fmt.Errorf("resolve retained output for role %q: %w", workflowRole, err)
			}
			if link.Scope != artifacts.ScopeAudit {
				return nil, invalidSubmission("retained role output is not Audit-owned")
			}
			result[slot] = link.Artifact.Clone()
		case auditprofile.AuditInputFromRoundResults:
			artifact, err := b.prepareRoundResults(ctx, snapshot)
			if err != nil {
				return nil, err
			}
			result[slot] = artifact
		case auditprofile.AuditInputFromItemPackage:
			return nil, invalidSubmission("non-check Audit role cannot receive an item package")
		default:
			return nil, invalidSubmission("Audit role input mapping is invalid")
		}
	}
	return result, nil
}

func (b *PinnedSubmissionBuilder) preparationOutput(ctx context.Context, auditID string, mapping auditprofile.AuditWorkflowInputMapping) (auditstore.ExactArtifact, error) {
	lookup, ok := b.outputs.(PreparationOutputLookup)
	if !ok {
		return auditstore.ExactArtifact{}, invalidSubmission("accepted preparation output lookup is unavailable")
	}
	output, err := lookup.GetPreparationOutput(ctx, auditID, mapping.Role, mapping.Name)
	if err != nil {
		return auditstore.ExactArtifact{}, err
	}
	return output.Output.Retained, nil
}

// Prepared and retained outputs do not exist at original Audit admission.
// Check their exact metadata once resolved, before an execution intent or Run
// can be submitted. The same boundary covers every Audit role and source kind.
func (b *PinnedSubmissionBuilder) validateRulepackInputs(ctx context.Context, audit auditstore.Audit, binding auditprofile.ResolvedAuditWorkflowBinding, inputs map[string]auditstore.ExactArtifact) error {
	workflow := binding.Workflow
	for _, slot := range resolved.NucleiRulepackInputs(workflow) {
		expected, exists := inputs[slot]
		if !exists {
			continue
		}
		var metadata auditstore.ExactArtifact
		var err error
		mapping := binding.Inputs[slot]
		if mapping.Source == auditprofile.AuditInputFromPreparation || mapping.Source == auditprofile.AuditInputFromRetainedOutput {
			metadata, err = b.artifacts.ResolveAuditExact(ctx, audit.AuditID, expected)
		} else {
			metadata, err = b.artifacts.ResolveProjectExact(ctx, audit.ProjectID, expected)
		}
		if err != nil {
			return err
		}
		if !scan.ValidNucleiRulepackMetadata(metadata.MediaType, metadata.SizeBytes) {
			return invalidSubmission("Audit rulepack input has unsupported media or exceeds its byte limit")
		}
	}
	return nil
}

func resolveRoleParameters(
	binding auditprofile.ResolvedAuditWorkflowBinding,
	scope auditservice.Scope,
) (map[string]string, error) {
	result := make(map[string]string, len(binding.Parameters))
	for name, mapping := range binding.Parameters {
		if mapping.Source == auditprofile.AuditParameterItemField {
			return nil, invalidSubmission("non-check Audit role cannot receive item parameters")
		}
		value, err := resolveParameterWithoutItem(mapping, scope)
		if err != nil {
			return nil, err
		}
		result[name] = value
	}
	return result, nil
}

func resolveParameterWithoutItem(
	mapping auditprofile.AuditWorkflowParameterMapping, scope auditservice.Scope,
) (string, error) {
	switch mapping.Source {
	case auditprofile.AuditParameterLiteral:
		return mapping.Value, nil
	case auditprofile.AuditParameterScopeField:
		switch mapping.Name {
		case "objective":
			return scope.Objective, nil
		case "target":
			return scope.Target, nil
		case "authorizationScope":
			return scope.AuthorizationScope, nil
		}
	}
	return "", invalidSubmission("Audit role parameter mapping is invalid")
}

// PrepareBatch may reduce candidates to a non-empty ordered prefix when the
// exact resolved task sizes cannot fit one deterministic item-task-set. The
// returned intent is the authoritative selected membership; omitted items have
// not consumed an attempt and remain ready for later reconciliation.
func (b *PinnedSubmissionBuilder) PrepareBatch(
	ctx context.Context,
	snapshot auditstore.ReconcileSnapshot,
	selected []CheckExecutionMember,
) (PreparedSubmission, error) {
	audit := snapshot.Audit
	if snapshot.Round == nil || audit.CurrentRoundID == nil ||
		snapshot.Round.RoundID != *audit.CurrentRoundID || len(selected) == 0 ||
		len(selected) > audit.Limits.BatchSize || len(selected) > auditstore.MaxCollectionItems {
		return PreparedSubmission{}, invalidSubmission("Audit batch is outside the pinned round")
	}
	profile, err := auditprofile.DecodeResolvedAuditProfileSnapshot(audit.ProfileSnapshot)
	if err != nil || profile.Ref.Name != audit.Profile.Name || profile.Ref.Version != audit.Profile.Version ||
		profile.Ref.Digest != audit.Profile.Digest || profile.Execution.BatchSize != audit.Limits.BatchSize {
		return PreparedSubmission{}, invalidSubmission("pinned AuditProfile cannot be decoded")
	}
	baseline, err := auditservice.DecodeBaseline(audit.BaselineSnapshot)
	if err != nil {
		return PreparedSubmission{}, invalidSubmission("pinned Audit baseline cannot be decoded")
	}
	first := selected[0]
	binding, exists := profile.Workflows[first.Item.WorkflowRole]
	if !exists || binding.Kind != auditprofile.AuditWorkflowCheck {
		return PreparedSubmission{}, invalidSubmission("Audit item does not name a check Workflow role")
	}
	var routed *auditservice.RoutedCheckContext
	if len(profile.Methods) != 0 {
		routed, err = auditservice.NewRoutedCheckContext(audit, profile, baseline)
		if err != nil {
			return PreparedSubmission{}, invalidSubmission("pinned typed method route closure is invalid")
		}
	}
	roundManifest, paged, err := b.readSelectedRoundExecutionManifest(ctx, snapshot, selected)
	if err != nil {
		return PreparedSubmission{}, err
	}
	if paged {
		selected = selected[:len(roundManifest.Items)]
	}
	manifestItems := make([]auditdomain.ExecutionItem, 0, len(selected))
	tasks := make([]auditstore.ExactArtifact, 0, len(selected))
	attemptIdentity := make([]string, 0, len(selected)*2)
	seenItems := make(map[string]struct{}, len(selected))
	previousOrdinal := -1
	var selectedTaskBytes int64
	for _, member := range selected {
		item, attempt := member.Item, member.Attempt
		if item.AuditID != audit.AuditID || item.RoundID != snapshot.Round.RoundID ||
			item.RoundID != *audit.CurrentRoundID || item.WorkflowRole != first.Item.WorkflowRole ||
			item.ApprovalKind != first.Item.ApprovalKind || item.ApprovalDigest != first.Item.ApprovalDigest ||
			attempt < 1 || attempt > audit.Limits.MaxItemRunAttempts || item.Ordinal <= previousOrdinal {
			return PreparedSubmission{}, invalidSubmission("Audit batch member is outside its shared dispatch envelope")
		}
		if _, duplicate := seenItems[item.ItemID]; duplicate {
			return PreparedSubmission{}, invalidSubmission("Audit batch repeats an item")
		}
		var manifestItem auditdomain.ExecutionItem
		var findErr error
		if paged {
			manifestItem, findErr = findSelectedPreparedManifestItem(roundManifest, item)
		} else {
			manifestItem, findErr = findManifestItem(roundManifest, item)
		}
		if findErr != nil {
			return PreparedSubmission{}, findErr
		}
		task, resolveErr := b.artifacts.ResolveProjectExact(ctx, audit.ProjectID, item.Task)
		if resolveErr != nil {
			return PreparedSubmission{}, resolveErr
		}
		if manifestItem.TaskRef == nil || !manifestItem.TaskRef.SameExact(task.Ref) ||
			manifestItem.TaskPackageDigest != task.Digest {
			return PreparedSubmission{}, invalidSubmission("item task does not match its execution manifest")
		}
		if len(tasks) != 0 && !fitsBatchedTaskPayload(selectedTaskBytes, task.SizeBytes) {
			break
		}
		index := len(tasks)
		seenItems[item.ItemID] = struct{}{}
		previousOrdinal = item.Ordinal
		manifestItem.Ordinal = index
		manifestItems = append(manifestItems, manifestItem)
		tasks = append(tasks, task)
		selectedTaskBytes += task.SizeBytes
		attemptIdentity = append(attemptIdentity, item.ItemID, strconv.Itoa(attempt))
	}
	selected = selected[:len(tasks)]
	// Finish the fitting descriptor prefix before reading any task body. The
	// local cache lives for this one preparation and is reused for assembly.
	verified, err := b.readSelectedTasks(ctx, audit.ProjectID, manifestItems, tasks, selected, baseline.Scope.Values(), paged)
	if err != nil {
		return PreparedSubmission{}, err
	}
	for index, member := range selected {
		if paged {
			task := verified[index].document
			if routed == nil || routed.Validate(*task.AcceptedPlan, task.Dependencies, member.Item.WorkflowRole) != nil {
				return PreparedSubmission{}, invalidSubmission("accepted check plan differs from its pinned route or authority")
			}
		}
		if err := b.validateItemDependencies(ctx, snapshot, binding, baseline, member.Item); err != nil {
			return PreparedSubmission{}, err
		}
	}

	manifest := auditdomain.ExecutionManifest{
		Schema: auditdomain.ExecutionManifestSchema,
		Items:  manifestItems,
	}
	encodedManifest, err := auditdomain.EncodeExecutionManifest(manifest)
	if err != nil || auditdomain.ValidateDispatchExecutionManifest(manifest) != nil {
		return PreparedSubmission{}, invalidSubmission("batch execution manifest is invalid")
	}
	manifestDigest := auditdomain.DigestBytes(encodedManifest)
	namespace := auditdomain.ArtifactNamespace(audit.AuditID)
	manifestArtifact, err := b.artifacts.PutImmutableProject(
		ctx, audit.ProjectID,
		contracts.ArtifactRef{Namespace: namespace, Name: "execution-manifest-" + stringsDigest(manifestDigest)},
		artifacts.Payload{MediaType: "application/json", Data: encodedManifest},
	)
	if err != nil {
		return PreparedSubmission{}, err
	}
	if manifestArtifact.Digest != manifestDigest {
		return PreparedSubmission{}, invalidSubmission("stored execution manifest digest differs")
	}
	taskInput, err := b.buildTaskInput(ctx, audit.ProjectID, namespace, manifestDigest, tasks, verified)
	if err != nil {
		return PreparedSubmission{}, err
	}
	var runInputs map[string]auditstore.ExactArtifact
	var parameters map[string]string
	members := make([]auditstore.ExecutionMemberIntent, len(selected))
	for index, member := range selected {
		candidateInputs, memberInputs, resolveErr := b.resolveInputs(
			ctx, snapshot, binding, baseline, manifestItems[index], member.Item.Dependencies, taskInput, manifestArtifact,
		)
		if resolveErr != nil {
			return PreparedSubmission{}, resolveErr
		}
		candidateParameters, parameterErr := resolveParameters(binding, baseline.Scope, member.Item)
		if parameterErr != nil {
			return PreparedSubmission{}, parameterErr
		}
		if index == 0 {
			runInputs, parameters = candidateInputs, candidateParameters
		} else if !sameExactInputMap(runInputs, candidateInputs) || !sameParameters(parameters, candidateParameters) {
			return PreparedSubmission{}, invalidSubmission("Audit batch members require different Workflow inputs or parameters")
		}
		if len(selected) > 1 {
			memberInputs = append(memberInputs, taskInput.Clone())
		}
		if member.Item.Dependencies != nil {
			memberInputs, err = uniqueDependencyInputs(memberInputs)
			if err != nil {
				return PreparedSubmission{}, err
			}
		}
		members[index] = auditstore.ExecutionMemberIntent{
			ItemID: member.Item.ItemID, BatchOrdinal: index, ItemAttempt: member.Attempt,
			Task: tasks[index], Inputs: memberInputs,
		}
	}
	if err := b.validateRulepackInputs(ctx, audit, binding, runInputs); err != nil {
		return PreparedSubmission{}, err
	}
	skills, err := selectSkills(binding.Workflow, baseline.Skills)
	if err != nil {
		return PreparedSubmission{}, err
	}

	executionIdentity := append([]string{audit.AuditID, snapshot.Round.RoundID}, attemptIdentity...)
	executionIdentity = append(executionIdentity, manifestDigest)
	executionID := auditdomain.DeterministicID("audit-execution", executionIdentity...)
	for index := range members {
		members[index].ExecutionItemID = auditdomain.DeterministicID("audit-execution-item", executionID, members[index].ItemID)
	}
	submissionKey := auditdomain.DeterministicID("audit-submission", executionID, manifestDigest)
	roundID := snapshot.Round.RoundID
	requestDigest, err := submissionDigest(struct {
		Labels                  map[string]string                    `json:"labels"`
		RequiredPlacementLabels []string                             `json:"requiredPlacementLabels"`
		Schema                  string                               `json:"schema"`
		AuditID                 string                               `json:"auditId"`
		ExecutionID             string                               `json:"executionId"`
		Members                 []auditstore.ExecutionMemberIntent   `json:"members"`
		ProfileDigest           string                               `json:"profileDigest"`
		Manifest                auditstore.ExactArtifact             `json:"manifest"`
		Workflow                resolved.ResolvedWorkflow            `json:"workflow"`
		Parameters              map[string]string                    `json:"parameters"`
		Inputs                  map[string]auditstore.ExactArtifact  `json:"inputs"`
		RuntimeConfig           any                                  `json:"runtimeConfig"`
		Skills                  []agentskills.RunSkillSnapshot       `json:"skills"`
		ProjectTarget           *runtimesettings.HTTPOriginTargetRef `json:"projectTarget,omitempty"`
	}{
		Labels:                  baseline.Labels.Clone(),
		RequiredPlacementLabels: append([]string{}, baseline.RequiredPlacementLabels...),
		Schema:                  "contractor.audit.submission.v1", AuditID: audit.AuditID,
		ExecutionID: executionID, Members: members,
		ProfileDigest: audit.Profile.Digest, Manifest: manifestArtifact,
		Workflow: binding.Workflow, Parameters: parameters, Inputs: runInputs,
		RuntimeConfig: baseline.RuntimeConfig, Skills: skills, ProjectTarget: baseline.ProjectHTTPTarget,
	})
	if err != nil {
		return PreparedSubmission{}, err
	}

	intent := auditstore.CreateExecutionIntentParams{
		ExecutionID: executionID, RoundID: &roundID, Role: auditstore.ExecutionCheck,
		WorkflowRole: first.Item.WorkflowRole, LaneClass: resolved.WorkflowLaneClass(binding.Workflow),
		Manifest: manifestArtifact, SubmissionKey: submissionKey, RequestDigest: requestDigest,
		Members: members,
	}
	run := runservice.AuditCreateParams{
		AuditInputs:             auditInputSlots(binding, selected[0].Item.Dependencies),
		RequiredPlacementLabels: append([]string{}, baseline.RequiredPlacementLabels...),
		ExecutionID:             executionID, Workflow: binding.Workflow,
		RuntimeConfig: baseline.RuntimeConfig, Skills: skills,
		ProjectHTTPTarget: baseline.ProjectHTTPTarget,
		Parameters:        parameters, Inputs: runInputs,
		ExecutionManifest: manifestArtifact, RequestDigest: requestDigest,
	}
	if selected[0].Item.Dependencies != nil {
		run.InputMaterializationID = auditdomain.DigestBytes([]byte("retained-dependency:" + manifestDigest))
		run.MaterializeInputs = runservice.MaterializeAuditRetainedDependencies
	}
	return PreparedSubmission{Intent: intent, Run: run}, nil
}

func fitsBatchedTaskPayload(selectedBytes, candidateBytes int64) bool {
	return selectedBytes > 0 && candidateBytes > 0 &&
		selectedBytes <= maximumBatchedTaskPayloadBytes &&
		candidateBytes <= maximumBatchedTaskPayloadBytes-selectedBytes
}

type verifiedTask struct {
	descriptor auditstore.ExactArtifact
	payload    []byte
	document   auditdomain.ItemTask
}

// readSelectedTasks reads a typed singleton or a final multi-item batch once.
// An ordinary singleton keeps the accepted SQL/exact-descriptor path; its
// Runtime verifies the task body before any model context is prepared.
func (b *PinnedSubmissionBuilder) readSelectedTasks(ctx context.Context, projectID string, manifest []auditdomain.ExecutionItem, tasks []auditstore.ExactArtifact, selected []CheckExecutionMember, scope map[string]string, typed bool) ([]verifiedTask, error) {
	if !typed && len(tasks) == 1 {
		return nil, nil
	}
	result := make([]verifiedTask, len(tasks))
	for index, descriptor := range tasks {
		payload, err := b.artifacts.ReadProjectExact(ctx, projectID, descriptor)
		if err != nil {
			return nil, err
		}
		pkg, err := auditdomain.ValidatePackage(payload.Data)
		if err != nil || payload.MediaType != auditdomain.PackageMediaType || pkg.Manifest.Kind != auditdomain.PackageKindTask || pkg.Manifest.PackageID != manifest[index].TaskPackageID || pkg.Digest != descriptor.Digest || pkg.Digest != manifest[index].TaskPackageDigest {
			return nil, invalidSubmission("Audit batch contains an invalid exact task package")
		}
		member, exists := pkg.MemberByID("task-document")
		if !exists || len(pkg.Members()) != 1 || member.Metadata().Path != "task.json" || member.Metadata().MediaType != auditdomain.JSONMediaType {
			return nil, invalidSubmission("Audit task document is absent")
		}
		document, err := auditdomain.DecodeItemTask(member.Data())
		if err != nil {
			return nil, invalidSubmission("Audit task document is invalid")
		}
		if err := validateTaskContent(document, selected[index].Item, scope, typed); err != nil {
			return nil, err
		}
		result[index] = verifiedTask{descriptor: descriptor.Clone(), payload: payload.Data, document: document}
	}
	return result, nil
}

func (b *PinnedSubmissionBuilder) buildTaskInput(
	ctx context.Context, projectID, namespace, manifestDigest string,
	tasks []auditstore.ExactArtifact, verified []verifiedTask,
) (auditstore.ExactArtifact, error) {
	if len(tasks) == 1 {
		return tasks[0].Clone(), nil
	}
	if len(tasks) < 2 || len(tasks) != len(verified) || len(tasks) > auditstore.MaxCollectionItems {
		return auditstore.ExactArtifact{}, invalidSubmission("Audit task set membership is invalid")
	}
	inputs := make([]auditdomain.PackageInput, len(tasks))
	for index, descriptor := range tasks {
		if verified[index].descriptor.Verify(descriptor) != nil {
			return auditstore.ExactArtifact{}, invalidSubmission("verified task identity differs during assembly")
		}
		inputs[index] = auditdomain.PackageInput{ID: fmt.Sprintf("task-%03d", index), Path: fmt.Sprintf("tasks/%03d.zip", index), MediaType: auditdomain.PackageMediaType, Data: verified[index].payload}
	}
	packageID := "task-set-" + stringsDigest(manifestDigest)
	payload, pkg, err := auditdomain.BuildPackage(
		packageID, auditdomain.PackageKindTaskSet, "", inputs,
	)
	if err != nil {
		return auditstore.ExactArtifact{}, invalidSubmission("Audit task set exceeds package bounds")
	}
	descriptor, err := b.artifacts.PutImmutableProject(
		ctx, projectID,
		contracts.ArtifactRef{Namespace: namespace, Name: "execution-tasks-" + stringsDigest(pkg.Digest)},
		artifacts.Payload{MediaType: auditdomain.PackageMediaType, Data: payload},
	)
	if err != nil {
		return auditstore.ExactArtifact{}, err
	}
	if descriptor.Digest != pkg.Digest {
		return auditstore.ExactArtifact{}, invalidSubmission("stored Audit task set digest differs")
	}
	return descriptor, nil
}

func sameExactInputMap(
	left map[string]auditstore.ExactArtifact,
	right map[string]auditstore.ExactArtifact,
) bool {
	if len(left) != len(right) {
		return false
	}
	for name, first := range left {
		second, exists := right[name]
		if !exists || first.Digest != second.Digest || first.MediaType != second.MediaType ||
			first.SizeBytes != second.SizeBytes || !first.Ref.SameExact(second.Ref) {
			return false
		}
	}
	return true
}

func sameParameters(left, right map[string]string) bool {
	if len(left) != len(right) {
		return false
	}
	for name, value := range left {
		candidate, exists := right[name]
		if !exists || candidate != value {
			return false
		}
	}
	return true
}

func (b *PinnedSubmissionBuilder) readRoundExecutionManifest(
	ctx context.Context,
	projectID string,
	descriptor auditstore.ExactArtifact,
) (auditdomain.ExecutionManifest, error) {
	// The durable Round row intentionally pins the immutable reference and
	// digest. Media type and size are Artifact metadata, so reconstruct the
	// complete descriptor from that exact revision before applying the strict
	// content gate. This remains fail-closed: ResolveProjectExact verifies the
	// stored digest and ReadProjectExact verifies every byte and metadata field.
	exact, err := b.artifacts.ResolveProjectExact(ctx, projectID, descriptor)
	if err != nil {
		return auditdomain.ExecutionManifest{}, err
	}
	payload, err := b.artifacts.ReadProjectExact(ctx, projectID, exact)
	if err != nil {
		return auditdomain.ExecutionManifest{}, err
	}
	if payload.MediaType != auditdomain.PackageMediaType {
		return auditdomain.ExecutionManifest{}, invalidSubmission("current Round manifest is not an Audit package")
	}
	pkg, err := auditdomain.ValidatePackage(payload.Data)
	if err != nil || pkg.Digest != exact.Digest || pkg.Manifest.Kind != auditdomain.PackageKindWorklist {
		return auditdomain.ExecutionManifest{}, invalidSubmission("current Round worklist package is invalid")
	}
	member, found := pkg.MemberByID("execution-manifest")
	if !found || member.Metadata().Path != "execution.json" ||
		member.Metadata().MediaType != auditdomain.JSONMediaType {
		return auditdomain.ExecutionManifest{}, invalidSubmission("current Round execution manifest is missing")
	}
	manifest, err := auditdomain.DecodeExecutionManifest(member.Data())
	if err != nil || auditdomain.ValidateDispatchExecutionManifest(manifest) != nil {
		return auditdomain.ExecutionManifest{}, invalidSubmission("current Round execution manifest is invalid")
	}
	return manifest, nil
}

func findManifestItem(
	manifest auditdomain.ExecutionManifest, item auditstore.Item,
) (auditdomain.ExecutionItem, error) {
	if auditdomain.ValidateDispatchExecutionManifest(manifest) != nil {
		return auditdomain.ExecutionItem{}, invalidSubmission("current Round execution manifest is invalid")
	}
	for _, candidate := range manifest.Items {
		if candidate.ItemKey != item.ItemKey {
			continue
		}
		if candidate.Ordinal != item.Ordinal || candidate.SubjectKey != item.SubjectKey {
			return auditdomain.ExecutionItem{}, invalidSubmission("Audit item differs from current Round manifest")
		}
		candidate.Inputs = append([]auditdomain.ExactInput(nil), candidate.Inputs...)
		if candidate.TaskRef != nil {
			ref := candidate.TaskRef.Clone()
			candidate.TaskRef = &ref
		}
		return candidate, nil
	}
	return auditdomain.ExecutionItem{}, invalidSubmission("Audit item is absent from current Round manifest")
}

func (b *PinnedSubmissionBuilder) resolveInputs(
	ctx context.Context,
	snapshot auditstore.ReconcileSnapshot,
	binding auditprofile.ResolvedAuditWorkflowBinding,
	baseline auditservice.BaselineSnapshot,
	manifest auditdomain.ExecutionItem,
	dependencies *auditdomain.RetainedDependencyManifest,
	task auditstore.ExactArtifact,
	executionManifest auditstore.ExactArtifact,
) (map[string]auditstore.ExactArtifact, []auditstore.ExactArtifact, error) {
	manifestInputs := make(map[string]auditdomain.ExactInput, len(manifest.Inputs))
	for _, input := range manifest.Inputs {
		manifestInputs[input.Name] = input
	}
	runInputs := make(map[string]auditstore.ExactArtifact)
	memberInputs := make([]auditstore.ExactArtifact, 0, len(manifest.Inputs))
	names := make([]string, 0, len(binding.Inputs))
	for name := range binding.Inputs {
		names = append(names, name)
	}
	sort.Strings(names)
	if dependencies != nil {
		for _, dependency := range dependencies.Dependencies {
			memberInputs = append(memberInputs, dependency.Retained.Clone())
		}
	}
	consumed := 0
	for _, slot := range names {
		mapping := binding.Inputs[slot]
		switch mapping.Source {
		case auditprofile.AuditInputFromItemPackage:
			runInputs[slot] = task
		case auditprofile.AuditInputFromExecutionManifest:
			runInputs[slot] = executionManifest
		case auditprofile.AuditInputFromAudit:
			descriptor, present := baseline.Inputs[mapping.Name]
			if !present {
				if binding.Workflow.Inputs[slot].Required {
					return nil, nil, invalidSubmission("required baseline input is missing")
				}
				continue
			}
			pinned, present := manifestInputs[slot]
			if !present || pinned.Digest != descriptor.Digest || !pinned.Ref.SameExact(descriptor.Ref) {
				return nil, nil, invalidSubmission("baseline input differs from execution manifest")
			}
			runInputs[slot] = descriptor.Clone()
			memberInputs = append(memberInputs, descriptor.Clone())
			consumed++
		case auditprofile.AuditInputFromPreparation:
			descriptor, err := b.preparationOutput(ctx, snapshot.Audit.AuditID, mapping)
			if err != nil {
				return nil, nil, err
			}
			pinned, present := manifestInputs[slot]
			if !present || pinned.Digest != descriptor.Digest || !pinned.Ref.SameExact(descriptor.Ref) {
				return nil, nil, invalidSubmission("prepared input differs from execution manifest")
			}
			runInputs[slot] = descriptor.Clone()
			memberInputs = append(memberInputs, descriptor.Clone())
			consumed++

		case auditprofile.AuditInputFromRetainedDependency:
			var dependency *auditdomain.RetainedDependency
			if dependencies != nil {
				for index := range dependencies.Dependencies {
					candidate := &dependencies.Dependencies[index]
					if candidate.Name == mapping.Name {
						dependency = candidate
						break
					}
				}
			}
			if dependency == nil {
				if binding.Workflow.Inputs[slot].Required {
					return nil, nil, invalidSubmission("required retained dependency is missing")
				}
				continue
			}
			pinned, present := manifestInputs[slot]
			if !present || pinned.Digest != dependency.Retained.Digest || !pinned.Ref.SameExact(dependency.Retained.Ref) || !contracts.AcceptsMediaType(binding.Workflow.Inputs[slot].MediaTypes, dependency.Retained.MediaType) {
				return nil, nil, invalidSubmission("retained dependency mapping differs or has incompatible media")
			}
			runInputs[slot] = dependency.Retained.Clone()
			consumed++
		case auditprofile.AuditInputFromRetainedOutput:
			if b.outputs == nil || snapshot.Round == nil {
				return nil, nil, invalidSubmission("retained-output lookup is unavailable")
			}
			link, err := b.outputs.GetArtifactLink(
				ctx, snapshot.Audit.AuditID,
				auditstore.RoleOutputLogicalKey(snapshot.Round.Ordinal, mapping.Role, mapping.Name),
			)
			if err != nil {
				return nil, nil, fmt.Errorf("resolve retained output for check role: %w", err)
			}
			if link.Scope != artifacts.ScopeAudit {
				return nil, nil, invalidSubmission("retained role output is not Audit-owned")
			}
			runInputs[slot] = link.Artifact.Clone()
		default:
			return nil, nil, invalidSubmission("Audit input mapping is invalid")
		}
	}
	if consumed != len(manifestInputs) {
		return nil, nil, invalidSubmission("execution manifest contains an unmapped input")
	}
	return runInputs, memberInputs, nil
}

func resolveParameters(
	binding auditprofile.ResolvedAuditWorkflowBinding,
	scope auditservice.Scope,
	item auditstore.Item,
) (map[string]string, error) {
	result := make(map[string]string, len(binding.Parameters))
	for name, mapping := range binding.Parameters {
		switch mapping.Source {
		case auditprofile.AuditParameterLiteral:
			result[name] = mapping.Value
		case auditprofile.AuditParameterItemField:
			switch mapping.Name {
			case "itemKey":
				result[name] = item.ItemKey
			case "subjectKey":
				result[name] = item.SubjectKey
			case "kind":
				result[name] = item.Kind
			default:
				return nil, invalidSubmission("Audit item parameter mapping is invalid")
			}
		case auditprofile.AuditParameterScopeField:
			switch mapping.Name {
			case "objective":
				result[name] = scope.Objective
			case "target":
				result[name] = scope.Target
			case "authorizationScope":
				result[name] = scope.AuthorizationScope
			default:
				return nil, invalidSubmission("Audit scope parameter mapping is invalid")
			}
		default:
			return nil, invalidSubmission("Audit parameter mapping is invalid")
		}
	}
	return result, nil
}

func selectSkills(
	workflow resolved.ResolvedWorkflow,
	baseline []agentskills.RunSkillSnapshot,
) ([]agentskills.RunSkillSnapshot, error) {
	refs, err := resolved.WorkflowSkillRefs(workflow)
	if err != nil {
		return nil, invalidSubmission("Workflow Skill references are invalid")
	}
	byName := make(map[string]agentskills.RunSkillSnapshot, len(baseline))
	for _, skill := range baseline {
		byName[skill.Name] = skill
	}
	result := make([]agentskills.RunSkillSnapshot, len(refs))
	for index, ref := range refs {
		skill, exists := byName[ref.Name]
		if !exists || skill.Source == nil || skill.Initialized() || skill.Validate() != nil {
			return nil, invalidSubmission("Workflow Skill is absent from pinned baseline")
		}
		result[index] = skill.Clone()
	}
	return result, nil
}

func submissionDigest(value any) (string, error) {
	encoded, err := json.Marshal(value)
	if err != nil {
		return "", fmt.Errorf("encode Audit submission identity: %w", err)
	}
	return auditdomain.DigestBytes(encoded), nil
}

func stringsDigest(value string) string {
	if len(value) == len("sha256:")+sha256.Size*2 {
		return value[len("sha256:"):]
	}
	return hex.EncodeToString([]byte(value))
}

func invalidSubmission(message string) error {
	return fmt.Errorf("%w: %s", ErrInvalidSubmission, message)
}

func auditInputSlots(binding auditprofile.ResolvedAuditWorkflowBinding, dependencies *auditdomain.RetainedDependencyManifest) map[string]bool {
	result := map[string]bool{}
	for slot, mapping := range binding.Inputs {
		if mapping.Source == auditprofile.AuditInputFromRetainedDependency && dependencies != nil {
			for _, dependency := range dependencies.Dependencies {
				if dependency.Name == mapping.Name && dependency.Kind == "prepare-output" {
					result[slot] = true
				}
			}
		}
		if mapping.Source == auditprofile.AuditInputFromPreparation || mapping.Source == auditprofile.AuditInputFromRetainedOutput {
			result[slot] = true
		}
	}
	return result
}
