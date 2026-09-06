package auditcontroller

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"sort"
	"strconv"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditservice"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
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

type PinnedSubmissionBuilder struct {
	artifacts ArtifactAccess
	outputs   RoleOutputLookup
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

func (b *PinnedSubmissionBuilder) PrepareRole(
	ctx context.Context,
	snapshot auditstore.ReconcileSnapshot,
	workflowRole string,
	attempt int,
) (PreparedSubmission, error) {
	audit := snapshot.Audit
	if snapshot.Round == nil || audit.CurrentRoundID == nil ||
		snapshot.Round.RoundID != *audit.CurrentRoundID || attempt < 1 || b.outputs == nil {
		return PreparedSubmission{}, invalidSubmission("Audit role is outside an immutable current Round")
	}
	profile, err := config.DecodeResolvedAuditProfileSnapshot(audit.ProfileSnapshot)
	if err != nil || profile.Ref.Name != audit.Profile.Name || profile.Ref.Version != audit.Profile.Version ||
		profile.Ref.Digest != audit.Profile.Digest {
		return PreparedSubmission{}, invalidSubmission("pinned AuditProfile cannot be decoded")
	}
	binding, exists := profile.Workflows[workflowRole]
	if !exists || binding.Kind != config.AuditWorkflowDiscovery && binding.Kind != config.AuditWorkflowAssessment {
		return PreparedSubmission{}, invalidSubmission("Audit role is not a discovery or assessment binding")
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
	manifestDigest := digestBytes(encodedManifest)
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
	parameters, err := resolveRoleParameters(binding, baseline.Scope)
	if err != nil {
		return PreparedSubmission{}, err
	}
	skills, err := selectSkills(binding.Workflow, baseline.Skills)
	if err != nil {
		return PreparedSubmission{}, err
	}
	kind := auditstore.ExecutionRole(binding.Kind)
	roundID := snapshot.Round.RoundID
	attemptText := strconv.Itoa(attempt)
	executionID := deterministicID(
		"audit-role-execution", audit.AuditID, roundID, string(kind), workflowRole, attemptText,
	)
	submissionKey := deterministicID("audit-role-submission", executionID, manifestDigest)
	requestDigest, err := submissionDigest(struct {
		Schema        string                              `json:"schema"`
		AuditID       string                              `json:"auditId"`
		RoundID       string                              `json:"roundId"`
		ExecutionID   string                              `json:"executionId"`
		Kind          auditstore.ExecutionRole            `json:"kind"`
		WorkflowRole  string                              `json:"workflowRole"`
		Attempt       int                                 `json:"attempt"`
		ProfileDigest string                              `json:"profileDigest"`
		Manifest      auditstore.ExactArtifact            `json:"manifest"`
		Workflow      config.ResolvedWorkflow             `json:"workflow"`
		Parameters    map[string]string                   `json:"parameters"`
		Inputs        map[string]auditstore.ExactArtifact `json:"inputs"`
		RuntimeConfig any                                 `json:"runtimeConfig"`
		Skills        []contracts.RunSkillSnapshot        `json:"skills"`
		ProjectTarget *contracts.HTTPOriginTargetRef      `json:"projectTarget,omitempty"`
	}{
		Schema: "contractor.audit.role-submission.v1", AuditID: audit.AuditID,
		RoundID: roundID, ExecutionID: executionID, Kind: kind,
		WorkflowRole: workflowRole, Attempt: attempt, ProfileDigest: audit.Profile.Digest,
		Manifest: manifestArtifact, Workflow: binding.Workflow, Parameters: parameters,
		Inputs: runInputs, RuntimeConfig: baseline.RuntimeConfig, Skills: skills,
		ProjectTarget: baseline.ProjectHTTPTarget,
	})
	if err != nil {
		return PreparedSubmission{}, err
	}
	return PreparedSubmission{
		Intent: auditstore.CreateExecutionIntentParams{
			ExecutionID: executionID, RoundID: &roundID, Role: kind,
			WorkflowRole: workflowRole, RoleAttempt: &attempt,
			Manifest: manifestArtifact, SubmissionKey: submissionKey,
			RequestDigest: requestDigest, Members: []auditstore.ExecutionMemberIntent{},
		},
		Run: runservice.AuditCreateParams{
			ExecutionID: executionID, Workflow: binding.Workflow,
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
	binding config.ResolvedAuditWorkflowBinding,
	baseline auditservice.BaselineSnapshot,
	executionManifest auditstore.ExactArtifact,
) (map[string]auditstore.ExactArtifact, error) {
	result := make(map[string]auditstore.ExactArtifact)
	for _, slot := range sortedKeys(binding.Inputs) {
		mapping := binding.Inputs[slot]
		switch mapping.Source {
		case config.AuditInputFromAudit:
			input, exists := baseline.Inputs[mapping.Name]
			if !exists {
				if binding.Workflow.Inputs[slot].Required {
					return nil, invalidSubmission("required baseline input is missing")
				}
				continue
			}
			result[slot] = cloneExact(input)
		case config.AuditInputFromExecutionManifest:
			result[slot] = cloneExact(executionManifest)
		case config.AuditInputFromRetainedOutput:
			link, err := b.outputs.GetArtifactLink(
				ctx, snapshot.Audit.AuditID,
				auditstore.RoleOutputLogicalKey(snapshot.Round.Ordinal, mapping.Role, mapping.Name),
			)
			if err != nil {
				return nil, fmt.Errorf("resolve retained output for role %q: %w", workflowRole, err)
			}
			result[slot] = cloneExact(link.Artifact)
		case config.AuditInputFromItemPackage:
			return nil, invalidSubmission("non-check Audit role cannot receive an item package")
		default:
			return nil, invalidSubmission("Audit role input mapping is invalid")
		}
	}
	return result, nil
}

func resolveRoleParameters(
	binding config.ResolvedAuditWorkflowBinding,
	scope auditservice.Scope,
) (map[string]string, error) {
	result := make(map[string]string, len(binding.Parameters))
	for name, mapping := range binding.Parameters {
		if mapping.Source == config.AuditParameterItemField {
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
	mapping config.AuditWorkflowParameterMapping, scope auditservice.Scope,
) (string, error) {
	switch mapping.Source {
	case config.AuditParameterLiteral:
		return mapping.Value, nil
	case config.AuditParameterScopeField:
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

func sortedKeys[T any](values map[string]T) []string {
	result := make([]string, 0, len(values))
	for key := range values {
		result = append(result, key)
	}
	sort.Strings(result)
	return result
}

func (b *PinnedSubmissionBuilder) Prepare(
	ctx context.Context,
	snapshot auditstore.ReconcileSnapshot,
	item auditstore.Item,
	attempt int,
) (PreparedSubmission, error) {
	return b.PrepareBatch(ctx, snapshot, []CheckExecutionMember{{Item: item, Attempt: attempt}})
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
	profile, err := config.DecodeResolvedAuditProfileSnapshot(audit.ProfileSnapshot)
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
	if !exists || binding.Kind != config.AuditWorkflowCheck {
		return PreparedSubmission{}, invalidSubmission("Audit item does not name a check Workflow role")
	}
	roundManifest, err := b.readRoundExecutionManifest(ctx, audit.ProjectID, snapshot.Round.Manifest)
	if err != nil {
		return PreparedSubmission{}, err
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
		manifestItem, findErr := findManifestItem(roundManifest, item)
		if findErr != nil {
			return PreparedSubmission{}, findErr
		}
		task, resolveErr := b.artifacts.ResolveProjectExact(ctx, audit.ProjectID, item.Task)
		if resolveErr != nil {
			return PreparedSubmission{}, resolveErr
		}
		if manifestItem.TaskRef == nil || !sameExactRef(*manifestItem.TaskRef, task.Ref) ||
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
	manifest := auditdomain.ExecutionManifest{
		Schema: auditdomain.ExecutionManifestSchema,
		Items:  manifestItems,
	}
	encodedManifest, err := auditdomain.EncodeExecutionManifest(manifest)
	if err != nil || auditdomain.ValidateDispatchExecutionManifest(manifest) != nil {
		return PreparedSubmission{}, invalidSubmission("batch execution manifest is invalid")
	}
	manifestDigest := digestBytes(encodedManifest)
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
	taskInput, err := b.buildTaskInput(ctx, audit.ProjectID, namespace, manifestDigest, manifestItems, tasks)
	if err != nil {
		return PreparedSubmission{}, err
	}
	var runInputs map[string]auditstore.ExactArtifact
	var parameters map[string]string
	members := make([]auditstore.ExecutionMemberIntent, len(selected))
	for index, member := range selected {
		candidateInputs, memberInputs, resolveErr := b.resolveInputs(
			ctx, snapshot, binding, baseline, manifestItems[index], taskInput, manifestArtifact,
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
			memberInputs = append(memberInputs, cloneExact(taskInput))
		}
		members[index] = auditstore.ExecutionMemberIntent{
			ItemID: member.Item.ItemID, BatchOrdinal: index, ItemAttempt: member.Attempt,
			Task: tasks[index], Inputs: memberInputs,
		}
	}
	skills, err := selectSkills(binding.Workflow, baseline.Skills)
	if err != nil {
		return PreparedSubmission{}, err
	}

	executionIdentity := append([]string{audit.AuditID, snapshot.Round.RoundID}, attemptIdentity...)
	executionIdentity = append(executionIdentity, manifestDigest)
	executionID := deterministicID("audit-execution", executionIdentity...)
	for index := range members {
		members[index].ExecutionItemID = deterministicID("audit-execution-item", executionID, members[index].ItemID)
	}
	submissionKey := deterministicID("audit-submission", executionID, manifestDigest)
	roundID := snapshot.Round.RoundID
	requestDigest, err := submissionDigest(struct {
		Schema        string                              `json:"schema"`
		AuditID       string                              `json:"auditId"`
		ExecutionID   string                              `json:"executionId"`
		Members       []auditstore.ExecutionMemberIntent  `json:"members"`
		ProfileDigest string                              `json:"profileDigest"`
		Manifest      auditstore.ExactArtifact            `json:"manifest"`
		Workflow      config.ResolvedWorkflow             `json:"workflow"`
		Parameters    map[string]string                   `json:"parameters"`
		Inputs        map[string]auditstore.ExactArtifact `json:"inputs"`
		RuntimeConfig any                                 `json:"runtimeConfig"`
		Skills        []contracts.RunSkillSnapshot        `json:"skills"`
		ProjectTarget *contracts.HTTPOriginTargetRef      `json:"projectTarget,omitempty"`
	}{
		Schema: "contractor.audit.submission.v1", AuditID: audit.AuditID,
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
		WorkflowRole: first.Item.WorkflowRole,
		Manifest:     manifestArtifact, SubmissionKey: submissionKey, RequestDigest: requestDigest,
		Members: members,
	}
	run := runservice.AuditCreateParams{
		ExecutionID: executionID, Workflow: binding.Workflow,
		RuntimeConfig: baseline.RuntimeConfig, Skills: skills,
		ProjectHTTPTarget: baseline.ProjectHTTPTarget,
		Parameters:        parameters, Inputs: runInputs,
		ExecutionManifest: manifestArtifact, RequestDigest: requestDigest,
	}
	return PreparedSubmission{Intent: intent, Run: run}, nil
}

func fitsBatchedTaskPayload(selectedBytes, candidateBytes int64) bool {
	return selectedBytes > 0 && candidateBytes > 0 &&
		selectedBytes <= maximumBatchedTaskPayloadBytes &&
		candidateBytes <= maximumBatchedTaskPayloadBytes-selectedBytes
}

func (b *PinnedSubmissionBuilder) buildTaskInput(
	ctx context.Context,
	projectID string,
	namespace string,
	manifestDigest string,
	manifestItems []auditdomain.ExecutionItem,
	tasks []auditstore.ExactArtifact,
) (auditstore.ExactArtifact, error) {
	if len(tasks) == 1 {
		return cloneExact(tasks[0]), nil
	}
	if len(tasks) < 2 || len(tasks) != len(manifestItems) || len(tasks) > auditstore.MaxCollectionItems {
		return auditstore.ExactArtifact{}, invalidSubmission("Audit task set membership is invalid")
	}
	inputs := make([]auditdomain.PackageInput, len(tasks))
	for index, descriptor := range tasks {
		payload, err := b.artifacts.ReadProjectExact(ctx, projectID, descriptor)
		if err != nil {
			return auditstore.ExactArtifact{}, err
		}
		pkg, err := auditdomain.ValidatePackage(payload.Data)
		if err != nil || payload.MediaType != auditdomain.PackageMediaType ||
			pkg.Manifest.Kind != auditdomain.PackageKindTask || pkg.Manifest.PackageID != manifestItems[index].TaskPackageID ||
			pkg.Digest != descriptor.Digest || pkg.Digest != manifestItems[index].TaskPackageDigest {
			return auditstore.ExactArtifact{}, invalidSubmission("Audit batch contains an invalid exact task package")
		}
		inputs[index] = auditdomain.PackageInput{
			ID: fmt.Sprintf("task-%03d", index), Path: fmt.Sprintf("tasks/%03d.zip", index),
			MediaType: auditdomain.PackageMediaType, Data: payload.Data,
		}
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
			first.SizeBytes != second.SizeBytes || !sameExactRef(first.Ref, second.Ref) {
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
	resolved, err := b.artifacts.ResolveProjectExact(ctx, projectID, descriptor)
	if err != nil {
		return auditdomain.ExecutionManifest{}, err
	}
	payload, err := b.artifacts.ReadProjectExact(ctx, projectID, resolved)
	if err != nil {
		return auditdomain.ExecutionManifest{}, err
	}
	if payload.MediaType != auditdomain.PackageMediaType {
		return auditdomain.ExecutionManifest{}, invalidSubmission("current Round manifest is not an Audit package")
	}
	pkg, err := auditdomain.ValidatePackage(payload.Data)
	if err != nil || pkg.Digest != resolved.Digest || pkg.Manifest.Kind != auditdomain.PackageKindWorklist {
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
			ref := cloneRef(*candidate.TaskRef)
			candidate.TaskRef = &ref
		}
		return candidate, nil
	}
	return auditdomain.ExecutionItem{}, invalidSubmission("Audit item is absent from current Round manifest")
}

func (b *PinnedSubmissionBuilder) resolveInputs(
	ctx context.Context,
	snapshot auditstore.ReconcileSnapshot,
	binding config.ResolvedAuditWorkflowBinding,
	baseline auditservice.BaselineSnapshot,
	manifest auditdomain.ExecutionItem,
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
	consumed := 0
	for _, slot := range names {
		mapping := binding.Inputs[slot]
		switch mapping.Source {
		case config.AuditInputFromItemPackage:
			runInputs[slot] = task
		case config.AuditInputFromExecutionManifest:
			runInputs[slot] = executionManifest
		case config.AuditInputFromAudit:
			descriptor, present := baseline.Inputs[mapping.Name]
			if !present {
				if binding.Workflow.Inputs[slot].Required {
					return nil, nil, invalidSubmission("required baseline input is missing")
				}
				continue
			}
			pinned, present := manifestInputs[slot]
			if !present || pinned.Digest != descriptor.Digest || !sameExactRef(pinned.Ref, descriptor.Ref) {
				return nil, nil, invalidSubmission("baseline input differs from execution manifest")
			}
			runInputs[slot] = cloneExact(descriptor)
			memberInputs = append(memberInputs, cloneExact(descriptor))
			consumed++
		case config.AuditInputFromRetainedOutput:
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
			runInputs[slot] = cloneExact(link.Artifact)
		default:
			return nil, nil, invalidSubmission("Audit input mapping is invalid")
		}
	}
	if consumed != len(manifestInputs) {
		return nil, nil, invalidSubmission("execution manifest contains an unmapped input")
	}
	return runInputs, memberInputs, nil
}

// resolveInputs remains a narrow pure helper for contract-focused unit tests.
// Production dispatch uses PinnedSubmissionBuilder.resolveInputs so a check
// may also consume an exact, Audit-retained discovery output.
func resolveInputs(
	binding config.ResolvedAuditWorkflowBinding,
	baseline auditservice.BaselineSnapshot,
	manifest auditdomain.ExecutionItem,
	task auditstore.ExactArtifact,
	executionManifest auditstore.ExactArtifact,
) (map[string]auditstore.ExactArtifact, []auditstore.ExactArtifact, error) {
	builder := &PinnedSubmissionBuilder{}
	return builder.resolveInputs(
		context.Background(), auditstore.ReconcileSnapshot{}, binding, baseline,
		manifest, task, executionManifest,
	)
}

func resolveParameters(
	binding config.ResolvedAuditWorkflowBinding,
	scope auditservice.Scope,
	item auditstore.Item,
) (map[string]string, error) {
	result := make(map[string]string, len(binding.Parameters))
	for name, mapping := range binding.Parameters {
		switch mapping.Source {
		case config.AuditParameterLiteral:
			result[name] = mapping.Value
		case config.AuditParameterItemField:
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
		case config.AuditParameterScopeField:
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
	workflow config.ResolvedWorkflow,
	baseline []contracts.RunSkillSnapshot,
) ([]contracts.RunSkillSnapshot, error) {
	refs, err := config.WorkflowSkillRefs(workflow)
	if err != nil {
		return nil, invalidSubmission("Workflow Skill references are invalid")
	}
	byName := make(map[string]contracts.RunSkillSnapshot, len(baseline))
	for _, skill := range baseline {
		byName[skill.Name] = skill
	}
	result := make([]contracts.RunSkillSnapshot, len(refs))
	for index, ref := range refs {
		skill, exists := byName[ref.Name]
		if !exists || skill.Source == nil || skill.Initialized() || skill.Validate() != nil {
			return nil, invalidSubmission("Workflow Skill is absent from pinned baseline")
		}
		result[index] = cloneSkill(skill)
	}
	return result, nil
}

func submissionDigest(value any) (string, error) {
	encoded, err := json.Marshal(value)
	if err != nil {
		return "", fmt.Errorf("encode Audit submission identity: %w", err)
	}
	return digestBytes(encoded), nil
}

func deterministicID(prefix string, values ...string) string {
	digest := sha256.New()
	_, _ = digest.Write([]byte("contractor.audit.identity.v1\x00" + prefix))
	for _, value := range values {
		_, _ = digest.Write([]byte{0})
		_, _ = digest.Write([]byte(value))
	}
	return prefix + "-" + hex.EncodeToString(digest.Sum(nil))
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

func sameExactRef(left, right contracts.ArtifactRef) bool {
	return left.Namespace == right.Namespace && left.Name == right.Name &&
		left.Revision != nil && right.Revision != nil && *left.Revision == *right.Revision
}

func cloneRef(source contracts.ArtifactRef) contracts.ArtifactRef {
	if source.Revision != nil {
		revision := *source.Revision
		source.Revision = &revision
	}
	return source
}

func cloneExact(source auditstore.ExactArtifact) auditstore.ExactArtifact {
	source.Ref = cloneRef(source.Ref)
	return source
}

func cloneSkill(source contracts.RunSkillSnapshot) contracts.RunSkillSnapshot {
	if source.Source != nil {
		value := cloneRef(*source.Source)
		source.Source = &value
	}
	if source.Artifact != nil {
		value := cloneRef(*source.Artifact)
		source.Artifact = &value
	}
	return source
}
