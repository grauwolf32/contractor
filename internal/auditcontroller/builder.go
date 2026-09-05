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

type PinnedSubmissionBuilder struct{ artifacts ArtifactAccess }

func NewPinnedSubmissionBuilder(access ArtifactAccess) (*PinnedSubmissionBuilder, error) {
	if access == nil {
		return nil, fmt.Errorf("Audit submission Artifact access is required")
	}
	return &PinnedSubmissionBuilder{artifacts: access}, nil
}

func (b *PinnedSubmissionBuilder) Prepare(
	ctx context.Context,
	snapshot auditstore.ReconcileSnapshot,
	item auditstore.Item,
	attempt int,
) (PreparedSubmission, error) {
	audit := snapshot.Audit
	if snapshot.Round == nil || audit.CurrentRoundID == nil ||
		item.AuditID != audit.AuditID || item.RoundID != snapshot.Round.RoundID ||
		item.RoundID != *audit.CurrentRoundID || attempt < 1 ||
		audit.Limits.BatchSize != 1 || attempt > audit.Limits.MaxItemRunAttempts {
		return PreparedSubmission{}, invalidSubmission("Audit item or attempt is outside the pinned round")
	}
	profile, err := config.DecodeResolvedAuditProfileSnapshot(audit.ProfileSnapshot)
	if err != nil || profile.Ref.Name != audit.Profile.Name || profile.Ref.Version != audit.Profile.Version ||
		profile.Ref.Digest != audit.Profile.Digest || profile.Execution.BatchSize != 1 {
		return PreparedSubmission{}, invalidSubmission("pinned AuditProfile cannot be decoded")
	}
	baseline, err := auditservice.DecodeBaseline(audit.BaselineSnapshot)
	if err != nil {
		return PreparedSubmission{}, invalidSubmission("pinned Audit baseline cannot be decoded")
	}
	binding, exists := profile.Workflows[item.WorkflowRole]
	if !exists {
		return PreparedSubmission{}, invalidSubmission("Audit item names an unknown Workflow role")
	}
	manifestItem, err := findManifestItem(baseline.Inventory.ExecutionManifest, item)
	if err != nil {
		return PreparedSubmission{}, err
	}
	task, err := b.artifacts.ResolveProjectExact(ctx, audit.ProjectID, item.Task)
	if err != nil {
		return PreparedSubmission{}, err
	}
	if manifestItem.TaskRef == nil || !sameExactRef(*manifestItem.TaskRef, task.Ref) ||
		manifestItem.TaskPackageDigest != task.Digest {
		return PreparedSubmission{}, invalidSubmission("item task does not match its execution manifest")
	}

	runInputs, memberInputs, err := resolveInputs(binding, baseline, manifestItem, task)
	if err != nil {
		return PreparedSubmission{}, err
	}
	parameters, err := resolveParameters(binding, baseline.Scope, item)
	if err != nil {
		return PreparedSubmission{}, err
	}
	skills, err := selectSkills(binding.Workflow, baseline.Skills)
	if err != nil {
		return PreparedSubmission{}, err
	}

	manifestItem.Ordinal = 0
	manifest := auditdomain.ExecutionManifest{
		Schema: auditdomain.ExecutionManifestSchema,
		Items:  []auditdomain.ExecutionItem{manifestItem},
	}
	encodedManifest, err := auditdomain.EncodeExecutionManifest(manifest)
	if err != nil || auditdomain.ValidateDispatchExecutionManifest(manifest) != nil {
		return PreparedSubmission{}, invalidSubmission("one-item execution manifest is invalid")
	}
	manifestDigest := digestBytes(encodedManifest)
	namespace := deterministicID("audit", audit.AuditID)
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

	attemptText := strconv.Itoa(attempt)
	executionID := deterministicID(
		"audit-execution", audit.AuditID, item.RoundID, item.ItemID, attemptText, manifestDigest,
	)
	executionItemID := deterministicID("audit-execution-item", executionID, item.ItemID)
	submissionKey := deterministicID("audit-submission", executionID, manifestDigest)
	roundID := item.RoundID
	requestDigest, err := submissionDigest(struct {
		Schema        string                              `json:"schema"`
		AuditID       string                              `json:"auditId"`
		ExecutionID   string                              `json:"executionId"`
		ItemID        string                              `json:"itemId"`
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
		Schema: "contractor.audit.submission.v1", AuditID: audit.AuditID,
		ExecutionID: executionID, ItemID: item.ItemID, Attempt: attempt,
		ProfileDigest: audit.Profile.Digest, Manifest: manifestArtifact,
		Workflow: binding.Workflow, Parameters: parameters, Inputs: runInputs,
		RuntimeConfig: baseline.RuntimeConfig, Skills: skills, ProjectTarget: baseline.ProjectHTTPTarget,
	})
	if err != nil {
		return PreparedSubmission{}, err
	}

	intent := auditstore.CreateExecutionIntentParams{
		ExecutionID: executionID, RoundID: &roundID, Role: auditstore.ExecutionCheck,
		Manifest: manifestArtifact, SubmissionKey: submissionKey, RequestDigest: requestDigest,
		Members: []auditstore.ExecutionMemberIntent{{
			ExecutionItemID: executionItemID, ItemID: item.ItemID,
			BatchOrdinal: 0, ItemAttempt: attempt, Task: task, Inputs: memberInputs,
		}},
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

func findManifestItem(
	manifest auditdomain.ExecutionManifest, item auditstore.Item,
) (auditdomain.ExecutionItem, error) {
	if auditdomain.ValidateDispatchExecutionManifest(manifest) != nil {
		return auditdomain.ExecutionItem{}, invalidSubmission("baseline execution manifest is invalid")
	}
	for _, candidate := range manifest.Items {
		if candidate.ItemKey != item.ItemKey {
			continue
		}
		if candidate.Ordinal != item.Ordinal || candidate.SubjectKey != item.SubjectKey {
			return auditdomain.ExecutionItem{}, invalidSubmission("Audit item differs from baseline manifest")
		}
		candidate.Inputs = append([]auditdomain.ExactInput(nil), candidate.Inputs...)
		if candidate.TaskRef != nil {
			ref := cloneRef(*candidate.TaskRef)
			candidate.TaskRef = &ref
		}
		return candidate, nil
	}
	return auditdomain.ExecutionItem{}, invalidSubmission("Audit item is absent from baseline manifest")
}

func resolveInputs(
	binding config.ResolvedAuditWorkflowBinding,
	baseline auditservice.BaselineSnapshot,
	manifest auditdomain.ExecutionItem,
	task auditstore.ExactArtifact,
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
			return nil, nil, invalidSubmission("retained-output is unsupported by the fixed-round Controller")
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
