package auditservice

import (
	"context"
	"errors"
	"fmt"
	artifactpolicy "github.com/grauwolf32/contractor/internal/artifacts/policy"
	"github.com/grauwolf32/contractor/internal/contracts"
	"maps"
	"slices"
	"strings"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/config/auditprofile"
	"github.com/grauwolf32/contractor/internal/contracts/auditcontract"
	"github.com/grauwolf32/contractor/internal/helpers/contentdigest"
	"github.com/grauwolf32/contractor/internal/helpers/strictjson"
)

// AcceptProposals validates the whole declared batch before any authoritative
// membership is accepted. The existing execution cursor and common Task receipt
// reconcile empty, hypothesis-only and checking batches without another Run.
func (s *Service) AcceptProposals(ctx context.Context, claim auditstore.ControllerClaim, snapshot auditstore.ReconcileSnapshot, execution auditstore.Execution, source auditcontract.Material, data []byte) (auditstore.TaskIntakeResult, error) {
	audit := snapshot.Audit
	if audit.AuditID != claim.AuditID || execution.AuditID != audit.AuditID || execution.RunID == nil || source.ScopeKind != "run" || source.ScopeID != *execution.RunID || source.Digest != contentdigest.Bytes(data) || source.SizeBytes != int64(len(data)) || len(data) > auditcontract.MaxMutationBytes {
		return auditstore.TaskIntakeResult{}, ErrInvalid
	}
	pinned, err := executionProfile(audit)
	if err != nil {
		return auditstore.TaskIntakeResult{}, err
	}
	binding, exists := pinned.Program.Workflows[execution.WorkflowRole]
	if !exists || source.Ref.Namespace != "outputs" || source.Ref.Name != binding.Outputs["proposals"] {
		return auditstore.TaskIntakeResult{}, ErrInvalid
	}
	key := "proposals-" + strings.TrimPrefix(contentdigest.Bytes([]byte(execution.ExecutionID)), "sha256:")[:48]
	digest, err := contentdigest.CanonicalJSON(map[string]any{"source": source, "execution": execution.ExecutionID})
	if err != nil {
		return auditstore.TaskIntakeResult{}, err
	}
	// The receipt precedes any producer Artifact read, including the frozen
	// role/context metadata. Replay still works after producer deletion.
	if _, found, err := s.owners().LookupTaskAcceptance(ctx, audit.OwnerID, audit.AuditID, key, digest); err != nil || found {
		return auditstore.TaskIntakeResult{Complete: found}, err
	}
	members, err := s.reads().ListExecutionTasks(ctx, audit.AuditID, execution.ExecutionID)
	if err != nil {
		return auditstore.TaskIntakeResult{}, err
	}
	if len(members) != 1 {
		return auditstore.TaskIntakeResult{}, ErrInvalid
	}
	artifactService := artifacts.NewService(s.storage.Repository(s.pool))
	scope, _ := artifacts.RunScope(*execution.RunID)
	run, err := artifactService.Run(*execution.RunID)
	if err != nil {
		return auditstore.TaskIntakeResult{}, err
	}
	rolesArtifact, err := run.Read(ctx, contracts.ArtifactRef{Namespace: artifactpolicy.AuditContextNamespace, Name: auditcontract.ProposalBindingName})
	if err != nil {
		return auditstore.TaskIntakeResult{}, err
	}
	rolesMetadata, err := run.Metadata(ctx, rolesArtifact.Ref)
	if err != nil {
		return auditstore.TaskIntakeResult{}, err
	}
	var roles auditcontract.ProposalBinding
	if !rolesMetadata.Frozen || strictjson.Decode(rolesArtifact.Payload.Data, &roles) != nil || roles.Schema != auditcontract.ProposalBindingSchema || roles.RunID != *execution.RunID || roles.Context.Namespace != "inputs" || roles.Context.ValidateExact() != nil {
		return auditstore.TaskIntakeResult{}, ErrInvalid
	}
	target, err := auditprofile.ResolveProposalTarget(pinned.Workflows[execution.WorkflowRole], binding)
	if err != nil || target == nil || roles.Context.Name != target.ContextSlot || roles.Result.Key() != target.Result.Key() {
		return auditstore.TaskIntakeResult{}, ErrInvalid
	}
	value, err := run.Read(ctx, roles.Context)
	if err != nil {
		return auditstore.TaskIntakeResult{}, err
	}
	contextMeta, err := run.Metadata(ctx, roles.Context)
	if err != nil {
		return auditstore.TaskIntakeResult{}, err
	}
	if !contextMeta.Frozen {
		return auditstore.TaskIntakeResult{}, ErrInvalid
	}
	trusted, err := auditcontract.Decode(value.Payload.Data, nil)
	if err != nil {
		return auditstore.TaskIntakeResult{}, err
	}
	var contextValue auditcontract.ProposalContext
	if strictjson.Decode(trusted, &contextValue) != nil || contextValue.Schema != auditcontract.ProposalContextSchema || contextValue.RunID != *execution.RunID {
		return auditstore.TaskIntakeResult{}, ErrInvalid
	}
	actual, _ := strictjson.Canonical(contextValue.Program)
	expected, _ := strictjson.Canonical(pinned.Program)
	if string(actual) != string(expected) {
		return auditstore.TaskIntakeResult{}, ErrInvalid
	}
	metadata, err := run.Metadata(ctx, source.Ref)
	if err != nil {
		return auditstore.TaskIntakeResult{}, err
	}
	if !metadata.Frozen || exactTaskMaterial(source).Verify(metadata.Exact()) != nil {
		return auditstore.TaskIntakeResult{}, ErrInvalid
	}
	canonical, err := auditcontract.Decode(data, trusted)
	if err != nil {
		return auditstore.TaskIntakeResult{}, err
	}
	var proposed auditcontract.Proposals
	if strictjson.Decode(canonical, &proposed) != nil || proposed.Schema != auditcontract.ProposalsSchema {
		return auditstore.TaskIntakeResult{}, ErrInvalid
	}
	// Resolve explicit own-Run refs through the ordinary metadata owner. JSON
	// carries no scope authority; original input identities share one budget.
	originalByLocal := map[string]auditcontract.Material{}
	definitions := []artifacts.MaterialInput{}
	for index, handle := range slices.Sorted(maps.Keys(contextValue.Materials)) {
		m := contextValue.Materials[handle]
		original := contextValue.InputSources[handle]
		originalByLocal[m.Ref.Key()] = original
		exact := m.Exact()
		definitions = append(definitions, artifacts.MaterialInput{Name: fmt.Sprintf("initial-%02d", index), Source: scope, Ref: m.Ref, Expected: &exact, Original: &original, Research: true})
	}
	selectedNames := map[string]string{}
	for index, handle := range slices.Sorted(maps.Keys(proposed.SelectedMaterials)) {
		ref := proposed.SelectedMaterials[handle]
		if ref.ValidateExact() != nil || artifactpolicy.IsModelHiddenBinding(ref.Namespace, ref.Name) || ref.Namespace == artifactpolicy.AuditContextNamespace || ref.Namespace == "inputs" && ref.Name == target.ContextSlot || ref.Key() == source.Ref.Key() || ref.Namespace == target.Result.Namespace && ref.Name == target.Result.Name {
			return auditstore.TaskIntakeResult{}, ErrInvalid
		}
		original, known := originalByLocal[ref.Key()]
		if ref.Namespace == "inputs" && !known {
			return auditstore.TaskIntakeResult{}, ErrInvalid
		}
		name := fmt.Sprintf("selected-%02d", index)
		selectedNames[handle] = name
		definition := artifacts.MaterialInput{Name: name, Source: scope, Ref: ref, Research: true}
		if known {
			// Both sides come from the frozen context; new refs supply neither.
			definition.Original = &original
			for _, m := range contextValue.Materials {
				if m.Ref.SameExact(ref) {
					exact := m.Exact()
					definition.Expected = &exact
					break
				}
			}
		}
		definitions = append(definitions, definition)
	}
	checked, err := artifactService.CheckInputMaterials(ctx, artifacts.MaterialAuthority{OwnerID: audit.OwnerID, ProjectID: audit.ProjectID, AuditID: audit.AuditID}, definitions, &contracts.MaterialBudget{})
	if err != nil {
		return auditstore.TaskIntakeResult{}, err
	}
	mappings := checked.MaterialMappings()
	for _, m := range mappings {
		originalByLocal[m.Local.Ref.Key()] = m.Source
	}
	for handle, name := range selectedNames {
		contextValue.Materials[handle] = mappings[name].Local
	}

	baseline, err := DecodeBaseline(audit.BaselineSnapshot)
	if err != nil {
		return auditstore.TaskIntakeResult{}, err
	}
	work := make([]auditcontract.PreparedSelection, 0, len(proposed.Work))
	for _, item := range proposed.Work {
		inputs := map[string]auditcontract.Material{}
		for slot, key := range item.Inputs {
			material := contextValue.Materials[key]
			if original, found := originalByLocal[material.Ref.Key()]; found {
				material = original
			}
			inputs[slot] = material
		}
		if err := auditprofile.ValidateTaskInputs(pinned.Program.Workflows[item.Workflow], pinned.Workflows[item.Workflow], inputs); err != nil {
			return auditstore.TaskIntakeResult{}, err
		}
		work = append(work, auditcontract.PreparedSelection{Key: item.Key, Workflow: item.Workflow, Objective: item.Objective, Inputs: inputs})
	}
	cursor, err := s.owners().RecordPreparedSource(ctx, auditstore.PreparedSourceParams{Claim: claim, ExecutionID: execution.ExecutionID, Source: source, Count: len(work)})
	if err != nil {
		return auditstore.TaskIntakeResult{}, err
	}
	if !cursor.Accepting {
		return auditstore.TaskIntakeResult{Complete: true}, nil
	}
	params := auditstore.AcceptTasksParams{OwnerID: audit.OwnerID, AuditID: audit.AuditID, Claim: &claim, IdempotencyKey: key, RequestDigest: digest, PreparedProgress: &auditstore.PreparedWorkProgress{ExecutionID: execution.ExecutionID, From: 0, To: len(work)}, AssertionSnapshots: map[string]auditcontract.Hypothesis{}}
	hypotheses := map[string]auditcontract.Hypothesis{}
	for _, alias := range slices.Sorted(maps.Keys(proposed.Hypotheses)) {
		seed := proposed.Hypotheses[alias]
		h := auditcontract.Hypothesis{Schema: auditcontract.HypothesisSchema, ID: auditdomain.DeterministicID("hypothesis", audit.AuditID, execution.ExecutionID, alias), Revision: 1, Statement: seed.Statement, Scope: seed.Scope, Grounds: seed.Grounds, Origin: auditcontract.HypothesisOrigin{Kind: "run", AuditID: audit.AuditID, TaskID: members[0].Document.TaskID, RunID: *execution.RunID, Source: &source}}
		grounds := &contracts.MaterialBudget{}
		for _, handle := range seed.Sources {
			material := contextValue.Materials[handle]
			if original, found := originalByLocal[material.Ref.Key()]; found {
				material = original
			}
			if err := grounds.Add(material); err != nil {
				return auditstore.TaskIntakeResult{}, err
			}
		}
		h.Sources = grounds.Materials()
		encoded, err := strictjson.Canonical(h)
		if err != nil {
			return auditstore.TaskIntakeResult{}, err
		}
		h, err = auditcontract.DecodeHypothesis(encoded)
		if err != nil {
			return auditstore.TaskIntakeResult{}, err
		}
		hypotheses[alias] = h
		params.HypothesisWrites = append(params.HypothesisWrites, auditstore.HypothesisWrite{Document: h, SourceKey: auditdomain.DeterministicID("seed", execution.ExecutionID, alias)})
	}
	for _, item := range proposed.Work {
		if item.Hypothesis == "" {
			continue
		}
		h, found := hypotheses[item.Hypothesis]
		if !found {
			seed := pinned.Program.Hypotheses[item.Hypothesis]
			h = auditdomain.InitialHypothesis(audit.AuditID, item.Hypothesis, seed)
			if baseline.InitialAssertions != nil {
				if _, ok := baseline.InitialAssertions.Hypotheses[item.Hypothesis]; ok {
					h.Origin.Source = &baseline.InitialAssertions.Source
				}
			}
		}
		params.AssertionSnapshots[item.Key] = h
	}
	accepted, err := s.acceptCommonWork(ctx, audit, pinned, baseline, work, params, func(item auditcontract.PreparedSelection) auditstore.TaskSource {
		return auditstore.TaskSource{Kind: "prepared", Key: item.Key, ExecutionID: execution.ExecutionID, Output: &source}
	})
	if errors.Is(err, auditstore.ErrTaskBudgetExhausted) {
		return auditstore.TaskIntakeResult{Complete: true, GapCode: "task-budget-exhausted"}, nil
	}
	return auditstore.TaskIntakeResult{Complete: err == nil, Changed: err == nil && !accepted.Replayed}, err
}
