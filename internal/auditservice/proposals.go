package auditservice

import (
	"context"
	"errors"
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
	members, err := s.reads().ListExecutionTasks(ctx, audit.AuditID, execution.ExecutionID)
	if err != nil {
		return auditstore.TaskIntakeResult{}, err
	}
	if len(members) != 1 {
		return auditstore.TaskIntakeResult{}, ErrInvalid
	}
	slot := ""
	for name, mapping := range binding.Inputs {
		if mapping.Source == "prior-results" {
			if slot != "" {
				return auditstore.TaskIntakeResult{}, ErrInvalid
			}
			slot = name
		}
	}
	selected, exists := members[0].Document.Inputs[slot]
	if slot == "" || !exists {
		return auditstore.TaskIntakeResult{}, ErrInvalid
	}
	key := "proposals-" + strings.TrimPrefix(contentdigest.Bytes([]byte(execution.ExecutionID)), "sha256:")[:48]
	digest, err := contentdigest.CanonicalJSON(map[string]any{"source": source, "context": selected})
	if err != nil {
		return auditstore.TaskIntakeResult{}, err
	}
	if _, found, err := s.owners().LookupTaskAcceptance(ctx, audit.OwnerID, audit.AuditID, key, digest); err != nil || found {
		return auditstore.TaskIntakeResult{Complete: found}, err
	}
	scope, _ := artifacts.AuditScope(audit.AuditID)
	store, _ := artifacts.NewService(s.storage.Repository(s.pool)).Scoped(scope)
	value, err := store.Read(ctx, selected.Ref)
	if err != nil {
		return auditstore.TaskIntakeResult{}, err
	}
	if exactTaskMaterial(selected).Verify(value.Exact()) != nil {
		return auditstore.TaskIntakeResult{}, ErrInvalid
	}
	trusted, err := auditcontract.Decode(value.Payload.Data, nil)
	if err != nil {
		return auditstore.TaskIntakeResult{}, err
	}
	var contextValue auditcontract.ProposalContext
	if strictjson.Decode(trusted, &contextValue) != nil || contextValue.Schema != auditcontract.ProposalContextSchema {
		return auditstore.TaskIntakeResult{}, ErrInvalid
	}
	actual, _ := strictjson.Canonical(contextValue.Program)
	expected, _ := strictjson.Canonical(pinned.Program)
	if string(actual) != string(expected) {
		return auditstore.TaskIntakeResult{}, ErrInvalid
	}
	run, err := artifacts.NewService(s.storage.Repository(s.pool)).Run(*execution.RunID)
	if err != nil {
		return auditstore.TaskIntakeResult{}, err
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
	baseline, err := DecodeBaseline(audit.BaselineSnapshot)
	if err != nil {
		return auditstore.TaskIntakeResult{}, err
	}
	work := make([]auditcontract.PreparedSelection, 0, len(proposed.Work))
	for _, item := range proposed.Work {
		inputs := map[string]auditcontract.Material{}
		for slot, key := range item.Inputs {
			inputs[slot] = contextValue.Materials[key]
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
	for alias, seed := range proposed.Hypotheses {
		h := auditcontract.Hypothesis{Schema: auditcontract.HypothesisSchema, ID: auditdomain.DeterministicID("hypothesis", audit.AuditID, execution.ExecutionID, alias), Revision: 1, Statement: seed.Statement, Scope: seed.Scope, Grounds: seed.Grounds, Origin: auditcontract.HypothesisOrigin{Kind: "run", AuditID: audit.AuditID, TaskID: members[0].Document.TaskID, RunID: *execution.RunID, Source: &source}}
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
