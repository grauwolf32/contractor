package auditservice

import (
	"context"
	"slices"
	"strings"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/auditstore/hookstore"
	"github.com/grauwolf32/contractor/internal/auditstore/ownerstore"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/auditcontract"
	"github.com/grauwolf32/contractor/internal/helpers/contentdigest"
	"github.com/grauwolf32/contractor/internal/helpers/strictjson"
	postgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/jackc/pgx/v5"
)

// AcceptContinuation records one ordinary Task at the idle barrier. The source
// cursor accounts for new work; Workflow inputs determine research materials.
func (s *Service) AcceptContinuation(ctx context.Context, claim auditstore.ControllerClaim, snapshot auditstore.ReconcileSnapshot) (auditstore.TaskIntakeResult, error) {
	audit := snapshot.Audit
	if audit.AuditID != claim.AuditID || audit.State != auditstore.AuditActive {
		return auditstore.TaskIntakeResult{}, auditstore.ErrPrecondition
	}
	pinned, err := executionProfile(audit)
	if err != nil {
		return auditstore.TaskIntakeResult{}, err
	}
	if pinned.Program.Continuation == nil {
		return auditstore.TaskIntakeResult{Complete: true}, nil
	}
	alias := pinned.Program.Continuation.Workflow
	sourceAliases := []string{}
	for name, binding := range pinned.Program.Workflows {
		producer := binding.Outputs["proposals"] != ""
		for _, stage := range pinned.Workflows[name].Stages {
			for _, agent := range stage.Agents {
				producer = producer || agent.Template.SelectsHypothesisProposals()
			}
		}
		if !producer {
			sourceAliases = append(sourceAliases, name)
		}
	}
	slices.Sort(sourceAliases)
	tasks, err := s.reads().ContinuationTasks(ctx, audit.OwnerID, audit.AuditID, alias, sourceAliases, contracts.MaxSelectedMaterials)
	if err != nil {
		return auditstore.TaskIntakeResult{}, err
	}
	if len(tasks) == 0 {
		return auditstore.TaskIntakeResult{Complete: true}, nil
	}
	if auditstore.RemainingTaskCapacity(audit) == 0 || audit.ReservedRunCount >= audit.Limits.MaxSubmittedRuns {
		return auditstore.TaskIntakeResult{Complete: true, GapCode: "task-budget-exhausted"}, nil
	}
	baseline, err := DecodeBaseline(audit.BaselineSnapshot)
	if err != nil {
		return auditstore.TaskIntakeResult{}, err
	}
	binding := pinned.Program.Workflows[alias]
	workflow := pinned.Workflows[alias]
	inputs := map[string]auditcontract.Material{}
	budget := &contracts.MaterialBudget{}
	service := artifacts.NewService(s.storage.Repository(s.pool))
	definitions := []artifacts.MaterialInput{}
	wantsResults := false
	for slot, mapping := range binding.Inputs {
		wantsResults = wantsResults || mapping.Source == "task-results"
		if mapping.Source != "audit-input" {
			continue
		}
		selected, found := baseline.Inputs[mapping.Name]
		if !found {
			continue
		}
		material := auditcontract.Material{ScopeKind: "project", ScopeID: audit.ProjectID, Ref: selected.Ref, Digest: selected.Digest, MediaType: selected.MediaType, SizeBytes: selected.SizeBytes}
		inputs[slot] = material
		contract := workflow.Inputs[slot]
		scope, _ := artifacts.ProjectScope(audit.ProjectID)
		exact := material.Exact()
		definitions = append(definitions, artifacts.MaterialInput{Name: slot, Source: scope, Ref: material.Ref, Expected: &exact, MediaTypes: contract.MediaTypes, Set: contract.MaterialSet, Required: contract.Required, Research: true})
	}
	if _, err := service.CheckInputMaterials(ctx, artifacts.MaterialAuthority{OwnerID: audit.OwnerID, ProjectID: audit.ProjectID, AuditID: audit.AuditID}, definitions, budget); err != nil {
		return auditstore.TaskIntakeResult{}, err
	}

	if wantsResults {
		empty, err := strictjson.Canonical(contracts.MaterialSet{Schema: contracts.MaterialSetSchema, Materials: []contracts.MaterialDescriptor{}})
		if err != nil {
			return auditstore.TaskIntakeResult{}, err
		}
		if budget.CheckSetDocumentBytes(int64(len(empty))) != nil {
			return auditstore.TaskIntakeResult{Complete: true, GapCode: "material-budget-exhausted"}, nil
		}
	}
	chosen := []auditstore.ContinuationSource{}
	results := []auditcontract.Material{}
	resultKeys := map[string]bool{}
	for _, source := range tasks {
		if wantsResults && source.Result != nil && !resultKeys[contracts.MaterialKey(*source.Result)] {
			preview, err := strictjson.Canonical(contracts.MaterialSet{Schema: contracts.MaterialSetSchema, Materials: append(results, *source.Result)})
			if err != nil {
				return auditstore.TaskIntakeResult{}, err
			}
			if budget.CheckSetDocumentBytes(int64(len(preview))) != nil {
				break
			}
			if err := budget.Add(*source.Result); err != nil {
				break
			}
			results = append(results, *source.Result)
			resultKeys[contracts.MaterialKey(*source.Result)] = true
		}
		chosen = append(chosen, source)
	}
	if len(chosen) == 0 {
		return auditstore.TaskIntakeResult{Complete: true, GapCode: "material-budget-exhausted"}, nil
	}
	if wantsResults {
		selected, err := s.retainResultSet(ctx, audit, &claim, results)
		if err != nil {
			return auditstore.TaskIntakeResult{}, err
		}
		for slot, mapping := range binding.Inputs {
			if mapping.Source == "task-results" {
				inputs[slot] = selected
			}
		}
	}
	ids := make([]string, 0, len(chosen))
	for _, source := range chosen {
		ids = append(ids, source.TaskID)
	}
	through := chosen[len(chosen)-1].Ordinal
	digest, err := contentdigest.CanonicalJSON(map[string]any{"workflow": alias, "through": through, "tasks": ids, "inputs": inputs})
	if err != nil {
		return auditstore.TaskIntakeResult{}, err
	}
	key := "continuation-" + strings.TrimPrefix(digest, "sha256:")[:48]
	work := []auditcontract.PreparedSelection{{Key: key, Workflow: alias, Objective: "Use the configured materials and tools to propose bounded next hypotheses/checks, or explicitly return no new work.", Inputs: inputs}}
	params := auditstore.AcceptTasksParams{OwnerID: audit.OwnerID, AuditID: audit.AuditID, Claim: &claim, ContinuationRevision: audit.Revision, IdempotencyKey: key, RequestDigest: digest}
	accepted, err := s.acceptCommonWork(ctx, audit, pinned, baseline, work, params, func(auditcontract.PreparedSelection) auditstore.TaskSource {
		observed := []string(nil)
		if wantsResults {
			observed = ids
		}
		return auditstore.TaskSource{Kind: "continuation", Key: key, ThroughOrdinal: through, ObservationTaskIDs: observed}
	})
	return auditstore.TaskIntakeResult{Changed: err == nil && !accepted.Replayed}, err
}

// retainResultSet uses the common Artifact material-set shape. It contains no
// copied reports or invented input slots; normal Run preparation binds members.
func (s *Service) retainResultSet(ctx context.Context, audit auditstore.Audit, claim *auditstore.ControllerClaim, materials []auditcontract.Material) (auditcontract.Material, error) {
	selected := contracts.MaterialSet{Schema: contracts.MaterialSetSchema, Materials: append([]contracts.MaterialDescriptor{}, materials...)}
	data, err := strictjson.Canonical(selected)
	if err != nil {
		return auditcontract.Material{}, err
	}
	normalized, err := contracts.DecodeMaterialSet(data)
	if err != nil {
		return auditcontract.Material{}, err
	}
	data, err = strictjson.Canonical(normalized)
	if err != nil {
		return auditcontract.Material{}, err
	}
	var retained auditcontract.Material
	err = postgres.InTx(ctx, s.pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		current, err := ownerstore.NewTx(tx, s.storage).LockTaskInputs(ctx, audit.OwnerID, audit.AuditID, claim)
		if err != nil {
			return err
		}
		if current.Revision != audit.Revision {
			return auditstore.ErrPrecondition
		}
		write := func(service *artifacts.Service) error {
			scope, _ := artifacts.AuditScope(audit.AuditID)
			target := contracts.ArtifactRef{Namespace: auditdomain.ArtifactNamespace(audit.AuditID), Name: "materials-" + strings.TrimPrefix(contentdigest.Bytes(data), "sha256:")}
			written, err := service.CreateFrozen(ctx, scope, target, artifacts.Payload{MediaType: "application/json", Data: data})
			if err != nil {
				return err
			}
			retained = auditcontract.Material{ScopeKind: "audit", ScopeID: audit.AuditID, Ref: written.Ref, Digest: contentdigest.Bytes(data), MediaType: "application/json", SizeBytes: int64(len(data))}
			return nil
		}
		if claim != nil {
			return hookstore.NewClaimedArtifacts(tx, s.storage).WithArtifactWrite(ctx, *claim, audit.ProjectID, auditdomain.ArtifactNamespace(audit.AuditID), write)
		}
		return write(artifacts.NewService(s.storage.TxRepository(tx)))
	})
	return retained, err
}
