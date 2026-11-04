package auditservice

import (
	"context"

	"fmt"
	"slices"
	"strings"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/auditstore/hookstore"
	"github.com/grauwolf32/contractor/internal/auditstore/ownerstore"
	"github.com/grauwolf32/contractor/internal/config/auditprofile"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/auditcontract"
	"github.com/grauwolf32/contractor/internal/helpers/contentdigest"
	"github.com/grauwolf32/contractor/internal/helpers/strictjson"
	postgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/jackc/pgx/v5"
)

// AcceptContinuation runs at the ordinary idle barrier, before completion.
// Acceptance creates a ready Task, so Pause and Resume use the same dispatch fence.
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
	for key, binding := range pinned.Program.Workflows {
		if binding.Outputs["proposals"] == "" {
			sourceAliases = append(sourceAliases, key)
		}
	}
	slices.Sort(sourceAliases)
	tasks, err := s.reads().ContinuationTasks(ctx, audit.OwnerID, audit.AuditID, alias, sourceAliases)
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
	inputs := map[string]auditcontract.Material{}
	for slot, mapping := range pinned.Program.Workflows[alias].Inputs {
		if mapping.Source == "audit-input" {
			if source, ok := baseline.Inputs[mapping.Name]; ok {
				inputs[slot] = auditcontract.Material{ScopeKind: "project", ScopeID: audit.ProjectID, Ref: source.Ref, Digest: source.Digest, MediaType: source.MediaType, SizeBytes: source.SizeBytes}
			}
		}
	}
	for index, task := range tasks {
		if task.Result != nil {
			slot := fmt.Sprintf("source_%02d", index+1)
			if pinned.Program.Workflows[alias].Inputs[slot].Source != "task-input" {
				return auditstore.TaskIntakeResult{}, ErrInvalid
			}
			inputs[slot] = *task.Result
		}
	}
	material, err := s.generationContext(ctx, audit, &claim, pinned, baseline, tasks, inputs)
	if err != nil {
		return auditstore.TaskIntakeResult{}, err
	}
	for slot, mapping := range pinned.Program.Workflows[alias].Inputs {
		if mapping.Source == "prior-results" {
			inputs[slot] = material
		}
	}
	key := "continuation-" + strings.TrimPrefix(material.Digest, "sha256:")[:48]
	work := []auditcontract.PreparedSelection{{Key: key, Workflow: alias, Objective: "Propose bounded next hypotheses and checks from the selected exact context, or explicitly return no new work.", Inputs: inputs}}
	through := tasks[len(tasks)-1].Ordinal
	params := auditstore.AcceptTasksParams{OwnerID: audit.OwnerID, AuditID: audit.AuditID, Claim: &claim, ContinuationRevision: audit.Revision, IdempotencyKey: key, RequestDigest: material.Digest}
	accepted, err := s.acceptCommonWork(ctx, audit, pinned, baseline, work, params, func(auditcontract.PreparedSelection) auditstore.TaskSource {
		return auditstore.TaskSource{Kind: "continuation", Key: key, ThroughOrdinal: through}
	})
	return auditstore.TaskIntakeResult{Changed: err == nil && !accepted.Replayed}, err
}

// generationContext pins only selected inputs and the next finite slice of
// retained results. It uses ordinary Artifact writes under Project/claim/Audit
// fences; model output cannot supply or replace this system input.
func (s *Service) generationContext(ctx context.Context, audit auditstore.Audit, claim *auditstore.ControllerClaim, pinned auditprofile.PinnedProgram, baseline BaselineSnapshot, tasks []auditstore.ContinuationSource, inputs map[string]auditcontract.Material) (auditcontract.Material, error) {
	selected := auditcontract.ProposalContext{Schema: auditcontract.ProposalContextSchema, Program: pinned.Program, Materials: map[string]auditcontract.Material{}}
	// Only inputs selected by the producing binding are included. A continuation
	// additionally carries the ordinary results chosen by the indexed source slice.
	for key, material := range inputs {
		selected.Materials[key] = material
	}
	if inputs == nil && pinned.Program.Continuation != nil {
		for slot, mapping := range pinned.Program.Workflows[pinned.Program.Continuation.Workflow].Inputs {
			if mapping.Source == "audit-input" {
				if source, ok := baseline.Inputs[mapping.Name]; ok {
					selected.Materials[slot] = auditcontract.Material{ScopeKind: "project", ScopeID: audit.ProjectID, Ref: source.Ref, Digest: source.Digest, MediaType: source.MediaType, SizeBytes: source.SizeBytes}
				}
			}
		}
	}
	for index, task := range tasks {
		observation := auditcontract.ProposalObservation{Key: fmt.Sprintf("source_%02d", index+1), TaskID: task.TaskID, Hypothesis: task.Hypothesis}
		if task.GapCode != nil {
			observation.Gap = *task.GapCode
		}
		selected.Observations = append(selected.Observations, observation)
	}

	data, err := strictjson.Canonical(selected)
	if err != nil || len(data) > auditcontract.MaximumDocumentBytes {
		return auditcontract.Material{}, ErrInvalid
	}
	data, err = auditcontract.Decode(data, nil)
	if err != nil {
		return auditcontract.Material{}, err
	}
	var material auditcontract.Material
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
			target := contracts.ArtifactRef{Namespace: auditdomain.ArtifactNamespace(audit.AuditID), Name: "generation-" + strings.TrimPrefix(contentdigest.Bytes(data), "sha256:")}
			written, err := service.CreateFrozen(ctx, scope, target, artifacts.Payload{MediaType: "application/json", Data: data})
			if err != nil {
				return err
			}
			material = auditcontract.Material{ScopeKind: "audit", ScopeID: audit.AuditID, Ref: written.Ref, Digest: contentdigest.Bytes(data), MediaType: "application/json", SizeBytes: int64(len(data))}
			return nil
		}
		if claim != nil {
			return hookstore.NewClaimedArtifacts(tx, s.storage).WithArtifactWrite(ctx, *claim, audit.ProjectID, auditdomain.ArtifactNamespace(audit.AuditID), write)
		}
		return write(artifacts.NewService(s.storage.TxRepository(tx)))
	})
	return material, err
}
