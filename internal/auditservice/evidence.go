package auditservice

import (
	"context"
	"encoding/json"
	"errors"
	"slices"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/auditstore/ownerstore"
	"github.com/grauwolf32/contractor/internal/auditstore/querystore"
	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/auditcontract"
	"github.com/grauwolf32/contractor/internal/controlplane"
	"github.com/grauwolf32/contractor/internal/helpers/strictjson"
	postgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/jackc/pgx/v5"
)

// RecordEvidence is called under the allocation's Registry write fence. The
// frozen annotation binding is the receipt; source forks and Audit links commit
// in its transaction. No target call or model identity is interpreted here.
func (s *Service) RecordEvidence(ctx context.Context, grant controlplane.AllocationGrant, input auditcontract.EvidenceSubmission) (auditcontract.EvidenceReceipt, error) {
	if input.Validate() != nil || grant.ReadPolicy != controlplane.ReadCurrentRun {
		return auditcontract.EvidenceReceipt{}, ErrInvalid
	}
	if receipt, found, err := s.replayEvidence(ctx, grant, input); err != nil || found {
		return receipt, err
	}
	if grant.WriteFenced || grant.Lost || grant.WritePolicy != controlplane.WriteInputsAndIntermediates {
		return auditcontract.EvidenceReceipt{}, auditstore.ErrPrecondition
	}
	var result auditcontract.EvidenceReceipt
	err := postgres.InTxWithRetry(ctx, s.pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		result = auditcontract.EvidenceReceipt{}
		runs := runstore.NewPostgresTxStore(tx, s.storage)
		if _, err := runs.LockRunState(ctx, grant.RunID, runstore.RunRunning); err != nil {
			return err
		}
		reads := runstore.NewPostgresStore(tx, s.storage)
		run, err := reads.GetRun(ctx, grant.RunID)
		if err != nil {
			return err
		}
		stage, err := reads.GetStageExecution(ctx, grant.StageExecutionID)
		if err != nil {
			return err
		}
		if run.AuditExecutionID == nil || stage.RunID != run.RunID {
			return ErrInvalid
		}
		spec, err := resolved.DecodeResolvedStageSnapshot(stage.StageSpecSnapshot)
		if err != nil {
			return err
		}
		binding, ok := spec.Agents[grant.LogicalAgentName]
		if !ok || binding.Namespace != grant.Namespace || !slices.ContainsFunc(binding.Template.Toolsets, func(sel contracts.ToolsetSelection) bool {
			return sel.Ref.ToolsetID == "audit-results" && sel.Ref.Version == "2" && slices.Contains(sel.Tools, "evidence")
		}) {
			return ErrInvalid
		}
		task, err := querystore.New(tx, s.storage).ExecutionTaskOrigin(ctx, *run.AuditExecutionID, run.RunID, input.ItemKey)
		if err != nil {
			return err
		}
		owners := ownerstore.NewTx(tx, s.storage)
		audit, err := owners.LockTaskInputs(ctx, run.OwnerID, task.AuditID, nil)
		if err != nil {
			return err
		}
		service := artifacts.NewService(s.storage.TxRepository(tx))
		scope, _ := artifacts.AuditScope(audit.AuditID)
		sourceScope, _ := artifacts.RunScope(run.RunID)
		receiptKey := "evidence/" + *run.AuditExecutionID + "/" + auditdomain.DeterministicID("receipt", grant.AllocationID, input.InvocationID, input.SubmissionID)
		name := auditdomain.DeterministicID("evidence", grant.AllocationID, input.InvocationID, input.SubmissionID)
		target := contracts.ArtifactRef{Namespace: auditdomain.ArtifactNamespace(audit.AuditID), Name: name}
		// Resolve a replay before reading source bytes: the source Run may have
		// disappeared only after normal allocation cleanup, while its receipt lasts.
		request, err := strictjson.Canonical(input)
		if err != nil {
			return err
		}
		requestDigest := auditdomain.DigestBytes(request)
		owned, _ := service.Scoped(scope)
		if existing, readErr := owned.Read(ctx, target); readErr == nil {
			var saved auditcontract.EvidenceRecord
			if strictjson.Decode(existing.Payload.Data, &saved) != nil || saved.Validate() != nil || saved.RequestDigest != requestDigest || saved.Origin.RunID != run.RunID {
				return auditstore.ErrConflict
			}
			encodedRef, _ := json.Marshal(existing.Ref)
			if err := owners.ValidateEvidenceReplay(ctx, audit.AuditID, receiptKey, encodedRef); err != nil {
				return err
			}
			result = auditcontract.EvidenceReceipt{
				APIVersion:      contracts.APIVersion,
				Material:        materialFromWrite(audit.AuditID, existing.Ref, existing.Payload),
				Sources:         saved.Sources,
				SelectedSources: saved.SelectedSources,
				ItemKey:         input.ItemKey,
				Replayed:        true,
			}
			return nil
		} else if !errors.Is(readErr, artifacts.ErrArtifactNotFound) {
			return readErr
		}

		links := []auditstore.ArtifactLink{}
		sources := []auditcontract.Material{}
		selectedSources := []auditcontract.Material{}
		var bytes int64
		for _, ref := range input.Sources {
			// The ordinary Run view rejects hidden/reserved material and requires an
			// exact revision. Merely embedding another scope never grants access.
			runScope, _ := service.Run(run.RunID)
			meta, err := runScope.Metadata(ctx, ref)
			if err != nil {
				return err
			}
			bytes += meta.Size
			if bytes > auditcontract.MaximumDocumentBytes {
				return auditstore.ErrEvidenceBudgetExhausted
			}
			name := auditdomain.DeterministicID("source", run.RunID, ref.Namespace, ref.Name, *ref.Revision)
			fork, err := service.Fork(ctx, artifacts.Fork{Source: sourceScope, SourceRef: ref, Target: scope, TargetRef: contracts.ArtifactRef{Namespace: target.Namespace, Name: name}, Frozen: true, Replay: true, Kind: artifacts.LineageAuditOutput})
			if err != nil {
				return err
			}
			exact := meta.Exact()
			exact.Ref = fork.TargetRef
			material := auditcontract.Material{ScopeKind: "audit", ScopeID: audit.AuditID, Ref: exact.Ref, Digest: exact.Digest, MediaType: exact.MediaType, SizeBytes: exact.SizeBytes}
			sources = append(sources, material)
			alias, err := auditcontract.EvidenceSnapshotRef(ref)
			if err != nil {
				return err
			}
			frozen, err := service.Fork(ctx, artifacts.Fork{Source: scope, SourceRef: material.Ref, Target: sourceScope, TargetRef: alias, Frozen: true, Replay: true, Kind: artifacts.LineageAuditInput})
			if err != nil {
				return err
			}
			selected := material
			selected.ScopeKind = "run"
			selected.ScopeID = run.RunID
			selected.Ref = frozen.TargetRef
			selectedSources = append(selectedSources, selected)
			links = append(links, auditstore.ArtifactLink{Scope: artifacts.ScopeAudit, LogicalKey: "evidence-source/" + name, Artifact: exact, SourceProvenance: []byte(`{"kind":"selected-source"}`)})
		}
		record := auditcontract.EvidenceRecord{
			Schema:        "contractor.audit.evidence.v1",
			RequestDigest: requestDigest,
			Origin: auditcontract.EvidenceOrigin{
				AllocationID:     grant.AllocationID,
				StageExecutionID: grant.StageExecutionID,
				LogicalAgentName: grant.LogicalAgentName,
				InvocationID:     input.InvocationID,
				SubmissionID:     input.SubmissionID,
				AuditID:          audit.AuditID,
				TaskID:           task.TaskID,
				RunID:            run.RunID,
				ExecutionID:      *run.AuditExecutionID,
				Hypothesis:       task.Hypothesis,
			},
			Sources:          sources,
			SelectedSources:  selectedSources,
			Interpretation:   input.Interpretation,
			TestedConditions: input.TestedConditions,
			Limitations:      input.Limitations,
		}
		data, err := strictjson.Canonical(record)
		if err != nil {
			return err
		}
		if err := owners.AdmitEvidenceCollection(ctx, audit.AuditID, *run.AuditExecutionID, len(input.Sources), int64(len(data))+bytes); err != nil {
			return err
		}
		payload := artifacts.Payload{MediaType: "application/json", Data: data}
		written, err := service.CreateFrozen(ctx, scope, target, payload)
		if err != nil {
			return err
		}
		exact := contracts.ExactContent(written.Ref, payload.MediaType, data)
		provenance, _ := json.Marshal(map[string]any{"audit_id": audit.AuditID, "task_id": task.TaskID, "run_id": run.RunID, "execution_id": *run.AuditExecutionID, "sourceCount": len(sources), "sourceBytes": bytes})
		links = append(links, auditstore.ArtifactLink{Scope: artifacts.ScopeAudit, LogicalKey: receiptKey, Artifact: exact, SourceProvenance: provenance})
		if err := owners.RetainEvidenceLinks(ctx, audit, links); err != nil {
			return err
		}
		result = auditcontract.EvidenceReceipt{
			APIVersion:      contracts.APIVersion,
			Material:        materialFromWrite(audit.AuditID, written.Ref, payload),
			Sources:         sources,
			SelectedSources: selectedSources,
			ItemKey:         input.ItemKey,
		}
		return nil
	})
	return result, err
}

func materialFromWrite(auditID string, ref contracts.ArtifactRef, payload artifacts.Payload) auditcontract.Material {
	return auditcontract.Material{ScopeKind: "audit", ScopeID: auditID, Ref: ref, Digest: auditdomain.DigestBytes(payload.Data), MediaType: payload.MediaType, SizeBytes: int64(len(payload.Data))}
}

// A committed receipt remains reconcilable under an authenticated read grant
// after write fencing. No source is reread and no append is repeated.
func (s *Service) replayEvidence(ctx context.Context, grant controlplane.AllocationGrant, input auditcontract.EvidenceSubmission) (auditcontract.EvidenceReceipt, bool, error) {
	run, err := runstore.NewPostgresStore(s.pool, s.storage).GetRun(ctx, grant.RunID)
	if err != nil {
		return auditcontract.EvidenceReceipt{}, false, err
	}
	if run.AuditExecutionID == nil {
		return auditcontract.EvidenceReceipt{}, false, ErrInvalid
	}
	task, err := querystore.New(s.pool, s.storage).ExecutionTaskOrigin(ctx, *run.AuditExecutionID, run.RunID, input.ItemKey)
	if err != nil {
		return auditcontract.EvidenceReceipt{}, false, err
	}
	key := "evidence/" + *run.AuditExecutionID + "/" + auditdomain.DeterministicID("receipt", grant.AllocationID, input.InvocationID, input.SubmissionID)
	link, err := s.reads().GetArtifactLink(ctx, task.AuditID, key)
	if errors.Is(err, auditstore.ErrNotFound) {
		return auditcontract.EvidenceReceipt{}, false, nil
	}
	if err != nil {
		return auditcontract.EvidenceReceipt{}, false, err
	}
	scope, _ := artifacts.AuditScope(task.AuditID)
	store, _ := artifacts.NewService(s.storage.Repository(s.pool)).Scoped(scope)
	read, err := store.ReadExact(ctx, link.Artifact)
	if err != nil {
		return auditcontract.EvidenceReceipt{}, false, err
	}
	var record auditcontract.EvidenceRecord
	raw, err := strictjson.Canonical(input)
	if err != nil {
		return auditcontract.EvidenceReceipt{}, false, ErrInvalid
	}
	if strictjson.Decode(read.Payload.Data, &record) != nil || record.Validate() != nil || record.RequestDigest != auditdomain.DigestBytes(raw) || record.Origin.RunID != run.RunID || record.Origin.TaskID != task.TaskID || record.Origin.AllocationID != grant.AllocationID || record.Origin.StageExecutionID != grant.StageExecutionID || record.Origin.LogicalAgentName != grant.LogicalAgentName || record.Origin.InvocationID != input.InvocationID || record.Origin.SubmissionID != input.SubmissionID {
		return auditcontract.EvidenceReceipt{}, false, auditstore.ErrConflict
	}
	return auditcontract.EvidenceReceipt{
		APIVersion:      contracts.APIVersion,
		Material:        materialFromWrite(task.AuditID, read.Ref, read.Payload),
		Sources:         record.Sources,
		SelectedSources: record.SelectedSources,
		ItemKey:         input.ItemKey,
		Replayed:        true,
	}, true, nil
}
