package auditservice

import (
	"context"
	"encoding/json"
	"reflect"
	"slices"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/auditcontract"
	"github.com/grauwolf32/contractor/internal/controlplane"
	"github.com/grauwolf32/contractor/internal/helpers/strictjson"
	postgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/jackc/pgx/v5"
)

func checkResultGrant(ctx context.Context, runs *runstore.PostgresStore, run runstore.WorkflowRun, grant controlplane.AllocationGrant, tool string) error {
	stage, err := runs.GetStageExecution(ctx, grant.StageExecutionID)
	if err != nil {
		return err
	}
	if stage.RunID != run.RunID {
		return ErrInvalid
	}
	spec, err := resolved.DecodeResolvedStageSnapshot(stage.StageSpecSnapshot)
	if err != nil {
		return err
	}
	binding, ok := spec.Agents[grant.LogicalAgentName]
	if !ok || binding.Namespace != grant.Namespace || !slices.ContainsFunc(binding.Template.Toolsets, func(sel contracts.ToolsetSelection) bool {
		return sel.Ref.ToolsetID == "audit-results" && sel.Ref.Version == "2" && slices.Contains(sel.Tools, tool)
	}) {
		return ErrInvalid
	}
	if tool != "evidence" || run.AuditCompletion != nil && run.AuditCompletion.Finding != nil {
		if run.AuditCompletion == nil || run.AuditCompletion.Stage != stage.StageName || run.AuditCompletion.Agent != grant.LogicalAgentName {
			return ErrInvalid
		}
	}
	return nil
}

// RecordEvidence keeps the selected bytes and acknowledgement in the concrete
// owner's scope atomically. Both callers use the same source/capture algorithm.
func (s *Service) RecordEvidence(ctx context.Context, grant controlplane.AllocationGrant, input auditcontract.EvidenceSubmission) (auditcontract.EvidenceReceipt, error) {
	if input.Validate() != nil || grant.ReadPolicy != controlplane.ReadCurrentRun {
		return auditcontract.EvidenceReceipt{}, ErrInvalid
	}
	run, err := runstore.NewPostgresStore(s.pool, s.storage).GetRun(ctx, grant.RunID)
	if err != nil {
		return auditcontract.EvidenceReceipt{}, err
	}
	owner, err := loadResultOwner(ctx, s.pool, s.storage, run)
	if err != nil {
		return auditcontract.EvidenceReceipt{}, err
	}
	if saved, found, err := replayEvidence(ctx, owner, grant, input); found || err != nil {
		return saved, err
	}
	if grant.WriteFenced || grant.Lost || grant.WritePolicy != controlplane.WriteInputsAndIntermediates {
		return auditcontract.EvidenceReceipt{}, auditstore.ErrPrecondition
	}
	var receipt auditcontract.EvidenceReceipt
	err = postgres.InTxWithRetry(ctx, s.pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		o, err := loadResultOwner(ctx, tx, s.storage, run)
		if err != nil {
			return err
		}
		if err = o.lock(ctx, tx, s.storage); err != nil {
			return err
		}
		if err = checkResultGrant(ctx, runstore.NewPostgresStore(tx, s.storage), run, grant, "evidence"); err != nil {
			return err
		}
		if saved, found, err := replayEvidence(ctx, o, grant, input); found || err != nil {
			receipt = saved
			return err
		}
		sourceScope, _ := artifacts.RunScope(run.RunID)
		runStore, _ := o.artifacts.Run(run.RunID)
		sources, selectedSources := []auditcontract.Material{}, []auditcontract.Material{}
		links := []auditstore.ArtifactLink{}
		var sourceBytes int64
		for _, ref := range input.Sources {
			meta, err := runStore.Metadata(ctx, ref)
			if err != nil {
				return err
			}
			sourceBytes += meta.Size
			if sourceBytes > auditcontract.MaximumDocumentBytes {
				return auditstore.ErrEvidenceBudgetExhausted
			}
			name := auditdomain.DeterministicID("source", run.RunID, ref.Namespace, ref.Name, *ref.Revision)
			outputKind, inputKind := artifacts.LineageAuditOutput, artifacts.LineageAuditInput
			if o.audit == nil {
				outputKind = artifacts.LineageProjectOutputPublish
				inputKind = artifacts.LineageInputFork
			}
			fork, err := o.artifacts.Fork(ctx, artifacts.Fork{Source: sourceScope, SourceRef: ref, Target: o.scope, TargetRef: contracts.ArtifactRef{Namespace: o.namespace, Name: name}, Frozen: true, Replay: true, Kind: outputKind})
			if err != nil {
				return err
			}
			exact := meta.Exact()
			exact.Ref = fork.TargetRef
			material := auditcontract.Material{ScopeKind: string(o.scope.Kind()), ScopeID: o.scope.ID(), Ref: exact.Ref, Digest: exact.Digest, MediaType: exact.MediaType, SizeBytes: exact.SizeBytes}
			sources = append(sources, material)
			alias, err := auditcontract.EvidenceSnapshotRef(ref)
			if err != nil {
				return err
			}
			frozen, err := o.artifacts.Fork(ctx, artifacts.Fork{Source: o.scope, SourceRef: material.Ref, Target: sourceScope, TargetRef: alias, Frozen: true, Replay: true, Kind: inputKind})
			if err != nil {
				return err
			}
			selected := material
			selected.ScopeKind = "run"
			selected.ScopeID = run.RunID
			selected.Ref = frozen.TargetRef
			selectedSources = append(selectedSources, selected)
			links = append(links, auditstore.ArtifactLink{Scope: o.scope.Kind(), LogicalKey: "evidence-source/" + name, Artifact: exact, SourceProvenance: []byte(`{"kind":"selected-source"}`)})
		}
		request, _ := strictjson.Canonical(input)
		record := auditcontract.EvidenceRecord{Schema: "contractor.audit.evidence.v1", RequestDigest: auditdomain.DigestBytes(request), Origin: o.origin(grant.AllocationID, grant.StageExecutionID, grant.LogicalAgentName, input.InvocationID, input.SubmissionID, input.ItemKey), Sources: sources, SelectedSources: selectedSources, Interpretation: input.Interpretation, TestedConditions: input.TestedConditions, Limitations: input.Limitations}
		if record.Validate() != nil {
			return ErrInvalid
		}
		data, err := strictjson.Canonical(record)
		if err != nil {
			return err
		}
		if err = o.admit(ctx, "evidence", len(sources), int64(len(data))+sourceBytes); err != nil {
			return err
		}
		target := contracts.ArtifactRef{Namespace: o.namespace, Name: auditdomain.DeterministicID("evidence", grant.AllocationID, input.InvocationID, input.SubmissionID)}
		payload := artifacts.Payload{MediaType: "application/json", Data: data}
		written, err := o.artifacts.CreateFrozen(ctx, o.scope, target, payload)
		if err != nil {
			return err
		}
		provenance, _ := json.Marshal(map[string]any{"audit_id": record.Origin.AuditID, "task_id": record.Origin.TaskID, "run_id": run.RunID, "execution_id": record.Origin.ExecutionID, "sourceCount": len(sources), "sourceBytes": sourceBytes})
		material := o.material(written.Ref, payload)
		key := "evidence/" + o.group + "/" + auditdomain.DeterministicID("receipt", grant.AllocationID, input.InvocationID, input.SubmissionID)
		links = append(links, auditstore.ArtifactLink{Scope: o.scope.Kind(), LogicalKey: key, Artifact: material.Exact(), SourceProvenance: provenance})
		if err = o.retain(ctx, "evidence", links, len(sources), sourceBytes); err != nil {
			return err
		}
		receipt = auditcontract.EvidenceReceipt{APIVersion: contracts.APIVersion, Material: material, Sources: sources, SelectedSources: selectedSources, ItemKey: input.ItemKey}
		return nil
	})
	return receipt, err
}

func replayEvidence(ctx context.Context, o *resultOwner, grant controlplane.AllocationGrant, input auditcontract.EvidenceSubmission) (auditcontract.EvidenceReceipt, bool, error) {
	if _, ok := o.members[input.ItemKey]; !ok {
		return auditcontract.EvidenceReceipt{}, false, ErrInvalid
	}
	key := "evidence/" + o.group + "/" + auditdomain.DeterministicID("receipt", grant.AllocationID, input.InvocationID, input.SubmissionID)
	link, found, err := o.readLink(ctx, key)
	if !found || err != nil {
		return auditcontract.EvidenceReceipt{}, found, err
	}
	store, _ := o.artifacts.Scoped(o.scope)
	saved, err := store.ReadExact(ctx, link.Artifact)
	if err != nil {
		return auditcontract.EvidenceReceipt{}, false, err
	}
	var record auditcontract.EvidenceRecord
	request, _ := strictjson.Canonical(input)
	expected := o.origin(grant.AllocationID, grant.StageExecutionID, grant.LogicalAgentName, input.InvocationID, input.SubmissionID, input.ItemKey)
	if strictjson.Decode(saved.Payload.Data, &record) != nil || record.Validate() != nil || !o.ownsEvidence(record) || record.RequestDigest != auditdomain.DigestBytes(request) || !reflect.DeepEqual(record.Origin, expected) {
		return auditcontract.EvidenceReceipt{}, false, auditstore.ErrConflict
	}
	return auditcontract.EvidenceReceipt{APIVersion: contracts.APIVersion, Material: o.material(saved.Ref, saved.Payload), Sources: record.Sources, SelectedSources: record.SelectedSources, ItemKey: input.ItemKey, Replayed: true}, true, nil
}
