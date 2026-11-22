package auditservice

import (
	"context"
	"errors"
	"reflect"
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

// JudgmentStaleError carries private authoritative generations. Runtime may use
// them only as the base of a later model response, never retry stale content.
type JudgmentStaleError struct {
	Generations []auditcontract.JudgmentGeneration
}

func (*JudgmentStaleError) Error() string { return "Audit judgment generation changed" }

func judgmentKey(executionID string, grant controlplane.AllocationGrant, input auditcontract.JudgmentSubmission) string {
	return "judgment/" + executionID + "/" + auditdomain.DeterministicID("receipt", grant.AllocationID, input.InvocationID, input.SubmissionID)
}

// RecordJudgments publishes an immutable ordinary result and its natural
// receipt atomically. Run success and all-item completion remain separate.
func (s *Service) RecordJudgments(ctx context.Context, grant controlplane.AllocationGrant, input auditcontract.JudgmentSubmission) (auditcontract.JudgmentReceipt, error) {
	if input.Validate() != nil || grant.ReadPolicy != controlplane.ReadCurrentRun {
		return auditcontract.JudgmentReceipt{}, ErrInvalid
	}
	run, err := runstore.NewPostgresStore(s.pool, s.storage).GetRun(ctx, grant.RunID)
	if err != nil {
		return auditcontract.JudgmentReceipt{}, err
	}
	if run.AuditExecutionID == nil {
		return auditcontract.JudgmentReceipt{}, ErrInvalid
	}
	reads := querystore.New(s.pool, s.storage)
	auditID, err := reads.ExecutionAuditOrigin(ctx, *run.AuditExecutionID, run.RunID)
	if err != nil {
		return auditcontract.JudgmentReceipt{}, err
	}
	if receipt, found, err := replayJudgment(ctx, grant, input, run, auditID, reads, artifacts.NewService(s.storage.Repository(s.pool))); found || err != nil {
		return receipt, err
	}
	if grant.WriteFenced || grant.Lost || grant.WritePolicy != controlplane.WriteInputsAndIntermediates {
		return auditcontract.JudgmentReceipt{}, auditstore.ErrPrecondition
	}
	var receipt auditcontract.JudgmentReceipt
	err = postgres.InTxWithRetry(ctx, s.pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		receipt = auditcontract.JudgmentReceipt{}
		if _, err := runstore.NewPostgresTxStore(tx, s.storage).LockRunState(ctx, grant.RunID, runstore.RunRunning); err != nil {
			return err
		}
		runs := runstore.NewPostgresStore(tx, s.storage)
		run, err := runs.GetRun(ctx, grant.RunID)
		if err != nil {
			return err
		}
		stage, err := runs.GetStageExecution(ctx, grant.StageExecutionID)
		if err != nil {
			return err
		}
		if run.AuditExecutionID == nil || run.AuditCompletion == nil || stage.RunID != run.RunID || run.AuditCompletion.Stage != stage.StageName || run.AuditCompletion.Agent != grant.LogicalAgentName || run.PublicationMode != runstore.PublicationAuditManaged {
			return ErrInvalid
		}
		spec, err := resolved.DecodeResolvedStageSnapshot(stage.StageSpecSnapshot)
		if err != nil {
			return err
		}
		binding, ok := spec.Agents[grant.LogicalAgentName]
		name := "submit_result"
		if input.Mode == "batch" {
			name = "submit_results"
		}
		if !ok || binding.Namespace != grant.Namespace || !slices.ContainsFunc(binding.Template.Toolsets, func(sel contracts.ToolsetSelection) bool {
			return sel.Ref.ToolsetID == "audit-results" && sel.Ref.Version == "2" && slices.Contains(sel.Tools, name)
		}) {
			return ErrInvalid
		}
		reads := querystore.New(tx, s.storage)
		actualAuditID, err := reads.ExecutionAuditOrigin(ctx, *run.AuditExecutionID, run.RunID)
		if err != nil {
			return err
		}
		owners := ownerstore.NewTx(tx, s.storage)
		audit, err := owners.LockTaskInputs(ctx, run.OwnerID, actualAuditID, nil)
		if err != nil {
			return err
		}
		if run.ProjectID == nil || *run.ProjectID != audit.ProjectID {
			return ErrInvalid
		}
		service := artifacts.NewService(s.storage.TxRepository(tx))
		if saved, found, err := replayJudgment(ctx, grant, input, run, audit.AuditID, reads, service); found || err != nil {
			receipt = saved
			return err
		}
		tasks, err := reads.ListExecutionTasks(ctx, audit.AuditID, *run.AuditExecutionID)
		if err != nil {
			return err
		}
		if input.Mode == "batch" && len(input.Items) != len(tasks) {
			return ErrInvalid
		}
		members := map[string]auditstore.Task{}
		for index, task := range tasks {
			members[auditcontract.ExecutionItemKey(index)] = task
		}
		for _, update := range input.Items {
			if _, ok := members[update.Conclusion.ItemKey]; !ok {
				return ErrInvalid
			}
		}
		latest, err := loadJudgments(ctx, reads, service, audit.AuditID, *run.AuditExecutionID, run.RunID)
		if err != nil {
			return err
		}
		normalized, err := normalizeJudgments(ctx, reads, service, audit, run, members, input.Items)
		if err != nil {
			return err
		}
		generations := make([]auditcontract.JudgmentGeneration, 0, len(input.Items))
		stale := false
		for index, update := range input.Items {
			key := update.Conclusion.ItemKey
			prior, exists := latest[key]
			current := uint64(0)
			if exists {
				if prior.Generation.TaskID != members[key].Document.TaskID {
					return ErrInvalid
				}
				current = prior.Generation.Generation
			}
			same := exists && reflect.DeepEqual(prior.Item.Conclusion, update.Conclusion)
			if same {
				normalized.Items[index] = prior.Item
			} else {
				if input.Mode == "batch" && exists {
					return auditstore.ErrConflict
				}
				if current != update.BaseGeneration {
					stale = true
				}
				if current == auditcontract.MaximumResultGeneration {
					return auditstore.ErrEvidenceBudgetExhausted
				}
			}
			generation := current
			if !same {
				generation++
			}
			generations = append(generations, auditcontract.JudgmentGeneration{ItemKey: key, TaskID: members[key].Document.TaskID, Generation: generation})
		}
		if stale {
			current := make([]auditcontract.JudgmentGeneration, 0, len(input.Items))
			for _, update := range input.Items {
				value := auditcontract.JudgmentGeneration{ItemKey: update.Conclusion.ItemKey, TaskID: members[update.Conclusion.ItemKey].Document.TaskID}
				if prior, ok := latest[update.Conclusion.ItemKey]; ok {
					value.Generation = prior.Generation.Generation
				}
				current = append(current, value)
			}
			return &JudgmentStaleError{Generations: current}
		}
		data, err := strictjson.Canonical(normalized)
		if err != nil {
			return err
		}
		if err := owners.AdmitEvidenceCollection(ctx, audit.AuditID, *run.AuditExecutionID, 0, int64(len(data))); err != nil {
			return err
		}
		request, _ := strictjson.Canonical(input)
		metadata := auditcontract.JudgmentMutation{Schema: "contractor.audit.result-receipt.v1", RequestDigest: auditdomain.DigestBytes(request), AuditID: audit.AuditID, ExecutionID: *run.AuditExecutionID, RunID: run.RunID, AllocationID: grant.AllocationID, StageExecutionID: grant.StageExecutionID, LogicalAgentName: grant.LogicalAgentName, InvocationID: input.InvocationID, SubmissionID: input.SubmissionID, Generations: generations}
		provenance, err := strictjson.Canonical(metadata)
		if err != nil || metadata.Validate() != nil {
			return ErrInvalid
		}
		scope, _ := artifacts.AuditScope(audit.AuditID)
		ref := contracts.ArtifactRef{Namespace: auditdomain.ArtifactNamespace(audit.AuditID), Name: auditdomain.DeterministicID("judgment", grant.AllocationID, input.InvocationID, input.SubmissionID)}
		payload := artifacts.Payload{MediaType: "application/json", Data: data}
		written, err := service.CreateFrozen(ctx, scope, ref, payload)
		if err != nil {
			return err
		}
		material := materialFromWrite(audit.AuditID, written.Ref, payload)
		link := auditstore.ArtifactLink{Scope: artifacts.ScopeAudit, LogicalKey: judgmentKey(*run.AuditExecutionID, grant, input), Artifact: material.Exact(), SourceProvenance: provenance}
		if err := owners.RetainJudgmentLinks(ctx, audit, []auditstore.ArtifactLink{link}); err != nil {
			return err
		}
		receipt = auditcontract.JudgmentReceipt{APIVersion: contracts.APIVersion, Material: material, Generations: generations}
		return nil
	})
	return receipt, err
}

func replayJudgment(ctx context.Context, grant controlplane.AllocationGrant, input auditcontract.JudgmentSubmission, run runstore.WorkflowRun, auditID string, reads *querystore.Store, service *artifacts.Service) (auditcontract.JudgmentReceipt, bool, error) {
	link, err := reads.GetArtifactLink(ctx, auditID, judgmentKey(*run.AuditExecutionID, grant, input))
	if errors.Is(err, auditstore.ErrNotFound) {
		return auditcontract.JudgmentReceipt{}, false, nil
	}
	if err != nil {
		return auditcontract.JudgmentReceipt{}, false, err
	}
	scope, _ := artifacts.AuditScope(auditID)
	owned, _ := service.Scoped(scope)
	read, err := owned.ReadExact(ctx, link.Artifact)
	if err != nil {
		return auditcontract.JudgmentReceipt{}, false, err
	}
	metadata, _, err := auditcontract.DecodeJudgment(read.Payload.Data, link.SourceProvenance)
	request, _ := strictjson.Canonical(input)
	if err != nil || metadata.RequestDigest != auditdomain.DigestBytes(request) || metadata.AuditID != auditID || metadata.ExecutionID != *run.AuditExecutionID || metadata.RunID != run.RunID || metadata.AllocationID != grant.AllocationID || metadata.StageExecutionID != grant.StageExecutionID || metadata.LogicalAgentName != grant.LogicalAgentName || metadata.InvocationID != input.InvocationID || metadata.SubmissionID != input.SubmissionID {
		return auditcontract.JudgmentReceipt{}, false, auditstore.ErrConflict
	}
	return auditcontract.JudgmentReceipt{APIVersion: contracts.APIVersion, Material: materialFromWrite(auditID, read.Ref, read.Payload), Generations: metadata.Generations, Replayed: true}, true, nil
}

func loadJudgments(ctx context.Context, reads *querystore.Store, service *artifacts.Service, auditID, executionID, runID string) (map[string]auditcontract.AcceptedJudgment, error) {
	links, err := reads.ListExecutionJudgments(ctx, auditID, executionID)
	if err != nil {
		return nil, err
	}
	scope, _ := artifacts.AuditScope(auditID)
	owned, _ := service.Scoped(scope)
	records := make([]auditcontract.StoredJudgment, 0, len(links))
	var total int64
	for _, link := range links {
		total += link.Artifact.SizeBytes
		if total > auditcontract.MaximumDocumentBytes {
			return nil, ErrInvalid
		}
		read, err := owned.ReadExact(ctx, link.Artifact)
		if err != nil {
			return nil, err
		}
		records = append(records, auditcontract.StoredJudgment{Material: materialFromWrite(auditID, read.Ref, read.Payload), Provenance: link.SourceProvenance, Data: read.Payload.Data})
	}
	return auditcontract.LatestJudgments(records, auditID, executionID, runID)
}

func normalizeJudgments(ctx context.Context, reads *querystore.Store, service *artifacts.Service, audit auditstore.Audit, run runstore.WorkflowRun, members map[string]auditstore.Task, updates []auditcontract.JudgmentUpdate) (auditcontract.ResultDocument, error) {
	links, err := reads.ListExecutionEvidence(ctx, audit.AuditID, *run.AuditExecutionID)
	if err != nil {
		return auditcontract.ResultDocument{}, err
	}
	scope, _ := artifacts.AuditScope(audit.AuditID)
	owned, _ := service.Scoped(scope)
	evidence := map[string][]auditcontract.Material{}
	var total int64
	for _, link := range links {
		total += link.Artifact.SizeBytes
		if total > auditcontract.MaximumDocumentBytes {
			return auditcontract.ResultDocument{}, ErrInvalid
		}
		read, err := owned.ReadExact(ctx, link.Artifact)
		if err != nil {
			return auditcontract.ResultDocument{}, err
		}
		var record auditcontract.EvidenceRecord
		if strictjson.Decode(read.Payload.Data, &record) != nil || record.Validate() != nil || record.Origin.AuditID != audit.AuditID || record.Origin.ExecutionID != *run.AuditExecutionID || record.Origin.RunID != run.RunID {
			return auditcontract.ResultDocument{}, ErrInvalid
		}
		evidence[record.Origin.TaskID] = append(evidence[record.Origin.TaskID], record.Sources...)
		evidence[record.Origin.TaskID] = append(evidence[record.Origin.TaskID], materialFromWrite(audit.AuditID, read.Ref, read.Payload))
	}
	contextItems := make([]map[string]any, 0, len(updates))
	items := make([]auditcontract.Conclusion, 0, len(updates))
	for _, update := range updates {
		task := members[update.Conclusion.ItemKey]
		allowed := []auditcontract.Material{}
		seen := map[string]bool{}
		for _, material := range evidence[task.Document.TaskID] {
			key, _ := strictjson.Canonical(material)
			if !seen[string(key)] {
				allowed = append(allowed, material)
				seen[string(key)] = true
			}
		}
		member := map[string]any{"item_key": update.Conclusion.ItemKey, "task_id": task.Document.TaskID, "objective": task.Document.Objective, "evidence": allowed}
		if task.Hypothesis != nil {
			member["hypothesis"] = task.Hypothesis
		}
		contextItems = append(contextItems, member)
		items = append(items, update.Conclusion)
	}
	digest, err := strictjson.Canonical(run.WorkflowSnapshot)
	if err != nil {
		return auditcontract.ResultDocument{}, err
	}
	trusted, _ := strictjson.Canonical(map[string]any{"schema": "contractor.audit.context.v1", "audit_id": audit.AuditID, "run_id": run.RunID, "workflow_ref": run.WorkflowName + "@" + run.WorkflowVersion, "workflow_digest": auditdomain.DigestBytes(digest), "items": contextItems})
	raw, _ := strictjson.Canonical(map[string]any{"schema": "contractor.audit.conclusions.v1", "items": items})
	data, err := auditcontract.Decode(raw, trusted)
	var result auditcontract.ResultDocument
	if err != nil || strictjson.Decode(data, &result) != nil {
		return result, ErrInvalid
	}
	return result, nil
}
