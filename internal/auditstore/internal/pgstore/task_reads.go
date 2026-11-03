package pgstore

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"

	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/config/auditprofile"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/auditcontract"
	"github.com/grauwolf32/contractor/internal/helpers/contentdigest"
	"github.com/grauwolf32/contractor/internal/helpers/pagelimit"
	"github.com/grauwolf32/contractor/internal/helpers/strictjson"
	"github.com/jackc/pgx/v5"
)

func scanTask(row scanner, extra ...any) (auditstore.Task, error) {
	var task auditstore.Task
	var document, source, result json.RawMessage
	values := []any{&document, &task.SourceKey, &source, &task.Ordinal, &task.Iteration, &task.BatchDigest, &task.State, &task.Attempts, &task.LastExecution, &result, &task.GapCode, &task.CreatedAt, &task.UpdatedAt}
	if err := row.Scan(append(values, extra...)...); err != nil {
		return auditstore.Task{}, err
	}
	var err error
	task.Document, err = auditcontract.DecodeTask(document)
	if err != nil || strictjson.Decode(source, &task.Source) != nil || !validTaskSource(task.Source) || task.Ordinal < 1 || task.Iteration < 1 || task.Attempts < 0 || task.State != auditstore.TaskReady && task.State != auditstore.TaskSelected && task.State != auditstore.TaskSettled {
		return auditstore.Task{}, errStoredTask
	}
	sourceKey, err := contentdigest.CanonicalJSON(task.Source)
	batch, batchErr := taskBatchDigest(task.Document)
	if err != nil || batchErr != nil || sourceKey != task.SourceKey || batch != task.BatchDigest {
		return auditstore.Task{}, errStoredTask
	}
	if result != nil {
		task.Result = &auditcontract.Material{}
		if strictjson.Decode(result, task.Result) != nil || task.Result.ScopeKind != "audit" || task.Result.ScopeID != task.Document.AuditID {
			return auditstore.Task{}, errStoredTask
		}
		exact := contracts.ExactArtifact{Ref: task.Result.Ref, Digest: task.Result.Digest, MediaType: task.Result.MediaType, SizeBytes: task.Result.SizeBytes}
		if !exact.HasMetadata() || exact.Validate() != nil {
			return auditstore.Task{}, errStoredTask
		}
	}
	for _, material := range task.Document.Inputs {
		if material.ScopeKind != "audit" || material.ScopeID != task.Document.AuditID {
			return auditstore.Task{}, errStoredTask
		}
	}
	if task.Ordinal > auditprofile.MaxAuditItemsTotal || task.Iteration > auditprofile.MaxAuditRounds || task.Attempts > auditprofile.MaxAuditRunAttempts || task.LastExecution != nil && !contracts.ValidResourceID(*task.LastExecution) || task.Attempts > 0 && task.LastExecution == nil || task.Attempts == 0 && task.LastExecution != nil || task.State == auditstore.TaskSelected && task.Attempts == 0 || task.State != auditstore.TaskSettled && (task.Result != nil || task.GapCode != nil) || task.GapCode != nil && validateText("stored Task gap", *task.GapCode, 128, true) != nil || task.State == auditstore.TaskSettled && task.Result == nil && task.GapCode == nil {
		return auditstore.Task{}, errStoredTask
	}
	return task, nil
}

// ListTasks returns a row/byte bounded keyset page. Its continuation belongs
// to this page, including when the byte budget selects a shorter prefix.
func (s *QueryStore) ListTasks(ctx context.Context, p auditstore.ListTasksParams) (auditstore.TaskPage, error) {
	if !pagelimit.ValidPage(p.Limit) || p.AfterOrdinal < 0 || p.State != nil && *p.State != auditstore.TaskReady && *p.State != auditstore.TaskSelected && *p.State != auditstore.TaskSettled {
		return auditstore.TaskPage{}, auditstore.ErrInvalid
	}
	audit, err := s.get(ctx, p.OwnerID, p.AuditID)
	if err != nil {
		return auditstore.TaskPage{}, err
	}
	pinned, err := decodeAuditProgram(audit)
	if err != nil {
		return auditstore.TaskPage{}, err
	}
	filter := taskStateFilterSQL(p.State)
	arguments := []any{p.AuditID, p.AfterOrdinal, pagelimit.FetchRows(p.Limit)}
	rows, err := s.db.Query(ctx, fmt.Sprintf(taskPageSQL, filter), arguments...)
	if err != nil {
		return auditstore.TaskPage{}, err
	}
	defer rows.Close()
	page := auditstore.TaskPage{Tasks: make([]auditstore.Task, 0, p.Limit)}
	var candidates int
	for rows.Next() {
		task, err := scanTask(rows, &candidates)
		if err != nil || task.Document.AuditID != p.AuditID {
			if err == nil {
				err = errStoredTask
			}
			return auditstore.TaskPage{}, err
		}
		if err := validateTaskBinding(pinned, task.Document); err != nil || task.Ordinal > pinned.Program.Limits.MaxTasks || task.Iteration > pinned.Program.Limits.MaxIterations || task.Attempts > pinned.Program.Limits.MaxAttempts {
			return auditstore.TaskPage{}, errStoredTask
		}
		page.Tasks = append(page.Tasks, task)
	}
	if err := rows.Err(); err != nil {
		return auditstore.TaskPage{}, err
	}
	if candidates > len(page.Tasks) || len(page.Tasks) > p.Limit {
		if len(page.Tasks) == 0 {
			return auditstore.TaskPage{}, errStoredTask
		}
		if len(page.Tasks) > p.Limit {
			page.Tasks = page.Tasks[:p.Limit]
		}
		ordinal := page.Tasks[len(page.Tasks)-1].Ordinal
		page.NextOrdinal = &ordinal
	}
	return page, nil
}

func (s *QueryStore) GetTask(ctx context.Context, ownerID, auditID, taskID string) (auditstore.Task, error) {
	audit, err := s.get(ctx, ownerID, auditID)
	if err != nil {
		return auditstore.Task{}, err
	}
	pinned, err := decodeAuditProgram(audit)
	if err != nil {
		return auditstore.Task{}, err
	}
	task, err := scanTask(s.db.QueryRow(ctx, getTaskSQL, auditID, taskID))
	if errors.Is(err, pgx.ErrNoRows) {
		return auditstore.Task{}, auditstore.ErrNotFound
	}
	if err != nil {
		return auditstore.Task{}, err
	}
	if task.Document.AuditID != auditID || task.Document.TaskID != taskID || validateTaskBinding(pinned, task.Document) != nil || task.Ordinal > pinned.Program.Limits.MaxTasks || task.Iteration > pinned.Program.Limits.MaxIterations || task.Attempts > pinned.Program.Limits.MaxAttempts {
		return auditstore.Task{}, errStoredTask
	}
	return task, nil
}

// ListExecutionTasks is the bounded immutable membership read used by trusted
// collection and dispatch callers. Public reads authorize the Audit owner.
func (s *QueryStore) ListExecutionTasks(ctx context.Context, auditID, executionID string) ([]auditstore.Task, error) {
	if !contracts.ValidResourceID(auditID) || !contracts.ValidResourceID(executionID) {
		return nil, auditstore.ErrInvalid
	}
	ids, err := taskExecutionIDs(ctx, s.db, auditID, executionID)
	if err != nil {
		return nil, err
	}
	return loadSelectedTasks(ctx, s.db, auditstore.Audit{AuditID: auditID}, ids)
}

func (s *QueryStore) CollectionReceipt(ctx context.Context, auditID, executionID string) (auditstore.CollectionReceipt, error) {
	if !contracts.ValidResourceID(auditID) || !contracts.ValidResourceID(executionID) {
		return auditstore.CollectionReceipt{}, auditstore.ErrInvalid
	}
	receipt, err := scanReceipt(s.db.QueryRow(ctx, `SELECT `+collectionReceiptProjection("execution")+` FROM audit_executions AS execution WHERE audit_id=$1 AND execution_id=$2 AND collection_receipt_id IS NOT NULL`, auditID, executionID))
	if errors.Is(err, pgx.ErrNoRows) {
		return auditstore.CollectionReceipt{}, auditstore.ErrNotFound
	}
	return receipt, err
}
