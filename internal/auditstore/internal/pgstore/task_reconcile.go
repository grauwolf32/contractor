package pgstore

import (
	"context"
	"sort"

	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/jackc/pgx/v5"
)

// readTaskReconcileSnapshot runs inside the existing bounded repeatable-read
// snapshot. Each allowed Workflow contributes at most 64 small candidate
// keys; only the first 64 keys fitting 1 MiB are hydrated. A full model lane
// never hides ready tool work behind an arbitrarily large model queue.
func (s *ClaimStore) readTaskReconcileSnapshot(ctx context.Context, result auditstore.ReconcileSnapshot) (auditstore.ReconcileSnapshot, error) {
	pinned, err := decodeAuditProgram(result.Audit)
	if err != nil {
		return auditstore.ReconcileSnapshot{}, err
	}
	if err := s.db.QueryRow(ctx, `SELECT EXISTS(SELECT 1 FROM audit_tasks WHERE audit_id=$1 AND state IN ('ready','selected'))`, result.Audit.AuditID).Scan(&result.HasUnsettledTasks); err != nil {
		return auditstore.ReconcileSnapshot{}, err
	}
	result.Executions, result.RunStates, err = s.listWindowExecutions(ctx, result.Audit.AuditID)
	if err != nil {
		return auditstore.ReconcileSnapshot{}, err
	}
	if len(result.Executions) > auditstore.MaxReconcileRows {
		result.Executions, result.MoreExecutions = result.Executions[:auditstore.MaxReconcileRows], true
	}
	if result.Audit.State != auditstore.AuditActive || result.Audit.Dispatch != auditstore.DispatchOpen {
		return result, nil
	}
	aliases := []string{}
	for alias, workflow := range pinned.Workflows {
		if result.DispatchWindowOpen(resolved.WorkflowLaneClass(workflow)) {
			aliases = append(aliases, alias)
		}
	}
	sort.Strings(aliases)
	if len(aliases) == 0 {
		return result, nil
	}
	rows, err := s.db.Query(ctx, readyTasksSQL, result.Audit.AuditID, aliases)
	if err != nil {
		return auditstore.ReconcileSnapshot{}, err
	}
	result.ReadyTasks, err = pgx.CollectRows(rows, func(row pgx.CollectableRow) (auditstore.Task, error) {
		task, err := scanTask(row)
		if err != nil {
			return auditstore.Task{}, err
		}
		if task.Document.AuditID != result.Audit.AuditID || validateTaskBinding(pinned, task.Document) != nil || task.State != auditstore.TaskReady {
			return auditstore.Task{}, errStoredTask
		}
		return task, nil
	})
	return result, err
}

// SettleUndispatchedTasks records a bounded closure gap on accepted work
// that has no current execution. Selected Tasks settle through collection.
func (s *ClaimStore) SettleUndispatchedTasks(ctx context.Context, claim auditstore.ControllerClaim, limit int) (int, error) {
	if validateClaimIdentity(claim) != nil || limit < 1 || limit > auditstore.MaxReconcileRows {
		return 0, auditstore.ErrInvalid
	}
	var authorized bool
	var count int
	err := s.db.QueryRow(ctx, settleUndispatchedTasksSQL, claim.AuditID, claim.HolderID, claim.Epoch, limit).Scan(&authorized, &count)
	if err != nil {
		return 0, writeError("settle undispatched Audit tasks", err)
	}
	if authorized {
		return count, nil
	}
	if live, err := s.claimLive(ctx, claim); err != nil {
		return 0, err
	} else if !live {
		return 0, auditstore.ErrClaimLost
	}
	return 0, auditstore.ErrPrecondition
}
