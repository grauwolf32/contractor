package pgstore

import (
	"context"
	"errors"
	"fmt"
	"slices"

	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/auditcontract"
	"github.com/grauwolf32/contractor/internal/helpers/contentdigest"
	"github.com/grauwolf32/contractor/internal/helpers/strictjson"
	postgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/grauwolf32/contractor/internal/settingsstore"
	"github.com/jackc/pgx/v5"
)

type taskExecutionMember struct {
	ID      string `json:"id"`
	TaskID  string `json:"task_id"`
	Ordinal int    `json:"ordinal"`
	Attempt int    `json:"attempt"`
}

// SelectTasks reserves an ordinary execution. This mutation derives attempts,
// identity, compatibility and lane from immutable accepted work. An ambiguous
// reservation is recovered from the same selected tasks, not a new Run slot.
func (s *ClaimStore) SelectTasks(ctx context.Context, p auditstore.SelectTasksParams) (auditstore.Execution, bool, error) {
	if validateClaimIdentity(p.Claim) != nil || len(p.TaskIDs) < 1 || len(p.TaskIDs) > auditstore.MaxCollectionItems {
		return auditstore.Execution{}, false, auditstore.ErrInvalid
	}
	p.TaskIDs = slices.Clone(p.TaskIDs)
	seen := map[string]bool{}
	for _, id := range p.TaskIDs {
		if !contracts.ValidResourceID(id) || seen[id] {
			return auditstore.Execution{}, false, auditstore.ErrInvalid
		}
		seen[id] = true
	}
	var execution auditstore.Execution
	var created bool
	err := postgres.Within(ctx, s.db, pgx.TxOptions{}, func(tx pgx.Tx) error {
		settings, err := settingsstore.NewPostgresTxStore(tx).LockSchedulerSettings(ctx)
		if err != nil {
			return err
		}
		var owner, project string
		if err := tx.QueryRow(ctx, `SELECT owner_id,project_id FROM audits WHERE audit_id=$1`, p.Claim.AuditID).Scan(&owner, &project); err != nil {
			if errors.Is(err, pgx.ErrNoRows) {
				return auditstore.ErrNotFound
			}
			return err
		}
		if _, err := projectstore.NewPostgresStore(tx).LockActiveAuditProject(ctx, owner, project); err != nil {
			if errors.Is(err, projectstore.ErrDeleting) {
				return auditstore.ErrProjectDeleting
			}
			return err
		}
		if err := auditClaims.Lock(ctx, tx, p.Claim.AuditID, p.Claim.HolderID, int64(p.Claim.Epoch)); err != nil {
			if errors.Is(err, postgres.ErrControllerClaimLost) {
				return auditstore.ErrClaimLost
			}
			return err
		}
		audit, err := scanAudit(tx.QueryRow(ctx, `SELECT `+prefixedAuditColumns("audit")+` FROM audits AS audit WHERE audit_id=$1 FOR UPDATE`, p.Claim.AuditID))
		if err != nil {
			return err
		}
		pinned, err := decodeAuditProgram(audit)
		if err != nil {
			return err
		}
		tasks, err := loadSelectedTasks(ctx, tx, audit, p.TaskIDs)
		if err != nil {
			return err
		}
		if tasks[0].State == auditstore.TaskSelected {
			prior := tasks[0].LastExecution
			if prior == nil {
				return errStoredTask
			}
			for _, task := range tasks {
				if task.State != auditstore.TaskSelected || task.LastExecution == nil || *task.LastExecution != *prior {
					return auditstore.ErrPrecondition
				}
			}
			execution, err = NewClaimStore(tx).getExecution(ctx, audit.AuditID, *prior)
			if err != nil {
				return err
			}
			members, err := taskExecutionIDs(ctx, tx, audit.AuditID, execution.ExecutionID)
			if err != nil {
				return err
			}
			expected := slices.Clone(p.TaskIDs)
			slices.Sort(expected)
			slices.Sort(members)
			if !slices.Equal(members, expected) || execution.Role != auditstore.ExecutionTask {
				return auditstore.ErrConflict
			}
			return nil
		}
		if audit.State != auditstore.AuditActive || audit.Dispatch != auditstore.DispatchOpen || len(tasks) > audit.Limits.BatchSize {
			return auditstore.ErrPrecondition
		}
		first := tasks[0]
		binding := pinned.Program.Workflows[first.Document.Workflow]
		closure := pinned.Workflows[first.Document.Workflow]
		contextInput := false
		for _, mapping := range binding.Inputs {
			contextInput = contextInput || mapping.Source == "task-context"
		}
		if len(tasks) > 1 && !contextInput {
			return auditstore.ErrInvalid
		}
		members := make([]taskExecutionMember, 0, len(tasks))
		for i, task := range tasks {
			if task.State != auditstore.TaskReady || task.Attempts >= binding.AttemptLimit(audit.Limits.MaxItemRunAttempts) || task.BatchDigest != first.BatchDigest || validateTaskBinding(pinned, task.Document) != nil {
				return auditstore.ErrPrecondition
			}
			members = append(members, taskExecutionMember{TaskID: task.Document.TaskID, Ordinal: i, Attempt: task.Attempts + 1})
		}
		lane := resolved.WorkflowLaneClass(closure)
		if lane == resolved.LaneClassTool && audit.OutstandingToolRunCount >= settings.MaxConcurrentToolRuns || lane == resolved.LaneClassModel && audit.OutstandingRunCount-audit.OutstandingToolRunCount >= settings.MaxConcurrentRuns || lane != resolved.LaneClassTool && lane != resolved.LaneClassModel {
			return auditstore.ErrPrecondition
		}
		digest, err := contentdigest.CanonicalJSON(map[string]any{"audit_id": audit.AuditID, "profile_digest": pinned.Ref.Digest, "workflow": first.Document.Workflow, "members": members})
		if err != nil {
			return err
		}
		id := auditdomain.DeterministicID("execution", audit.AuditID, digest)
		for i := range members {
			members[i].ID = auditdomain.DeterministicID("attempt", id, members[i].TaskID)
		}
		execution, err = ScanExecution(tx.QueryRow(ctx, insertTaskExecutionSQL+ExecutionColumns("audit_executions"), id, audit.AuditID, first.Document.Workflow, "audit-"+id, digest, string(lane)))
		if err != nil {
			return err
		}
		encoded, err := strictjson.Canonical(members)
		if err != nil {
			return err
		}
		if _, err = tx.Exec(ctx, insertTaskMembersSQL, id, audit.AuditID, encoded); err != nil {
			return err
		}
		var ignored uint64
		if err = tx.QueryRow(ctx, reserveTaskRunSQL, audit.AuditID, id, string(lane), len(tasks), first.Document.Workflow).Scan(&ignored); errors.Is(err, pgx.ErrNoRows) {
			return auditstore.ErrPrecondition
		} else if err != nil {
			return err
		}
		changed, err := tx.Exec(ctx, selectTaskBatchSQL, audit.AuditID, p.TaskIDs, id)
		if err != nil {
			return err
		}
		if changed.RowsAffected() != int64(len(tasks)) {
			return auditstore.ErrConflict
		}
		if live, err := auditClaims.Live(ctx, tx, audit.AuditID, p.Claim.HolderID, int64(p.Claim.Epoch)); err != nil {
			return err
		} else if !live {
			return auditstore.ErrClaimLost
		}
		created = true
		return nil
	})
	if err != nil {
		return auditstore.Execution{}, false, writeError("select Audit tasks", err)
	}
	return execution, created, nil
}

func loadSelectedTasks(ctx context.Context, db postgres.DBTX, audit auditstore.Audit, ids []string) ([]auditstore.Task, error) {
	var count int
	var bytes int64
	if err := db.QueryRow(ctx, taskSelectionSizeSQL, audit.AuditID, ids).Scan(&count, &bytes); err != nil {
		return nil, err
	}
	if count != len(ids) {
		return nil, auditstore.ErrNotFound
	}
	if bytes > auditcontract.MaxMutationBytes {
		return nil, auditstore.ErrInvalid
	}
	rows, err := db.Query(ctx, selectedTasksSQL, audit.AuditID, ids)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	tasks := make([]auditstore.Task, 0, count)
	for rows.Next() {
		task, err := scanTask(rows)
		if err != nil {
			return nil, err
		}
		tasks = append(tasks, task)
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	if len(tasks) != count {
		return nil, auditstore.ErrConflict
	}
	return tasks, nil
}

func taskExecutionIDs(ctx context.Context, db postgres.DBTX, auditID, executionID string) ([]string, error) {
	rows, err := db.Query(ctx, taskExecutionMembersSQL, auditID, executionID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	ids := []string{}
	for rows.Next() {
		var id string
		if err := rows.Scan(&id); err != nil {
			return nil, err
		}
		ids = append(ids, id)
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	if len(ids) < 1 || len(ids) > auditstore.MaxCollectionItems {
		return nil, fmt.Errorf("%w: invalid Task execution membership", errStoredTask)
	}
	return ids, nil
}
