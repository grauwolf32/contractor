package evalstore

import (
	"context"
	"errors"

	"github.com/grauwolf32/contractor/internal/evaldomain"
	pg "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/jackc/pgx/v5"
)

// purgeBatchBytes is the stored-row-byte budget of one Eval purge pass
// (LIM-EVM-20). It holds the largest row a validated write can store, about
// 10.8 MiB, so a fresh pass always fits the next row; the row budget is the
// shared CleanupBatchRows.
const purgeBatchBytes = 16 << 20

// PurgeProgress reports one bounded Eval purge pass (EVM-R243): the rows it
// deleted or rewrote, counting the rows its triggers and cascades wrote, their
// stored bytes, and whether nothing of the purged scope is left.
type PurgeProgress struct {
	Rows, Bytes int64
	Done        bool
}

func newPurgeBudget() pg.CleanupBudget {
	return pg.CleanupBudget{RowLimit: pg.CleanupBatchRows, ByteLimit: purgeBatchBytes}
}

func purgeProgress(budget pg.CleanupBudget, done bool) PurgeProgress {
	return PurgeProgress{Rows: budget.Rows, Bytes: budget.Bytes, Done: done}
}

// purgeDelete is the bounded delete of the rows of table that filter selects,
// charged at the row cost and bytes the purge declares for that table.
func purgeDelete(table, filter string, args ...any) pg.BoundedDelete {
	return pg.BoundedDelete{
		Table: table, Filter: filter, RowBytes: purgeChargedBytes(table), RowCost: purgeRowCost(table), Args: args,
	}
}

func (s *Store) BeginDeletion(ctx context.Context, scope Scope, id string, mutation evaldomain.MutationIdentity) (Receipt, error) {
	return mutate(ctx, s, scope, id, "delete", mutation, func() (ExperimentReceipt, error) {
		e, err := s.locked(ctx, scope, id)
		if err != nil {
			return ExperimentReceipt{}, err
		}
		if err = checkMutable(e, mutation); err != nil {
			return ExperimentReceipt{}, err
		}
		_, err = s.db.Exec(ctx, `UPDATE eval_experiments SET deletion_requested_at=clock_timestamp(),state='cancelling',`+advance+` WHERE experiment_id=$1`, id)
		return ExperimentReceipt{ExperimentID: id, Revision: e.Revision + 1, State: "cancelling"}, err
	})
}

// ProjectBlocked includes unresolved intents and exact Audit workspace
// dependencies. The ordinary Project controller must wait before Run purge.
func (s *Store) ProjectBlocked(ctx context.Context, owner, project string) (bool, error) {
	var blocked bool
	err := s.db.QueryRow(ctx, `
SELECT EXISTS(SELECT 1 FROM eval_experiments e WHERE e.owner_id=$1 AND e.project_id=$2 AND
    (e.outstanding_count>0 OR EXISTS(SELECT 1 FROM eval_submissions s JOIN projects p ON p.project_id=s.execution_project_id WHERE s.experiment_id=e.experiment_id)))
    OR EXISTS(SELECT 1 FROM eval_submissions s JOIN eval_experiments e USING(experiment_id) WHERE e.owner_id=$1 AND s.execution_project_id=$2 AND s.state IN ('intent','accepted'))
`, owner, project).Scan(&blocked)
	return blocked, err
}

// Purge runs one bounded pass of an experiment's purge under the Eval
// controller's claim (EVM-R243). It removes only private eval records:
// execution deletion belongs to the ordinary services, and every owned Audit
// workspace must already have drained. Nothing is removed unless the
// experiment is fenced and drained. The experiment's mutation receipts stay
// until Project purge, so retries replay instead of recreating effects
// (EVM-R226). A pass commits on its own; the caller starts the next one in a
// later transaction until Done.
func (s *Store) Purge(ctx context.Context, scope Scope, id string, claim Claim) (PurgeProgress, error) {
	if err := s.requireTx(); err != nil {
		return PurgeProgress{}, err
	}
	if _, err := s.project(ctx, scope, true); err != nil {
		return PurgeProgress{}, err
	}
	e, err := s.locked(ctx, scope, id)
	if err != nil {
		return PurgeProgress{}, err
	}
	if err = s.checkClaim(ctx, id, claim); err != nil {
		return PurgeProgress{}, err
	}
	if e.DeletionRequestedAt == nil || e.Outstanding != 0 {
		return PurgeProgress{}, ErrDrain
	}
	var blocked bool
	if err = s.db.QueryRow(ctx, purgeBlockedSQL, id).Scan(&blocked); err != nil {
		return PurgeProgress{}, err
	}
	if blocked {
		return PurgeProgress{}, ErrDrain
	}
	if err = s.allowPurge(ctx); err != nil {
		return PurgeProgress{}, err
	}
	budget := newPurgeBudget()
	done, err := s.purgeExperiment(ctx, e, &budget)
	return purgeProgress(budget, done), err
}

// PurgeProject runs one bounded pass of the private Eval data purge of a
// deleting Project (EVM-R243, DEL-R60). It runs in the Project controller's
// transaction, after its row lock and the ordinary Run and Audit purge, when
// every experiment is fenced. Its experiments, in identifier order, then its
// dataset revisions, its collection row and its Eval mutation receipts share
// one budget. Done reports that no Eval data of the Project is left.
func (s *Store) PurgeProject(ctx context.Context, scope Scope) (PurgeProgress, error) {
	if err := s.requireTx(); err != nil {
		return PurgeProgress{}, err
	}
	var lifecycle string
	err := s.db.QueryRow(ctx, `SELECT lifecycle_state FROM projects WHERE owner_id=$1 AND project_id=$2 FOR SHARE`, scope.OwnerID, scope.ProjectID).Scan(&lifecycle)
	if err != nil {
		return PurgeProgress{}, normalize(err)
	}
	if lifecycle != "deleting" {
		return PurgeProgress{}, evaldomain.Failure("eval_not_ready")
	}
	blocked, err := s.ProjectBlocked(ctx, scope.OwnerID, scope.ProjectID)
	if err != nil {
		return PurgeProgress{}, err
	}
	if blocked {
		return PurgeProgress{}, ErrDrain
	}
	var unsafe bool
	if err = s.db.QueryRow(ctx, purgeProjectSQL, scope.OwnerID, scope.ProjectID).Scan(&unsafe); err != nil {
		return PurgeProgress{}, err
	}
	if unsafe {
		return PurgeProgress{}, ErrDrain
	}
	if err = s.allowPurge(ctx); err != nil {
		return PurgeProgress{}, err
	}
	budget := newPurgeBudget()
	for {
		var id string
		err = s.db.QueryRow(ctx, nextPurgedExperimentSQL, scope.OwnerID, scope.ProjectID).Scan(&id)
		if errors.Is(err, pgx.ErrNoRows) {
			break
		}
		if err != nil {
			return PurgeProgress{}, err
		}
		e, err := s.locked(ctx, scope, id)
		if evaldomain.IsCode(err, "eval_not_found") {
			continue // An experiment purge finished it concurrently.
		}
		if err != nil {
			return PurgeProgress{}, err
		}
		if gone, err := s.purgeExperiment(ctx, e, &budget); err != nil || !gone {
			return purgeProgress(budget, false), err
		}
	}
	// Every experiment is gone, so no dataset revision is still referenced.
	for _, table := range projectPurgeTables {
		gone, err := pg.DeleteWithinBudget(ctx, s.db, &budget, purgeDelete(table,
			"owner_id = $5 AND project_id = $6", scope.OwnerID, scope.ProjectID))
		if err != nil || !gone {
			return purgeProgress(budget, false), err
		}
	}
	gone, err := pg.DeleteEvalReceiptsWithinBudget(ctx, s.db, &budget, scope.OwnerID, scope.ProjectID)
	return purgeProgress(budget, gone), err
}

// Purging reports whether a purge may already have removed rows that the
// experiment's member-level reads need: its deletion is requested and it has
// no selected view, which the first purge pass removes before anything else
// (EVM-R244). Such reads answer eval_project_deleting instead of a partial
// result.
func (s *Store) Purging(ctx context.Context, owner, id string) (bool, error) {
	var purging bool
	err := s.db.QueryRow(ctx, `
SELECT e.deletion_requested_at IS NOT NULL AND NOT EXISTS(SELECT 1
    FROM eval_projection_queue q
    WHERE q.experiment_id = e.experiment_id AND q.generation IS NOT NULL)
FROM eval_experiments e
WHERE e.owner_id = $1 AND e.experiment_id = $2`, owner, id).Scan(&purging)
	return purging, normalize(err)
}

// allowPurge lets this transaction delete the immutable Eval rows and Eval
// receipts; nothing else may.
func (s *Store) allowPurge(ctx context.Context) error {
	_, err := s.db.Exec(ctx, `SELECT set_config('contractor.eval_purge','on',true)`)
	return err
}

// purgeExperiment spends budget on one fenced, drained experiment that the
// caller has locked, leaf first, and reports whether the experiment is gone.
// It stops at the first table the budget cannot empty: no parent is deleted
// while a child row is left, so no cascade removes uncounted rows. The first
// pass removes the selected view pointer before deleting a row of that view.
// The experiment row lock and the projection queue row lock keep every writer
// of a dependent row out of the pass, and the experiment row, with its frozen
// plan, queue and claim, goes last.
func (s *Store) purgeExperiment(ctx context.Context, e Experiment, budget *pg.CleanupBudget) (bool, error) {
	var selected bool
	var queueBytes int64
	err := s.db.QueryRow(ctx, lockPurgedQueueSQL, e.ID).Scan(&selected, &queueBytes)
	switch {
	case errors.Is(err, pgx.ErrNoRows):
	case err != nil:
		return false, err
	case selected:
		if !budget.Fits(1, queueBytes) {
			return false, nil
		}
		if _, err = s.db.Exec(ctx, invalidatePurgedViewSQL, e.ID); err != nil {
			return false, err
		}
		budget.Charge(1, queueBytes)
	}
	if done, err := s.clearPurgedSelections(ctx, e.ID, budget); err != nil || !done {
		return false, err
	}
	for _, table := range experimentPurgePhases {
		gone, err := pg.DeleteWithinBudget(ctx, s.db, budget, purgeDelete(table, "experiment_id = $5", e.ID))
		if err != nil || !gone {
			return false, err
		}
	}
	var rows, bytes int64
	if err = s.db.QueryRow(ctx, finalPurgeUnitSQL, e.ID).Scan(&rows, &bytes); err != nil {
		return false, err
	}
	if !budget.Fits(rows, bytes) {
		return false, nil
	}
	if _, err = s.db.Exec(ctx, `DELETE FROM eval_experiments WHERE experiment_id=$1 AND owner_id=$2`, e.ID, e.OwnerID); err != nil {
		return false, err
	}
	budget.Charge(rows, bytes)
	return true, nil
}

// clearPurgedSelections spends the same pass budget as the deletes that follow
// it; a stopped pass resumes on the remaining selected members.
func (s *Store) clearPurgedSelections(ctx context.Context, id string, budget *pg.CleanupBudget) (bool, error) {
	remaining := budget.RowLimit - budget.Rows
	if remaining < 1 {
		return false, nil
	}
	var changed, bytes, read int64
	err := s.db.QueryRow(ctx, clearPurgedSelectionsSQL, id, remaining,
		budget.ByteLimit-budget.Bytes, budget.Rows == 0).Scan(&changed, &bytes, &read)
	if err != nil {
		return false, err
	}
	budget.Charge(changed, bytes)
	return changed == read, nil
}
