package evalstore

import (
	"context"

	"github.com/grauwolf32/contractor/internal/evaldomain"
)

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
	err := s.db.QueryRow(ctx, `SELECT EXISTS(SELECT 1 FROM eval_experiments e WHERE e.owner_id=$1 AND e.project_id=$2 AND
 (e.outstanding_count>0 OR EXISTS(SELECT 1 FROM eval_project_dependencies d JOIN projects p ON p.project_id=d.project_id WHERE d.experiment_id=e.experiment_id)))
 OR EXISTS(SELECT 1 FROM eval_project_dependencies d JOIN eval_submissions s USING(experiment_id,member_id) WHERE d.owner_id=$1 AND d.project_id=$2 AND s.state IN ('intent','accepted'))`, owner, project).Scan(&blocked)
	return blocked, err
}

// Purge removes only private eval records. Execution deletion belongs to the
// ordinary services and every owned Audit workspace must already have drained.
func (s *Store) Purge(ctx context.Context, scope Scope, id string) error {
	if err := s.requireTx(); err != nil {
		return err
	}
	if _, err := s.project(ctx, scope, true); err != nil {
		return err
	}
	e, err := s.locked(ctx, scope, id)
	if err != nil {
		return err
	}
	if e.DeletionRequestedAt == nil || e.Outstanding != 0 {
		return ErrDrain
	}
	var blocked bool
	err = s.db.QueryRow(ctx, `SELECT EXISTS(SELECT 1 FROM eval_suboperations WHERE experiment_id=$1 AND state='intent') OR EXISTS(SELECT 1 FROM eval_project_dependencies d JOIN projects p ON p.project_id=d.project_id WHERE d.experiment_id=$1)`, id).Scan(&blocked)
	if err != nil {
		return err
	}
	if blocked {
		return ErrDrain
	}
	if _, err = s.db.Exec(ctx, `SELECT set_config('contractor.eval_purge','on',true)`); err != nil {
		return err
	}
	if _, err = s.db.Exec(ctx, `DELETE FROM eval_selections WHERE experiment_id=$1`, id); err != nil {
		return err
	}
	// Keep mutation receipts until Project purge: retries must not recreate effects.
	_, err = s.db.Exec(ctx, `DELETE FROM eval_experiments WHERE experiment_id=$1 AND owner_id=$2`, id, scope.OwnerID)
	return err
}
func (s *Store) PurgeProject(ctx context.Context, scope Scope) error {
	if err := s.requireTx(); err != nil {
		return err
	}
	var lifecycle string
	err := s.db.QueryRow(ctx, `SELECT lifecycle_state FROM projects WHERE owner_id=$1 AND project_id=$2 FOR SHARE`, scope.OwnerID, scope.ProjectID).Scan(&lifecycle)
	if err != nil {
		return normalize(err)
	}
	if lifecycle != "deleting" {
		return evaldomain.Failure("eval_not_ready")
	}
	blocked, err := s.ProjectBlocked(ctx, scope.OwnerID, scope.ProjectID)
	if err != nil {
		return err
	}
	if blocked {
		return ErrDrain
	}
	// Final purge is invoked in the Project controller's transaction, after its
	// row lock and ordinary Run/Audit drain. All experiments are already fenced.
	var unsafe bool
	err = s.db.QueryRow(ctx, `SELECT EXISTS(SELECT 1 FROM eval_experiments WHERE owner_id=$1 AND project_id=$2 AND deletion_requested_at IS NULL) OR EXISTS(SELECT 1 FROM eval_suboperations op JOIN eval_experiments e USING(experiment_id) WHERE e.owner_id=$1 AND e.project_id=$2 AND op.state='intent')`, scope.OwnerID, scope.ProjectID).Scan(&unsafe)
	if err != nil {
		return err
	}
	if unsafe {
		return ErrDrain
	}
	if _, err = s.db.Exec(ctx, `SELECT set_config('contractor.eval_purge','on',true)`); err != nil {
		return err
	}
	if _, err = s.db.Exec(ctx, `DELETE FROM eval_selections WHERE experiment_id IN (SELECT experiment_id FROM eval_experiments WHERE owner_id=$1 AND project_id=$2)`, scope.OwnerID, scope.ProjectID); err != nil {
		return err
	}
	for _, table := range []string{"eval_experiments", "eval_dataset_revisions", "eval_mutation_receipts", "eval_collections"} {
		if _, err = s.db.Exec(ctx, `DELETE FROM `+table+` WHERE owner_id=$1 AND project_id=$2`, scope.OwnerID, scope.ProjectID); err != nil {
			return err
		}
	}
	return nil
}
