package evalstore

import (
	"context"
	"github.com/grauwolf32/contractor/internal/evaldomain"
)

type ExecutionTombstone struct {
	Kind, ID, TerminalState string
	NeverStarted            bool
}

func (s *Store) Tombstone(ctx context.Context, owner, id, member string) (ExecutionTombstone, error) {
	var t ExecutionTombstone
	err := s.db.QueryRow(ctx, `SELECT t.execution_kind,t.execution_id,t.terminal_state,t.never_started FROM eval_execution_tombstones t JOIN eval_experiments e USING(experiment_id) WHERE e.owner_id=$1 AND e.experiment_id=$2 AND t.member_id=$3`, owner, id, member).Scan(&t.Kind, &t.ID, &t.TerminalState, &t.NeverStarted)
	return t, normalize(err)
}
func (s *Store) BindTombstone(ctx context.Context, scope Scope, id, member string, claim Claim) error {
	if _, err := s.recovery(ctx, scope, id, claim); err != nil {
		return err
	}
	t, err := s.Tombstone(ctx, scope.OwnerID, id, member)
	if err != nil {
		return err
	}
	if t.NeverStarted {
		return evaldomain.Failure("eval_not_ready")
	}
	tag, err := s.db.Exec(ctx, `UPDATE eval_submissions SET state='accepted',execution_id=$3,updated_at=clock_timestamp() WHERE experiment_id=$1 AND member_id=$2 AND (state='intent' OR (state='accepted' AND execution_id=$3))`, id, member, t.ID)
	if err == nil && tag.RowsAffected() != 1 {
		return evaldomain.Failure("eval_member_conflict")
	}
	return normalize(err)
}
