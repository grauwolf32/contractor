package postgres

import (
	"context"
	"time"

	"github.com/jackc/pgx/v5"
)

// AfterCommit schedules best-effort cleanup only after a definite commit.
// Effects share one cancellation-independent five-second budget. If it expires,
// the first unstarted effect's deferred callback reports how many effects were
// left unstarted; no later effect or deferred callback runs. A blocking backend
// call must itself honor the supplied context; admission between calls cannot
// interrupt an operating-system call already in progress.
// False means the caller owns an unwrapped transaction; it must arrange its
// own cleanup. Never run effects after rollback or an ambiguous commit error.
func AfterCommit(tx pgx.Tx, effect func(context.Context), deferred func(int)) bool {
	current, ok := tx.(*commitTx)
	if !ok {
		return false
	}
	current.effects = append(current.effects, commitEffect{run: effect, deferred: deferred})
	return true
}

type commitTx struct {
	pgx.Tx
	effects []commitEffect
}

type commitEffect struct {
	run      func(context.Context)
	deferred func(int)
}

const commitCleanupTimeout = 5 * time.Second

func (tx *commitTx) Commit(ctx context.Context) error {
	err := tx.Tx.Commit(ctx)
	if err == nil && len(tx.effects) > 0 {
		cleanup, cancel := context.WithTimeout(context.WithoutCancel(ctx), commitCleanupTimeout)
		tx.runEffects(cleanup)
		cancel()
	}
	tx.effects = nil
	return err
}

func (tx *commitTx) runEffects(ctx context.Context) {
	for index, effect := range tx.effects {
		if ctx.Err() != nil {
			if effect.deferred != nil {
				effect.deferred(len(tx.effects) - index)
			}
			return
		}
		effect.run(ctx)
	}
}

func (tx *commitTx) Rollback(ctx context.Context) error { tx.effects = nil; return tx.Tx.Rollback(ctx) }
