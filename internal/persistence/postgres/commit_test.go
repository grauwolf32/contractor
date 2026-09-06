package postgres

import (
	"context"
	"errors"
	"github.com/jackc/pgx/v5"
	"testing"
)

type commitResultTx struct {
	pgx.Tx
	err error
}

func (tx commitResultTx) Commit(context.Context) error   { return tx.err }
func (tx commitResultTx) Rollback(context.Context) error { return nil }

func TestCommitEffectsNeverRunAfterRollbackOrAmbiguousFailure(t *testing.T) {
	for _, outcome := range []string{"commit", "rollback", "ambiguous"} {
		t.Run(outcome, func(t *testing.T) {
			raw := commitResultTx{}
			if outcome == "ambiguous" {
				raw.err = errors.New("lost commit acknowledgement")
			}
			tx := &commitTx{Tx: raw}
			called := false
			if !AfterCommit(tx, func() { called = true }) {
				t.Fatal("callback not registered")
			}
			if outcome == "rollback" {
				_ = tx.Rollback(context.Background())
			} else {
				_ = tx.Commit(context.Background())
			}
			if called != (outcome == "commit") {
				t.Fatalf("effect called=%t", called)
			}
		})
	}
}
