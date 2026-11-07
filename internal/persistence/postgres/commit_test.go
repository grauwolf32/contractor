package postgres

import (
	"context"
	"errors"
	"testing"

	"github.com/jackc/pgx/v5"
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
			if !AfterCommit(tx, func(context.Context) { called = true }, nil) {
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

func TestCommitCleanupSurvivesCallerCancellation(t *testing.T) {
	caller, cancel := context.WithCancel(t.Context())
	cancel()
	tx := &commitTx{Tx: commitResultTx{}}
	called := false
	AfterCommit(tx, func(cleanup context.Context) {
		called = true
		if cleanup.Err() != nil {
			t.Fatal("definite commit cleanup inherited caller cancellation")
		}
		if _, bounded := cleanup.Deadline(); !bounded {
			t.Fatal("cleanup has no operation budget")
		}
	}, nil)
	if err := tx.Commit(caller); err != nil || !called {
		t.Fatalf("definite commit: called=%t err=%v", called, err)
	}
}

func TestCommitCleanupStopsAtSharedBudget(t *testing.T) {
	// Advancing this injected budget after three effects is deterministic;
	// no wall-clock delay or timing threshold is involved.
	budget, exhaust := context.WithCancel(t.Context())
	defer exhaust()
	tx := &commitTx{}
	called, diagnostics, deferred := 0, 0, 0
	const candidates = 1000
	for range candidates {
		AfterCommit(tx, func(ctx context.Context) {
			if ctx != budget {
				t.Fatal("each effect must use the shared budget")
			}
			called++
			if called == 3 {
				exhaust()
			}
		}, func(remaining int) {
			diagnostics++
			deferred = remaining
		})
	}
	tx.runEffects(budget)
	if called != 3 || diagnostics != 1 || deferred != candidates-called {
		t.Fatalf("effects=%d diagnostics=%d deferred=%d", called, diagnostics, deferred)
	}
	t.Logf("shared budget admitted %d of %d effects; one diagnostic deferred %d", called, candidates, deferred)
}
