package pgstore

import (
	"context"
	"strings"
	"sync"
	"testing"

	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/settingsstore"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

// commitAfterFirstStatement runs an armed change, from another connection,
// right after the first statement that is not a BEGIN completes on the traced
// pool. Armed before a reconcile read, the change commits between the
// snapshot's first statement and the rest of it.
type commitAfterFirstStatement struct {
	mu     sync.Mutex
	change func(context.Context) error
	fired  bool
	err    error
}

type tracedStatement struct{}

func (c *commitAfterFirstStatement) arm(change func(context.Context) error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.change, c.fired, c.err = change, false, nil
}

func (c *commitAfterFirstStatement) result() (bool, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.fired, c.err
}

func (c *commitAfterFirstStatement) TraceQueryStart(
	ctx context.Context, _ *pgx.Conn, data pgx.TraceQueryStartData,
) context.Context {
	if strings.HasPrefix(strings.ToLower(strings.TrimSpace(data.SQL)), "begin") {
		return ctx
	}
	return context.WithValue(ctx, tracedStatement{}, true)
}

func (c *commitAfterFirstStatement) TraceQueryEnd(ctx context.Context, _ *pgx.Conn, data pgx.TraceQueryEndData) {
	if ctx.Value(tracedStatement{}) == nil || data.Err != nil {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.change == nil {
		return
	}
	change := c.change
	c.change = nil
	c.fired, c.err = true, change(context.WithoutCancel(ctx))
}

// TestPostgresReconcileSnapshotReadsOneCommitOnAPoolAndInACallerTransaction
// commits a Scheduler settings change from another connection right after
// the snapshot's first statement. On a pool the snapshot is one read-only
// REPEATABLE READ transaction, so its dispatch window still shows the value
// from before the change (AUX-R95); in a caller's REPEATABLE READ transaction
// it runs inside that transaction and sees the same (PER-R18). Statements
// that each took their own snapshot would report the new value.
func TestPostgresReconcileSnapshotReadsOneCommitOnAPoolAndInACallerTransaction(t *testing.T) {
	f := newCollectingAuditFixture(t, "snapshotcommit", 1<<20)
	settings := settingsstore.NewPostgresStore(f.pool)
	trace := &commitAfterFirstStatement{}
	configuration := f.pool.Config()
	configuration.ConnConfig.Tracer = trace
	traced, err := pgxpool.NewWithConfig(f.ctx, configuration)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(traced.Close)
	changeWindow := func(ctx context.Context) error {
		current, err := settings.GetSchedulerSettings(ctx)
		if err != nil {
			return err
		}
		_, err = settings.UpdateSchedulerSettings(ctx, settingsstore.UpdateSchedulerSettingsParams{
			MaxConcurrentRuns: current.MaxConcurrentRuns + 1, ExpectedRevision: current.Revision,
		})
		return err
	}
	window := func(t *testing.T) int {
		t.Helper()
		current, err := settings.GetSchedulerSettings(f.ctx)
		if err != nil {
			t.Fatal(err)
		}
		return current.MaxConcurrentRuns
	}
	requireOneCommit := func(t *testing.T, before int, snapshot auditstore.ReconcileSnapshot, err error) {
		t.Helper()
		if err != nil {
			t.Fatal(err)
		}
		fired, changeErr := trace.result()
		if !fired || changeErr != nil {
			t.Fatalf("concurrent settings change = (fired %t, %v), want committed", fired, changeErr)
		}
		if after := window(t); after != before+1 {
			t.Fatalf("dispatch window after the concurrent change = %d, want %d", after, before+1)
		}
		if snapshot.MaxConcurrentRuns != before || snapshot.Audit.AuditID != f.audit.AuditID {
			t.Fatalf("snapshot = (window %d, Audit %s), want window %d of Audit %s from before the change",
				snapshot.MaxConcurrentRuns, snapshot.Audit.AuditID, before, f.audit.AuditID)
		}
	}

	t.Run("pool", func(t *testing.T) {
		before := window(t)
		trace.arm(changeWindow)
		snapshot, err := NewClaimStore(traced).GetReconcileSnapshot(f.ctx, f.claim)
		requireOneCommit(t, before, snapshot, err)
		fresh, err := NewClaimStore(traced).GetReconcileSnapshot(f.ctx, f.claim)
		if err != nil || fresh.MaxConcurrentRuns != before+1 {
			t.Fatalf("next snapshot = (window %d, %v), want the committed window %d", fresh.MaxConcurrentRuns, err, before+1)
		}
	})

	t.Run("caller transaction", func(t *testing.T) {
		before := window(t)
		tx, err := traced.BeginTx(f.ctx, pgx.TxOptions{IsoLevel: pgx.RepeatableRead})
		if err != nil {
			t.Fatal(err)
		}
		defer func() { _ = tx.Rollback(context.WithoutCancel(f.ctx)) }()
		trace.arm(changeWindow)
		snapshot, err := NewClaimStore(tx).GetReconcileSnapshot(f.ctx, f.claim)
		requireOneCommit(t, before, snapshot, err)
		// The snapshot ran inside the caller's transaction, which stays usable
		// and keeps its own snapshot.
		var inside int
		if err := tx.QueryRow(f.ctx, `SELECT max_concurrent_runs FROM scheduler_settings WHERE singleton`).
			Scan(&inside); err != nil || inside != before {
			t.Fatalf("caller transaction window = (%d, %v), want %d", inside, err, before)
		}
		if err := tx.Commit(f.ctx); err != nil {
			t.Fatal(err)
		}
	})
}
