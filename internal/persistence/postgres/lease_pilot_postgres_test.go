package postgres

import (
	"context"
	"errors"
	"slices"
	"testing"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

// The model's shared-lock set must represent every transaction, not merely
// one old writer. A fresh old proof queues behind the successor's exclusive
// waiter, then checks the now-replaced session and commits nothing.
func TestServerLeaseHandoverPilotMultipleWritersAndQueuedProof(t *testing.T) {
	for _, first := range []string{"commit", "rollback"} {
		for _, second := range []string{"commit", "rollback"} {
			t.Run(first+"-"+second, func(t *testing.T) {
				ctx, fixturePool := leaseTestPool(t)
				config := fixturePool.Config()
				// Two writers, handover, queued proof and a lock observer.
				config.MaxConns = 8
				pool, err := pgxpool.NewWithConfig(ctx, config)
				if err != nil {
					t.Fatal(err)
				}
				t.Cleanup(pool.Close)
				if _, err := pool.Exec(ctx, `CREATE TABLE lease_pilot_writes (id integer PRIMARY KEY)`); err != nil {
					t.Fatal(err)
				}
				key := randomLeaseKey(t)
				previous := connectLeaseSession(t, ctx, pool)
				takeLease(t, ctx, previous, key)
				old, err := NewServerLease(ctx, previous, key)
				if err != nil {
					t.Fatal(err)
				}
				var writers [2]pgx.Tx
				var pids [2]int32
				for i := range writers {
					tx, err := BeginTx(ctx, pool, pgx.TxOptions{})
					if err != nil {
						t.Fatal(err)
					}
					writers[i] = tx
					t.Cleanup(func() { _ = tx.Rollback(context.WithoutCancel(ctx)) })
					if err := old.Prove(ctx, tx); err != nil {
						t.Fatal(err)
					}
					if _, err := tx.Exec(ctx, `INSERT INTO lease_pilot_writes (id) VALUES ($1)`, i+1); err != nil {
						t.Fatal(err)
					}
					if err := tx.QueryRow(ctx, `SELECT pg_backend_pid()`).Scan(&pids[i]); err != nil {
						t.Fatal(err)
					}
				}
				terminateLeaseSession(t, ctx, pool, previous)
				successor := connectLeaseSession(t, ctx, pool)
				takeLease(t, ctx, successor, key)
				next, err := NewServerLease(ctx, successor, key)
				if err != nil {
					t.Fatal(err)
				}
				waiting, cancel := context.WithCancel(ctx)
				handover, joined := make(chan error, 1), make(chan struct{})
				go func() {
					defer close(joined)
					handover <- AwaitLeaseHandover(waiting, pool, next)
				}()
				t.Cleanup(func() { cancel(); <-joined })
				handoverPID := awaitLeasePilotWaiter(t, ctx, pool, 0, pids[0])
				awaitLeasePilotWaiter(t, ctx, pool, handoverPID, pids[1])

				late, err := BeginTx(ctx, pool, pgx.TxOptions{})
				if err != nil {
					t.Fatal(err)
				}
				t.Cleanup(func() { _ = late.Rollback(context.WithoutCancel(ctx)) })
				var latePID int32
				if err := late.QueryRow(ctx, `SELECT pg_backend_pid()`).Scan(&latePID); err != nil {
					t.Fatal(err)
				}
				proof, proofJoined := make(chan error, 1), make(chan struct{})
				go func() {
					defer close(proofJoined)
					err := old.Prove(waiting, late)
					if err == nil {
						_, err = late.Exec(waiting, `INSERT INTO lease_pilot_writes (id) VALUES (3)`)
					}
					if err == nil {
						err = late.Commit(waiting)
					}
					proof <- err
				}()
				t.Cleanup(func() { cancel(); <-proofJoined })
				awaitLeasePilotWaiter(t, ctx, pool, latePID, handoverPID)

				var want []int32
				for i, outcome := range []string{first, second} {
					var err error
					if outcome == "commit" {
						err = writers[i].Commit(ctx)
						want = append(want, int32(i+1))
					} else {
						err = writers[i].Rollback(ctx)
					}
					if err != nil {
						t.Fatal(err)
					}
					if i == 0 {
						awaitLeasePilotWaiter(t, ctx, pool, handoverPID, pids[1])
						select {
						case err := <-handover:
							t.Fatalf("handover ignored the second writer: %v", err)
						default:
						}
					}
				}
				if err := <-handover; err != nil {
					t.Fatal(err)
				}
				if err := <-proof; !errors.Is(err, ErrLeaseLost) {
					t.Fatalf("queued old proof = %v, want lease lost", err)
				}
				if err := late.Rollback(ctx); err != nil {
					t.Fatal(err)
				}
				// This is the snapshot boundary; domain recovery is exercised by
				// the credential COMMIT-proxy tests mapped in the pilot evidence.
				if err := InTx(ctx, pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
					if err := next.Prove(ctx, tx); err != nil {
						return err
					}
					rows, err := tx.Query(ctx, `SELECT id FROM lease_pilot_writes ORDER BY id`)
					if err != nil {
						return err
					}
					got, err := pgx.CollectRows(rows, pgx.RowTo[int32])
					if err == nil && !slices.Equal(got, want) {
						t.Errorf("successor snapshot = %v, want %v", got, want)
					}
					return err
				}); err != nil {
					t.Fatal(err)
				}
			})
		}
	}
}

// Observe actual PostgreSQL hard/soft lock dependencies, not elapsed time.
// waiter=0 discovers the handover backend; subsequent calls pin that backend.
func awaitLeasePilotWaiter(t *testing.T, ctx context.Context, pool *pgxpool.Pool, waiter, blocker int32) int32 {
	t.Helper()
	for {
		var pid int32
		err := pool.QueryRow(ctx, `
SELECT pid FROM pg_locks
WHERE locktype = 'advisory' AND NOT granted
  AND ($1::integer = 0 OR pid = $1)
  AND $2::integer = ANY (pg_blocking_pids(pid))
ORDER BY pid LIMIT 1`, waiter, blocker).Scan(&pid)
		if err == nil {
			return pid
		}
		if !errors.Is(err, pgx.ErrNoRows) {
			t.Fatal(err)
		}
		select {
		case <-ctx.Done():
			t.Fatalf("backend %d did not wait for %d: %v", waiter, blocker, ctx.Err())
		case <-time.After(10 * time.Millisecond):
		}
	}
}
