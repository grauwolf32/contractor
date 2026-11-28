package postgres

import (
	"bytes"
	"context"
	"errors"
	"io"
	"testing"
	"time"

	"github.com/jackc/pgx/v5"
)

// The fixture table has no uniqueness guard: only the production transaction
// and receipt protocol can prevent a duplicate committed effect.
func newKeyedPilot(t *testing.T) keyedFixture {
	t.Helper()
	f := newKeyedFixture(t)
	if _, err := f.pool.Exec(f.ctx, `CREATE TABLE keyed_pilot_effects (scope text NOT NULL, value text NOT NULL)`); err != nil {
		t.Fatal(err)
	}
	return f
}

func pilotEffectCount(t *testing.T, f keyedFixture, scope string, want int) {
	t.Helper()
	var count int
	if err := f.pool.QueryRow(f.ctx, `SELECT count(*) FROM keyed_pilot_effects WHERE scope=$1`, scope).Scan(&count); err != nil || count != want {
		t.Fatalf("committed effects = %d, want %d: %v", count, want, err)
	}
}

// The COMMIT really reaches PostgreSQL. Only its successful acknowledgement
// is replaced; this does not pretend to simulate arbitrary server/network faults.
type keyedPilotLostCommitReply struct{ pgx.Tx }

func (tx keyedPilotLostCommitReply) Commit(ctx context.Context) error {
	if err := tx.Tx.Commit(ctx); err != nil {
		return err
	}
	return io.ErrUnexpectedEOF
}

func TestPostgresKeyedMutationPilotAtomicityAndLostReply(t *testing.T) {
	f := newKeyedPilot(t)
	families := keyedFamilies()
	families = append(families, struct {
		name    string
		key     ReceiptKey
		receipt Receipt
	}{
		"project", ProjectReceiptKey("owner", "keyed-project", "inventory", "inventory.merge", "pilot"),
		Receipt{OwnerID: "owner", ProjectID: "keyed-project", ResourceID: "inventory", Operation: "inventory.merge", Response: []byte(`{"revision":"original"}`)},
	})
	for _, family := range families {
		for _, edge := range []string{"after-effect", "after-receipt", "lost-before-commit", "lost-commit-reply"} {
			t.Run(family.name+"/"+edge, func(t *testing.T) {
				key := family.key
				// Runtime management keys must remain digests; use another valid digest per edge.
				if family.name == "runtime-management" {
					key.key = keyedDigest(map[string]string{"after-effect": "1", "after-receipt": "2", "lost-before-commit": "4", "lost-commit-reply": "3"}[edge])
				} else {
					key.key += "-" + edge
				}
				scope := family.name + "/" + edge
				refused := errors.New("injected transaction refusal")
				mutationCalls, attempts := 0, 0
				mutate := func(tx pgx.Tx, fail bool) func() (Receipt, error) {
					return func() (Receipt, error) {
						mutationCalls++
						if _, err := tx.Exec(f.ctx, `INSERT INTO keyed_pilot_effects VALUES ($1,$2)`, scope, "original"); err != nil {
							return Receipt{}, err
						}
						if fail {
							return Receipt{}, refused
						}
						return family.receipt, nil
					}
				}
				err := retryTransaction(f.ctx, func() error {
					attempts++
					tx, err := BeginTx(f.ctx, f.pool, pgx.TxOptions{})
					if err != nil {
						return err
					}
					defer func() { _ = tx.Rollback(context.WithoutCancel(f.ctx)) }()
					_, replayed, err := KeyedMutation(f.ctx, tx, key, keyedDigest("a"), mutate(tx, edge == "after-effect"))
					if err != nil {
						return err
					}
					if replayed {
						return errors.New("first request replayed")
					}
					if edge == "after-receipt" {
						return refused
					}
					if edge == "lost-before-commit" {
						if err := tx.Rollback(f.ctx); err != nil {
							return err
						}
						return WrapError("commit keyed pilot", io.ErrUnexpectedEOF)
					}
					return WrapError("commit keyed pilot", keyedPilotLostCommitReply{tx}.Commit(f.ctx))
				})
				if attempts != 1 || mutationCalls != 1 {
					t.Fatalf("ambiguous/permanent outcome retried: attempts=%d mutations=%d", attempts, mutationCalls)
				}
				committed := edge == "lost-commit-reply"
				if committed {
					if !errors.Is(err, io.ErrUnexpectedEOF) || Classify(err) != ClassUnavailable {
						t.Fatalf("lost commit classification: %v", err)
					}
					pilotEffectCount(t, f, scope, 1)
				} else {
					if edge == "lost-before-commit" {
						if !errors.Is(err, io.ErrUnexpectedEOF) || Classify(err) != ClassUnavailable {
							t.Fatalf("lost before commit: %v", err)
						}
					} else if !errors.Is(err, refused) {
						t.Fatalf("rollback: %v", err)
					}
					pilotEffectCount(t, f, scope, 0)
				}
				if _, found, err := FindReceipt(f.ctx, f.second, key, keyedDigest("a")); err != nil || found != committed {
					t.Fatalf("effect/receipt atomicity: found=%v err=%v", found, err)
				}
				// A fresh transaction/connection reconciles the same key. It must not
				// infer from a lost response that the original transaction rolled back.
				var got Receipt
				var replayed bool
				err = InTxWithRetry(f.ctx, f.second, pgx.TxOptions{}, func(tx pgx.Tx) error {
					var err error
					got, replayed, err = KeyedMutation(f.ctx, tx, key, keyedDigest("a"), mutate(tx, false))
					return err
				})
				if err != nil || replayed != committed || !bytes.Equal(got.Response, family.receipt.Response) {
					t.Fatalf("reconciliation: replay=%v receipt=%+v err=%v", replayed, got, err)
				}
				wantCalls := 2
				if committed {
					wantCalls = 1
				}
				if mutationCalls != wantCalls {
					t.Fatalf("mutation calls=%d want=%d", mutationCalls, wantCalls)
				}
				pilotEffectCount(t, f, scope, 1)
				// A mismatching request cannot append an effect or replace the receipt.
				err = InTx(f.ctx, f.second, pgx.TxOptions{}, func(tx pgx.Tx) error {
					_, _, err := KeyedMutation(f.ctx, tx, key, keyedDigest("b"), mutate(tx, false))
					return err
				})
				if !errors.Is(err, ErrReceiptMismatch) || mutationCalls != wantCalls {
					t.Fatalf("changed request: calls=%d err=%v", mutationCalls, err)
				}
				stored, found, err := FindReceipt(f.ctx, f.pool, key, keyedDigest("a"))
				if err != nil || !found || !bytes.Equal(stored.Response, got.Response) {
					t.Fatalf("receipt changed: %+v %v", stored, err)
				}
				pilotEffectCount(t, f, scope, 1)
			})
		}
	}
}

func TestPostgresKeyedMutationPilotContentionAndSnapshot(t *testing.T) {
	f := newKeyedPilot(t)
	for _, iso := range []pgx.TxIsoLevel{pgx.ReadCommitted, pgx.RepeatableRead} {
		isoName := "read-committed"
		if iso == pgx.RepeatableRead {
			isoName = "repeatable-read"
		}
		for _, mode := range []string{"same", "different", "cancel"} {
			t.Run(isoName+"/"+mode, func(t *testing.T) {
				scope := isoName + "-" + mode
				key := ProjectReceiptKey("owner", "keyed-project", "inventory", "inventory.merge", scope)
				receipt := Receipt{OwnerID: "owner", ProjectID: "keyed-project", ResourceID: "inventory", Operation: "inventory.merge", Response: []byte(`{"value":"winner"}`)}
				first, err := BeginTx(f.ctx, f.pool, pgx.TxOptions{})
				if err != nil {
					t.Fatal(err)
				}
				defer func() { _ = first.Rollback(context.WithoutCancel(f.ctx)) }()
				if _, _, err := KeyedMutation(f.ctx, first, key, keyedDigest("a"), func() (Receipt, error) {
					_, err := first.Exec(f.ctx, `INSERT INTO keyed_pilot_effects VALUES ($1,'winner')`, scope)
					return receipt, err
				}); err != nil {
					t.Fatal(err)
				}
				ctx, cancel := context.WithCancel(f.ctx)
				defer cancel()
				type outcome struct {
					attempts, calls int
					replay          bool
					receipt         Receipt
					err             error
				}
				done := make(chan outcome, 1)
				started := make(chan int, 1)
				go func() {
					var result outcome
					result.err = InTxWithRetry(ctx, f.second, pgx.TxOptions{IsoLevel: iso}, func(tx pgx.Tx) error {
						result.attempts++
						if result.attempts == 1 {
							// Establish the RR snapshot before waiting on the winner's advisory lock.
							var pid, count int
							if err := tx.QueryRow(ctx, `SELECT pg_backend_pid(),count(*) FROM keyed_pilot_effects`).Scan(&pid, &count); err != nil {
								return err
							}
							started <- pid
						}
						digest := keyedDigest("a")
						if mode == "different" {
							digest = keyedDigest("b")
						}
						var err error
						result.receipt, result.replay, err = KeyedMutation(ctx, tx, key, digest, func() (Receipt, error) {
							result.calls++
							_, err := tx.Exec(ctx, `INSERT INTO keyed_pilot_effects VALUES ($1,'contender')`, scope)
							return receipt, err
						})
						return err
					})
					done <- result
				}()
				var pid int
				select {
				case pid = <-started:
				case result := <-done:
					t.Fatalf("contender ended before snapshot: %v", result.err)
				case <-f.ctx.Done():
					t.Fatal(f.ctx.Err())
				}
				waitKeyedPilotLock(t, f, pid, done)
				var result outcome
				if mode == "cancel" {
					cancel()
					select {
					case result = <-done:
					case <-f.ctx.Done():
						t.Fatal(f.ctx.Err())
					}
				}
				if err := first.Commit(f.ctx); err != nil {
					t.Fatal(err)
				}
				if mode != "cancel" {
					select {
					case result = <-done:
					case <-f.ctx.Done():
						t.Fatal(f.ctx.Err())
					}
				}
				if mode == "cancel" {
					if !errors.Is(result.err, context.Canceled) || result.attempts != 1 || result.calls != 0 {
						t.Fatalf("cancelled waiter: %+v", result)
					}
					var replayed bool
					err := InTx(f.ctx, f.second, pgx.TxOptions{}, func(tx pgx.Tx) error {
						var err error
						_, replayed, err = KeyedMutation(f.ctx, tx, key, keyedDigest("a"), func() (Receipt, error) { return Receipt{}, errors.New("cancel replay ran mutation") })
						return err
					})
					if err != nil || !replayed {
						t.Fatalf("fresh replay after cancellation: %v %v", replayed, err)
					}
				} else {
					wantAttempts, wantCalls := 1, 0
					if iso == pgx.RepeatableRead {
						wantAttempts, wantCalls = 2, 1
					}
					if result.attempts != wantAttempts || result.calls != wantCalls {
						t.Fatalf("snapshot attempts/calls: %+v", result)
					}
					if mode == "same" {
						if result.err != nil || !result.replay || !bytes.Equal(result.receipt.Response, receipt.Response) {
							t.Fatalf("same request: %+v", result)
						}
					} else if !errors.Is(result.err, ErrReceiptMismatch) {
						t.Fatalf("different request: %+v", result)
					}
				}
				pilotEffectCount(t, f, scope, 1)
			})
		}
	}
}

// Poll PostgreSQL's observed lock wait, never infer ordering from elapsed time.
func waitKeyedPilotLock[T any](t *testing.T, f keyedFixture, pid int, done <-chan T) {
	t.Helper()
	ticker := time.NewTicker(5 * time.Millisecond)
	defer ticker.Stop()
	for {
		var waiting bool
		if err := f.pool.QueryRow(f.ctx, `SELECT COALESCE(wait_event_type='Lock',false) FROM pg_stat_activity WHERE pid=$1`, pid).Scan(&waiting); err != nil {
			t.Fatal(err)
		}
		if waiting {
			return
		}
		select {
		case <-done:
			t.Fatal("contender did not wait for the keyed lock")
		case <-f.ctx.Done():
			t.Fatal(f.ctx.Err())
		case <-ticker.C:
		}
	}
}
