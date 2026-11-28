package postgres

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgconn"
)

// These real-store cases bind the local V498 TLA+ pilot to the common SQL
// used by both controllers. They do not load the ignored model at test time.
func TestPostgresControllerClaimsRenewAndDelayedOperations(t *testing.T) {
	for _, table := range controllerClaimPilotTables {
		t.Run(table.name, func(t *testing.T) {
			for _, scenario := range []string{"live", "expired", "released", "takeover", "same-holder-new-epoch"} {
				t.Run(scenario, func(t *testing.T) {
					f := newClaimPilotFixture(t, table)
					first := f.lease(t, f.pool, "holder-a")
					if len(first) != 1 {
						t.Fatalf("initial lease = %+v", first)
					}
					old := first[0]
					switch scenario {
					case "expired", "takeover", "same-holder-new-epoch":
						f.expire(t, f.pool)
					case "released":
						if err := f.claims.Release(f.ctx, f.pool, table.id, old.HolderID, old.Epoch); err != nil {
							t.Fatal(err)
						}
					}
					if scenario == "takeover" || scenario == "same-holder-new-epoch" {
						holder := "holder-b"
						if scenario == "same-holder-new-epoch" {
							holder = "holder-a"
						}
						replacement := f.lease(t, f.second, holder)
						if len(replacement) != 1 || replacement[0].Epoch != old.Epoch+1 {
							t.Fatalf("replacement = %+v", replacement)
						}
					}
					before := f.snapshot(t)
					err := f.claims.Renew(f.ctx, f.second, table.id, old.HolderID, old.Epoch, 5*time.Minute)
					if scenario == "live" {
						if err != nil {
							t.Fatal(err)
						}
						after := f.snapshot(t)
						if after.Holder != before.Holder || after.Epoch != before.Epoch || !after.Start.Equal(before.Start) || !after.End.After(before.End) {
							t.Fatalf("renew changed identity or failed to extend lease: before=%+v after=%+v", before, after)
						}
						if err := f.write(old); err != nil {
							t.Fatal(err)
						}
						f.wantEffects(t, 1)
						return
					}
					if !errors.Is(err, ErrControllerClaimLost) || f.snapshot(t) != before {
						t.Fatalf("stale renew changed claim or returned %v", err)
					}
					if err := f.write(old); !errors.Is(err, ErrControllerClaimLost) {
						t.Fatalf("late write = %v, want lost claim", err)
					}
					f.wantEffects(t, 0)
					err = f.claims.Release(f.ctx, f.second, table.id, old.HolderID, old.Epoch)
					if scenario == "expired" {
						if err != nil || f.snapshot(t).Holder != "" {
							t.Fatalf("expired holder release = %v", err)
						}
					} else if !errors.Is(err, ErrControllerClaimLost) || f.snapshot(t) != before {
						t.Fatalf("stale release changed claim or returned %v", err)
					}
				})
			}
		})
	}
}

func TestPostgresControllerClaimsTransactionFencesTakeover(t *testing.T) {
	for _, table := range controllerClaimPilotTables {
		t.Run(table.name, func(t *testing.T) {
			for _, outcome := range []string{"commit", "rollback"} {
				t.Run(outcome, func(t *testing.T) {
					f := newClaimPilotFixture(t, table)
					leased := f.lease(t, f.pool, "holder-a")
					if len(leased) != 1 {
						t.Fatalf("lease = %+v", leased)
					}
					old := leased[0]
					tx, err := f.pool.Begin(f.ctx)
					if err != nil {
						t.Fatal(err)
					}
					defer tx.Rollback(f.ctx)
					if err := f.claims.Lock(f.ctx, tx, table.id, old.HolderID, old.Epoch); err != nil {
						t.Fatal(err)
					}
					// Prove the actual row lock from a second connection, without
					// sleeps or elapsed-time assertions. NOWAIT must reject it.
					_, err = f.second.Exec(f.ctx, "SELECT 1 FROM "+table.name+" WHERE "+table.subject+"=$1 FOR UPDATE NOWAIT", table.id)
					var pgErr *pgconn.PgError
					if !errors.As(err, &pgErr) || pgErr.Code != "55P03" {
						t.Fatalf("claim lock not retained until transaction end: %v", err)
					}
					// Move the lease into the past deterministically while retaining
					// the admission lock. This represents Tick after BeginWrite.
					f.expire(t, tx)
					if got := f.lease(t, f.second, "holder-b"); len(got) != 0 {
						t.Fatalf("takeover before transaction end = %+v", got)
					}
					if _, err := tx.Exec(f.ctx, "INSERT INTO claim_pilot_effects DEFAULT VALUES"); err != nil {
						t.Fatal(err)
					}
					want := 0
					if outcome == "commit" {
						err = tx.Commit(f.ctx)
						want = 1
					} else {
						err = tx.Rollback(f.ctx)
					}
					if err != nil {
						t.Fatal(err)
					}
					f.wantEffects(t, want)
					// Rollback also rolls back our test-only expiry update.
					if outcome == "rollback" {
						f.expire(t, f.pool)
					}
					replacement := f.lease(t, f.second, "holder-b")
					if len(replacement) != 1 || replacement[0].Epoch != old.Epoch+1 {
						t.Fatalf("takeover after transaction end = %+v", replacement)
					}
					if err := f.write(old); !errors.Is(err, ErrControllerClaimLost) {
						t.Fatalf("write after takeover = %v", err)
					}
					f.wantEffects(t, want)
				})
			}
		})
	}
}

// A stale request can start before takeover commits and wait on its row.
// READ COMMITTED must recheck the holder/epoch predicate after that wait.
func TestPostgresControllerClaimsWaitingOperationsRecheckEpoch(t *testing.T) {
	for _, table := range controllerClaimPilotTables {
		t.Run(table.name, func(t *testing.T) {
			for _, operation := range []string{"renew", "release", "write"} {
				t.Run(operation, func(t *testing.T) {
					f := newClaimPilotFixture(t, table)
					leased := f.lease(t, f.pool, "holder-a")
					if len(leased) != 1 {
						t.Fatalf("lease = %+v", leased)
					}
					old := leased[0]
					blocker, err := f.pool.Begin(f.ctx)
					if err != nil {
						t.Fatal(err)
					}
					defer blocker.Rollback(f.ctx)
					// Stage a new epoch with the SAME holder to expose ABA bugs.
					if _, err := blocker.Exec(f.ctx, "UPDATE "+table.name+" SET epoch=epoch+1 WHERE "+table.subject+"=$1", table.id); err != nil {
						t.Fatal(err)
					}
					conn, err := f.second.Acquire(f.ctx)
					if err != nil {
						t.Fatal(err)
					}
					defer conn.Release()
					waiterPID := conn.Conn().PgConn().PID()
					ctx, cancel := context.WithCancel(f.ctx)
					done := make(chan error, 1)
					joined := false
					defer func() {
						cancel()
						_ = blocker.Rollback(f.ctx)
						if !joined {
							<-done
						}
					}()
					go func() {
						var err error
						switch operation {
						case "renew":
							err = f.claims.Renew(ctx, conn, table.id, old.HolderID, old.Epoch, 5*time.Minute)
						case "release":
							err = f.claims.Release(ctx, conn, table.id, old.HolderID, old.Epoch)
						case "write":
							var tx pgx.Tx
							tx, err = conn.Begin(ctx)
							if err == nil {
								err = f.claims.Lock(ctx, tx, table.id, old.HolderID, old.Epoch)
								if err == nil {
									_, err = tx.Exec(ctx, "INSERT INTO claim_pilot_effects DEFAULT VALUES")
								}
								if err == nil {
									err = tx.Commit(ctx)
								}
								_ = tx.Rollback(f.ctx)
							}
						}
						done <- err
					}()
					waiting := false
					// Observe a real lock dependency, with bounded queries and the
					// fixture context as deadline; no scheduling sleeps or timing claim.
					for attempt := 0; attempt < 1000 && !waiting; attempt++ {
						if err := f.pool.QueryRow(f.ctx, "SELECT cardinality(pg_blocking_pids($1)) > 0", waiterPID).Scan(&waiting); err != nil {
							t.Fatal(err)
						}
					}
					if !waiting {
						t.Fatal("operation never reached its claim row lock")
					}
					if err := blocker.Commit(f.ctx); err != nil {
						t.Fatal(err)
					}
					err = <-done
					joined = true
					if !errors.Is(err, ErrControllerClaimLost) {
						t.Fatalf("waiting %s = %v, want claim lost", operation, err)
					}
					current := f.snapshot(t)
					if current.Holder != old.HolderID || current.Epoch != old.Epoch+1 || !current.End.Equal(old.ExpiresAt) {
						t.Fatalf("stale operation changed replacement: %+v", current)
					}
					f.wantEffects(t, 0)
				})
			}
		})
	}
}

type claimPilotTable struct{ name, subject, id string }

var controllerClaimPilotTables = []claimPilotTable{
	{"audit_controller_claims", "audit_id", "keyed-audit"},
	{"eval_controller_claims", "experiment_id", "keyed-experiment"},
}

type claimPilotFixture struct {
	keyedFixture
	table  claimPilotTable
	claims ControllerClaims
}

func newClaimPilotFixture(t *testing.T, table claimPilotTable) claimPilotFixture {
	t.Helper()
	f := claimPilotFixture{keyedFixture: newKeyedFixture(t), table: table, claims: NewControllerClaims(table.name, table.subject)}
	if _, err := f.pool.Exec(f.ctx, "INSERT INTO "+table.name+" ("+table.subject+") VALUES ($1)", table.id); err != nil {
		t.Fatal(err)
	}
	if _, err := f.pool.Exec(f.ctx, "CREATE TABLE claim_pilot_effects (effect integer DEFAULT 1)"); err != nil {
		t.Fatal(err)
	}
	return f
}

func (f claimPilotFixture) lease(t *testing.T, db DBTX, holder string) []ControllerClaim {
	t.Helper()
	rows, err := db.Query(f.ctx, `WITH candidates AS (
    SELECT claim.ctid AS claim_tid FROM `+f.table.name+` AS claim
    WHERE `+f.claims.AvailableCondition("claim")+`
    FOR UPDATE SKIP LOCKED LIMIT 1
)`+f.claims.LeaseSQL(), holder, time.Minute.Milliseconds())
	if err != nil {
		t.Fatal(err)
	}
	defer rows.Close()
	var out []ControllerClaim
	for rows.Next() {
		claim, err := ScanControllerClaim(rows)
		if err != nil {
			t.Fatal(err)
		}
		out = append(out, claim)
	}
	if err := rows.Err(); err != nil {
		t.Fatal(err)
	}
	return out
}

func (f claimPilotFixture) expire(t *testing.T, db DBTX) {
	t.Helper()
	if _, err := db.Exec(f.ctx, "UPDATE "+f.table.name+` SET
claimed_at = clock_timestamp() - interval '2 minutes',
expires_at = clock_timestamp() - interval '1 minute' WHERE `+f.table.subject+"=$1", f.table.id); err != nil {
		t.Fatal(err)
	}
}

type claimPilotSnapshot struct {
	Holder     string
	Epoch      int64
	Start, End time.Time
}

func (f claimPilotFixture) snapshot(t *testing.T) claimPilotSnapshot {
	t.Helper()
	var s claimPilotSnapshot
	if err := f.pool.QueryRow(f.ctx, "SELECT COALESCE(holder_id,''), epoch, COALESCE(claimed_at,'epoch'), COALESCE(expires_at,'epoch') FROM "+f.table.name+" WHERE "+f.table.subject+"=$1", f.table.id).Scan(&s.Holder, &s.Epoch, &s.Start, &s.End); err != nil {
		t.Fatal(err)
	}
	return s
}

func (f claimPilotFixture) write(c ControllerClaim) error {
	return InTx(f.ctx, f.second, pgx.TxOptions{}, func(tx pgx.Tx) error {
		if err := f.claims.Lock(f.ctx, tx, f.table.id, c.HolderID, c.Epoch); err != nil {
			return err
		}
		_, err := tx.Exec(f.ctx, "INSERT INTO claim_pilot_effects DEFAULT VALUES")
		return err
	})
}

func (f claimPilotFixture) wantEffects(t *testing.T, want int) {
	t.Helper()
	var got int
	if err := f.pool.QueryRow(f.ctx, "SELECT count(*) FROM claim_pilot_effects").Scan(&got); err != nil || got != want {
		t.Fatalf("effects = %d, %v; want %d", got, err, want)
	}
}
