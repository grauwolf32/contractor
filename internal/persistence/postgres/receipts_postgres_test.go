package postgres

import (
	"bytes"
	"context"
	"errors"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

// keyedFixture is a migrated isolated schema with a Project holding an
// Audit and an evaluation Project, so a receipt of every family and both
// controller claim tables have their parents.
type keyedFixture struct {
	ctx    context.Context
	pool   *pgxpool.Pool
	second *pgxpool.Pool
}

func newKeyedFixture(t *testing.T) keyedFixture {
	t.Helper()
	url := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if url == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	ctx, cancel := context.WithTimeout(t.Context(), 90*time.Second)
	t.Cleanup(cancel)
	pool, second := isolatedPools(t, ctx, url)
	if _, err := ApplyMigrations(ctx, pool); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `
INSERT INTO projects (project_id, owner_id, kind, name, request_idempotency_key, request_digest)
VALUES ('keyed-project', 'owner', 'project', 'Keyed', 'keyed-project', 'sha256:'||repeat('a', 64)),
       ('keyed-evaluation', 'owner', 'evaluation', 'Keyed evaluation', 'keyed-evaluation', 'sha256:'||repeat('b', 64));
INSERT INTO audits (
    audit_id, owner_id, project_id, profile_name, profile_version, profile_digest,
    profile_snapshot, input_selection, max_rounds, batch_size, max_items_per_round,
    max_items_total, max_submitted_runs, max_item_run_attempts, max_evidence_bytes
) VALUES ('keyed-audit', 'owner', 'keyed-project', 'profile', '1', 'sha256:'||repeat('a', 64),
          '{}', '{}', 1, 1, 10, 10, 10, 1, 1024);
INSERT INTO eval_experiments (
    experiment_id, owner_id, project_id, portable_id, control_mode, name, state, max_in_flight, wall_ms
) VALUES ('keyed-experiment', 'owner', 'keyed-evaluation', 'keyed-experiment', 'external', 'Keyed', 'running', 1, 1000)`); err != nil {
		t.Fatal(err)
	}
	return keyedFixture{ctx: ctx, pool: pool, second: second}
}

func keyedDigest(fill string) string { return "sha256:" + strings.Repeat(fill, 64) }

// keyedFamilies are a valid receipt of each keyed mutation family under its
// receipt key.
func keyedFamilies() []struct {
	name    string
	key     ReceiptKey
	receipt Receipt
} {
	return []struct {
		name    string
		key     ReceiptKey
		receipt Receipt
	}{
		{"audit", AuditReceiptKey("owner", "audit.create", "create-1"), Receipt{
			Operation: "audit.create", ResourceID: "keyed-audit", Response: []byte(`{"auditId":"keyed-audit"}`),
			OwnerID: "owner", AuditID: "keyed-audit",
		}},
		{"eval", EvalReceiptKey("owner", "keyed-evaluation", "experiments", "experiment-create", "create-1"), Receipt{
			Operation: "experiment-create", ResourceID: "experiments", Response: []byte(`{"experimentId":"keyed-experiment"}`),
			OwnerID: "owner", ProjectID: "keyed-evaluation",
		}},
		{"runtime-management", RuntimeManagementReceiptKey(keyedDigest("c")), Receipt{
			Operation: "runtime-label.create", ResourceID: "label", Response: []byte(`{"deleted":true}`),
		}},
	}
}

// TestPostgresKeyedMutationReplaysAndRefusesUnderEveryFamily pins lock ->
// replay -> claim for every family: a failed mutation claims nothing, the
// first success runs once and claims the key, the same request replays the
// stored receipt without running again, and another request under the key is
// the mismatch, without running either.
func TestPostgresKeyedMutationReplaysAndRefusesUnderEveryFamily(t *testing.T) {
	f := newKeyedFixture(t)
	for _, family := range keyedFamilies() {
		t.Run(family.name, func(t *testing.T) {
			run := func(requestDigest string, mutationErr error) (Receipt, bool, int, error) {
				calls := 0
				var receipt Receipt
				var replayed bool
				err := InTx(f.ctx, f.pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
					var err error
					receipt, replayed, err = KeyedMutation(f.ctx, tx, family.key, requestDigest, func() (Receipt, error) {
						calls++
						if mutationErr != nil {
							return Receipt{}, mutationErr
						}
						return family.receipt, nil
					})
					return err
				})
				return receipt, replayed, calls, err
			}
			refused := errors.New("refused mutation")
			if _, _, calls, err := run(keyedDigest("1"), refused); !errors.Is(err, refused) || calls != 1 {
				t.Fatalf("refused mutation = %d call(s), %v", calls, err)
			}
			if _, found, err := FindReceipt(f.ctx, f.pool, family.key, keyedDigest("1")); err != nil || found {
				t.Fatalf("refused mutation left a receipt: %t, %v", found, err)
			}
			first, replayed, calls, err := run(keyedDigest("1"), nil)
			if err != nil || replayed || calls != 1 || !bytes.Equal(first.Response, family.receipt.Response) {
				t.Fatalf("first mutation = (%+v, %t, %d call(s), %v)", first, replayed, calls, err)
			}
			again, replayed, calls, err := run(keyedDigest("1"), nil)
			want := family.receipt
			want.RequestDigest = keyedDigest("1")
			if err != nil || !replayed || calls != 0 || again.Operation != want.Operation ||
				again.ResourceID != want.ResourceID || again.RequestDigest != want.RequestDigest ||
				!bytes.Equal(again.Response, want.Response) || again.OwnerID != want.OwnerID ||
				again.ProjectID != want.ProjectID || again.AuditID != want.AuditID {
				t.Fatalf("replay = (%+v, %t, %d call(s), %v), want %+v", again, replayed, calls, err, want)
			}
			if _, _, calls, err := run(keyedDigest("2"), nil); !errors.Is(err, ErrReceiptMismatch) || calls != 0 {
				t.Fatalf("another request under the key = %d call(s), %v", calls, err)
			}
			if _, _, err := FindReceipt(f.ctx, f.pool, family.key, keyedDigest("2")); !errors.Is(err, ErrReceiptMismatch) {
				t.Fatalf("lookup of another request = %v", err)
			}
		})
	}
}

// TestPostgresKeyedMutationSerializesOneKey runs two requests under one key
// in two transactions: the second waits for the key's lock while the first
// is open and then replays the first's receipt instead of running again.
func TestPostgresKeyedMutationSerializesOneKey(t *testing.T) {
	f := newKeyedFixture(t)
	key := AuditReceiptKey("owner", "audit.transition", "concurrent")
	receipt := Receipt{
		Operation: "audit.transition", ResourceID: "keyed-audit", Response: []byte(`{"auditId":"keyed-audit"}`),
		OwnerID: "owner", AuditID: "keyed-audit",
	}
	first, err := BeginTx(f.ctx, f.pool, pgx.TxOptions{})
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = first.Rollback(context.WithoutCancel(f.ctx)) }()
	if _, replayed, err := KeyedMutation(f.ctx, first, key, keyedDigest("1"), func() (Receipt, error) {
		return receipt, nil
	}); err != nil || replayed {
		t.Fatalf("first request = (%t, %v)", replayed, err)
	}
	second, err := BeginTx(f.ctx, f.second, pgx.TxOptions{})
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = second.Rollback(context.WithoutCancel(f.ctx)) }()
	var pid int
	if err := second.QueryRow(f.ctx, `SELECT pg_backend_pid()`).Scan(&pid); err != nil {
		t.Fatal(err)
	}
	type outcome struct {
		replayed bool
		calls    int
		err      error
	}
	done := make(chan outcome, 1)
	go func() {
		calls := 0
		_, replayed, err := KeyedMutation(f.ctx, second, key, keyedDigest("1"), func() (Receipt, error) {
			calls++
			return receipt, nil
		})
		done <- outcome{replayed: replayed, calls: calls, err: err}
	}()
	for waiting := false; !waiting; {
		select {
		case result := <-done:
			t.Fatalf("second request finished while the first held the key: %+v", result)
		case <-time.After(10 * time.Millisecond):
		}
		var event *string
		if err := f.pool.QueryRow(f.ctx, `SELECT wait_event_type FROM pg_stat_activity WHERE pid = $1`, pid).Scan(&event); err != nil {
			t.Fatal(err)
		}
		waiting = event != nil && *event == "Lock"
	}
	if err := first.Commit(f.ctx); err != nil {
		t.Fatal(err)
	}
	if result := <-done; result.err != nil || !result.replayed || result.calls != 0 {
		t.Fatalf("second request after the first committed = %+v, want a replay", result)
	}
}

// TestPostgresMutationReceiptsKeepTheirShapeAndRetention pins the table's own
// rules: a receipt whose attributes do not match its family's key scope is
// invalid, no receipt changes, a Runtime management receipt is never deleted,
// an Eval receipt goes only in its Project's purge and an Audit receipt with
// its Audit.
func TestPostgresMutationReceiptsKeepTheirShapeAndRetention(t *testing.T) {
	f := newKeyedFixture(t)
	for _, invalid := range []struct {
		name    string
		key     ReceiptKey
		receipt Receipt
	}{
		{"audit without its Audit", AuditReceiptKey("owner", "audit.create", "invalid"), Receipt{
			Operation: "audit.create", ResourceID: "keyed-audit", RequestDigest: keyedDigest("1"),
			Response: []byte(`{}`), OwnerID: "owner",
		}},
		{"audit under another operation", AuditReceiptKey("owner", "audit.create", "invalid"), Receipt{
			Operation: "audit.start", ResourceID: "keyed-audit", RequestDigest: keyedDigest("1"),
			Response: []byte(`{}`), OwnerID: "owner", AuditID: "keyed-audit",
		}},
		{"eval of another Project", EvalReceiptKey("owner", "keyed-evaluation", "experiments", "experiment-create", "invalid"), Receipt{
			Operation: "experiment-create", ResourceID: "experiments", RequestDigest: keyedDigest("1"),
			Response: []byte(`{}`), OwnerID: "owner", ProjectID: "keyed-project",
		}},
		{"runtime management with an owner", RuntimeManagementReceiptKey(keyedDigest("c")), Receipt{
			Operation: "runtime-label.create", ResourceID: "label", RequestDigest: keyedDigest("1"),
			Response: []byte(`{}`), OwnerID: "owner",
		}},
		{"runtime management under a raw key", RuntimeManagementReceiptKey("raw-key"), Receipt{
			Operation: "runtime-label.create", ResourceID: "label", RequestDigest: keyedDigest("1"), Response: []byte(`{}`),
		}},
		{"malformed request digest", RuntimeManagementReceiptKey(keyedDigest("c")), Receipt{
			Operation: "runtime-label.create", ResourceID: "label", RequestDigest: "sha256:short", Response: []byte(`{}`),
		}},
	} {
		if err := claimReceipt(f.ctx, f.pool, invalid.key, invalid.receipt); Classify(err) != ClassInvalid {
			t.Fatalf("%s claimed = %v, want invalid", invalid.name, err)
		}
	}
	for _, family := range keyedFamilies() {
		receipt := family.receipt
		receipt.RequestDigest = keyedDigest("1")
		if err := claimReceipt(f.ctx, f.pool, family.key, receipt); err != nil {
			t.Fatalf("claim %s receipt: %v", family.name, err)
		}
		if err := claimReceipt(f.ctx, f.pool, family.key, receipt); !errors.Is(err, ErrReceiptClaimed) {
			t.Fatalf("second claim of the %s key = %v", family.name, err)
		}
	}
	if _, err := f.pool.Exec(f.ctx, `UPDATE mutation_receipts SET response = '\x7b7d'`); Classify(err) != ClassRejected {
		t.Fatalf("receipt update = %v, want rejected", err)
	}
	for _, family := range []string{"runtime-management", "eval"} {
		if _, err := f.pool.Exec(f.ctx, `DELETE FROM mutation_receipts WHERE key_scope[1] = $1`, family); Classify(err) != ClassRejected {
			t.Fatalf("%s receipt deletion = %v, want rejected", family, err)
		}
	}
	if project, found, err := EvalReceiptProject(f.ctx, f.pool, "owner", "experiments", "experiment-create", "create-1"); err != nil ||
		!found || project != "keyed-evaluation" {
		t.Fatalf("Eval receipt Project = (%q, %t, %v)", project, found, err)
	}
	if err := InTx(f.ctx, f.pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		if _, err := tx.Exec(f.ctx, `SELECT set_config('contractor.eval_purge', 'on', true)`); err != nil {
			return err
		}
		return PurgeEvalReceipts(f.ctx, tx, "owner", "keyed-evaluation")
	}); err != nil {
		t.Fatalf("purge Eval receipts: %v", err)
	}
	if _, err := f.pool.Exec(f.ctx, `DELETE FROM audits WHERE audit_id = 'keyed-audit'`); err != nil {
		t.Fatalf("delete Audit: %v", err)
	}
	var families []string
	if err := f.pool.QueryRow(f.ctx, `
SELECT COALESCE(array_agg(key_scope[1] ORDER BY key_scope[1]), '{}') FROM mutation_receipts`).Scan(&families); err != nil {
		t.Fatal(err)
	}
	if strings.Join(families, ",") != "runtime-management" {
		t.Fatalf("receipts after the Project purge and the Audit deletion = %v, want only Runtime management", families)
	}
}

// TestPostgresControllerClaimsLeaseProveAndRelease runs one lease life cycle
// on each controller's claim table: a lease advances the epoch, the holder
// proves it while it lasts, a stale holder or epoch finds it lost, a release
// accepts an expired lease of its holder, and the next lease takes over.
func TestPostgresControllerClaimsLeaseProveAndRelease(t *testing.T) {
	f := newKeyedFixture(t)
	if _, err := f.pool.Exec(f.ctx, `
INSERT INTO audit_controller_claims (audit_id) VALUES ('keyed-audit');
INSERT INTO eval_controller_claims (experiment_id) VALUES ('keyed-experiment')`); err != nil {
		t.Fatal(err)
	}
	for _, table := range []struct{ name, subject, id string }{
		{"audit_controller_claims", "audit_id", "keyed-audit"},
		{"eval_controller_claims", "experiment_id", "keyed-experiment"},
	} {
		t.Run(table.name, func(t *testing.T) {
			claims := NewControllerClaims(table.name, table.subject)
			lease := func(holder string) []ControllerClaim {
				t.Helper()
				rows, err := f.pool.Query(f.ctx, `
WITH candidates AS (
    SELECT claim.ctid AS claim_tid
      FROM `+table.name+` AS claim
     WHERE `+claims.AvailableCondition("claim")+`
     FOR UPDATE SKIP LOCKED
     LIMIT $3
)`+claims.LeaseSQL(), holder, time.Minute.Milliseconds(), 10)
				if err != nil {
					t.Fatal(err)
				}
				defer rows.Close()
				var result []ControllerClaim
				for rows.Next() {
					claim, err := ScanControllerClaim(rows)
					if err != nil {
						t.Fatal(err)
					}
					result = append(result, claim)
				}
				if err := rows.Err(); err != nil {
					t.Fatal(err)
				}
				return result
			}
			live := func(holder string, epoch int64) bool {
				t.Helper()
				result, err := claims.Live(f.ctx, f.pool, table.id, holder, epoch)
				if err != nil {
					t.Fatal(err)
				}
				return result
			}
			lock := func(holder string, epoch int64) error {
				return InTx(f.ctx, f.pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
					return claims.Lock(f.ctx, tx, table.id, holder, epoch)
				})
			}
			first := lease("holder-a")
			if len(first) != 1 || first[0].SubjectID != table.id || first[0].HolderID != "holder-a" || first[0].Epoch != 1 {
				t.Fatalf("first lease = %+v", first)
			}
			if held := lease("holder-b"); len(held) != 0 {
				t.Fatalf("lease of a held claim = %+v", held)
			}
			if !live("holder-a", 1) || live("holder-b", 1) || live("holder-a", 2) {
				t.Fatal("live claim reported for the wrong holder or epoch")
			}
			if err := lock("holder-a", 1); err != nil {
				t.Fatalf("lock of the live claim: %v", err)
			}
			if err := lock("holder-a", 2); !errors.Is(err, ErrControllerClaimLost) {
				t.Fatalf("lock under another epoch = %v", err)
			}
			for _, stale := range []struct {
				holder string
				epoch  int64
			}{{"holder-b", 1}, {"holder-a", 2}} {
				if err := claims.Release(f.ctx, f.pool, table.id, stale.holder, stale.epoch); !errors.Is(err, ErrControllerClaimLost) {
					t.Fatalf("release by %s at %d = %v", stale.holder, stale.epoch, err)
				}
			}
			if err := claims.Release(f.ctx, f.pool, table.id, "holder-a", 1); err != nil || live("holder-a", 1) {
				t.Fatalf("release of the live claim = %v", err)
			}
			second := lease("holder-b")
			if len(second) != 1 || second[0].Epoch != 2 {
				t.Fatalf("lease after release = %+v", second)
			}
			if err := lock("holder-a", 1); !errors.Is(err, ErrControllerClaimLost) {
				t.Fatalf("lock of a taken-over claim = %v", err)
			}
			if _, err := f.pool.Exec(f.ctx, `
UPDATE `+table.name+`
   SET claimed_at = clock_timestamp() - interval '2 minutes', expires_at = clock_timestamp() - interval '1 minute'
 WHERE `+table.subject+` = $1`, table.id); err != nil {
				t.Fatal(err)
			}
			if live("holder-b", 2) {
				t.Fatal("an expired lease is live")
			}
			if err := lock("holder-b", 2); !errors.Is(err, ErrControllerClaimLost) {
				t.Fatalf("lock of an expired lease = %v", err)
			}
			if err := claims.Release(f.ctx, f.pool, table.id, "holder-b", 2); err != nil {
				t.Fatalf("release of its holder's expired lease = %v", err)
			}
			if third := lease("holder-c"); len(third) != 1 || third[0].Epoch != 3 {
				t.Fatalf("lease after an expired lease = %+v", third)
			}
			if _, err := f.pool.Exec(f.ctx, `
UPDATE `+table.name+` SET claimed_at = NULL WHERE `+table.subject+` = $1`, table.id); Classify(err) != ClassInvalid {
				t.Fatalf("held claim without its lease start = %v, want invalid", err)
			}
		})
	}
}
