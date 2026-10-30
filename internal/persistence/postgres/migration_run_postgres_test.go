package postgres

import (
	"context"
	"crypto/sha256"
	"errors"
	"fmt"
	"os"
	"strings"
	"testing"
	"time"
)

// A migration run applies every pending migration and its ledger row in one
// transaction under the budget the operator chose (PER-R29, PER-R31). A
// reader of eval_controller_claims keeps a later pending migration
// (controller_claim_shape) from locking the table within the selected lock
// timeout: the run fails as a timeout, and the migrations it had already
// applied, run_artifact_revision_limit among them, roll back with it, so the
// ledger and the schema stay at the version the run started from. A drifted ledger is refused without
// running anything; once the reader is gone and the ledger matches, the next
// run applies the same migrations.
func TestPostgresMigrationRunCommitsEveryPendingMigrationOrNone(t *testing.T) {
	url := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if url == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	ctx, cancel := context.WithTimeout(t.Context(), 2*time.Minute)
	defer cancel()
	pool, other := isolatedPools(t, ctx, url)
	start, _ := embeddedMigration(t, "run_artifact_revision_limit")
	if blocked, _ := embeddedMigration(t, "controller_claim_shape"); blocked <= start {
		t.Fatal("controller_claim_shape does not follow run_artifact_revision_limit")
	}
	installMigrationPrefix(t, ctx, pool, start)
	available, err := loadMigrations()
	if err != nil {
		t.Fatal(err)
	}
	var startChecksum []byte
	for _, item := range available {
		if item.version == start {
			startChecksum = item.checksum[:]
		}
	}
	reader, err := other.Begin(ctx)
	if err != nil {
		t.Fatal(err)
	}
	defer reader.Rollback(context.Background())
	if _, err := reader.Exec(ctx, `LOCK TABLE eval_controller_claims IN ACCESS SHARE MODE`); err != nil {
		t.Fatal(err)
	}

	_, err = ApplyMigrations(ctx, pool, MigrationBudgets{StatementTimeout: 30 * time.Second, LockTimeout: 200 * time.Millisecond})
	if Classify(err) != ClassTimeout || !strings.Contains(err.Error(), "_controller_claim_shape.sql") {
		t.Fatalf("run blocked by a reader = %v, want the lock timeout of controller_claim_shape", err)
	}
	version := func() (int64, bool) {
		t.Helper()
		var latest int64
		var counted bool
		if err := pool.QueryRow(ctx, `
SELECT (SELECT max(version) FROM contractor_schema_migrations),
       EXISTS (SELECT 1 FROM information_schema.columns
                WHERE table_schema = current_schema() AND table_name = 'artifact_scopes'
                  AND column_name = 'revision_count')`).Scan(&latest, &counted); err != nil {
			t.Fatal(err)
		}
		return latest, counted
	}
	// run_artifact_revision_limit adds artifact_scopes.revision_count; it ran
	// before controller_claim_shape failed.
	if latest, counted := version(); latest != start || counted {
		t.Fatalf("after the failed run: ledger at %d, revision counter present %t; want %d and absent", latest, counted, start)
	}

	if err := reader.Rollback(ctx); err != nil {
		t.Fatal(err)
	}
	// A ledger row that no longer matches its embedded file is drift: the
	// run refuses it before any pending migration runs (PER-R30).
	if _, err := pool.Exec(ctx, `UPDATE contractor_schema_migrations SET checksum = sha256(checksum) WHERE version = $1`, start); err != nil {
		t.Fatal(err)
	}
	if _, err := ApplyMigrations(ctx, pool, MigrationBudgets{}); !errors.Is(err, ErrMigrationDrift) {
		t.Fatalf("run on a drifted ledger = %v, want migration drift", err)
	}
	if latest, counted := version(); latest != start || counted {
		t.Fatalf("after the refused run: ledger at %d, revision counter present %t; want %d and absent", latest, counted, start)
	}
	if _, err := pool.Exec(ctx, `UPDATE contractor_schema_migrations SET checksum = $1 WHERE version = $2`, startChecksum, start); err != nil {
		t.Fatal(err)
	}
	result, err := ApplyMigrations(ctx, pool, MigrationBudgets{})
	if err != nil {
		t.Fatalf("run after the reader left: %v", err)
	}
	last := available[len(available)-1].version
	if len(result.AppliedVersions) == 0 || result.AppliedVersions[0] != start+1 || result.CurrentVersion != last {
		t.Fatalf("run after the reader left = %+v, want versions %d to %d", result, start+1, last)
	}
	if latest, counted := version(); latest != last || !counted {
		t.Fatalf("after the second run: ledger at %d, revision counter present %t", latest, counted)
	}
}

// The migrator's client gives each of its own single-statement calls a
// deadline of the statement timeout plus migrationQueryMargin, but sends a
// migration file as one call whose client deadline is the run deadline
// (PER-R31, LIM-PER-11): PostgreSQL applies the statement timeout to each
// statement of the file, so the file has no client bound of its own
// (PER-R32). On a pool opened as the migrator opens it, with a statement
// timeout of 2 s, so a per-call deadline of 7 s: a file of eight 1 s
// statements, 8 s together, applies and records its ledger row; a statement
// beyond the statement timeout is cancelled by PostgreSQL (SQLSTATE 57014);
// and the caller's deadline, which the run deadline never outlasts, still
// ends a file call, with no SQLSTATE. Both failures are timeouts that leave
// nothing of the run behind. The test asserts outcomes, not elapsed times:
// each fixture is at least a second from the bound it tests, and pg_sleep
// never returns early.
func TestPostgresMigrationFileOutlastsTheCallDeadline(t *testing.T) {
	url := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if url == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	budgets := MigrationBudgets{StatementTimeout: 2 * time.Second, LockTimeout: 100 * time.Millisecond}
	callDeadline := budgets.StatementTimeout + migrationQueryMargin
	const sleep, sleeps = time.Second, 8
	const callerDeadline = 4 * time.Second
	if sleep >= budgets.StatementTimeout || sleeps*sleep < callDeadline+sleep || sleeps*sleep < callerDeadline+sleep {
		t.Fatalf("the fixtures do not straddle the statement timeout %s, the call deadline %s and the caller's deadline %s",
			budgets.StatementTimeout, callDeadline, callerDeadline)
	}
	file := func(sleeps int, seconds float64) []byte {
		statements := fmt.Sprintf("CREATE TABLE migration_marker (id integer);\n%s",
			strings.Repeat(fmt.Sprintf("SELECT pg_sleep(%g);\n", seconds), sleeps))
		return []byte(statements)
	}
	for _, test := range []struct {
		name     string
		contents []byte
		deadline time.Duration // of the caller; zero leaves the run deadline
		applied  bool
		sqlstate string // empty for a deadline of the client
	}{
		{name: "statements beyond the call deadline together", contents: file(sleeps, sleep.Seconds()), applied: true},
		{name: "one statement beyond the statement timeout", contents: file(1, 2*budgets.StatementTimeout.Seconds()), sqlstate: SQLStateQueryCanceled},
		{name: "statements beyond the caller's deadline together", contents: file(sleeps, sleep.Seconds()), deadline: callerDeadline},
	} {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			ctx, cancel := context.WithTimeout(t.Context(), time.Minute)
			defer cancel()
			base, _ := isolatedPools(t, ctx, url)
			migrator := budgetTestPool(t, ctx, base, PoolOptions{})
			slow := migration{version: 1, name: "000001_slow_file.sql", contents: test.contents, checksum: sha256.Sum256(test.contents)}
			run := ctx
			if test.deadline > 0 {
				var stop context.CancelFunc
				run, stop = context.WithTimeout(ctx, test.deadline)
				defer stop()
			}
			result, err := applyMigrations(run, migrator, budgets, []migration{slow})
			// The run creates the ledger in its own transaction, so a failed
			// run leaves no ledger at all.
			var marker, ledger bool
			if err := base.QueryRow(ctx, `
SELECT to_regclass('migration_marker') IS NOT NULL, to_regclass('contractor_schema_migrations') IS NOT NULL`).Scan(&marker, &ledger); err != nil {
				t.Fatal(err)
			}
			if test.applied {
				if err != nil || result.CurrentVersion != 1 || !marker || !ledger {
					t.Fatalf("file beyond the call deadline = %+v, %v; marker %t, ledger %t", result, err, marker, ledger)
				}
				applied, err := readAppliedMigrations(ctx, base)
				if err != nil {
					t.Fatal(err)
				}
				if state, err := compareMigrationLedger([]migration{slow}, applied); err != nil || state.currentVersion != 1 || len(state.pending) != 0 {
					t.Fatalf("ledger after the file = %+v, %v", state, err)
				}
				return
			}
			if Classify(err) != ClassTimeout || SQLState(err) != test.sqlstate ||
				errors.Is(err, context.DeadlineExceeded) != (test.sqlstate == "") ||
				!strings.Contains(err.Error(), "apply migration 000001_slow_file.sql") {
				t.Fatalf("slow file = %v (SQLSTATE %q), want a timeout with SQLSTATE %q", err, SQLState(err), test.sqlstate)
			}
			if marker || ledger {
				t.Fatalf("the failed run left marker %t, ledger %t", marker, ledger)
			}
		})
	}
}
