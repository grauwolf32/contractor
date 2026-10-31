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

// Pending files and ledger rows commit in one transaction (PER-R29, PER-R31).
// Synthetic versions keep this protocol independent of the current baseline:
// a reader blocks the last file after an earlier pending file ran, so both
// pending files must roll back. Drift refuses the same run before any DDL.
func TestPostgresMigrationRunCommitsEveryPendingMigrationOrNone(t *testing.T) {
	url := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if url == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	ctx, cancel := context.WithTimeout(t.Context(), 2*time.Minute)
	defer cancel()
	pool, other := isolatedPools(t, ctx, url)
	available := []migration{}
	for i, sql := range []string{
		"CREATE TABLE migration_fixture (id integer PRIMARY KEY);",
		"CREATE TABLE migration_marker (id integer);",
		"ALTER TABLE migration_fixture ADD COLUMN value text;",
	} {
		contents := []byte(sql)
		available = append(available, migration{version: int64(i + 1), name: fmt.Sprintf("%06d_fixture.sql", i+1), contents: contents, checksum: sha256.Sum256(contents)})
	}
	if _, err := applyMigrations(ctx, pool, MigrationBudgets{}, available[:1]); err != nil {
		t.Fatal(err)
	}
	reader, err := other.Begin(ctx)
	if err != nil {
		t.Fatal(err)
	}
	defer reader.Rollback(context.Background())
	if _, err := reader.Exec(ctx, `LOCK TABLE migration_fixture IN ACCESS SHARE MODE`); err != nil {
		t.Fatal(err)
	}
	_, err = applyMigrations(ctx, pool, MigrationBudgets{StatementTimeout: 30 * time.Second, LockTimeout: 200 * time.Millisecond}, available)
	if Classify(err) != ClassTimeout || !strings.Contains(err.Error(), "000003_fixture.sql") {
		t.Fatalf("blocked run=%v, want lock timeout of the last file", err)
	}
	state := func(wantVersion int64, wantMarker bool) {
		t.Helper()
		var latest int64
		var marker bool
		if err := pool.QueryRow(ctx, `SELECT (SELECT max(version) FROM contractor_schema_migrations), to_regclass('migration_marker') IS NOT NULL`).Scan(&latest, &marker); err != nil {
			t.Fatal(err)
		}
		if latest != wantVersion || marker != wantMarker {
			t.Fatalf("ledger=%d marker=%t, want %d/%t", latest, marker, wantVersion, wantMarker)
		}
	}
	state(1, false)
	if err := reader.Rollback(ctx); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `UPDATE contractor_schema_migrations SET checksum=sha256(checksum) WHERE version=1`); err != nil {
		t.Fatal(err)
	}
	if _, err := applyMigrations(ctx, pool, MigrationBudgets{}, available); !errors.Is(err, ErrMigrationDrift) {
		t.Fatalf("drifted run=%v", err)
	}
	state(1, false)
	if _, err := pool.Exec(ctx, `UPDATE contractor_schema_migrations SET checksum=$1 WHERE version=1`, available[0].checksum[:]); err != nil {
		t.Fatal(err)
	}
	result, err := applyMigrations(ctx, pool, MigrationBudgets{}, available)
	if err != nil || result.CurrentVersion != 3 || len(result.AppliedVersions) != 2 || result.AppliedVersions[0] != 2 || result.AppliedVersions[1] != 3 {
		t.Fatalf("retry=%+v, %v", result, err)
	}
	state(3, true)
	var column bool
	if err := pool.QueryRow(ctx, `SELECT EXISTS(SELECT 1 FROM information_schema.columns WHERE table_schema=current_schema() AND table_name='migration_fixture' AND column_name='value')`).Scan(&column); err != nil || !column {
		t.Fatalf("last file missing after retry: column=%t, %v", column, err)
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
