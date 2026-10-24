package postgres

import (
	"context"
	"errors"
	"os"
	"strings"
	"testing"
	"time"
)

// A migration run applies every pending migration and its ledger row in one
// transaction under the budget the operator chose (PER-R29, PER-R31). A
// reader of eval_controller_claims keeps a later pending migration
// (000014_controller_claim_shape.sql) from locking the table within the
// selected lock timeout: the run fails as a timeout, and the migrations it
// had already applied roll back with it, so the ledger and the schema stay
// at the version the run started from. A drifted ledger is refused without
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
	installMigrationPrefix(t, ctx, pool, 7)
	available, err := loadMigrations()
	if err != nil || available[6].version != 7 {
		t.Fatalf("embedded migrations: %v", err)
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
	if Classify(err) != ClassTimeout || !strings.Contains(err.Error(), "000014_controller_claim_shape.sql") {
		t.Fatalf("run blocked by a reader = %v, want the lock timeout of 000014", err)
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
	// 000008 adds artifact_scopes.revision_count; it ran before 000014 failed.
	if latest, counted := version(); latest != 7 || counted {
		t.Fatalf("after the failed run: ledger at %d, revision counter present %t; want 7 and absent", latest, counted)
	}

	if err := reader.Rollback(ctx); err != nil {
		t.Fatal(err)
	}
	// A ledger row that no longer matches its embedded file is drift: the
	// run refuses it before any pending migration runs (PER-R30).
	if _, err := pool.Exec(ctx, `UPDATE contractor_schema_migrations SET checksum = sha256(checksum) WHERE version = 7`); err != nil {
		t.Fatal(err)
	}
	if _, err := ApplyMigrations(ctx, pool, MigrationBudgets{}); !errors.Is(err, ErrMigrationDrift) {
		t.Fatalf("run on a drifted ledger = %v, want migration drift", err)
	}
	if latest, counted := version(); latest != 7 || counted {
		t.Fatalf("after the refused run: ledger at %d, revision counter present %t; want 7 and absent", latest, counted)
	}
	if _, err := pool.Exec(ctx, `UPDATE contractor_schema_migrations SET checksum = $1 WHERE version = 7`, available[6].checksum[:]); err != nil {
		t.Fatal(err)
	}
	result, err := ApplyMigrations(ctx, pool, MigrationBudgets{})
	if err != nil {
		t.Fatalf("run after the reader left: %v", err)
	}
	last := available[len(available)-1].version
	if len(result.AppliedVersions) == 0 || result.AppliedVersions[0] != 8 || result.CurrentVersion != last {
		t.Fatalf("run after the reader left = %+v, want versions 8 to %d", result, last)
	}
	if latest, counted := version(); latest != last || !counted {
		t.Fatalf("after the second run: ledger at %d, revision counter present %t", latest, counted)
	}
}
