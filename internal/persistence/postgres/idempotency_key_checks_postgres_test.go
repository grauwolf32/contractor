package postgres

import (
	"context"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
)

// idempotencyKeyCheck is the Idempotency-Key grammar as the schema's CHECK
// constraints spell it; contracts.ValidIdempotencyKey is its Go definition.
const idempotencyKeyCheck = `'^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$'`

// TestSharedPostgresIdempotencyKeyChecksMatchSharedCases evaluates the
// Idempotency-Key CHECK pattern of the migrated schema with PostgreSQL's own
// regular expressions against the shared cases, so a key that the Go stores
// accept is never refused by a constraint, and finds the constraints that
// carry it, so a vacuous pass is impossible.
func TestSharedPostgresIdempotencyKeyChecksMatchSharedCases(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 45*time.Second)
	defer cancel()
	pool, _ := isolatedPools(t, ctx, storageReviewDatabaseURL(t))
	if _, err := ApplyMigrations(ctx, pool, MigrationBudgets{}); err != nil {
		t.Fatal(err)
	}
	rows, err := pool.Query(ctx, `
SELECT conrelid::regclass::text, pg_get_constraintdef(oid)
FROM pg_constraint
WHERE contype = 'c' AND connamespace = current_schema()::regnamespace`)
	if err != nil {
		t.Fatal(err)
	}
	tables := map[string]bool{}
	for rows.Next() {
		var table, definition string
		if err := rows.Scan(&table, &definition); err != nil {
			t.Fatal(err)
		}
		if strings.Contains(definition, idempotencyKeyCheck) {
			tables[table] = true
		}
	}
	if err := rows.Err(); err != nil {
		t.Fatal(err)
	}
	for _, table := range []string{"projects", "workflow_runs", "credential_operations", "mutation_receipts"} {
		if !tables[table] {
			t.Errorf("no CHECK constraint of %s carries the Idempotency-Key pattern %s", table, idempotencyKeyCheck)
		}
	}

	cases := contractstest.ReadIdempotencyKeyCases(t)
	for _, group := range []struct {
		values []string
		valid  bool
	}{{cases.Valid, true}, {cases.Invalid, false}} {
		for _, value := range group.values {
			if strings.ContainsRune(value, 0) {
				continue // PostgreSQL text cannot hold NUL, so no column stores it.
			}
			var matches bool
			if err := pool.QueryRow(ctx, `SELECT $1::text ~ `+idempotencyKeyCheck, value).Scan(&matches); err != nil {
				t.Fatal(err)
			}
			if matches != group.valid {
				t.Errorf("PostgreSQL Idempotency-Key CHECK on %q = %v, want %v", value, matches, group.valid)
			}
		}
	}
}
