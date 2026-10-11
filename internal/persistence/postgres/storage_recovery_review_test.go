package postgres

import (
	"context"
	"errors"
	"os"
	"testing"
	"time"
)

func TestStorageReviewMigrationRejectsNewerAndDrift(t *testing.T) {
	for _, invalid := range []string{"newer-version", "checksum-drift"} {
		t.Run(invalid, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(t.Context(), 30*time.Second)
			defer cancel()
			pool, _ := isolatedPools(t, ctx, storageReviewDatabaseURL(t))
			result, err := ApplyMigrations(ctx, pool)
			if err != nil {
				t.Fatal(err)
			}
			if invalid == "newer-version" {
				_, err = pool.Exec(ctx, `INSERT INTO contractor_schema_migrations(version,name,checksum) VALUES (999999,'999999_future.sql',decode(repeat('00',32),'hex'))`)
			} else {
				_, err = pool.Exec(ctx, `UPDATE contractor_schema_migrations SET checksum=decode(repeat('00',32),'hex') WHERE version=1`)
			}
			if err != nil {
				t.Fatal(err)
			}
			if _, err := ApplyMigrations(ctx, pool); !errors.Is(err, ErrMigrationDrift) {
				t.Fatalf("unsupported migration ledger: %v", err)
			}
			var count int
			if err := pool.QueryRow(ctx, `SELECT count(*) FROM contractor_schema_migrations WHERE version < 999999`).Scan(&count); err != nil || count != len(result.AppliedVersions) {
				t.Fatalf("rejected migration changed ledger: %d %v", count, err)
			}
		})
	}
}

func storageReviewDatabaseURL(t *testing.T) string {
	t.Helper()
	value := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if value == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	return value
}
