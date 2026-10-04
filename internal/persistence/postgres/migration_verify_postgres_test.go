package postgres

import (
	"context"
	"errors"
	"fmt"
	"os"
	"strings"
	"testing"
	"time"
)

func TestPostgresVerifyMigrationsRejectsMissingBehindAndDriftedLedgers(t *testing.T) {
	databaseURL := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if databaseURL == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	ctx, cancel := context.WithTimeout(t.Context(), 45*time.Second)
	defer cancel()
	pool, _ := isolatedPools(t, ctx, databaseURL)
	if err := VerifyMigrations(ctx, pool); !errors.Is(err, ErrMigrationsPending) ||
		!strings.Contains(err.Error(), "contractor server migrate") {
		t.Fatalf("missing ledger verification: %v", err)
	}
	var ledger *string
	if err := pool.QueryRow(ctx, `SELECT to_regclass('contractor_schema_migrations')::text`).Scan(&ledger); err != nil || ledger != nil {
		t.Fatalf("read-only verification created a ledger: %v, %v", ledger, err)
	}
	if _, err := ApplyMigrations(ctx, pool); err != nil {
		t.Fatal(err)
	}
	if err := VerifyMigrations(ctx, pool); err != nil {
		t.Fatalf("matching ledger: %v", err)
	}
	available, err := loadMigrations()
	if err != nil {
		t.Fatal(err)
	}
	latest := available[len(available)-1]
	if _, err := pool.Exec(ctx, `DELETE FROM contractor_schema_migrations WHERE version=$1`, latest.version); err != nil {
		t.Fatal(err)
	}
	if err := VerifyMigrations(ctx, pool); !errors.Is(err, ErrMigrationsPending) ||
		!strings.Contains(err.Error(), fmt.Sprintf("%06d", latest.version)) {
		t.Fatalf("behind ledger verification: %v", err)
	}
	if _, err := pool.Exec(ctx, `INSERT INTO contractor_schema_migrations(version,name,checksum) VALUES ($1,$2,$3)`,
		latest.version, latest.name, latest.checksum[:]); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `INSERT INTO contractor_schema_migrations(version,name,checksum) VALUES (999999,'999999_future.sql',decode(repeat('00',32),'hex'))`); err != nil {
		t.Fatal(err)
	}
	if err := VerifyMigrations(ctx, pool); !errors.Is(err, ErrMigrationDrift) {
		t.Fatalf("newer ledger verification: %v", err)
	}
	if _, err := pool.Exec(ctx, `DELETE FROM contractor_schema_migrations WHERE version=999999`); err != nil {
		t.Fatal(err)
	}
	first := available[0]
	if _, err := pool.Exec(ctx, `UPDATE contractor_schema_migrations SET name='renamed.sql' WHERE version=$1`, first.version); err != nil {
		t.Fatal(err)
	}
	if err := VerifyMigrations(ctx, pool); !errors.Is(err, ErrMigrationDrift) {
		t.Fatalf("renamed ledger verification: %v", err)
	}
	if _, err := pool.Exec(ctx, `UPDATE contractor_schema_migrations SET name=$2,checksum=decode(repeat('00',32),'hex') WHERE version=$1`,
		first.version, first.name); err != nil {
		t.Fatal(err)
	}
	if err := VerifyMigrations(ctx, pool); !errors.Is(err, ErrMigrationDrift) {
		t.Fatalf("checksum-drifted ledger verification: %v", err)
	}
	if _, err := pool.Exec(ctx, `UPDATE contractor_schema_migrations SET checksum=$2 WHERE version=$1`, first.version, first.checksum[:]); err != nil {
		t.Fatal(err)
	}
	if err := VerifyMigrations(ctx, pool); err != nil {
		t.Fatalf("restored ledger verification: %v", err)
	}
}
