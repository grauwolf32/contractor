package postgres

import (
	"context"
	"errors"
	"os"
	"testing"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

// Version 48 changes the storage representation of every existing inline blob.
// Interrupt its actual DDL after earlier statements ran, then use the normal
// migrator to prove both the schema ledger and existing bytes are recoverable.
func TestStorageReviewInterruptedBlobUpgrade(t *testing.T) {
	for _, interrupt := range []string{"cancel", "terminate-backend"} {
		t.Run(interrupt, func(t *testing.T) {
			ctx, stop := context.WithTimeout(t.Context(), 45*time.Second)
			defer stop()
			base, observer := isolatedPools(t, ctx, storageReviewDatabaseURL(t))
			installStorageReviewPrefix(t, ctx, base, 47)
			if _, err := base.Exec(ctx, `INSERT INTO artifact_blobs (sha256,payload,size_bytes)
VALUES (sha256('historical-content'::bytea),'historical-content'::bytea,18)`); err != nil {
				t.Fatal(err)
			}
			config := base.Config()
			config.MaxConns = 1
			migrator, err := pgxpool.NewWithConfig(ctx, config)
			if err != nil {
				t.Fatal(err)
			}
			defer migrator.Close()
			connection, err := migrator.Acquire(ctx)
			if err != nil {
				t.Fatal(err)
			}
			pid := connection.Conn().PgConn().PID()
			connection.Release()
			locker, err := base.Begin(ctx)
			if err != nil {
				t.Fatal(err)
			}
			defer locker.Rollback(context.Background())
			if _, err := locker.Exec(ctx, `LOCK TABLE artifact_blobs IN ACCESS SHARE MODE`); err != nil {
				t.Fatal(err)
			}
			migrationCtx, cancel := context.WithCancel(ctx)
			defer cancel()
			done := make(chan error, 1)
			go func() { _, err := ApplyMigrations(migrationCtx, migrator); done <- err }()
			waitStorageReviewMigrationLock(t, ctx, observer, pid)
			if interrupt == "cancel" {
				cancel()
			} else {
				var terminated bool
				if err := observer.QueryRow(ctx, `SELECT pg_terminate_backend($1)`, pid).Scan(&terminated); err != nil || !terminated {
					t.Fatalf("terminate migrator=%v: %v", terminated, err)
				}
			}
			if err := <-done; err == nil {
				t.Fatal("interrupted migration succeeded")
			}
			if err := locker.Rollback(ctx); err != nil {
				t.Fatal(err)
			}
			var latest int64
			var settings *string
			if err := observer.QueryRow(ctx, `SELECT max(version),to_regclass('artifact_blob_settings')::text FROM contractor_schema_migrations`).Scan(&latest, &settings); err != nil {
				t.Fatal(err)
			}
			if latest != 47 || settings != nil {
				t.Fatalf("partial migration escaped rollback: version=%d settings=%v", latest, settings)
			}
			result, err := ApplyMigrations(ctx, migrator)
			if err != nil || len(result.AppliedVersions) == 0 || result.AppliedVersions[0] != 48 {
				t.Fatalf("resume upgrade: %+v %v", result, err)
			}
			var payload, backend string
			var intact bool
			if err := observer.QueryRow(ctx, `SELECT convert_from(payload,'UTF8'),backend,sha256=sha256(payload) AND size_bytes=octet_length(payload) FROM artifact_blobs`).Scan(&payload, &backend, &intact); err != nil {
				t.Fatal(err)
			}
			if payload != "historical-content" || backend != "postgresql" || !intact {
				t.Fatalf("historical blob changed: %q %s %v", payload, backend, intact)
			}
			if err := observer.QueryRow(ctx, `SELECT backend FROM artifact_blob_settings WHERE singleton`).Scan(&backend); err != nil || backend != "postgresql" {
				t.Fatalf("populated-store backend=%s: %v", backend, err)
			}
			again, err := ApplyMigrations(ctx, migrator)
			if err != nil || len(again.AppliedVersions) != 0 {
				t.Fatalf("idempotent migration: %+v %v", again, err)
			}
		})
	}
}

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

func installStorageReviewPrefix(t *testing.T, ctx context.Context, pool *pgxpool.Pool, last int64) {
	t.Helper()
	available, err := loadMigrations()
	if err != nil {
		t.Fatal(err)
	}
	err = InTx(ctx, pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		if _, err := tx.Exec(ctx, `CREATE TABLE contractor_schema_migrations (
version bigint PRIMARY KEY CHECK(version>0), name text NOT NULL,
checksum bytea NOT NULL CHECK(octet_length(checksum)=32),
applied_at timestamptz NOT NULL DEFAULT clock_timestamp())`); err != nil {
			return err
		}
		for _, item := range available {
			if item.version > last {
				break
			}
			if _, err := tx.Exec(ctx, string(item.contents)); err != nil {
				return err
			}
			if _, err := tx.Exec(ctx, `INSERT INTO contractor_schema_migrations(version,name,checksum) VALUES($1,$2,$3)`, item.version, item.name, item.checksum[:]); err != nil {
				return err
			}
		}
		return nil
	})
	if err != nil {
		t.Fatalf("install historical schema: %v", err)
	}
}

func waitStorageReviewMigrationLock(t *testing.T, ctx context.Context, pool *pgxpool.Pool, pid uint32) {
	t.Helper()
	deadline := time.NewTimer(5 * time.Second)
	defer deadline.Stop()
	tick := time.NewTicker(5 * time.Millisecond)
	defer tick.Stop()
	for {
		var waiting bool
		if err := pool.QueryRow(ctx, `SELECT EXISTS(SELECT 1 FROM pg_stat_activity WHERE pid=$1 AND wait_event_type='Lock' AND query LIKE '%ALTER TABLE artifact_blobs%')`, pid).Scan(&waiting); err != nil {
			t.Fatal(err)
		}
		if waiting {
			return
		}
		select {
		case <-deadline.C:
			t.Fatal("migration never reached blocked blob DDL")
		case <-ctx.Done():
			t.Fatal(ctx.Err())
		case <-tick.C:
		}
	}
}
