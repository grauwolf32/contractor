package app

import (
	"context"
	cryptorand "crypto/rand"
	"encoding/hex"
	"os"
	"testing"
	"time"

	"github.com/jackc/pgx/v5"
)

// Advisory locks are scoped per database while pg_locks lists the whole
// cluster, so another database holding the same lease key must not be
// reported as this database's holder.
func TestPostgresControlPlaneLeaseHolderIgnoresOtherDatabases(t *testing.T) {
	ctx, stop := context.WithTimeout(t.Context(), 30*time.Second)
	defer stop()
	pool := isolatedAppPool(t, ctx)
	databaseURL := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")

	admin, err := pgx.Connect(ctx, databaseURL)
	if err != nil {
		t.Fatal(err)
	}
	// Cleanups run last-in first-out: the database is dropped before the
	// admin connection closes.
	t.Cleanup(func() { _ = admin.Close(context.Background()) })
	random := make([]byte, 6)
	if _, err := cryptorand.Read(random); err != nil {
		t.Fatal(err)
	}
	name := "contractor_lease_" + hex.EncodeToString(random)
	identifier := pgx.Identifier{name}.Sanitize()
	if _, err := admin.Exec(ctx, "CREATE DATABASE "+identifier); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		cleanup, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		if _, err := admin.Exec(cleanup, "DROP DATABASE IF EXISTS "+identifier+" WITH (FORCE)"); err != nil {
			t.Errorf("drop lease database: %v", err)
		}
	})

	// The foreign holder connects first, so it usually has the lower PID and
	// an unscoped lookup ordered by PID would report it.
	foreignConfig, err := pgx.ParseConfig(databaseURL)
	if err != nil {
		t.Fatal(err)
	}
	foreignConfig.Database = name
	foreign, err := pgx.ConnectConfig(ctx, foreignConfig)
	if err != nil {
		t.Fatal(err)
	}
	defer foreign.Close(context.Background())
	if _, err := foreign.Exec(ctx, `SELECT pg_advisory_lock($1)`, controlPlaneLeaseKey); err != nil {
		t.Fatal(err)
	}

	active, err := openControlPlaneLease(ctx, pool)
	if err != nil {
		t.Fatal(err)
	}
	defer active.Close()
	if acquired, err := active.tryAcquire(ctx); err != nil || !acquired {
		t.Fatalf("lease held for another database blocked this one = (%v, %v)", acquired, err)
	}
	pid := int32(active.conn.PgConn().PID())
	if foreignPID := int32(foreign.PgConn().PID()); foreignPID >= pid {
		t.Logf("foreign holder PID %d is not below the active PID %d; the check still requires an exact match", foreignPID, pid)
	}
	if got := active.holderPID(ctx); got != pid {
		t.Fatalf("holderPID = %d, want this database's holder %d", got, pid)
	}
}
