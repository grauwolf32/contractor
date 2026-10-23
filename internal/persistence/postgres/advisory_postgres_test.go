package postgres

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"os"
	"testing"
	"time"

	"github.com/jackc/pgx/v5"
)

// TestPostgresTryLockAdvisoryNeverWaitsForAHolder pins what credential
// deletion reconciliation relies on: the try variant shares LockAdvisory's
// key, reports false at once while another transaction holds that key,
// exclusively or shared, and takes it once the holder's transaction ends.
func TestPostgresTryLockAdvisoryNeverWaitsForAHolder(t *testing.T) {
	url := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if url == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	ctx, cancel := context.WithTimeout(t.Context(), 15*time.Second)
	defer cancel()
	pool, _ := isolatedPools(t, ctx, url)
	suffix := make([]byte, 8)
	if _, err := rand.Read(suffix); err != nil {
		t.Fatal(err)
	}
	// Advisory locks are database-wide, so the key is unique to this run.
	key := NewAdvisoryKey("try-lock-test", hex.EncodeToString(suffix))

	tryLock := func() bool {
		t.Helper()
		var locked bool
		if err := InTx(ctx, pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
			var err error
			locked, err = TryLockAdvisory(ctx, tx, key)
			return err
		}); err != nil {
			t.Fatal(err)
		}
		return locked
	}
	for _, holder := range []struct {
		name string
		lock func(context.Context, DBTX, AdvisoryKey) error
	}{{"exclusive", LockAdvisory}, {"shared", LockAdvisoryShared}} {
		tx, err := BeginTx(ctx, pool, pgx.TxOptions{})
		if err != nil {
			t.Fatal(err)
		}
		if err := holder.lock(ctx, tx, key); err != nil {
			t.Fatal(err)
		}
		if tryLock() {
			t.Fatalf("try lock was taken while a %s holder was open", holder.name)
		}
		if err := tx.Rollback(ctx); err != nil {
			t.Fatal(err)
		}
		if !tryLock() {
			t.Fatalf("try lock was refused after the %s holder ended", holder.name)
		}
	}
}
