package runtimeconfig

import (
	"context"
	"os"
	"reflect"
	"testing"
	"time"

	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/jackc/pgx/v5"
)

func TestPostgresBindingPinsShareLocksAndExcludeWriters(t *testing.T) {
	url := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if url == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	ctx, cancel := context.WithTimeout(t.Context(), 20*time.Second)
	defer cancel()
	pool := isolatedRuntimeConfigPool(t, ctx, url)
	if _, err := persistencepostgres.ApplyMigrations(ctx, pool); err != nil {
		t.Fatal(err)
	}
	repository := NewRepository(pool)
	for _, label := range []string{"alpha", "zulu"} {
		if _, err := repository.CreateBinding(ctx, label, BuiltInRunSnapshot().Default.Config, "operator", time.Now()); err != nil {
			t.Fatal(err)
		}
	}
	first, err := pool.BeginTx(ctx, pgx.TxOptions{IsoLevel: pgx.RepeatableRead})
	if err != nil {
		t.Fatal(err)
	}
	defer first.Rollback(context.Background())
	second, err := pool.BeginTx(ctx, pgx.TxOptions{IsoLevel: pgx.RepeatableRead})
	if err != nil {
		t.Fatal(err)
	}
	defer second.Rollback(context.Background())
	pin, err := PinRunSnapshot(ctx, first, []string{"zulu", "alpha"}, allowRuntimeCredentialCatalog{}, TransactionLLMCredentialLookup{})
	if err != nil {
		t.Fatal(err)
	}
	// The deadline is a failure bound, not a performance assertion. The first
	// transaction remains open while the second pins the overlapping set.
	readCtx, cancelRead := context.WithTimeout(ctx, time.Second)
	other, err := PinRunSnapshot(readCtx, second, []string{"alpha", "zulu"}, allowRuntimeCredentialCatalog{}, TransactionLLMCredentialLookup{})
	cancelRead()
	if err != nil || !reflect.DeepEqual(pin, other) {
		t.Fatalf("concurrent pins differ: first=%+v second=%+v error=%v", pin, other, err)
	}
	if err := first.Commit(ctx); err != nil {
		t.Fatal(err)
	}
	// A remaining reader must exclude both non-key updates and deletion.
	for _, sql := range []string{
		`UPDATE runtime_label_bindings SET updated_by = 'blocked-writer' WHERE label = 'default'`,
		`DELETE FROM runtime_label_bindings WHERE label = 'alpha'`,
	} {
		writer, err := pool.Begin(ctx)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := writer.Exec(ctx, `SET LOCAL lock_timeout = '100ms'`); err != nil {
			t.Fatal(err)
		}
		_, err = writer.Exec(ctx, sql)
		writer.Rollback(ctx)
		if persistencepostgres.SQLState(err) != "55P03" {
			t.Fatalf("writer was not excluded by shared pin: %v", err)
		}
	}
	if err := second.Commit(ctx); err != nil {
		t.Fatal(err)
	}
	if err := repository.DeleteBinding(ctx, "alpha", 1); err != nil {
		t.Fatalf("writer did not resume after both readers committed: %v", err)
	}
}

func TestPostgresBindingCancelledPinReleasesEarlierLocks(t *testing.T) {
	url := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if url == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	ctx, cancel := context.WithTimeout(t.Context(), 15*time.Second)
	defer cancel()
	pool := isolatedRuntimeConfigPool(t, ctx, url)
	if _, err := persistencepostgres.ApplyMigrations(ctx, pool); err != nil {
		t.Fatal(err)
	}
	if _, err := NewRepository(pool).CreateBinding(ctx, "zulu", BuiltInRunSnapshot().Default.Config, "operator", time.Now()); err != nil {
		t.Fatal(err)
	}
	writer, err := pool.Begin(ctx)
	if err != nil {
		t.Fatal(err)
	}
	defer writer.Rollback(context.Background())
	if _, err := NewRepository(writer).LockBindingsForUpdate(ctx, []string{"zulu"}); err != nil {
		t.Fatal(err)
	}
	reader, err := pool.Begin(ctx)
	if err != nil {
		t.Fatal(err)
	}
	readCtx, cancelRead := context.WithTimeout(ctx, 100*time.Millisecond)
	_, err = PinRunSnapshot(readCtx, reader, []string{"zulu"}, allowRuntimeCredentialCatalog{}, TransactionLLMCredentialLookup{})
	cancelRead()
	reader.Rollback(ctx)
	if err == nil {
		t.Fatal("reader unexpectedly passed the exclusive writer")
	}
	// The cancelled transaction acquired default before waiting on zulu.
	// Rollback must release that partial acquisition.
	probeCtx, cancelProbe := context.WithTimeout(ctx, time.Second)
	defer cancelProbe()
	if _, err := NewRepository(writer).LockBindingsForUpdate(probeCtx, []string{DefaultLabel}); err != nil {
		t.Fatalf("cancelled reader orphaned default lock: %v", err)
	}
}
