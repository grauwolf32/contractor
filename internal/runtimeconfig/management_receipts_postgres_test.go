package runtimeconfig

import (
	"context"
	"errors"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/contracts"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/jackc/pgx/v5"
)

// One receipt helper records and replays every Runtime management mutation:
// a key replays only its own request, operation and resource, any other use
// of the key is the permanent idempotency conflict, and a recorded key cannot
// be recorded twice.
func TestPostgresManagementReceiptsReplayOnlyTheirOwnRequest(t *testing.T) {
	databaseURL := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if databaseURL == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	pool := isolatedRuntimeConfigPool(t, ctx, databaseURL)
	if _, err := persistencepostgres.ApplyMigrations(ctx, pool); err != nil {
		t.Fatal(err)
	}
	type result struct {
		Value string `json:"value"`
	}
	digest := func(fill string) string { return "sha256:" + strings.Repeat(fill, 64) }
	receipt := ManagementReceipt{
		KeyDigest: digest("a"), RequestDigest: digest("b"), Kind: "runtime-label.create", ResourceID: "receipt-label",
	}
	at := time.Date(2026, 9, 1, 0, 0, 0, 0, time.UTC)
	var replay result
	if found, err := ReadManagementReplay(ctx, pool, receipt, &replay); err != nil || found {
		t.Fatalf("unused key replay = (%t, %v)", found, err)
	}
	if err := persistencepostgres.InTx(ctx, pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		if found, err := LockManagementReplay(ctx, tx, receipt, &replay); err != nil || found {
			t.Fatalf("locked unused key replay = (%t, %v)", found, err)
		}
		if err := RecordManagementResult(ctx, tx, receipt, result{Value: "first"}, " ", at); !errors.Is(err, ErrInvalid) {
			t.Fatalf("blank actor = %v, want ErrInvalid", err)
		}
		return RecordManagementResult(ctx, tx, receipt, result{Value: "first"}, "operator", at)
	}); err != nil {
		t.Fatal(err)
	}
	if found, err := ReadManagementReplay(ctx, pool, receipt, &replay); err != nil || !found || replay.Value != "first" {
		t.Fatalf("recorded key replay = (%t, %+v, %v)", found, replay, err)
	}
	for name, other := range map[string]ManagementReceipt{
		"request":   {KeyDigest: receipt.KeyDigest, RequestDigest: digest("c"), Kind: receipt.Kind, ResourceID: receipt.ResourceID},
		"operation": {KeyDigest: receipt.KeyDigest, RequestDigest: receipt.RequestDigest, Kind: "runtime-agent.delete", ResourceID: receipt.ResourceID},
		"resource":  {KeyDigest: receipt.KeyDigest, RequestDigest: receipt.RequestDigest, Kind: receipt.Kind, ResourceID: "other-label"},
	} {
		var ignored result
		if found, err := ReadManagementReplay(ctx, pool, other, &ignored); found ||
			!errors.Is(err, contracts.ErrIdempotencyConflict) || errors.Is(err, ErrConflict) {
			t.Fatalf("key reused with another %s = (%t, %v), want only the idempotency conflict", name, found, err)
		}
	}
	tx, err := pool.Begin(ctx)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = tx.Rollback(ctx) }()
	if err := RecordManagementResult(ctx, tx, receipt, result{Value: "second"}, "operator", at); !errors.Is(err, ErrConflict) {
		t.Fatalf("second record of a key = %v, want ErrConflict", err)
	}
}
