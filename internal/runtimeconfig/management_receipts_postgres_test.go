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
// of the key is the permanent idempotency conflict, a repeated mutation
// replays without running again, and a failed mutation records nothing.
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
	mutate := func(ran *int, value string) func() (any, error) {
		return func() (any, error) {
			*ran++
			return result{Value: value}, nil
		}
	}
	var replay result
	if found, err := ReadManagementReplay(ctx, pool, receipt, &replay); err != nil || found {
		t.Fatalf("unused key replay = (%t, %v)", found, err)
	}
	ran := 0
	if err := persistencepostgres.InTx(ctx, pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		replayed, err := KeyedManagementMutation(ctx, tx, receipt, &replay, mutate(&ran, "first"))
		if err != nil || replayed {
			t.Fatalf("first mutation = (%t, %v)", replayed, err)
		}
		return nil
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
		if err := persistencepostgres.InTx(ctx, pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
			_, err := KeyedManagementMutation(ctx, tx, other, &ignored, mutate(&ran, "other"))
			return err
		}); !errors.Is(err, contracts.ErrIdempotencyConflict) {
			t.Fatalf("mutation under a key of another %s = %v, want the idempotency conflict", name, err)
		}
	}
	var repeated result
	if err := persistencepostgres.InTx(ctx, pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		replayed, err := KeyedManagementMutation(ctx, tx, receipt, &repeated, mutate(&ran, "second"))
		if err != nil || !replayed {
			t.Fatalf("repeated mutation = (%t, %v)", replayed, err)
		}
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	if repeated.Value != "first" || ran != 1 {
		t.Fatalf("repeated mutation replayed %+v after %d runs, want the first outcome after one run", repeated, ran)
	}
	failing := ManagementReceipt{
		KeyDigest: digest("d"), RequestDigest: digest("e"), Kind: "runtime-label.delete", ResourceID: "failing-label",
	}
	refused := errors.New("refused")
	if err := persistencepostgres.InTx(ctx, pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		_, err := KeyedManagementMutation(ctx, tx, failing, &replay, func() (any, error) { return nil, refused })
		return err
	}); !errors.Is(err, refused) {
		t.Fatalf("failed mutation = %v, want its own error", err)
	}
	if found, err := ReadManagementReplay(ctx, pool, failing, &replay); err != nil || found {
		t.Fatalf("failed mutation recorded a receipt: (%t, %v)", found, err)
	}
}
