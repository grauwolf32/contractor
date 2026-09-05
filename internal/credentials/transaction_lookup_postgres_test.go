package credentials

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

func TestTransactionLookupPinsRuntimeConfigWithoutAnotherPoolConnection(t *testing.T) {
	databaseURL := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if databaseURL == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	pool := isolatedCredentialPool(t, ctx, databaseURL)
	record, _ := sealedTestRecord(t, "transaction-worker", "secret-not-used-by-pinning")
	repository := NewRepository(pool)
	if err := repository.ReserveCredentialID(ctx, record.CredentialID, record.CreatedAt); err != nil {
		t.Fatal(err)
	}
	if err := repository.InsertCredential(ctx, record); err != nil {
		t.Fatal(err)
	}

	publisher, err := runtimeconfig.NewPublisher(runtimeconfig.PublisherOptions{
		Pool: pool,
		GatewayResolver: runtimeconfig.GatewayResolverFunc(func(
			context.Context, string,
		) (contracts.ResolvedLLMGatewayConfig, error) {
			return contracts.ResolvedLLMGatewayConfig{
				Ref: record.LLMGateway, Protocol: contracts.OpenAICompatibleProtocol,
				URL: "http://127.0.0.1:4000/v1",
			}, nil
		}),
		RuntimeCredentials: transactionTestRuntimeCredentialCatalog{},
		PlannerTelemetryAdapters: runtimeconfig.PlannerTelemetryAdapterCatalogFunc(
			func(string) bool { return true },
		),
	})
	if err != nil {
		t.Fatal(err)
	}
	publishBinding := func(label, credentialID string) {
		t.Helper()
		document := []byte(fmt.Sprintf(`{
  "apiVersion":"contractor/v1alpha1","kind":"RuntimeConfig",
  "metadata":{"name":"%s-config","version":"1"},
  "spec":{"worker":{"llmGateway":{"gateway":"%s@%s","credential":"%s"}}}
}`, label, record.LLMGateway.GatewayID, record.LLMGateway.Version, credentialID))
		published, publishErr := publisher.Publish(ctx, document, "publish-"+label, "operator")
		if publishErr != nil {
			t.Fatal(publishErr)
		}
		if _, bindErr := runtimeconfig.NewRepository(pool).CreateBinding(
			ctx, label, published.Version.Ref, "operator", time.Now().UTC(),
		); bindErr != nil {
			t.Fatal(bindErr)
		}
	}
	publishBinding("valid", record.CredentialID)
	publishBinding("missing", "missing-transaction-worker")

	development, err := NewStaticProvider(nil)
	if err != nil {
		t.Fatal(err)
	}
	factory, err := NewTransactionLookupFactory(development)
	if err != nil {
		t.Fatal(err)
	}

	limitedConfig := pool.Config()
	limitedConfig.MaxConns = 2
	limited, err := pgxpool.NewWithConfig(ctx, limitedConfig)
	if err != nil {
		t.Fatal(err)
	}
	defer limited.Close()
	warmFirst, err := limited.Acquire(ctx)
	if err != nil {
		t.Fatal(err)
	}
	warmSecond, err := limited.Acquire(ctx)
	if err != nil {
		warmFirst.Release()
		t.Fatal(err)
	}
	warmSecond.Release()
	warmFirst.Release()
	listener, err := limited.Acquire(ctx)
	if err != nil {
		t.Fatal(err)
	}
	defer listener.Release()
	if _, err := listener.Exec(ctx, `LISTEN contractor_transaction_lookup_test`); err != nil {
		t.Fatal(err)
	}

	start := time.Now()
	tx, err := limited.BeginTx(ctx, pgx.TxOptions{IsoLevel: pgx.RepeatableRead})
	if err != nil {
		t.Fatal(err)
	}
	bound, err := runtimeconfig.BindTransactionLLMCredentialLookup(tx, factory)
	if err != nil {
		_ = tx.Rollback(ctx)
		t.Fatal(err)
	}
	store := runstore.NewRunCreationPostgresStore(tx, bound)
	pinned, err := store.PinRuntimeLabels(ctx, []string{"valid"})
	if err != nil {
		_ = tx.Rollback(ctx)
		t.Fatal(err)
	}
	if len(pinned.LLMCredentialIDs) != 1 || pinned.LLMCredentialIDs[0] != record.CredentialID {
		_ = tx.Rollback(ctx)
		t.Fatalf("pinned LLM credential IDs = %v", pinned.LLMCredentialIDs)
	}
	if _, err := store.CreateRun(ctx, runstore.CreateRunParams{
		RunID: "run-transaction-lookup", OwnerID: "owner-transaction-lookup",
		WorkflowName: "test", WorkflowVersion: "1", WorkflowSchemaVersion: contracts.APIVersion,
		WorkflowSnapshot: json.RawMessage(`{"ref":{"name":"test","version":"1"}}`),
		Parameters:       map[string]string{}, RuntimeConfig: pinned,
	}); err != nil {
		_ = tx.Rollback(ctx)
		t.Fatal(err)
	}
	if err := tx.Commit(ctx); err != nil {
		t.Fatal(err)
	}
	if elapsed := time.Since(start); elapsed > 2*time.Second {
		t.Fatalf("transaction-bound valid lookup took %s under saturated pool", elapsed)
	}

	missingCtx, missingCancel := context.WithTimeout(ctx, 2*time.Second)
	defer missingCancel()
	missingTx, err := limited.BeginTx(missingCtx, pgx.TxOptions{IsoLevel: pgx.RepeatableRead})
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = missingTx.Rollback(context.Background()) }()
	missingLookup, err := runtimeconfig.BindTransactionLLMCredentialLookup(missingTx, factory)
	if err != nil {
		t.Fatal(err)
	}
	_, err = runstore.NewRunCreationPostgresStore(missingTx, missingLookup).
		PinRuntimeLabels(missingCtx, []string{"missing"})
	if !errors.Is(err, runtimeconfig.ErrInvalid) || errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("missing transaction credential error = %v", err)
	}

	duplicateDevelopment, err := NewStaticProvider([]StaticEntry{{
		Metadata: config.CredentialMetadata{
			Ref: recordCredentialRef(record), LLMGateway: record.LLMGateway, Unrestricted: true,
		},
		Token: contracts.NewSecretString("development-duplicate"),
	}})
	if err != nil {
		t.Fatal(err)
	}
	duplicateFactory, err := NewTransactionLookupFactory(duplicateDevelopment)
	if err != nil {
		t.Fatal(err)
	}
	duplicateLookup, err := runtimeconfig.BindTransactionLLMCredentialLookup(missingTx, duplicateFactory)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := duplicateLookup.LookupLLMCredential(missingCtx, record.CredentialID); !errors.Is(err, ErrConflict) {
		t.Fatalf("static/managed duplicate lookup error = %v", err)
	}
}

type transactionTestRuntimeCredentialCatalog struct{}

func (transactionTestRuntimeCredentialCatalog) ValidateRuntimeCredential(context.Context, string, ...string) error {
	return nil
}

func (transactionTestRuntimeCredentialCatalog) WithCredentialReferences(
	_ context.Context, fn func() error,
) error {
	return fn()
}

func recordCredentialRef(record Record) contracts.LLMCredentialRef {
	return contracts.LLMCredentialRef{CredentialID: record.CredentialID}
}
