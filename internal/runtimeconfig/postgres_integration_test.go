package runtimeconfig

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"errors"
	"os"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/contracts"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgconn"
	"github.com/jackc/pgx/v5/pgxpool"
)

func TestPostgresRuntimeConfigBootstrapPublicationReplayAndBindingCAS(t *testing.T) {
	databaseURL := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if databaseURL == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	pool := isolatedRuntimeConfigPool(t, ctx, databaseURL)
	if _, err := persistencepostgres.ApplyMigrations(ctx, pool); err != nil {
		t.Fatalf("apply migrations: %v", err)
	}
	repository := NewRepository(pool)
	builtIn, err := repository.GetVersion(ctx, BuiltInName, BuiltInVersion)
	if err != nil || builtIn.Ref.Digest != BuiltInDigest || string(builtIn.CanonicalDocument) != BuiltInCanonicalDocument || !builtIn.BuiltIn {
		t.Fatalf("built-in RuntimeConfig = (%+v, %v)", builtIn, err)
	}
	defaultBinding, err := repository.GetBinding(ctx, DefaultLabel)
	if err != nil || defaultBinding.Revision != 1 || defaultBinding.Ref != builtIn.Ref {
		t.Fatalf("default binding = (%+v, %v)", defaultBinding, err)
	}
	if _, err := persistencepostgres.ApplyMigrations(ctx, pool); err != nil {
		t.Fatalf("reapply migrations: %v", err)
	}
	defaultAgain, err := repository.GetBinding(ctx, DefaultLabel)
	if err != nil || defaultAgain.Revision != 1 {
		t.Fatalf("default binding after replay = (%+v, %v)", defaultAgain, err)
	}

	var resolverCalls atomic.Int32
	var gatewayDigest atomic.Value
	gatewayDigest.Store("sha256:" + strings.Repeat("a", 64))
	resolver := GatewayResolverFunc(func(_ context.Context, selector string) (contracts.ResolvedLLMGatewayConfig, error) {
		resolverCalls.Add(1)
		id, version, _ := strings.Cut(selector, "@")
		return contracts.ResolvedLLMGatewayConfig{
			Ref:      contracts.LLMGatewayConfigRef{GatewayID: id, Version: version, Digest: gatewayDigest.Load().(string)},
			Protocol: contracts.OpenAICompatibleProtocol, URL: "http://127.0.0.1:4000/v1",
		}, nil
	})
	now := time.Date(2026, 8, 31, 12, 0, 0, 0, time.UTC)
	publisher, err := NewPublisher(PublisherOptions{
		Pool: pool, GatewayResolver: resolver,
		RuntimeCredentials:       allowRuntimeCredentialCatalog{},
		PlannerTelemetryAdapters: PlannerTelemetryAdapterCatalogFunc(func(ref string) bool { return ref == "otlp-http@1" }),
		Now:                      func() time.Time { return now },
	})
	if err != nil {
		t.Fatal(err)
	}
	document := []byte(`{
  "apiVersion":"contractor/v1alpha1","kind":"RuntimeConfig",
  "metadata":{"name":"debug","version":"1"},
  "spec":{"worker":{"llmGateway":{"gateway":"local-litellm@1","credential":"worker-local"},"caido":{"adapter":"caido-graphql@1","endpoint":"https://caido.internal/prefix","credential":"caido-lab"}}}
}`)
	created, err := publisher.Publish(ctx, document, "publish-debug-1", "operator")
	if err != nil || created.Replayed || resolverCalls.Load() != 1 {
		t.Fatalf("publish RuntimeConfig = (%+v, calls=%d, %v)", created, resolverCalls.Load(), err)
	}
	if created.Version.Spec.Worker.LLMGateway.Gateway.Value.Digest != "sha256:"+strings.Repeat("a", 64) {
		t.Fatalf("published Gateway ref = %+v", created.Version.Spec.Worker.LLMGateway.Gateway.Value)
	}
	if caido := created.Version.Spec.Worker.Caido; !caido.Present || caido.Clear ||
		caido.Value.Endpoint != "https://caido.internal/prefix" || caido.Value.Credential != "caido-lab" {
		t.Fatalf("published Caido config = %+v", caido)
	}
	gatewayDigest.Store("sha256:" + strings.Repeat("b", 64))
	replayed, err := publisher.Publish(ctx, document, "publish-debug-1", "another-operator")
	if err != nil || !replayed.Replayed || replayed.Version.Ref != created.Version.Ref || resolverCalls.Load() != 1 {
		t.Fatalf("replay after catalog change = (%+v, calls=%d, %v)", replayed, resolverCalls.Load(), err)
	}
	changed := bytesReplace(document, `"worker-local"`, `"other-worker"`)
	if _, err := publisher.Publish(ctx, changed, "publish-debug-1", "operator"); !errors.Is(err, ErrConflict) || resolverCalls.Load() != 1 {
		t.Fatalf("changed idempotency replay = (calls=%d, %v)", resolverCalls.Load(), err)
	}

	// Both requests pass the initial lookup before either transaction exists,
	// and the mutable resolver deliberately returns different exact refs. The
	// losing transaction must still replay the committed publication.
	concurrentDocument := bytesReplace(
		bytesReplace(document, `"name":"debug"`, `"name":"concurrent"`),
		`"credential":"worker-local"`, `"credential":"concurrent-worker"`,
	)
	var concurrentCalls atomic.Int32
	var resolvedTogether sync.WaitGroup
	resolvedTogether.Add(2)
	concurrentResolver := GatewayResolverFunc(func(_ context.Context, selector string) (contracts.ResolvedLLMGatewayConfig, error) {
		call := concurrentCalls.Add(1)
		resolvedTogether.Done()
		resolvedTogether.Wait()
		id, version, _ := strings.Cut(selector, "@")
		digit := "c"
		if call == 2 {
			digit = "d"
		}
		return contracts.ResolvedLLMGatewayConfig{
			Ref:      contracts.LLMGatewayConfigRef{GatewayID: id, Version: version, Digest: "sha256:" + strings.Repeat(digit, 64)},
			Protocol: contracts.OpenAICompatibleProtocol, URL: "http://127.0.0.1:4000/v1",
		}, nil
	})
	concurrentPublisher, err := NewPublisher(PublisherOptions{
		Pool: pool, GatewayResolver: concurrentResolver,
		RuntimeCredentials:       allowRuntimeCredentialCatalog{},
		PlannerTelemetryAdapters: PlannerTelemetryAdapterCatalogFunc(func(ref string) bool { return ref == "otlp-http@1" }),
		Now:                      func() time.Time { return now.Add(30 * time.Second) },
	})
	if err != nil {
		t.Fatal(err)
	}
	concurrentResults := make([]PublishResult, 2)
	concurrentErrors := make([]error, 2)
	var concurrentWait sync.WaitGroup
	for index := range concurrentResults {
		concurrentWait.Add(1)
		go func(index int) {
			defer concurrentWait.Done()
			concurrentResults[index], concurrentErrors[index] = concurrentPublisher.Publish(
				ctx, concurrentDocument, "publish-concurrent-1", "operator",
			)
		}(index)
	}
	concurrentWait.Wait()
	if concurrentErrors[0] != nil || concurrentErrors[1] != nil ||
		concurrentResults[0].Version.Ref != concurrentResults[1].Version.Ref ||
		concurrentResults[0].Replayed == concurrentResults[1].Replayed {
		t.Fatalf("concurrent idempotency results = (%+v, %+v), errors = %v", concurrentResults[0], concurrentResults[1], concurrentErrors)
	}

	secondDocument := []byte(`{
  "apiVersion":"contractor/v1alpha1","kind":"RuntimeConfig",
  "metadata":{"name":"no-debug","version":"1"},
  "spec":{"worker":{"telemetry":null}}
}`)
	second, err := publisher.Publish(ctx, secondDocument, "publish-no-debug-1", "operator")
	if err != nil {
		t.Fatal(err)
	}
	binding, err := repository.CreateBinding(ctx, "investigate", builtIn.Ref, "operator", now)
	if err != nil || binding.Revision != 1 {
		t.Fatalf("create binding = (%+v, %v)", binding, err)
	}

	refs := []Ref{created.Version.Ref, second.Version.Ref}
	errorsFound := make([]error, 2)
	results := make([]Binding, 2)
	var wait sync.WaitGroup
	for index := range refs {
		wait.Add(1)
		go func(index int) {
			defer wait.Done()
			results[index], errorsFound[index] = NewRepository(pool).Rebind(
				ctx, "investigate", 1, refs[index], "operator", now.Add(time.Minute),
			)
		}(index)
	}
	wait.Wait()
	successes, stale := 0, 0
	for _, err := range errorsFound {
		switch {
		case err == nil:
			successes++
		case errors.Is(err, ErrPrecondition):
			stale++
		default:
			t.Fatalf("unexpected concurrent CAS error: %v", err)
		}
	}
	if successes != 1 || stale != 1 {
		t.Fatalf("concurrent CAS outcomes = %v", errorsFound)
	}
	current, err := repository.GetBinding(ctx, "investigate")
	if err != nil || current.Revision != 2 {
		t.Fatalf("binding after CAS = (%+v, %v)", current, err)
	}
	noOp, err := repository.Rebind(ctx, current.Label, current.Revision, current.Ref, "operator", now.Add(2*time.Minute))
	if err != nil || noOp.Revision != current.Revision {
		t.Fatalf("semantic replay advanced binding = (%+v, %v)", noOp, err)
	}
	if err := repository.DeleteBinding(ctx, DefaultLabel, 1); !errors.Is(err, ErrReserved) {
		t.Fatalf("delete default error = %v", err)
	}
	_, err = pool.Exec(ctx, `DELETE FROM runtime_label_bindings WHERE label = 'default'`)
	assertRuntimeConfigSQLState(t, err, "23514")
	_, err = pool.Exec(ctx, `UPDATE runtime_config_versions SET actor_id = 'changed' WHERE name = 'debug' AND version = '1'`)
	assertRuntimeConfigSQLState(t, err, "23514")
}

func TestPostgresRuntimeManagementBindingMutationIsCASAndReplaySafe(t *testing.T) {
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
	now := time.Date(2026, 9, 1, 0, 0, 0, 0, time.UTC)
	publisher, err := NewPublisher(PublisherOptions{
		Pool: pool, RuntimeCredentials: allowRuntimeCredentialCatalog{}, Now: func() time.Time { return now },
		PlannerTelemetryAdapters: PlannerTelemetryAdapterCatalogFunc(func(ref string) bool { return ref == "otlp-http@1" }),
	})
	if err != nil {
		t.Fatal(err)
	}
	bindings, err := NewBindingService(pool, allowRuntimeCredentialCatalog{})
	if err != nil {
		t.Fatal(err)
	}
	management, err := NewManagementService(pool, publisher, bindings)
	if err != nil {
		t.Fatal(err)
	}
	refs := make([]Ref, 3)
	for index, name := range []string{"managed-base", "managed-a", "managed-b"} {
		document := []byte(`{"apiVersion":"contractor/v1alpha1","kind":"RuntimeConfig","metadata":{"name":"` + name + `","version":"1"},"spec":{"worker":{"telemetry":null}}}`)
		published, publishErr := management.Publish(ctx, document, "publish-"+name, "operator")
		if publishErr != nil {
			t.Fatal(publishErr)
		}
		refs[index] = published.Version.Ref
	}
	created, err := management.CreateBinding(
		ctx, "managed", refs[0], "binding-create", "operator", now,
	)
	if err != nil || created.Binding == nil || created.Binding.Revision != 1 || created.Replayed {
		t.Fatalf("managed binding create = (%+v, %v)", created, err)
	}
	replayedCreate, err := management.CreateBinding(
		ctx, "managed", refs[0], "binding-create", "another-operator", now.Add(time.Minute),
	)
	if err != nil || replayedCreate.Binding == nil || !replayedCreate.Replayed ||
		replayedCreate.Binding.Revision != 1 {
		t.Fatalf("managed binding replay = (%+v, %v)", replayedCreate, err)
	}
	if _, err := management.CreateBinding(
		ctx, "managed", refs[1], "binding-create", "operator", now,
	); !errors.Is(err, ErrConflict) {
		t.Fatalf("changed create replay error = %v", err)
	}

	results := make([]BindingMutationResult, 2)
	errorsFound := make([]error, 2)
	var wait sync.WaitGroup
	for index := range results {
		wait.Add(1)
		go func(index int) {
			defer wait.Done()
			results[index], errorsFound[index] = management.Rebind(
				ctx, "managed", 1, refs[index+1], "binding-race-"+string(rune('a'+index)),
				"operator", now.Add(2*time.Minute),
			)
		}(index)
	}
	wait.Wait()
	winner := -1
	for index, found := range errorsFound {
		if found == nil {
			winner = index
			continue
		}
		if !errors.Is(found, ErrPrecondition) {
			t.Fatalf("unexpected managed CAS error %d: %v", index, found)
		}
	}
	if winner < 0 || errorsFound[1-winner] == nil || results[winner].Binding == nil ||
		results[winner].Binding.Revision != 2 {
		t.Fatalf("managed CAS = results %+v errors %v", results, errorsFound)
	}
	winnerKey := "binding-race-" + string(rune('a'+winner))
	replay, err := management.Rebind(
		ctx, "managed", 1, refs[winner+1], winnerKey, "operator", now.Add(3*time.Minute),
	)
	if err != nil || !replay.Replayed || replay.Binding == nil || replay.Binding.Revision != 2 {
		t.Fatalf("managed CAS replay = (%+v, %v)", replay, err)
	}
	deleted, err := management.DeleteBinding(
		ctx, "managed", 2, "binding-delete", "operator", now.Add(4*time.Minute),
	)
	if err != nil || !deleted.Deleted || deleted.Replayed {
		t.Fatalf("managed delete = (%+v, %v)", deleted, err)
	}
	deletedReplay, err := management.DeleteBinding(
		ctx, "managed", 2, "binding-delete", "operator", now.Add(5*time.Minute),
	)
	if err != nil || !deletedReplay.Deleted || !deletedReplay.Replayed {
		t.Fatalf("managed delete replay = (%+v, %v)", deletedReplay, err)
	}
	if _, err := pool.Exec(ctx, `UPDATE runtime_management_operations SET actor_id = 'changed'`); err == nil {
		t.Fatal("immutable Runtime management operation was updated")
	}
}

func isolatedRuntimeConfigPool(t *testing.T, ctx context.Context, databaseURL string) *pgxpool.Pool {
	t.Helper()
	admin, err := pgxpool.New(ctx, databaseURL)
	if err != nil {
		t.Fatal(err)
	}
	if err := admin.Ping(ctx); err != nil {
		admin.Close()
		t.Fatal(err)
	}
	randomBytes := make([]byte, 8)
	if _, err := rand.Read(randomBytes); err != nil {
		admin.Close()
		t.Fatal(err)
	}
	schema := "runtime_config_test_" + hex.EncodeToString(randomBytes)
	identifier := pgx.Identifier{schema}.Sanitize()
	if _, err := admin.Exec(ctx, `CREATE SCHEMA `+identifier); err != nil {
		admin.Close()
		t.Fatal(err)
	}
	config, err := pgxpool.ParseConfig(databaseURL)
	if err != nil {
		admin.Close()
		t.Fatal(err)
	}
	config.ConnConfig.RuntimeParams["search_path"] = schema
	pool, err := pgxpool.NewWithConfig(ctx, config)
	if err != nil {
		admin.Close()
		t.Fatal(err)
	}
	t.Cleanup(func() {
		pool.Close()
		cleanupCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		if _, err := admin.Exec(cleanupCtx, `DROP SCHEMA `+identifier+` CASCADE`); err != nil {
			t.Logf("drop isolated schema: %v", err)
		}
		admin.Close()
	})
	return pool
}

func bytesReplace(input []byte, oldValue, newValue string) []byte {
	return []byte(strings.Replace(string(input), oldValue, newValue, 1))
}

func assertRuntimeConfigSQLState(t *testing.T, err error, want string) {
	t.Helper()
	var postgresError *pgconn.PgError
	if !errors.As(err, &postgresError) || postgresError.Code != want {
		t.Fatalf("PostgreSQL error = %v, want SQLSTATE %s", err, want)
	}
}

type allowRuntimeCredentialCatalog struct{}

func (allowRuntimeCredentialCatalog) ValidateRuntimeCredential(context.Context, string, ...string) error {
	return nil
}

func (allowRuntimeCredentialCatalog) WithCredentialReferences(_ context.Context, fn func() error) error {
	return fn()
}
