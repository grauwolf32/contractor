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
	publisher := NewPublisher(pool, resolver, func() time.Time { return now })
	document := []byte(`{
  "apiVersion":"contractor/v1alpha1","kind":"RuntimeConfig",
  "metadata":{"name":"debug","version":"1"},
  "spec":{"worker":{"llmGateway":{"gateway":"local-litellm@1","credential":"worker-local"}}}
}`)
	created, err := publisher.Publish(ctx, document, "publish-debug-1", "operator")
	if err != nil || created.Replayed || resolverCalls.Load() != 1 {
		t.Fatalf("publish RuntimeConfig = (%+v, calls=%d, %v)", created, resolverCalls.Load(), err)
	}
	if created.Version.Spec.Worker.LLMGateway.Gateway.Value.Digest != "sha256:"+strings.Repeat("a", 64) {
		t.Fatalf("published Gateway ref = %+v", created.Version.Spec.Worker.LLMGateway.Gateway.Value)
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
	concurrentPublisher := NewPublisher(pool, concurrentResolver, func() time.Time { return now.Add(30 * time.Second) })
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
