package agentprincipal

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"errors"
	"os"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/apierror"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/llmgateway"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

type countingPrincipalQueryTx struct {
	pgx.Tx
	queries []string
	args    [][]any
}

func (tx *countingPrincipalQueryTx) Query(ctx context.Context, sql string, args ...any) (pgx.Rows, error) {
	tx.queries = append(tx.queries, sql)
	tx.args = append(tx.args, args)
	return tx.Tx.Query(ctx, sql, args...)
}

func TestPostgresPrincipalAdapterPagesBatchAndPreserveSnapshot(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 45*time.Second)
	defer cancel()
	pool := isolatedPrincipalPool(t, ctx)
	ref := runtimeconfig.BuiltInRunSnapshot().Default.Config
	bindings := runtimeconfig.NewRepository(pool)
	for _, label := range []string{"alpha", "beta", "gamma"} {
		if _, err := bindings.CreateBinding(ctx, label, ref, "operator", time.Now().UTC()); err != nil {
			t.Fatal(err)
		}
	}
	ids := []string{strings.Repeat("a", 64), strings.Repeat("b", 64), strings.Repeat("c", 64), strings.Repeat("d", 64)}
	labelSets := [][]string{{"alpha", "beta"}, {"beta"}, {"gamma"}, {}}
	for index, id := range ids {
		at := time.Now().UTC()
		inserted, err := NewRepository(pool).Insert(ctx, Principal{
			RuntimeAgentID: id, Labels: labelSets[index], LabelRevision: 1,
			CreatedBy: "operator", CreatedAt: at, UpdatedBy: "operator", UpdatedAt: at,
		})
		if err != nil || !inserted {
			t.Fatalf("insert principal %s: inserted=%v error=%v", id, inserted, err)
		}
	}
	service, err := NewService(ServiceOptions{Pool: pool})
	if err != nil {
		t.Fatal(err)
	}

	// The visible principals share two bindings and one version. The third
	// principal is only a pagination sentinel and its gamma binding is omitted.
	read, err := pool.BeginTx(ctx, pgx.TxOptions{IsoLevel: pgx.RepeatableRead, AccessMode: pgx.ReadOnly})
	if err != nil {
		t.Fatal(err)
	}
	counted := &countingPrincipalQueryTx{Tx: read}
	page, hasMore, err := listAdapterPage(ctx, counted, "", 2)
	if err != nil || !hasMore || len(page) != 2 || page[0].Principal.RuntimeAgentID != ids[0] || page[1].Principal.RuntimeAgentID != ids[1] {
		t.Fatalf("first page = (%+v, %v, %v)", page, hasMore, err)
	}
	if len(counted.queries) != 3 {
		t.Fatalf("first page ran %d catalog queries, want 3", len(counted.queries))
	}
	if labels, ok := counted.args[1][0].([]string); !ok || !slices.Equal(labels, []string{"alpha", "beta"}) {
		t.Fatalf("first page requested bindings = %v", counted.args[1])
	}
	if refs, ok := counted.args[2][0].([]string); !ok || !slices.Equal(refs, []string{ref.Name}) {
		t.Fatalf("first page requested versions = %v", counted.args[2])
	}
	for _, row := range page {
		expected, err := service.RequiredRuntimeAdapters(ctx, row.Principal.Labels)
		if err != nil || !slices.Equal(row.RequiredRuntimeAdapters, expected) {
			t.Fatalf("adapter projection for %s = %v, want %v, error=%v", row.Principal.RuntimeAgentID, row.RequiredRuntimeAdapters, expected, err)
		}
	}
	counted.queries, counted.args = nil, nil
	fullPage, hasMore, err := listAdapterPage(ctx, counted, "", 3)
	if err != nil || !hasMore || len(fullPage) != 3 || len(counted.queries) != 3 {
		t.Fatalf("three-row page = (%+v, %v, %v), queries=%d", fullPage, hasMore, err, len(counted.queries))
	}
	if refs, ok := counted.args[2][0].([]string); !ok || !slices.Equal(refs, []string{ref.Name}) {
		t.Fatalf("shared version was not deduplicated: %v", counted.args[2])
	}
	if err := read.Commit(ctx); err != nil {
		t.Fatal(err)
	}
	last, more, err := service.ListWithRequiredRuntimeAdapters(ctx, ids[2], 1)
	if err != nil || more || len(last) != 1 || last[0].Principal.RuntimeAgentID != ids[3] ||
		last[0].RequiredRuntimeAdapters == nil || len(last[0].RequiredRuntimeAdapters) != 0 {
		t.Fatalf("unlabeled final page = (%+v, %v, %v)", last, more, err)
	}

	// A repeatable-read list must see the same old label and binding even
	// after another connection removes both in a newer transaction.
	snapshot, err := pool.BeginTx(ctx, pgx.TxOptions{IsoLevel: pgx.RepeatableRead, AccessMode: pgx.ReadOnly})
	if err != nil {
		t.Fatal(err)
	}
	defer snapshot.Rollback(context.Background())
	var count int
	if err := snapshot.QueryRow(ctx, `SELECT count(*) FROM runtime_agent_principals`).Scan(&count); err != nil || count != 4 {
		t.Fatalf("establish principal snapshot: count=%d error=%v", count, err)
	}
	if _, err := pool.Exec(ctx, `UPDATE runtime_agent_principals SET labels = '{}'::text[], label_revision = 2 WHERE runtime_agent_id = $1`, ids[2]); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `DELETE FROM runtime_label_bindings WHERE label = 'gamma'`); err != nil {
		t.Fatal(err)
	}
	oldPage, more, err := listAdapterPage(ctx, snapshot, ids[1], 1)
	if err != nil || !more || len(oldPage) != 1 || !slices.Equal(oldPage[0].Principal.Labels, []string{"gamma"}) {
		t.Fatalf("repeatable-read page after concurrent deletion = (%+v, %v, %v)", oldPage, more, err)
	}
	freshPage, more, err := service.ListWithRequiredRuntimeAdapters(ctx, ids[1], 1)
	if err != nil || !more || len(freshPage) != 1 || len(freshPage[0].Principal.Labels) != 0 {
		t.Fatalf("fresh page after concurrent deletion = (%+v, %v, %v)", freshPage, more, err)
	}
}

func TestPostgresRuntimeAgentPrincipalSeedCASAndDelete(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	pool := isolatedPrincipalPool(t, ctx)

	now := time.Date(2026, 9, 1, 1, 0, 0, 0, time.UTC)
	publisher, err := runtimeconfig.NewPublisher(runtimeconfig.PublisherOptions{
		Pool:                     pool,
		PlannerTelemetryAdapters: runtimeconfig.PlannerTelemetryAdapterCatalogFunc(func(ref string) bool { return ref == "otlp-http@1" }),
		GatewayResolver: runtimeconfig.GatewayResolverFunc(func(_ context.Context, selector string) (llmgateway.ResolvedLLMGatewayConfig, error) {
			gatewayID, version, _ := strings.Cut(selector, "@")
			return llmgateway.ResolvedLLMGatewayConfig{
				Ref: llmgateway.LLMGatewayConfigRef{
					GatewayID: gatewayID, Version: version,
					Digest: "sha256:" + strings.Repeat("a", 64),
				},
				Protocol: llmgateway.OpenAICompatibleProtocol,
				URL:      "http://127.0.0.1:4000/v1",
			}, nil
		}),
		RuntimeCredentials: allowRuntimeCredentialCatalog{},
		Now:                func() time.Time { return now },
	})
	if err != nil {
		t.Fatal(err)
	}
	published, err := publisher.Publish(ctx, []byte(`{
  "apiVersion":"contractor/v1alpha1","kind":"RuntimeConfig",
  "metadata":{"name":"agent-route","version":"1"},
  "spec":{"worker":{"llmGateway":{"gateway":"local-litellm@1"}}}
}`), "principal-config", "operator")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := runtimeconfig.NewRepository(pool).CreateBinding(
		ctx, "agent-route", published.Version.Ref, "operator", now,
	); err != nil {
		t.Fatal(err)
	}

	guard := &recordingDeletionGuard{}
	service, err := NewService(ServiceOptions{
		Pool: pool, DeletionGuard: guard, Now: func() time.Time { return now },
	})
	if err != nil {
		t.Fatal(err)
	}
	principalID := strings.Repeat("a", 64)
	created, err := service.Register(ctx, principalID, []string{"agent-route"})
	if err != nil || created.LabelRevision != 1 || !slices.Equal(created.Labels, []string{"agent-route"}) {
		t.Fatalf("created principal = (%+v, %v)", created, err)
	}
	// Startup arguments are seed-only after the first successful registration.
	replayed, err := service.Register(ctx, principalID, []string{})
	if err != nil || replayed.LabelRevision != 1 || !slices.Equal(replayed.Labels, created.Labels) {
		t.Fatalf("replayed principal = (%+v, %v)", replayed, err)
	}
	// The refusal still matches runtimeconfig.ErrConflict, but the public API
	// answers its own 409 runtime_agent_in_use.
	if _, err := service.DeleteIdempotent(
		ctx, principalID, 1, "principal-delete-labeled", "operator", now.Add(time.Minute),
	); !errors.Is(err, ErrInUse) || !errors.Is(err, runtimeconfig.ErrConflict) {
		t.Fatalf("delete labeled principal error = %v", err)
	} else if classified, ok := apierror.Of(err); !ok || classified.ErrorKind() != apierror.RuntimeAgentInUse {
		t.Fatalf("delete labeled principal answers %v, want runtime_agent_in_use", err)
	}
	if _, err := service.ReplaceLabelsIdempotent(
		ctx, principalID, 1, []string{"missing"}, "principal-unknown-label", "operator", now.Add(time.Minute),
	); !errors.Is(err, runtimeconfig.ErrUnknownLabel) || !errors.Is(err, runtimeconfig.ErrNotFound) {
		t.Fatalf("unknown body label error = %v", err)
	}
	if _, err := service.ReplaceLabelsIdempotent(
		ctx, strings.Repeat("b", 64), 1, []string{"missing"}, "principal-missing", "operator", now.Add(time.Minute),
	); !errors.Is(err, runtimeconfig.ErrNotFound) || errors.Is(err, runtimeconfig.ErrUnknownLabel) {
		t.Fatalf("missing principal error = %v", err)
	}

	type concurrentResult struct {
		key    string
		result MutationResult
		err    error
	}
	results := make(chan concurrentResult, 2)
	for _, key := range []string{"principal-labels-a", "principal-labels-b"} {
		go func(key string) {
			result, replaceErr := service.ReplaceLabelsIdempotent(
				ctx, principalID, 1, []string{}, key, "operator", now.Add(time.Minute),
			)
			results <- concurrentResult{key: key, result: result, err: replaceErr}
		}(key)
	}
	var successful concurrentResult
	successes, preconditions := 0, 0
	for range 2 {
		result := <-results
		switch {
		case result.err == nil:
			successes++
			successful = result
		case errors.Is(result.err, runtimeconfig.ErrPrecondition):
			preconditions++
		default:
			t.Fatalf("concurrent label replacement error = %v", result.err)
		}
	}
	if successes != 1 || preconditions != 1 || successful.result.Principal == nil ||
		successful.result.Principal.LabelRevision != 2 || len(successful.result.Principal.Labels) != 0 {
		t.Fatalf("concurrent label replacements = success %d precondition %d result %+v", successes, preconditions, successful)
	}
	replayedMutation, err := service.ReplaceLabelsIdempotent(
		ctx, principalID, 1, []string{}, successful.key, "operator", now.Add(time.Minute),
	)
	if err != nil || !replayedMutation.Replayed || replayedMutation.Principal == nil ||
		replayedMutation.Principal.LabelRevision != 2 {
		t.Fatalf("label response-loss replay = (%+v, %v)", replayedMutation, err)
	}
	// The same key with other labels is a different request: permanent, and
	// the replayed result stays the original one.
	if _, err := service.ReplaceLabelsIdempotent(
		ctx, principalID, 1, []string{"agent-route"}, successful.key, "operator", now.Add(time.Minute),
	); !errors.Is(err, contracts.ErrIdempotencyConflict) || errors.Is(err, runtimeconfig.ErrConflict) {
		t.Fatalf("label key reused with another request error = %v", err)
	}
	if _, err := service.ReplaceLabelsIdempotent(
		ctx, principalID, 1, []string{}, "principal-stale", "operator", now.Add(2*time.Minute),
	); !errors.Is(err, runtimeconfig.ErrPrecondition) {
		t.Fatalf("stale principal CAS error = %v", err)
	}
	deleted, err := service.DeleteIdempotent(
		ctx, principalID, 2, "principal-delete", "operator", now.Add(2*time.Minute),
	)
	if err != nil || !deleted.Deleted || deleted.Replayed {
		t.Fatalf("delete empty offline principal = (%+v, %v)", deleted, err)
	}
	deleteReplay, err := service.DeleteIdempotent(
		ctx, principalID, 2, "principal-delete", "operator", now.Add(2*time.Minute),
	)
	if err != nil || !deleteReplay.Deleted || !deleteReplay.Replayed {
		t.Fatalf("delete response-loss replay = (%+v, %v)", deleteReplay, err)
	}
	if _, err := NewRepository(pool).Get(ctx, principalID); !errors.Is(err, runtimeconfig.ErrNotFound) {
		t.Fatalf("deleted principal lookup error = %v", err)
	}
	if guard.begins != 2 || guard.releases != 2 {
		t.Fatalf("deletion guard calls = begin %d release %d", guard.begins, guard.releases)
	}
}

type recordingDeletionGuard struct {
	mu       sync.Mutex
	begins   int
	releases int
}

func (g *recordingDeletionGuard) BeginPrincipalDeletion(string) (func(), error) {
	g.mu.Lock()
	g.begins++
	g.mu.Unlock()
	return func() {
		g.mu.Lock()
		g.releases++
		g.mu.Unlock()
	}, nil
}

type allowRuntimeCredentialCatalog struct{}

func (catalog allowRuntimeCredentialCatalog) ForRuntimeTransaction(pgx.Tx) (runtimeconfig.RuntimeCredentialValidator, error) {
	return catalog, nil
}

func (allowRuntimeCredentialCatalog) ValidateRuntimeCredential(context.Context, string, ...string) error {
	return nil
}

func (allowRuntimeCredentialCatalog) WithCredentialReferences(_ context.Context, fn func() error) error {
	return fn()
}

// isolatedPrincipalPool opens a pool on a fresh, migrated schema that is
// dropped after the test.
func isolatedPrincipalPool(t *testing.T, ctx context.Context) *pgxpool.Pool {
	t.Helper()
	databaseURL := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if databaseURL == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
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
	schema := "agent_principal_test_" + hex.EncodeToString(randomBytes)
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
	if _, err := persistencepostgres.ApplyMigrations(ctx, pool); err != nil {
		t.Fatalf("apply migrations: %v", err)
	}
	return pool
}
