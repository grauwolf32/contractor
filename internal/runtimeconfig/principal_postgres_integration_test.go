package runtimeconfig

import (
	"context"
	"errors"
	"os"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/contracts"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/jackc/pgx/v5"
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
	databaseURL := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if databaseURL == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	ctx, cancel := context.WithTimeout(t.Context(), 45*time.Second)
	defer cancel()
	pool := isolatedRuntimeConfigPool(t, ctx, databaseURL)
	if _, err := persistencepostgres.ApplyMigrations(ctx, pool); err != nil {
		t.Fatal(err)
	}
	ref := BuiltInRunSnapshot().Default.Config
	bindings := NewRepository(pool)
	for _, label := range []string{"alpha", "beta", "gamma"} {
		if _, err := bindings.CreateBinding(ctx, label, ref, "operator", time.Now().UTC()); err != nil {
			t.Fatal(err)
		}
	}
	ids := []string{strings.Repeat("a", 64), strings.Repeat("b", 64), strings.Repeat("c", 64), strings.Repeat("d", 64)}
	labelSets := [][]string{{"alpha", "beta"}, {"beta"}, {"gamma"}, {}}
	for index, id := range ids {
		at := time.Now().UTC()
		inserted, err := NewPrincipalRepository(pool).Insert(ctx, RuntimeAgentPrincipal{
			RuntimeAgentID: id, Labels: labelSets[index], LabelRevision: 1,
			CreatedBy: "operator", CreatedAt: at, UpdatedBy: "operator", UpdatedAt: at,
		})
		if err != nil || !inserted {
			t.Fatalf("insert principal %s: inserted=%v error=%v", id, inserted, err)
		}
	}
	service, err := NewPrincipalService(PrincipalServiceOptions{Pool: pool})
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
	page, hasMore, err := listPrincipalAdapterPage(ctx, counted, "", 2)
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
	fullPage, hasMore, err := listPrincipalAdapterPage(ctx, counted, "", 3)
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
	if err != nil || more || len(last) != 1 || last[0].Principal.RuntimeAgentID != ids[3] || len(last[0].RequiredRuntimeAdapters) != 0 {
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
	oldPage, more, err := listPrincipalAdapterPage(ctx, snapshot, ids[1], 1)
	if err != nil || !more || len(oldPage) != 1 || !slices.Equal(oldPage[0].Principal.Labels, []string{"gamma"}) {
		t.Fatalf("repeatable-read page after concurrent deletion = (%+v, %v, %v)", oldPage, more, err)
	}
	freshPage, more, err := service.ListWithRequiredRuntimeAdapters(ctx, ids[1], 1)
	if err != nil || !more || len(freshPage) != 1 || len(freshPage[0].Principal.Labels) != 0 {
		t.Fatalf("fresh page after concurrent deletion = (%+v, %v, %v)", freshPage, more, err)
	}
}

func TestPostgresRuntimeAgentPrincipalSeedCASAndDelete(t *testing.T) {
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

	now := time.Date(2026, 9, 1, 1, 0, 0, 0, time.UTC)
	publisher, err := NewPublisher(PublisherOptions{
		Pool:                     pool,
		PlannerTelemetryAdapters: PlannerTelemetryAdapterCatalogFunc(func(ref string) bool { return ref == "otlp-http@1" }),
		GatewayResolver: GatewayResolverFunc(func(_ context.Context, selector string) (contracts.ResolvedLLMGatewayConfig, error) {
			gatewayID, version, _ := strings.Cut(selector, "@")
			return contracts.ResolvedLLMGatewayConfig{
				Ref: contracts.LLMGatewayConfigRef{
					GatewayID: gatewayID, Version: version,
					Digest: "sha256:" + strings.Repeat("a", 64),
				},
				Protocol: contracts.OpenAICompatibleProtocol,
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
	if _, err := NewRepository(pool).CreateBinding(
		ctx, "agent-route", published.Version.Ref, "operator", now,
	); err != nil {
		t.Fatal(err)
	}

	guard := &recordingDeletionGuard{}
	service, err := NewPrincipalService(PrincipalServiceOptions{
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
	if _, err := service.DeleteIdempotent(
		ctx, principalID, 1, "principal-delete-labeled", "operator", now.Add(time.Minute),
	); !errors.Is(err, ErrConflict) {
		t.Fatalf("delete labeled principal error = %v", err)
	}
	if _, err := service.ReplaceLabelsIdempotent(
		ctx, principalID, 1, []string{"missing"}, "principal-unknown-label", "operator", now.Add(time.Minute),
	); !errors.Is(err, ErrUnknownLabel) || !errors.Is(err, ErrNotFound) {
		t.Fatalf("unknown body label error = %v", err)
	}
	if _, err := service.ReplaceLabelsIdempotent(
		ctx, strings.Repeat("b", 64), 1, []string{"missing"}, "principal-missing", "operator", now.Add(time.Minute),
	); !errors.Is(err, ErrNotFound) || errors.Is(err, ErrUnknownLabel) {
		t.Fatalf("missing principal error = %v", err)
	}

	type concurrentResult struct {
		key    string
		result PrincipalMutationResult
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
		case errors.Is(result.err, ErrPrecondition):
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
	if _, err := service.ReplaceLabelsIdempotent(
		ctx, principalID, 1, []string{}, "principal-stale", "operator", now.Add(2*time.Minute),
	); !errors.Is(err, ErrPrecondition) {
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
	if _, err := NewPrincipalRepository(pool).Get(ctx, principalID); !errors.Is(err, ErrNotFound) {
		t.Fatalf("deleted principal lookup error = %v", err)
	}
	if guard.begins != 2 || guard.releases != 2 {
		t.Fatalf("deletion guard calls = begin %d release %d", guard.begins, guard.releases)
	}
}

func TestPostgresLabelRebindCannotInvalidateAssignedPrincipalLayer(t *testing.T) {
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
	now := time.Date(2026, 9, 1, 2, 0, 0, 0, time.UTC)
	resolver := GatewayResolverFunc(func(_ context.Context, selector string) (contracts.ResolvedLLMGatewayConfig, error) {
		gatewayID, version, _ := strings.Cut(selector, "@")
		digit := "b"
		if gatewayID == "other-litellm" {
			digit = "c"
		}
		return contracts.ResolvedLLMGatewayConfig{
			Ref: contracts.LLMGatewayConfigRef{
				GatewayID: gatewayID, Version: version,
				Digest: "sha256:" + strings.Repeat(digit, 64),
			},
			Protocol: contracts.OpenAICompatibleProtocol,
			URL:      "http://127.0.0.1:4000/v1",
		}, nil
	})
	publisher, err := NewPublisher(PublisherOptions{
		Pool: pool, GatewayResolver: resolver,
		RuntimeCredentials: allowRuntimeCredentialCatalog{}, Now: func() time.Time { return now },
		PlannerTelemetryAdapters: PlannerTelemetryAdapterCatalogFunc(func(ref string) bool { return ref == "otlp-http@1" }),
	})
	if err != nil {
		t.Fatal(err)
	}
	publish := func(name, gateway string) Ref {
		t.Helper()
		document := []byte(`{"apiVersion":"contractor/v1alpha1","kind":"RuntimeConfig","metadata":{"name":"` +
			name + `","version":"1"},"spec":{"worker":{"llmGateway":{"gateway":"` + gateway + `@1"}}}}`)
		result, publishErr := publisher.Publish(ctx, document, "publish-"+name, "operator")
		if publishErr != nil {
			t.Fatal(publishErr)
		}
		return result.Version.Ref
	}
	firstRef := publish("route-one", "local-litellm")
	identicalRef := publish("route-two", "local-litellm")
	conflictingRef := publish("route-other", "other-litellm")
	repository := NewRepository(pool)
	for label, ref := range map[string]Ref{"route-one": firstRef, "route-two": identicalRef} {
		if _, err := repository.CreateBinding(ctx, label, ref, "operator", now); err != nil {
			t.Fatal(err)
		}
	}
	principalService, err := NewPrincipalService(PrincipalServiceOptions{
		Pool: pool, DeletionGuard: &recordingDeletionGuard{}, Now: func() time.Time { return now },
	})
	if err != nil {
		t.Fatal(err)
	}
	principalID := strings.Repeat("d", 64)
	if _, err := principalService.Register(
		ctx, principalID, []string{"route-one", "route-two"},
	); err != nil {
		t.Fatal(err)
	}
	bindingService, err := NewBindingService(allowRuntimeCredentialCatalog{})
	if err != nil {
		t.Fatal(err)
	}
	management, err := NewManagementService(pool, publisher, bindingService)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := management.Rebind(
		ctx, "route-two", 1, conflictingRef, "rebind-conflict", "operator", now.Add(time.Minute),
	); err == nil {
		t.Fatal("conflicting label rebind was accepted")
	} else {
		var conflict *MergeConflictError
		if !errors.As(err, &conflict) || conflict.Path != "worker.llmGateway.gateway" {
			t.Fatalf("conflicting rebind error = %v", err)
		}
	}
	current, err := repository.GetBinding(ctx, "route-two")
	if err != nil || current.Revision != 1 || current.Ref != identicalRef {
		t.Fatalf("binding after rolled-back conflict = (%+v, %v)", current, err)
	}
	if _, err := management.DeleteBinding(
		ctx, "route-one", 1, "delete-assigned", "operator", now.Add(2*time.Minute),
	); !errors.Is(err, ErrConflict) {
		t.Fatalf("delete assigned label error = %v", err)
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
