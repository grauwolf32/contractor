package projectsitemap

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/artifacts/artifactstest"
	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contracts"
	postgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/persistence/postgrestest"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/runstore/runstoretest"
	"github.com/grauwolf32/contractor/internal/runtimeconfig/runtimeconfigtest"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

func sitemapPool(t *testing.T, tracers ...pgx.QueryTracer) *pgxpool.Pool {
	t.Helper()
	ctx, cancel := context.WithTimeout(t.Context(), 2*time.Minute)
	defer cancel()
	pool := postgrestest.NewPool(t, ctx, "", postgrestest.Options{
		CleanupTimeout: 30 * time.Second,
		Configure: func(poolConfig *pgxpool.Config) {
			if len(tracers) > 0 {
				poolConfig.ConnConfig.Tracer = tracers[0]
			}
		},
	})
	if _, err := postgres.ApplyMigrations(ctx, pool, postgres.MigrationBudgets{}); err != nil {
		t.Fatal(err)
	}
	if _, _, err := projectstore.NewPostgresStore(pool).Create(ctx, projectstore.CreateParams{
		ProjectID: "project", OwnerID: "owner", Kind: projectstore.KindProject, Name: "Sitemap", IdempotencyKey: "create", RequestDigest: "sha256:" + strings.Repeat("a", 64),
	}); err != nil {
		t.Fatal(err)
	}
	return pool
}

func createSitemapRun(t *testing.T, ctx context.Context, pool *pgxpool.Pool, id string) {
	t.Helper()
	project := "project"
	_, err := runstoretest.CreateRun(ctx, runstore.NewPostgresStore(pool), runstore.CreateRunParams{
		RunID: id, OwnerID: "owner", ProjectID: &project, LaneClass: resolved.LaneClassModel,
		WorkflowName: "sitemap-test", WorkflowVersion: "1", WorkflowSchemaVersion: "contractor/v1alpha1",
		WorkflowSnapshot: []byte(`{"ref":{"name":"sitemap-test","version":"1"}}`), RuntimeConfig: runtimeconfigtest.BuiltInRunSnapshot(),
	})
	if err != nil {
		t.Fatal(err)
	}
}

func TestPostgresSitemapSelectedContextUsesOrdinaryExactForkHold(t *testing.T) {
	pool := sitemapPool(t)
	ctx := t.Context()
	service := NewService(pool)
	createSitemapRun(t, ctx, pool, "context-consumer")
	policy, _ := artifacts.NewHistoryPolicy(2, artifacts.DefaultHistoryMaxBytes)
	ctx = artifacts.WithHistoryPolicy(ctx, policy)
	a := httpAuthority("producer")
	first, err := service.Mutate(ctx, a, "first", Mutation{Operation: "observe", Observations: []Observation{observation("/selected")}})
	if err != nil {
		t.Fatal(err)
	}
	artifactsService := artifacts.NewService(artifacts.NewPostgresRepository(pool))
	fork, err := artifactstest.ForkProjectInput(ctx, artifactsService, "project", first.Ref, "context-consumer", "map")
	if err != nil {
		t.Fatal(err)
	}
	for i := range 4 {
		if _, err := service.Mutate(ctx, a, fmt.Sprintf("next-%d", i), Mutation{Operation: "observe", Observations: []Observation{observation(fmt.Sprintf("/next/%d", i))}}); err != nil {
			t.Fatal(err)
		}
	}
	if _, err := readSitemap(service, ctx, "project", *first.Ref.Revision); err != nil {
		t.Fatalf("ordinary input lineage lost its source: %v", err)
	}
	run, _ := artifactsService.Run("context-consumer")
	if _, err := run.Read(ctx, fork.TargetRef); err != nil {
		t.Fatal(err)
	}
	// Selecting an already evicted exact source cannot acquire a hold by using
	// cached bytes or following the current binding.
	unheld, err := service.Mutate(ctx, a, "unheld", Mutation{Operation: "observe", Observations: []Observation{observation("/unheld")}})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := service.Mutate(ctx, a, "evict-unheld", Mutation{Operation: "observe", Observations: []Observation{observation("/last")}}); err != nil {
		t.Fatal(err)
	}
	if _, err := artifactstest.ForkProjectInput(ctx, artifactsService, "project", unheld.Ref, "context-consumer", "missing"); !errors.Is(err, artifacts.ErrArtifactNotFound) {
		t.Fatalf("lost exact selection race: %v", err)
	}
	view, err := readSitemap(service, ctx, "project", "")
	if err != nil {
		t.Fatal(err)
	}
	current := mapRef()
	current.Revision = &view.Revision
	if _, err := artifactstest.ForkProjectInput(ctx, artifactsService, "project", current, "context-consumer", "last"); err != nil {
		t.Fatal(err)
	}
	if _, err := service.Mutate(ctx, a, "all-protected", Mutation{Operation: "observe", Observations: []Observation{observation("/blocked")}}); !errors.Is(err, artifacts.ErrHistoryProtected) {
		t.Fatalf("protected history was not permanent: %v", err)
	}
	after, err := readSitemap(service, ctx, "project", "")
	if err != nil || after.Revision != view.Revision {
		t.Fatal("protected-history refusal changed current")
	}
}

// exhaustedHistoryWork injects the ordinary transaction work counter once,
// through a real pgx connection. It creates deterministic maintenance pressure
// without a fake repository or a second retention implementation.
type exhaustedHistoryWork struct {
	armed atomic.Bool
	err   error
}

func (t *exhaustedHistoryWork) TraceQueryStart(ctx context.Context, _ *pgx.Conn, _ pgx.TraceQueryStartData) context.Context {
	return ctx
}
func (t *exhaustedHistoryWork) TraceQueryEnd(ctx context.Context, conn *pgx.Conn, data pgx.TraceQueryEndData) {
	if data.Err == nil && t.armed.Load() && strings.Contains(data.CommandTag.String(), "SELECT") {
		// Only the explicit marker on the request context consumes the budget;
		// setup, receipt replay and separately committed maintenance stay normal.
		if inject, _ := ctx.Value(historyWorkTestKey{}).(bool); inject && t.armed.Swap(false) {
			_, t.err = conn.Exec(ctx, `SELECT set_config('contractor.artifact_history_work','64,0',true)`)
		}
	}
}

type historyWorkTestKey struct{}

func TestPostgresSitemapCommitsCommonMaintenanceAfterRollback(t *testing.T) {
	tracer := &exhaustedHistoryWork{}
	pool := sitemapPool(t, tracer)
	ctx := t.Context()
	service := NewService(pool)
	policy, _ := artifacts.NewHistoryPolicy(2, artifacts.DefaultHistoryMaxBytes)
	ctx = artifacts.WithHistoryPolicy(ctx, policy)
	a := httpAuthority("producer")
	for i := range 2 {
		if _, err := service.Mutate(ctx, a, fmt.Sprintf("seed-%d", i), Mutation{Operation: "observe", Observations: []Observation{observation(fmt.Sprintf("/seed/%d", i))}}); err != nil {
			t.Fatal(err)
		}
	}
	before, err := readSitemap(service, ctx, "project", "")
	if err != nil {
		t.Fatal(err)
	}
	tracer.armed.Store(true)
	command := Mutation{Operation: "observe", Observations: []Observation{observation("/pending")}}
	_, err = service.Mutate(context.WithValue(ctx, historyWorkTestKey{}, true), a, "pending", command)
	if tracer.err != nil {
		t.Fatalf("inject work budget: %v", tracer.err)
	}
	var pending *artifacts.HistoryMaintenanceError
	if !errors.As(err, &pending) || !pending.Maintained {
		t.Fatalf("ordinary committed maintenance outcome: %v", err)
	}
	after, err := readSitemap(service, ctx, "project", "")
	if err != nil || after.Revision != before.Revision {
		t.Fatal("failed publication changed map")
	}
	var receipts, retained int
	if err := pool.QueryRow(ctx, `SELECT count(*) FROM mutation_receipts WHERE idempotency_key='pending'`).Scan(&receipts); err != nil || receipts != 0 {
		t.Fatalf("failed publication claimed a receipt: %d %v", receipts, err)
	}
	if err := pool.QueryRow(ctx, `SELECT retained_revisions FROM artifact_bindings WHERE scope_kind='project' AND scope_id='project' AND namespace='sitemap' AND name='map'`).Scan(&retained); err != nil || retained != 1 {
		t.Fatalf("separate common maintenance did not commit: %d %v", retained, err)
	}
	result, err := service.Mutate(ctx, a, "pending", command)
	if err != nil || !result.Changed || result.Replayed {
		t.Fatalf("bounded caller continuation: %+v %v", result, err)
	}
}

func TestPostgresSitemapReceiptsOwnershipAndHistory(t *testing.T) {
	pool := sitemapPool(t)
	service := NewService(pool)
	ctx := t.Context()
	policy, err := artifacts.NewHistoryPolicy(2, artifacts.DefaultHistoryMaxBytes)
	if err != nil {
		t.Fatal(err)
	}
	ctx = artifacts.WithHistoryPolicy(ctx, policy)
	a := httpAuthority("producer")
	command := Mutation{Operation: "observe", Observations: []Observation{observation("/first")}}
	first, err := service.Mutate(ctx, a, "first", command)
	if err != nil {
		t.Fatal(err)
	}
	for i := range 4 {
		_, err := service.Mutate(ctx, a, fmt.Sprintf("next-%d", i), Mutation{Operation: "observe", Observations: []Observation{observation(fmt.Sprintf("/next/%d", i))}})
		if err != nil {
			t.Fatal(err)
		}
	}
	replayed, err := service.Mutate(ctx, a, "first", command)
	if err != nil || !replayed.Replayed || !replayed.Ref.SameExact(first.Ref) {
		t.Fatalf("receipt replay: %+v %v", replayed, err)
	}
	if _, err := readSitemap(service, ctx, "project", *first.Ref.Revision); !errors.Is(err, artifacts.ErrArtifactNotFound) {
		t.Fatalf("evicted read: %v", err)
	}
	foreign := a
	foreign.OwnerID = "foreign"
	if _, err := service.Mutate(ctx, foreign, "first", command); !errors.Is(err, projectstore.ErrNotFound) {
		t.Fatalf("foreign receipt: %v", err)
	}
	if _, err := service.Mutate(ctx, a, "first", Mutation{Operation: "observe", Observations: []Observation{observation("/changed")}}); !errors.Is(err, contracts.ErrIdempotencyConflict) {
		t.Fatalf("key reuse: %v", err)
	}
	var count int
	var holds int64
	if err := pool.QueryRow(ctx, `SELECT count(*),COALESCE(sum(hold_count),0) FROM artifact_binding_revisions WHERE scope_kind='project' AND scope_id='project'`).Scan(&count, &holds); err != nil {
		t.Fatal(err)
	}
	if count != 2 || holds != 0 {
		t.Fatalf("history/implicit pins: %d/%d", count, holds)
	}
	// Both generic publication adapters refuse a direct sitemap overwrite.
	store, _ := artifacts.NewService(artifacts.NewPostgresRepository(pool)).Project("project")
	if _, err := store.Write(ctx, mapRef(), artifacts.Payload{MediaType: MediaType, Data: []byte(`{}`)}, nil); !errors.Is(err, artifacts.ErrReservedNamespace) {
		t.Fatalf("generic bypass: %v", err)
	}
	view, err := readSitemap(service, ctx, "project", "")
	if err != nil {
		t.Fatal(err)
	}
	if len(view.Document.Groups) != 5 {
		t.Fatalf("lost observations: %d", len(view.Document.Groups))
	}
	if _, err := service.Mutate(ctx, a, "bad", Mutation{Operation: "observe", Observations: []Observation{{URL: "bad"}}}); !errors.Is(err, ErrInvalid) {
		t.Fatalf("invalid observation: %v", err)
	}
	after, err := readSitemap(service, ctx, "project", "")
	if err != nil || after.Revision != view.Revision {
		t.Fatal("rejected write changed current")
	}
}

func TestPostgresSitemapTenWritersAndPermanentSourceConflict(t *testing.T) {
	pool := sitemapPool(t)
	service := NewService(pool)
	ctx := t.Context()
	// Distinct source keys allow bounded CAS rebases. A caller retains a batch
	// after a contention refusal and may submit it again under the same key.
	var wg sync.WaitGroup
	failures := make(chan error, 10)
	for i := range 10 {
		wg.Go(func() {
			a := httpAuthority(fmt.Sprintf("writer-%d", i))
			m := Mutation{Operation: "observe", Observations: []Observation{observation(fmt.Sprintf("/writer/%d", i))}}
			var err error
			for range 10 {
				_, err = service.Mutate(ctx, a, fmt.Sprintf("write-%d", i), m)
				if err == nil || !errors.Is(err, artifacts.ErrArtifactConflict) {
					break
				}
			}
			if err != nil {
				failures <- err
			}
		})
	}
	wg.Wait()
	close(failures)
	for err := range failures {
		t.Error(err)
	}
	if t.Failed() {
		return
	}
	view, err := readSitemap(service, ctx, "project", "")
	if err != nil || len(view.Document.Groups) != 10 {
		t.Fatalf("concurrent appends: %d %v", len(view.Document.Groups), err)
	}
	a := Authority{OwnerID: "owner", ProjectID: "project", SourceID: "schema", Kind: "openapi", Deployment: "prod"}
	u := SourceUpdate{SourceID: "schema", Rules: []Rule{{SourceID: "schema", Origin: "https://example.test", Method: "GET", Path: "/writer/{id}", Parameters: []string{}}}}
	_, err = service.Mutate(ctx, a, "source-one", Mutation{Operation: "replace-source", Source: &u})
	if err != nil {
		t.Fatal(err)
	}
	before, _ := readSitemap(service, ctx, "project", "")
	if _, err := service.Mutate(ctx, a, "stale-source", Mutation{Operation: "replace-source", Source: &u}); !errors.Is(err, ErrConflict) {
		t.Fatalf("stale replacement: %v", err)
	}
	after, _ := readSitemap(service, ctx, "project", "")
	if before.Revision != after.Revision {
		t.Fatal("stale source changed current")
	}
}

func TestPostgresSitemapMalformedAndProjectFence(t *testing.T) {
	pool := sitemapPool(t)
	ctx := t.Context()
	service := NewService(pool)
	scope, _ := artifacts.ProjectScope("project")
	if _, err := artifacts.NewService(artifacts.NewPostgresRepository(pool)).Create(ctx, scope, mapRef(), artifacts.Payload{MediaType: MediaType, Data: []byte(`{}`)}); err != nil {
		t.Fatal(err)
	}
	if _, err := readSitemap(service, ctx, "project", ""); !errors.Is(err, ErrInvalid) {
		t.Fatalf("malformed read: %v", err)
	}
	if _, err := service.Mutate(ctx, httpAuthority("producer"), "cannot-replace", Mutation{Operation: "observe", Observations: []Observation{observation("/new")}}); !errors.Is(err, ErrInvalid) {
		t.Fatalf("malformed replaced: %v", err)
	}
	project, err := projectstore.NewPostgresStore(pool).Get(ctx, "owner", "project")
	if err != nil {
		t.Fatal(err)
	}
	_, _, err = projectstore.NewPostgresStore(pool).BeginDeletion(ctx, projectstore.BeginDeletionParams{OwnerID: "owner", ProjectID: "project", ExpectedRevision: project.Revision})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := service.Mutate(ctx, httpAuthority("producer"), "fenced", Mutation{Operation: "observe", Observations: []Observation{observation("/new")}}); !errors.Is(err, projectstore.ErrDeleting) {
		t.Fatalf("fenced write: %v", err)
	}
}

func TestPostgresSitemapReplayKeepsOriginalStructuralPolicy(t *testing.T) {
	pool := sitemapPool(t)
	ctx := t.Context()
	service := NewService(pool)
	owner := Authority{OwnerID: "owner", ProjectID: "project", SourceID: "rules", Kind: "manual"}
	rule := Rule{SourceID: "rules", Origin: "https://example.test", Method: "GET", Path: "/dispatch", Parameters: []string{}, Structural: []StructuralParameter{{Name: "action", Allowed: []string{"read", "delete"}}}}
	if _, err := service.Mutate(ctx, owner, "configure", Mutation{Operation: "replace-source", Source: &SourceUpdate{SourceID: "rules", Rules: []Rule{rule}}}); err != nil {
		t.Fatal(err)
	}
	a := httpAuthority("producer")
	command := Mutation{Operation: "observe", Observations: []Observation{observation("/dispatch?action=read")}}
	first, err := service.Mutate(ctx, a, "observation", command)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := service.Mutate(ctx, a, "observation", Mutation{Operation: "observe", Observations: []Observation{observation("/dispatch?action=delete")}}); !errors.Is(err, contracts.ErrIdempotencyConflict) {
		t.Fatalf("different structural function replayed old intent: %v", err)
	}
	view, err := readSitemap(service, ctx, "project", "")
	if err != nil {
		t.Fatal(err)
	}
	var expected Source
	for _, source := range view.Document.Sources {
		if source.Provenance.SourceID == "rules" {
			expected = source
		}
	}
	if _, err := service.Mutate(ctx, owner, "remove-rules", Mutation{Operation: "replace-source", Source: &SourceUpdate{SourceID: "rules", Expected: &expected, Rules: []Rule{}}}); err != nil {
		t.Fatal(err)
	}
	replayed, err := service.Mutate(ctx, a, "observation", command)
	if err != nil || !replayed.Replayed || !replayed.Ref.SameExact(first.Ref) {
		t.Fatalf("changed grouping altered receipt: %+v %v", replayed, err)
	}
}

func TestPostgresSitemapUsesSharedTransferAdmission(t *testing.T) {
	pool := sitemapPool(t)
	service := NewService(pool)
	ctx := t.Context()
	a := httpAuthority("producer")
	command := Mutation{Operation: "observe", Observations: []Observation{observation("/selected")}}
	first, err := service.Mutate(ctx, a, "first", command)
	if err != nil {
		t.Fatal(err)
	}
	runtime := artifacts.NewBlobRuntime(nil, nil)
	ctx = artifacts.WithBlobRuntime(ctx, runtime)
	var releases []func()
	for range 4 {
		_, release, err := artifacts.AcquireTransfer(ctx)
		if err != nil {
			t.Fatal(err)
		}
		releases = append(releases, release)
	}
	defer func() {
		for _, release := range releases {
			release()
		}
	}()
	if _, err := readSitemap(service, ctx, "project", ""); !errors.Is(err, artifacts.ErrTransferCapacity) {
		t.Fatalf("read bypassed common admission: %v", err)
	}
	if _, err := service.Mutate(ctx, a, "second", Mutation{Operation: "observe", Observations: []Observation{observation("/another")}}); !errors.Is(err, artifacts.ErrTransferCapacity) {
		t.Fatalf("reducer bypassed common admission: %v", err)
	}
	replayed, err := service.Mutate(ctx, a, "first", command)
	if err != nil || !replayed.Replayed || !replayed.Ref.SameExact(first.Ref) {
		t.Fatalf("metadata receipt replay needed a new payload: %+v %v", replayed, err)
	}
}
