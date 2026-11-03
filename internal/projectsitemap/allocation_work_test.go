package projectsitemap

import (
	"context"
	"fmt"
	"strings"
	"sync"
	"testing"

	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/artifacts/artifactstest"
	"github.com/grauwolf32/contractor/internal/contracts"
	postgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/jackc/pgx/v5"
)

type allocationReadCounter struct {
	mu      sync.Mutex
	queries []string
}

func (c *allocationReadCounter) TraceQueryStart(ctx context.Context, _ *pgx.Conn, data pgx.TraceQueryStartData) context.Context {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.queries = append(c.queries, data.SQL)
	return ctx
}

func (*allocationReadCounter) TraceQueryEnd(context.Context, *pgx.Conn, pgx.TraceQueryEndData) {}

func (c *allocationReadCounter) reset() []string {
	c.mu.Lock()
	defer c.mu.Unlock()
	queries := c.queries
	c.queries = nil
	return queries
}

func TestPostgresSitemapReadsDoNotResolveSnapshotInputs(t *testing.T) {
	counter := &allocationReadCounter{}
	pool := sitemapPool(t, counter)
	ctx := t.Context()
	createSitemapRun(t, ctx, pool, "consumer")
	stores := artifacts.NewService(artifactpgtest.NewStorage().Repository(pool))
	scope, _ := artifacts.ProjectScope("project")
	want := make(map[string]contracts.ArtifactRef)
	policy := readerPolicy()
	for i := range contracts.MaxSitemapSnapshotInputs {
		name := fmt.Sprintf("schema-%d", i)
		source, err := stores.Create(ctx, scope, contracts.ArtifactRef{Namespace: "schemas", Name: name}, artifacts.Payload{MediaType: "application/json", Data: []byte(fmt.Sprintf(`{"schema":%d}`, i))})
		if err != nil {
			t.Fatal(err)
		}
		fork, err := artifactstest.ForkProjectInput(ctx, stores, "project", source.Ref, "consumer", name)
		if err != nil {
			t.Fatal(err)
		}
		want[name] = fork.TargetRef
		policy.SnapshotInputs = append(policy.SnapshotInputs, name)
	}
	service := NewService(pool, artifactpgtest.NewStorage())
	counter.reset()
	if _, err := service.AllocationAccess(ctx, "consumer", "research", readerPolicy()); err != nil {
		t.Fatal(err)
	}
	emptyQueries := counter.reset()
	one := *policy.Clone()
	one.SnapshotInputs = one.SnapshotInputs[:1]
	if _, err := service.AllocationAccess(ctx, "consumer", "research", one); err != nil {
		t.Fatal(err)
	}
	oneQueries := counter.reset()
	access, err := service.AllocationAccess(ctx, "consumer", "research", policy)
	if err != nil {
		t.Fatal(err)
	}
	allQueries := counter.reset()
	if len(emptyQueries) != len(allQueries) || len(oneQueries) != len(allQueries) || len(allQueries) > 4 {
		t.Fatalf("access reads grew with input count: zero=%d one=%d all=%d", len(emptyQueries), len(oneQueries), len(allQueries))
	}
	for _, query := range allQueries {
		if strings.Contains(query, "artifact_") || strings.Contains(query, "payload") {
			t.Fatal("map access resolved snapshot inputs or read artifact content")
		}
	}
	resolve := func(access AllocationAccess) ([]contracts.ExactArtifact, []string) {
		t.Helper()
		counter.reset()
		var inputs []contracts.ExactArtifact
		err := postgres.InTx(ctx, pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
			var err error
			inputs, err = resolveSnapshotInputs(ctx, tx, access, artifactpgtest.NewStorage())
			return err
		})
		if err != nil {
			t.Fatal(err)
		}
		return inputs, counter.reset()
	}
	access.Policy = one
	_, oneInputQueries := resolve(access)
	access.Policy = policy
	inputs, allInputQueries := resolve(access)
	if len(oneInputQueries) != len(allInputQueries) || len(allInputQueries) > 4 {
		t.Fatalf("snapshot metadata reads grew with input count: one=%d all=%d", len(oneInputQueries), len(allInputQueries))
	}
	if len(inputs) != len(want) {
		t.Fatalf("selected inputs: %d", len(inputs))
	}
	for i, input := range inputs {
		if input.Ref.Name != policy.SnapshotInputs[i] || !input.Ref.SameExact(want[input.Ref.Name]) {
			t.Fatalf("original input/order changed: %+v", input)
		}
	}
	for _, query := range allInputQueries {
		if strings.Contains(query, "payload") {
			t.Fatal("snapshot metadata lookup fetched content")
		}
	}
	t.Logf("access statements=%d with zero artifact reads; snapshot metadata statements=%d for 1 or 8 inputs", len(allQueries), len(allInputQueries))
}
