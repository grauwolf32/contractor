package projectsitemap

import (
	"context"
	"fmt"
	"strings"
	"sync"
	"testing"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/artifacts/artifactstest"
	"github.com/grauwolf32/contractor/internal/contracts"
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

func TestPostgresSitemapAllocationInputReadsHaveConstantQueryBudget(t *testing.T) {
	counter := &allocationReadCounter{}
	pool := sitemapPool(t, counter)
	ctx := t.Context()
	createSitemapRun(t, ctx, pool, "consumer")
	stores := artifacts.NewService(artifacts.NewPostgresRepository(pool))
	scope, _ := artifacts.ProjectScope("project")
	want := make(map[string]contracts.ArtifactRef)
	policy := readerPolicy("blackbox")
	for i := range contracts.MaxSitemapOpenAPIInputs {
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
		policy.OpenAPIInputs = append(policy.OpenAPIInputs, name)
	}
	service := NewService(pool)
	counter.reset()
	one := *policy.Clone()
	one.OpenAPIInputs = one.OpenAPIInputs[:1]
	if _, err := service.AllocationAccess(ctx, "consumer", "research", one); err != nil {
		t.Fatal(err)
	}
	oneQueries := counter.reset()
	access, err := service.AllocationAccess(ctx, "consumer", "research", policy)
	if err != nil {
		t.Fatal(err)
	}
	allQueries := counter.reset()
	if len(oneQueries) != len(allQueries) || len(allQueries) > 6 {
		t.Fatalf("authority reads grew with input count: one=%d all=%d", len(oneQueries), len(allQueries))
	}
	if len(access.Inputs) != len(want) {
		t.Fatalf("selected inputs: %d", len(access.Inputs))
	}
	for i, input := range access.Inputs {
		if input.Ref.Name != policy.OpenAPIInputs[i] || !input.Ref.SameExact(want[input.Ref.Name]) {
			t.Fatalf("original input/order changed: %+v", input)
		}
	}
	for _, query := range allQueries {
		if strings.Contains(query, "payload") {
			t.Fatal("authority read fetched content")
		}
	}
}
