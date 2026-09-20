package runtimeconfig

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"reflect"
	"strings"
	"testing"
	"time"

	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/jackc/pgx/v5"
)

func coverageVersion(t *testing.T, name string, worker any) Version {
	t.Helper()
	p, err := PreparePublication(toolsetDocument(name, worker))
	if err != nil {
		t.Fatal(err)
	}
	v, err := p.Resolve(context.Background(), nil)
	if err != nil {
		t.Fatal(err)
	}
	return v
}

func TestToolsetCoverageCanonicalValuesClearsAndAllMixedSections(t *testing.T) {
	a := coverageVersion(t, "first", map[string]any{"toolsets": map[string]any{"mcp@1": map[string]any{"endpoint": "https://private.example/mcp", "credential": "private-credential"}}})
	b := coverageVersion(t, "second", map[string]any{"telemetry": nil, "httpProxy": nil, "caido": nil, "toolsets": map[string]any{"mcp@1": map[string]any{"endpoint": "https://private.example/mcp", "credential": "private-credential", "connectTimeoutSeconds": 10, "requestTimeoutSeconds": 60}}})
	project := func(v Version) BindingProjection {
		p, err := ProjectBinding(Binding{Label: "test", Ref: v.Ref, Revision: 1}, v)
		if err != nil {
			t.Fatal(err)
		}
		return p
	}
	left, right := project(a), project(b)
	if a.Ref.Digest == b.Ref.Digest || !reflect.DeepEqual(left.ToolsetConfigurationCoverage, right.ToolsetConfigurationCoverage) {
		t.Fatal("coverage hashes document identity instead of normalized value")
	}
	if !reflect.DeepEqual(right.ChangedSections, []string{"worker.caido", "worker.httpProxy", "worker.telemetry", "worker.toolsets"}) {
		t.Fatalf("mixed sections: %v", right.ChangedSections)
	}
	encoded, _ := json.Marshal(right.ToolsetConfigurationCoverage)
	if strings.Contains(string(encoded), "private") || strings.Contains(string(encoded), "endpoint") {
		t.Fatal("connection material entered picker")
	}
	b.Spec.Worker.LLMGateway.Present = true
	b.Spec.Planner.Telemetry.Present = true
	if len(project(b).ChangedSections) != 6 {
		t.Fatal("projection omitted a mixed section")
	}
	clear := project(coverageVersion(t, "clear", map[string]any{"toolsets": map[string]any{"mcp@1": nil}}))
	if !reflect.DeepEqual(clear.ToolsetConfigurationCoverage, []ToolsetCoverage{{Ref: "mcp@1", State: "cleared"}}) {
		t.Fatal("clear lost")
	}
	empty, err := DecodeStoredDocument([]byte(BuiltInCanonicalDocument))
	if err != nil || len(project(empty).ToolsetConfigurationCoverage) != 0 || len(project(empty).ChangedSections) != 0 {
		t.Fatal("empty default is not inheritance")
	}
	if _, err := ProjectBinding(Binding{Ref: a.Ref}, b); err == nil {
		t.Fatal("mismatched projection identity accepted")
	}
}

type coverageQueryCounter struct {
	persistencepostgres.DBTX
	queries int
}

func (c *coverageQueryCounter) Query(ctx context.Context, sql string, args ...any) (pgx.Rows, error) {
	c.queries++
	return c.DBTX.Query(ctx, sql, args...)
}
func (c *coverageQueryCounter) QueryRow(ctx context.Context, sql string, args ...any) pgx.Row {
	c.queries++
	return c.DBTX.QueryRow(ctx, sql, args...)
}

func TestPostgresToolsetCoverageOneQueryAndConcurrentBindingSnapshot(t *testing.T) {
	dsn := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if dsn == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	pool := isolatedRuntimeConfigPool(t, ctx, dsn)
	if _, err := persistencepostgres.ApplyMigrations(ctx, pool); err != nil {
		t.Fatal(err)
	}
	repository := NewRepository(pool)
	versions := []Version{
		coverageVersion(t, "a", map[string]any{"toolsets": map[string]any{"mcp@1": map[string]any{"endpoint": "https://a.example/mcp"}}}),
		coverageVersion(t, "b", map[string]any{"telemetry": nil, "toolsets": map[string]any{"mcp@1": nil}}),
	}
	for _, v := range versions {
		v.ActorID, v.CreatedAt = "operator", time.Now().UTC()
		if _, err := repository.InsertVersion(ctx, v); err != nil {
			t.Fatal(err)
		}
	}
	for i := 0; i < 60; i++ {
		if _, err := repository.CreateBinding(ctx, fmt.Sprintf("label-%02d", i), versions[0].Ref, "operator", time.Now().UTC()); err != nil {
			t.Fatal(err)
		}
	}
	counter := &coverageQueryCounter{DBTX: pool}
	reader := NewRepository(counter)
	rows, err := reader.ListBindingProjections(ctx, "", 200)
	if err != nil || len(rows) != 61 || counter.queries != 1 {
		t.Fatalf("page should be one query: len=%d queries=%d err=%v", len(rows), counter.queries, err)
	}
	if rows[0].Label != DefaultLabel || len(rows[0].ToolsetConfigurationCoverage) != 0 {
		t.Fatal("default projection missing")
	}
	before := counter.queries
	if _, err := reader.GetBindingProjection(ctx, "label-00"); err != nil || counter.queries != before+1 {
		t.Fatal("selected label requires more than one query")
	}
	writerDone := make(chan error, 1)
	go func() {
		for i := 0; i < 40; i++ {
			ref := versions[(i+1)%2].Ref
			_, err := repository.Rebind(ctx, "label-00", uint64(i+1), ref, "operator", time.Now().UTC())
			if err != nil {
				writerDone <- err
				return
			}
		}
		writerDone <- nil
	}()
	for i := 0; i < 40; i++ {
		row, err := reader.GetBindingProjection(ctx, "label-00")
		if err != nil {
			t.Fatal(err)
		}
		v := versions[0]
		if row.Ref == versions[1].Ref {
			v = versions[1]
		}
		expected, err := ProjectBinding(row.Binding, v)
		if err != nil || !reflect.DeepEqual(row, expected) {
			t.Fatal("coverage describes a different binding revision")
		}
	}
	if err := <-writerDone; err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `DELETE FROM runtime_label_bindings WHERE label='label-00'`); err != nil {
		t.Fatal(err)
	}
	if _, err := reader.GetBindingProjection(ctx, "label-00"); !errors.Is(err, ErrNotFound) {
		t.Fatal("deleted retained label did not return not found")
	}
}
