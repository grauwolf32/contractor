package runservice

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"reflect"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
)

type toolsetRuntimeCatalog struct{ acceptRuntimeCredentials }

func (toolsetRuntimeCatalog) WithCredentialReferences(_ context.Context, fn func() error) error {
	return fn()
}

type toolsetWorkflowResolver struct {
	workflow config.ResolvedWorkflow
	calls    atomic.Int64
}

func (r *toolsetWorkflowResolver) ResolveRunWorkflow(context.Context, string, config.ExecutionConfigPatch, config.CredentialLookup) (config.ResolvedWorkflow, error) {
	r.calls.Add(1)
	return r.workflow, nil
}

func TestPostgresPublicRunValidatesLateToolsetAndReplaysPinnedBindings(t *testing.T) {
	dsn := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if dsn == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	ctx, cancel := context.WithTimeout(t.Context(), 45*time.Second)
	defer cancel()
	pool := isolatedRunServicePool(t, ctx, dsn)
	snapshot, err := config.Load("../../testdata/configs", config.MVPDescriptors())
	if err != nil {
		t.Fatal(err)
	}
	workflow, err := snapshot.Workflow("artifact-copy@1")
	if err != nil {
		t.Fatal(err)
	}
	workflow.Inputs, workflow.Parameters = nil, nil
	late := workflow.Stages[workflow.EntryStage]
	late.Agents = make(map[string]config.ResolvedAgentBinding)
	binding := workflow.Stages[workflow.EntryStage].Agents["builder"]
	binding.Template.Toolsets = []contracts.ToolsetSelection{{Ref: contracts.ToolsetRef{ToolsetID: "mcp", Version: "1"}, Tools: []string{"search"}}}
	late.Agents["researcher"] = binding
	workflow.Stages["later"] = late
	resolver := &toolsetWorkflowResolver{workflow: workflow}
	service, err := New(Options{Runs: runstore.NewPostgresStore(pool), Workflows: resolver,
		LLMCredentials: emptyCredentialLookup{}, CredentialGuard: openCredentialGuard{}, RuntimeCredentials: acceptRuntimeCredentials{},
		Projects: projectstore.NewPostgresStore(pool), PublicTransaction: postgresPublicTransaction(pool), AuditTransaction: postgresAuditTransaction(pool)})
	if err != nil {
		t.Fatal(err)
	}
	var sequence atomic.Int64
	params := PublicCreateParams{OwnerID: "toolset-owner", Workflow: "artifact-copy@1", IdempotencyKey: "create-shared", RequestDigest: testDigest("1"), NewRunID: func() (string, error) { return fmt.Sprintf("toolset-run-%d", sequence.Add(1)), nil }}
	assertNoRun := func() {
		t.Helper()
		var count int
		if err := pool.QueryRow(ctx, `SELECT count(*) FROM workflow_runs`).Scan(&count); err != nil || count != 0 {
			t.Fatalf("invalid create persisted Run: %d, %v", count, err)
		}
	}
	if _, err := service.CreatePublic(ctx, params); err == nil {
		t.Fatal("late-stage missing MCP configuration accepted")
	}
	assertNoRun()
	publisher, err := runtimeconfig.NewPublisher(runtimeconfig.PublisherOptions{Pool: pool, RuntimeCredentials: toolsetRuntimeCatalog{}, PlannerTelemetryAdapters: runtimeconfig.PlannerTelemetryAdapterCatalogFunc(func(string) bool { return true })})
	if err != nil {
		t.Fatal(err)
	}
	publish := func(name string, value any) runtimeconfig.Ref {
		t.Helper()
		raw, _ := json.Marshal(map[string]any{"apiVersion": contracts.APIVersion, "kind": "RuntimeConfig", "metadata": map[string]string{"name": name, "version": "1"}, "spec": map[string]any{"worker": map[string]any{"toolsets": map[string]any{"mcp@1": value}}}})
		published, err := publisher.Publish(ctx, raw, "publish-"+name, "operator")
		if err != nil {
			t.Fatal(err)
		}
		return published.Version.Ref
	}
	firstRef := publish("knowledge-a", map[string]any{"endpoint": "https://first.example/mcp"})
	secondRef := publish("knowledge-b", map[string]any{"endpoint": "https://second.example/mcp"})
	cleared := publish("knowledge-clear", nil)
	bindings, err := runtimeconfig.NewBindingService(pool, toolsetRuntimeCatalog{})
	if err != nil {
		t.Fatal(err)
	}
	for label, ref := range map[string]runtimeconfig.Ref{"knowledge": firstRef, "clear": cleared, "other": secondRef} {
		if _, err := bindings.Create(ctx, label, ref, "operator", time.Now()); err != nil {
			t.Fatal(err)
		}
	}
	for _, labels := range [][]string{{"clear"}, {"knowledge", "clear"}, {"knowledge", "other"}, {"other", "knowledge"}} {
		params.RuntimeLabels = labels
		if _, err := service.CreatePublic(ctx, params); err == nil {
			t.Fatalf("invalid labels accepted: %v", labels)
		}
		assertNoRun()
	}
	params.RuntimeLabels = []string{"knowledge"}
	var results [2]CreateResult
	var failures [2]error
	var group sync.WaitGroup
	for i := range results {
		group.Add(1)
		go func(i int) { defer group.Done(); results[i], failures[i] = service.CreatePublic(ctx, params) }(i)
	}
	group.Wait()
	for _, err := range failures {
		if err != nil {
			t.Fatal(err)
		}
	}
	if results[0].Run.RunID != results[1].Run.RunID {
		t.Fatal("concurrent create duplicated Run")
	}
	accepted := results[0].Run
	if accepted.RuntimeConfig.Labels[0].Config != firstRef {
		t.Fatal("accepted Run lost locked pin")
	}
	if _, err := bindings.Rebind(ctx, "knowledge", 1, secondRef, "operator", time.Now()); err != nil {
		t.Fatal(err)
	}
	// Reconstruct the service's durable reader after the binding changes.
	service.runs = runstore.NewPostgresStore(pool)
	calls := resolver.calls.Load()
	replay, err := service.CreatePublic(ctx, params)
	if err != nil || !replay.Replayed || !reflect.DeepEqual(replay.Run.RuntimeConfig, accepted.RuntimeConfig) || resolver.calls.Load() != calls {
		t.Fatalf("exact replay consulted current workflow/binding: %v", err)
	}
	params.IdempotencyKey, params.RequestDigest = "repeat-reviewed", testDigest("2")
	repeated, err := service.CreatePublic(ctx, params)
	if err != nil || !repeated.Created || repeated.Run.RuntimeConfig.Labels[0].Config != secondRef {
		t.Fatalf("new Run did not use current binding: %v", err)
	}
	stored, err := runstore.NewPostgresStore(pool).GetRun(ctx, accepted.RunID)
	if err != nil || !reflect.DeepEqual(stored.RuntimeConfig, accepted.RuntimeConfig) {
		t.Fatal("late-stage/replacement authority changed after rebind")
	}
	old, err := runtimeconfig.NewRepository(pool).GetVersionByRef(ctx, stored.RuntimeConfig.Labels[0].Config)
	if err != nil || old.Spec.Worker.Toolsets["mcp@1"].Value.(runtimeconfig.MCPConfig).Endpoint != "https://first.example/mcp" {
		t.Fatal("original immutable configuration is unavailable")
	}
	params.RuntimeLabels = []string{"does-not-exist"}
	params.IdempotencyKey = "missing-label"
	params.RequestDigest = testDigest("3")
	if _, err := service.CreatePublic(ctx, params); !errors.Is(err, runtimeconfig.ErrNotFound) {
		t.Fatalf("missing binding: %v", err)
	}
}
