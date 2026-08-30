package streamline_test

import (
	"context"
	cryptorand "crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"sync"
	"testing"
	"time"

	workflowconfig "github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/planner"
	plannersession "github.com/grauwolf32/contractor/internal/planner/session"
	"github.com/grauwolf32/contractor/internal/planner/streamline"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

func TestPostgresGatewayWorkerFlowRecoversWithoutSemanticReplay(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	pool := isolatedPool(t, ctx)
	store := runstore.NewPostgresStore(pool)
	createStage(t, ctx, store)

	const gatewayToken = "sk-streamline-integration-secret"
	gateway := newScriptedGateway(t, gatewayToken)
	defer gateway.server.Close()
	llm, err := streamline.NewOpenAICompatibleModel(streamline.GatewaySettings{
		URL: gateway.server.URL + "/v1", Token: contracts.NewSecretString(gatewayToken),
		Model: "planner-model", HTTPClient: gateway.server.Client(),
	})
	if err != nil {
		t.Fatal(err)
	}
	sessions, err := plannersession.New(store, plannersession.Options{})
	if err != nil {
		t.Fatal(err)
	}
	workers := &workerInvoker{}
	inspector := artifactInspector{}
	factory, err := streamline.NewFactory(
		sessions, sessions, workers, inspector, llm,
		streamline.Limits{MaxModelCalls: 8, MaxTokens: 10_000, MaxWorkerCalls: 8, MaxWallTime: time.Minute},
	)
	if err != nil {
		t.Fatal(err)
	}
	invocation := testInvocation()
	first, err := factory.Create(invocation)
	if err != nil {
		t.Fatal(err)
	}
	result, err := first.Run(ctx)
	if err != nil || result.Outcome != contracts.StageSucceeded || result.Summary != "final report accepted" {
		t.Fatalf("first run = (%+v, %v)", result, err)
	}
	if gateway.calls() != 3 || workers.calls() != 2 {
		t.Fatalf("semantic calls gateway=%d Workers=%d", gateway.calls(), workers.calls())
	}

	// Simulate a process restart with fresh adapters over the same PostgreSQL
	// state. Completed recovery must bypass ADK, Gateway, and Workers.
	restartedSessions, err := plannersession.New(runstore.NewPostgresStore(pool), plannersession.Options{})
	if err != nil {
		t.Fatal(err)
	}
	restartedFactory, err := streamline.NewFactory(
		restartedSessions, restartedSessions, workers, inspector, llm, streamline.DefaultLimits(),
	)
	if err != nil {
		t.Fatal(err)
	}
	restarted, err := restartedFactory.Create(invocation)
	if err != nil {
		t.Fatal(err)
	}
	recovered, err := restarted.Run(ctx)
	if err != nil || recovered.Summary != result.Summary || gateway.calls() != 3 || workers.calls() != 2 {
		t.Fatalf("recovery=(%+v,%v) gateway=%d Workers=%d", recovered, err, gateway.calls(), workers.calls())
	}

	execution, err := store.GetStageExecution(ctx, "stage-streamline")
	if err != nil || execution.PlannerSessionID == nil {
		t.Fatalf("StageExecution = (%+v, %v)", execution, err)
	}
	events, err := store.ListPlannerEvents(ctx, *execution.PlannerSessionID, 0)
	if err != nil || len(events) < 8 {
		t.Fatalf("Planner events = (%d, %v)", len(events), err)
	}
	for _, event := range events {
		payload := string(event.Event)
		if strings.Contains(payload, gatewayToken) || strings.Contains(payload, "sensitive objective") ||
			strings.Contains(payload, "sensitive planner guidance") || strings.Contains(payload, "strict-secret-value") {
			t.Fatalf("durable Planner event leaked model/provider content: %s", payload)
		}
	}
}

type scriptedGateway struct {
	t      *testing.T
	server *httptest.Server
	mu     sync.Mutex
	count  int
	token  string
}

func newScriptedGateway(t *testing.T, token string) *scriptedGateway {
	t.Helper()
	result := &scriptedGateway{t: t, token: token}
	result.server = httptest.NewServer(http.HandlerFunc(result.serveHTTP))
	return result
}

func (g *scriptedGateway) serveHTTP(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path != "/v1/chat/completions" || r.Header.Get("Authorization") != "Bearer "+g.token {
		g.t.Errorf("gateway request path=%q authorization=%q", r.URL.Path, r.Header.Get("Authorization"))
		http.Error(w, "bad request", http.StatusBadRequest)
		return
	}
	var request struct {
		Tools []any `json:"tools"`
	}
	if err := json.NewDecoder(r.Body).Decode(&request); err != nil || len(request.Tools) != 4 {
		g.t.Errorf("gateway tool request = (%+v, %v)", request, err)
		http.Error(w, "bad request", http.StatusBadRequest)
		return
	}
	g.mu.Lock()
	step := g.count
	g.count++
	g.mu.Unlock()
	var name, arguments string
	switch step {
	case 0:
		name = "worker_analyzer"
		arguments = `{"objective":"analyze source","instructions":"write an exact draft","parameters":{"mode":"strict-secret-value"},"artifacts":{}}`
	case 1:
		name = "worker_reviewer"
		arguments = `{"objective":"review draft","instructions":"write the final report","parameters":{},"artifacts":{"draft":{"namespace":"analysis","name":"draft","revision":"draft-r1"}}}`
	case 2:
		name = "finish"
		arguments = `{"outcome":"succeeded","summary":"final report accepted","artifacts":{"report":{"namespace":"review","name":"report","revision":"report-r1"}}}`
	default:
		g.t.Errorf("unexpected Gateway call %d", step+1)
		http.Error(w, "unexpected call", http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	_, _ = fmt.Fprintf(w, `{
  "model":"planner-model",
  "choices":[{"finish_reason":"tool_calls","message":{"content":"","tool_calls":[{
    "id":"call-%d","type":"function","function":{"name":%q,"arguments":%q}
  }]}}],
  "usage":{"prompt_tokens":20,"completion_tokens":10,"total_tokens":30}
}`, step+1, name, arguments)
}

func (g *scriptedGateway) calls() int {
	g.mu.Lock()
	defer g.mu.Unlock()
	return g.count
}

type workerInvoker struct {
	mu    sync.Mutex
	count int
}

func (w *workerInvoker) Invoke(
	_ context.Context,
	binding string,
	_ contracts.WorkerHandle,
	request contracts.StageContentRequest,
) (contracts.StageContentResult, error) {
	w.mu.Lock()
	defer w.mu.Unlock()
	w.count++
	switch binding {
	case "analyzer":
		if request.Parameters["mode"] != "strict-secret-value" {
			return contracts.StageContentResult{}, errors.New("structured parameters missing")
		}
		return successfulResult("draft ready", "draft", exactRef("analysis", "draft", "draft-r1")), nil
	case "reviewer":
		if ref, ok := request.Artifacts["draft"]; !ok || ref.Revision == nil || *ref.Revision != "draft-r1" {
			return contracts.StageContentResult{}, errors.New("exact draft ref missing")
		}
		return successfulResult("report ready", "report", exactRef("review", "report", "report-r1")), nil
	default:
		return contracts.StageContentResult{}, errors.New("unknown Worker")
	}
}

func (w *workerInvoker) calls() int {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.count
}

type artifactInspector struct{}

func (artifactInspector) Inspect(
	_ context.Context, runID string, ref contracts.ArtifactRef,
) (planner.ArtifactMetadata, error) {
	if runID != "run-streamline" || ref.Revision == nil {
		return planner.ArtifactMetadata{}, errors.New("artifact outside RunScope")
	}
	switch ref.Namespace + "/" + ref.Name + "@" + *ref.Revision {
	case "analysis/draft@draft-r1", "review/report@report-r1":
		return planner.ArtifactMetadata{MediaType: "application/json"}, nil
	default:
		return planner.ArtifactMetadata{}, errors.New("artifact absent")
	}
}

func testInvocation() planner.Invocation {
	agents := map[string]workflowconfig.ResolvedAgentBinding{}
	workers := map[string]contracts.WorkerHandle{}
	for _, name := range []string{"analyzer", "reviewer"} {
		template := contracts.AgentTemplateRef{
			TemplateID: name, Version: "1", Digest: "sha256:" + strings.Repeat("a", 64),
		}
		runtime := contracts.WorkerRuntimeRef{RuntimeID: "adk", Version: "1"}
		agents[name] = workflowconfig.ResolvedAgentBinding{
			Namespace: name,
			Template: contracts.ResolvedAgentTemplate{
				Ref: template, Runtime: runtime, Description: name + " Worker",
			},
		}
		workers[name] = contracts.WorkerHandle{
			AllocationID: "allocation-" + name, AgentTemplateRef: template,
			WorkerRuntimeRef: runtime, AgentCard: map[string]any{"name": name},
			LeaseExpiresAt: time.Now().Add(5 * time.Minute),
		}
	}
	return planner.Invocation{
		StageExecutionID: "stage-streamline", RunID: "run-streamline",
		Deadline: time.Now().Add(time.Minute),
		Stage: workflowconfig.ResolvedStage{
			Objective:    "sensitive objective",
			Instructions: contracts.ResolvedInstructions{Text: "sensitive planner guidance"},
			Planner:      workflowconfig.PlannerRef{PlannerID: "streamline", Version: "1"},
			Agents:       agents, Context: workflowconfig.StageContext{Artifacts: map[string]workflowconfig.ContextArtifact{}},
			Result: workflowconfig.StageResultContract{Artifacts: map[string]workflowconfig.ArtifactSlot{
				"report": {Required: true, MediaTypes: []string{"application/json"}},
			}},
		},
		Context: planner.StageContext{
			Parameters: map[string]string{"mode": "strict-secret-value"},
			Artifacts:  map[string]*contracts.ArtifactRef{},
		},
		Workers: workers,
	}
}

func successfulResult(
	summary, slot string, ref contracts.ArtifactRef,
) contracts.StageContentResult {
	return contracts.StageContentResult{
		APIVersion: contracts.APIVersion, Outcome: contracts.StageSucceeded,
		Summary: summary, Artifacts: map[string]contracts.ArtifactRef{slot: ref},
	}
}

func exactRef(namespace, name, revision string) contracts.ArtifactRef {
	return contracts.ArtifactRef{Namespace: namespace, Name: name, Revision: &revision}
}

func createStage(t *testing.T, ctx context.Context, store *runstore.PostgresStore) {
	t.Helper()
	if _, err := store.CreateRun(ctx, runstore.CreateRunParams{
		RunID: "run-streamline", OwnerID: "user", WorkflowName: "workflow", WorkflowVersion: "1",
		WorkflowSchemaVersion: contracts.APIVersion, WorkflowSnapshot: json.RawMessage(`{"name":"workflow"}`),
		Parameters: map[string]string{"mode": "strict-secret-value"},
	}); err != nil {
		t.Fatal(err)
	}
	if _, err := store.TransitionRun(
		ctx, "run-streamline", runstore.RunInitializing, runstore.RunRunning,
		runstore.Reason{Code: "initialized"},
	); err != nil {
		t.Fatal(err)
	}
	if _, err := store.CreateStageExecution(ctx, runstore.CreateStageExecutionParams{
		StageExecutionID: "stage-streamline", RunID: "run-streamline", StageName: "plan", Attempt: 1,
		StageSpecSchemaVersion: contracts.APIVersion, StageSpecSnapshot: json.RawMessage(`{"objective":"plan"}`),
		StageContextSchemaVersion: contracts.APIVersion,
		StageContext: runstore.StageContextSnapshot{
			Parameters: map[string]string{"mode": "strict-secret-value"},
			Artifacts:  map[string]runstore.PinnedContextArtifact{},
		},
	}); err != nil {
		t.Fatal(err)
	}
}

func isolatedPool(t *testing.T, ctx context.Context) *pgxpool.Pool {
	t.Helper()
	databaseURL := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if databaseURL == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	adminConfig, err := pgxpool.ParseConfig(databaseURL)
	if err != nil {
		t.Fatal(err)
	}
	admin, err := pgxpool.NewWithConfig(ctx, adminConfig)
	if err != nil {
		t.Fatal(err)
	}
	if err := admin.Ping(ctx); err != nil {
		admin.Close()
		t.Fatal(err)
	}
	random := make([]byte, 8)
	if _, err := cryptorand.Read(random); err != nil {
		admin.Close()
		t.Fatal(err)
	}
	schema := "contractor_streamline_" + hex.EncodeToString(random)
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
	if _, err := persistencepostgres.ApplyMigrations(ctx, pool); err != nil {
		pool.Close()
		admin.Close()
		t.Fatal(err)
	}
	t.Cleanup(func() {
		pool.Close()
		cleanup, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		if _, err := admin.Exec(cleanup, `DROP SCHEMA `+identifier+` CASCADE`); err != nil {
			t.Logf("drop Streamline test schema: %v", err)
		}
		admin.Close()
	})
	return pool
}
