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
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

const (
	runtimeEndpointCanary = "https://runtime-placement-secret.invalid"
	runtimeTokenCanary    = "runtime-handle-token-secret"
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
		sessions, sessions, workers, inspector, unavailableWorkerStateReader{}, llm,
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
	if gateway.calls() != 5 || workers.calls() != 2 {
		t.Fatalf("semantic calls gateway=%d Workers=%d", gateway.calls(), workers.calls())
	}

	// Simulate a process restart with fresh adapters over the same PostgreSQL
	// state. Completed recovery must bypass ADK, Gateway, and Workers.
	restartedSessions, err := plannersession.New(runstore.NewPostgresStore(pool), plannersession.Options{})
	if err != nil {
		t.Fatal(err)
	}
	restartedFactory, err := streamline.NewFactory(
		restartedSessions, restartedSessions, workers, inspector,
		unavailableWorkerStateReader{}, llm, streamline.DefaultLimits(),
	)
	if err != nil {
		t.Fatal(err)
	}
	restarted, err := restartedFactory.Create(invocation)
	if err != nil {
		t.Fatal(err)
	}
	recovered, err := restarted.Run(ctx)
	if err != nil || recovered.Summary != result.Summary || gateway.calls() != 5 || workers.calls() != 2 {
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
			strings.Contains(payload, "sensitive planner guidance") || strings.Contains(payload, "strict-secret-value") ||
			strings.Contains(payload, "draft ready") || strings.Contains(payload, "report ready") ||
			strings.Contains(payload, runtimeEndpointCanary) || strings.Contains(payload, runtimeTokenCanary) {
			t.Fatalf("durable Planner event leaked model/provider content: %s", payload)
		}
	}
	identity := planner.SessionIdentity{
		SessionID: *execution.PlannerSessionID, StageExecutionID: execution.StageExecutionID,
		InvocationID: *execution.PlannerInvocationID,
	}
	plan, ok, err := sessions.LoadPlan(ctx, identity)
	if err != nil || !ok || plan.Revision != 6 || len(plan.Subtasks) != 2 ||
		plan.Subtasks[0].Objective != "subtask-sensitive-analysis" ||
		plan.Subtasks[0].Instructions != "write an exact draft" ||
		plan.Subtasks[0].Status != planner.PlannerSubtaskSucceeded ||
		plan.Subtasks[1].Status != planner.PlannerSubtaskSucceeded {
		t.Fatalf("typed durable plan = (%+v, %t, %v)", plan, ok, err)
	}
	runEvents, err := store.ListRunEvents(ctx, "run-streamline", 0, 1000)
	if err != nil || len(runEvents) < len(events) {
		t.Fatalf("Run events = (%d, %v), Planner events = %d", len(runEvents), err, len(events))
	}
	runEventsBySequence := make(map[int64]runstore.WorkflowRunEvent, len(runEvents))
	for index, event := range runEvents {
		runEventsBySequence[event.SequenceNumber] = event
		payload := string(event.Data)
		for _, forbidden := range []string{
			gatewayToken, "sensitive objective", "sensitive planner guidance",
			"strict-secret-value", "draft ready", "report ready",
			runtimeEndpointCanary, runtimeTokenCanary,
		} {
			if strings.Contains(payload, forbidden) {
				t.Fatalf("Run event %d leaked %q: %s", index, forbidden, payload)
			}
		}
	}
	for index, event := range events {
		if event.RunEventSequence == nil {
			t.Fatalf("Planner event %d has no Run event link: %+v", index, event)
		}
		linked, ok := runEventsBySequence[*event.RunEventSequence]
		if !ok || linked.EventID != event.EventID {
			t.Fatalf("Planner/Run event linkage %d: %+v / %+v", index, event, linked)
		}
	}
	storedSession, err := store.GetPlannerSession(ctx, *execution.PlannerSessionID)
	if err != nil || !strings.Contains(string(storedSession.State), "subtask-sensitive-analysis") ||
		strings.Contains(string(storedSession.State), "sensitive objective") ||
		strings.Contains(string(storedSession.State), "strict-secret-value") ||
		strings.Contains(string(storedSession.State), "draft ready") ||
		strings.Contains(string(storedSession.State), "report ready") ||
		strings.Contains(string(storedSession.State), runtimeEndpointCanary) ||
		strings.Contains(string(storedSession.State), runtimeTokenCanary) {
		t.Fatalf("durable Planner state redaction = (%s, %v)", storedSession.State, err)
	}
}

type unavailableWorkerStateReader struct{}

func (unavailableWorkerStateReader) ReadWorkerState(
	context.Context,
	contracts.WorkerHandle,
	string,
) (planner.WorkerStateReadResult, error) {
	return planner.WorkerStateReadResult{}, &planner.WorkerStateReadError{
		Code: "runtime_unavailable", Retryable: true,
	}
}

var _ planner.WorkerStateReader = unavailableWorkerStateReader{}

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
		name = "add_subtask"
		arguments = `{"objective":"subtask-sensitive-analysis","instructions":"write an exact draft"}`
	case 1:
		name = "execute_current_subtask"
		arguments = `{"subtask_id":"0"}`
	case 2:
		name = "add_subtask"
		arguments = `{"objective":"review draft","instructions":"write the final report"}`
	case 3:
		name = "execute_current_subtask"
		arguments = `{"subtask_id":"1"}`
	case 4:
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
) (contracts.WorkerCompletion, error) {
	w.mu.Lock()
	defer w.mu.Unlock()
	w.count++
	if binding != "builder" {
		return contracts.WorkerCompletion{}, errors.New("unknown Worker")
	}
	if request.Parameters["mode"] != "strict-secret-value" {
		return contracts.WorkerCompletion{}, errors.New("complete Stage parameters missing")
	}
	switch w.count {
	case 1:
		if request.Objective != "subtask-sensitive-analysis" || request.Instructions != "write an exact draft" {
			return contracts.WorkerCompletion{}, errors.New("stored first subtask missing")
		}
		return successfulResult(
			request.SubtaskID, "draft ready", "draft", exactRef("analysis", "draft", "draft-r1"),
		), nil
	case 2:
		if request.Objective != "review draft" || request.Instructions != "write the final report" {
			return contracts.WorkerCompletion{}, errors.New("stored second subtask missing")
		}
		return successfulResult(
			request.SubtaskID, "report ready", "report", exactRef("review", "report", "report-r1"),
		), nil
	default:
		return contracts.WorkerCompletion{}, errors.New("unexpected Worker call")
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
	const name = "builder"
	template := contracts.AgentTemplateRef{
		TemplateID: name, Version: "1", Digest: "sha256:" + strings.Repeat("a", 64),
	}
	runtime := contracts.WorkerRuntimeRef{RuntimeID: "adk", Version: "1"}
	agents := map[string]workflowconfig.ResolvedAgentBinding{name: {
		Namespace: name,
		Template: contracts.ResolvedAgentTemplate{
			Ref: template, Runtime: runtime, Description: name + " Worker",
		},
	}}
	workers := map[string]contracts.WorkerHandle{name: {
		AllocationID: "allocation-" + name, AgentTemplateRef: template,
		WorkerRuntimeRef: runtime, AgentCard: map[string]any{
			"name": name, "url": runtimeEndpointCanary, "token": runtimeTokenCanary,
		},
		LeaseExpiresAt: time.Now().Add(5 * time.Minute),
	}}
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
	subtaskID, result, slot string, ref contracts.ArtifactRef,
) contracts.WorkerCompletion {
	return contracts.WorkerCompletion{
		APIVersion: contracts.APIVersion,
		Result: &contracts.WorkerResult{
			SubtaskID: subtaskID, Result: result,
			Observations: contracts.WorkerObservations{
				Profile: contracts.WorkerObservationProfileLeanV1,
				Tools:   map[string]contracts.ToolObservationCount{},
			},
			Artifacts: map[string]contracts.ArtifactRef{slot: ref}, Summarized: false,
		},
		InvocationID: "worker-integration", StateRevision: 2,
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
		Parameters:    map[string]string{"mode": "strict-secret-value"},
		RuntimeConfig: runtimeconfig.BuiltInRunSnapshot(),
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
