//go:build e2e

package e2e

import (
	"context"
	"encoding/json"
	"fmt"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"

	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/localpki"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/stageresult"
	"github.com/grauwolf32/contractor/internal/telemetry"
	"github.com/jackc/pgx/v5/pgxpool"
)

// compactionReasoningCanary is provider reasoning of the first Worker
// response. It may reach only the compactor, as a tagged and redacted part,
// because the Workflow selects includeReasoning (WRK-R188).
const compactionReasoningCanary = "ephemeral-compaction-reasoning-canary"

// TestWorkerCompactionContinuesIsolatedWorker runs the opt-in compact@1
// Workflow through the real Server, Runtime and PostgreSQL: the Worker
// compresses its context twice, continues the same subtask and finishes with a
// normal trusted result (V417-003 A7).
func TestWorkerCompactionContinuesIsolatedWorker(t *testing.T) {
	for _, layer := range []string{"default", "binding", "escalation", "router"} {
		t.Run(layer, func(t *testing.T) { workerCompactionSelectionScenario(t, layer) })
	}
}

func workerCompactionSelectionScenario(t *testing.T, layer string) {
	if testing.Short() {
		t.Skip("end-to-end process test")
	}
	databaseURL := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if databaseURL == "" {
		t.Fatal("CONTRACTOR_TEST_DATABASE_URL is required")
	}
	repositoryRoot := repoRoot(t)
	temporaryRoot := t.TempDir()
	ctx, cancel := context.WithTimeout(context.Background(), 240*time.Second)
	defer cancel()

	isolateURL := isolatedDatabase(t, ctx, databaseURL)
	serverBinary := sharedServerBinary(t)
	runChecked(t, repositoryRoot, map[string]string{
		"CONTRACTOR_DATABASE_URL": isolateURL,
	}, serverBinary, "migrate")

	pkiRoot := filepath.Join(temporaryRoot, "pki")
	generator := localpki.Generator{}
	caPaths, err := generator.InitCA(pkiRoot, false)
	if err != nil {
		t.Fatalf("initialize test CA: %v", err)
	}
	leaf := localpki.LeafOptions{IPAddresses: []net.IP{net.ParseIP("127.0.0.1")}}
	controlPlanePaths, err := generator.IssueControlPlane(pkiRoot, localpki.ControlPlaneOptions{
		LeafOptions: leaf, URI: "urn:contractor:control-plane:worker-compaction-e2e",
	})
	if err != nil {
		t.Fatalf("issue Control Plane certificate: %v", err)
	}
	agentPaths, err := generator.IssueAgent(pkiRoot, "worker-compaction-e2e-agent", leaf)
	if err != nil {
		t.Fatalf("issue Runtime Agent certificate: %v", err)
	}

	gateway := newCompactionGateway(llmGatewayToken)
	t.Cleanup(gateway.close)
	configRoot := stageE2EConfiguration(
		t, filepath.Join(repositoryRoot, "configs", "e2e"),
		filepath.Join(temporaryRoot, "configs"), gateway.URL(),
	)
	expectedOrigin := "workflow.executionConfig.workers"
	expectedAttempts := 1
	expectedWorkers := 1
	if layer == "router" {
		expectedWorkers = 2
		path := filepath.Join(configRoot, "workflows/worker_compaction.yaml")
		source, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		text := strings.Replace(string(source), "      compactorModelPolicy: worker_compactor@1\n", "", 1)
		text = strings.Replace(text, "    workers:", `    planner:
      modelPolicy: planner@1
      llmGateway: local-litellm@1
      credential: development-planner
    stages:
      compact:
        agents:
          builder: {compactorModelPolicy: worker_compactor@1}
    workers:`, 1)
		text = strings.Replace(text, "planner: passthrough@1", "planner: router@1", 1)
		text = strings.Replace(text, "      agents:\n        builder:", "      agents:\n        observer: {template: artifact_builder@1}\n        builder:", 1)
		if err := os.WriteFile(path, []byte(text), 0600); err != nil {
			t.Fatal(err)
		}
		expectedOrigin = "workflow.executionConfig.stages.compact.agents.builder"
	}
	if layer != "default" && layer != "router" {
		policyPath := filepath.Join(configRoot, "model-policies/worker_compactor.yaml")
		original, err := os.ReadFile(policyPath)
		if err != nil {
			t.Fatal(err)
		}
		unused := strings.Replace(string(original), "name: worker_compactor", "name: unused_compactor", 1)
		unused = strings.Replace(unused, "model: worker-compactor-model", "model: unselected-compactor-model", 1)
		if err := os.WriteFile(filepath.Join(configRoot, "model-policies/unused_compactor.yaml"), []byte(unused), 0600); err != nil {
			t.Fatal(err)
		}
		path := filepath.Join(configRoot, "workflows/worker_compaction.yaml")
		data, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		source := strings.Replace(string(data), "compactorModelPolicy: worker_compactor@1", "compactorModelPolicy: unused_compactor@1", 1)
		if layer == "binding" {
			source = strings.Replace(source, "  entryStage: compact", `    stages:
      compact:
        agents:
          builder: {compactorModelPolicy: worker_compactor@1}

  entryStage: compact`, 1)
			expectedOrigin = "workflow.executionConfig.stages.compact.agents.builder"
		} else {
			source = strings.Replace(source, "        failed:\n          fail: {}", `        failed:
          escalate:
            maxAttempts: 1
            executionConfig:
              agents:
                builder: {compactorModelPolicy: worker_compactor@1}
            then: {fail: {}}`, 1)
			expectedOrigin = "workflow.stages.compact.on.failed.escalate.executionConfig.agents.builder"
			expectedAttempts = 2
			gateway.mu.Lock()
			gateway.failFirstMain = true
			gateway.mu.Unlock()
		}
		if err := os.WriteFile(path, []byte(source), 0600); err != nil {
			t.Fatal(err)
		}
	}
	publicAddress := freeAddress(t)
	privateAddress := freeAddress(t)
	runtimeAddress := freeAddress(t)
	publicBaseURL := "http://" + publicAddress
	privateBaseURL := "https://" + privateAddress
	runtimeBaseURL := "https://" + runtimeAddress
	userID := "compaction-e2e-user-" + randomHex(t, 8)
	localAuthFile := writeE2ELocalAuth(t, temporaryRoot, userID)
	server := startProcess(t, "Worker compaction Go Server", repositoryRoot, map[string]string{
		"CONTRACTOR_DATABASE_URL":            isolateURL,
		"CONTRACTOR_OPERATOR_CONFIG_ROOT":    configRoot,
		"CONTRACTOR_PUBLIC_LISTEN":           publicAddress,
		"CONTRACTOR_PRIVATE_LISTEN":          privateAddress,
		"CONTRACTOR_PRIVATE_URL":             privateBaseURL,
		"CONTRACTOR_CA_FILE":                 caPaths.Certificate,
		"CONTRACTOR_CONTROL_PLANE_CERT_FILE": controlPlanePaths.Certificate,
		"CONTRACTOR_CONTROL_PLANE_KEY_FILE":  controlPlanePaths.PrivateKey,
		"CONTRACTOR_LLM_GATEWAY_TOKEN":       llmGatewayToken,
		"CONTRACTOR_PUBLIC_BEARER_TOKEN":     publicToken,
		"CONTRACTOR_LOCAL_AUTH_FILE":         localAuthFile,
		"CONTRACTOR_BROWSER_ORIGINS":         "https://ui.contractor.invalid",
	}, serverBinary, "serve")

	publicClient := &http.Client{Timeout: 5 * time.Second}
	waitForHTTP(t, ctx, server, publicClient, publicBaseURL+"/readyz", http.StatusOK)
	python := filepath.Join(repositoryRoot, "runtime", ".venv", "bin", "python")
	if info, statErr := os.Stat(python); statErr != nil || info.IsDir() {
		t.Fatalf("Python Runtime environment is missing at %s; run 'cd runtime && uv sync --locked'", python)
	}
	workRoot := filepath.Join(temporaryRoot, "runtime-work")
	runtimeProcess := startProcess(
		t, "Worker compaction Python Runtime", filepath.Join(repositoryRoot, "runtime"),
		map[string]string{"PYTHONUNBUFFERED": "1"},
		python, "-m", "contractor_runtime",
		"--control-plane-url", privateBaseURL,
		"--advertised-control-url", runtimeBaseURL,
		"--advertised-a2a-url", runtimeBaseURL,
		"--ca-file", caPaths.Certificate,
		"--certificate-file", agentPaths.Certificate,
		"--private-key-file", agentPaths.PrivateKey,
		"--listen", runtimeAddress,
		"--work-root", workRoot,
		"--request-timeout-seconds", "10",
		"--shutdown-grace-seconds", "5",
	)
	controlClient := newMTLSClient(t, caPaths.Certificate, controlPlanePaths)
	waitForHTTP(t, ctx, runtimeProcess, controlClient, runtimeBaseURL+"/healthz", http.StatusOK)
	waitForProcessLog(t, ctx, runtimeProcess, "runtime agent registered")

	var observerProcess *childProcess
	var observerURL, observerWork string
	if layer == "router" {
		paths, err := generator.IssueAgent(pkiRoot, "worker-compaction-router-observer", leaf)
		if err != nil {
			t.Fatal(err)
		}
		address := freeAddress(t)
		observerURL = "https://" + address
		observerWork = filepath.Join(temporaryRoot, "observer-work")
		observerProcess = startProcess(t, "Router observer Python Runtime", filepath.Join(repositoryRoot, "runtime"), map[string]string{"PYTHONUNBUFFERED": "1"},
			python, "-m", "contractor_runtime", "--control-plane-url", privateBaseURL,
			"--advertised-control-url", observerURL, "--advertised-a2a-url", observerURL,
			"--ca-file", caPaths.Certificate, "--certificate-file", paths.Certificate,
			"--private-key-file", paths.PrivateKey, "--listen", address, "--work-root", observerWork,
			"--request-timeout-seconds", "10", "--shutdown-grace-seconds", "5")
		waitForHTTP(t, ctx, observerProcess, controlClient, observerURL+"/healthz", http.StatusOK)
		waitForProcessLog(t, ctx, observerProcess, "runtime agent registered")
	}

	pool, err := pgxpool.New(ctx, isolateURL)
	if err != nil {
		t.Fatalf("open assertion database: %v", err)
	}
	t.Cleanup(pool.Close)
	input := uploadInput(t, publicClient, publicBaseURL)

	runID := createWorkflowRunWithParameters(
		t, publicClient, publicBaseURL, "worker-compaction@1",
		"worker-compaction-two-checkpoints", input, map[string]string{"mode": "compaction"},
	)
	status := waitForRunState(
		t, ctx, server, runtimeProcess, gateway, publicClient, publicBaseURL, runID, "succeeded",
	)
	if failures := gateway.Failures(); len(failures) != 0 {
		t.Fatalf("compaction Gateway fixture failures: %v", failures)
	}
	if len(status.Attempts) != expectedAttempts || status.Attempts[len(status.Attempts)-1].State != "succeeded" {
		t.Fatalf("compaction attempts = %+v", status.Attempts)
	}
	attempt := status.Attempts[len(status.Attempts)-1]
	var stageResult stageresult.StageContentResult
	if err := json.Unmarshal(attempt.Result, &stageResult); err != nil {
		t.Fatalf("Stage result: %v (%s)", err, attempt.Result)
	}
	// A useful continuation is a normal result: nothing is summarized and no
	// handoff is fabricated (WFL-R95).
	if stageResult.Outcome != stageresult.StageSucceeded || stageResult.Error != nil ||
		stageResult.Summarized || stageResult.SummaryArtifact != nil {
		t.Fatalf("compaction Stage result = %+v", stageResult)
	}
	body, mediaType := download(t, publicClient, publicBaseURL+"/v1/runs/"+runID+"/outputs/result")
	if string(body) != "artifact produced after compaction" || mediaType != "text/plain" {
		t.Fatalf("compaction output = (%q, %q)", body, mediaType)
	}
	main, compactor, finalizer := gateway.Counts()
	if main != 6 || compactor != 2 || finalizer != 1 {
		t.Fatalf("Gateway calls main/compactor/finalizer = %d/%d/%d, want 6/2/1", main, compactor, finalizer)
	}
	// Planner calls are charged separately from the same nine Worker calls.
	expectedModelCalls := int64(9)
	if layer == "router" {
		expectedModelCalls += 3
	}
	if attempt.Metrics == nil || !attempt.Metrics.ReportsComplete ||
		attempt.Metrics.ModelCalls != expectedModelCalls || attempt.Metrics.TotalTokens != gateway.TotalTokens() {
		t.Fatalf("public compaction metrics = %+v, want %d calls and %d tokens",
			attempt.Metrics, expectedModelCalls, gateway.TotalTokens())
	}
	executions, err := runstore.NewPostgresStore(pool, artifactpgtest.NewStorage()).ListStageExecutions(ctx, runID)
	if err != nil || len(executions) != expectedAttempts {
		t.Fatalf("compaction StageExecutions = (%d, %v)", len(executions), err)
	}
	completed := executions[len(executions)-1]
	reports, err := telemetry.NewRepository(pool).ListAllocationReports(ctx, completed.StageExecutionID)
	if err != nil || len(reports) != expectedWorkers {
		t.Fatalf("compaction execution reports = (%d, %v)", len(reports), err)
	}
	var report telemetry.StoredAllocationReport
	for _, candidate := range reports {
		if candidate.LogicalAgentName == "builder" {
			report = candidate
		}
		if candidate.LogicalAgentName == "observer" && (candidate.Report.Worker.Compaction != nil || candidate.Report.Worker.Metrics.ModelCalls != nil && *candidate.Report.Worker.Metrics.ModelCalls != 0) {
			t.Fatal("unused Router binding inherited compaction or model work")
		}
	}
	budget := report.Report.Worker.Metrics.WorkerBudget
	if !report.Report.Worker.Complete || budget == nil || budget.ObservedModelCalls != 9 ||
		budget.ObservedToolCalls != 5 || report.Report.Worker.Metrics.Summarizer != nil {
		t.Fatalf("trusted compaction Worker report = %+v", report.Report.Worker)
	}
	// Public diagnostics consume the same final observation as the report;
	// phase counters are breakdowns of the nine charged calls.
	var diagnostics struct {
		Status string                        `json:"status"`
		Items  []telemetry.RuntimeDiagnostic `json:"items"`
		Page   json.RawMessage               `json:"page"`
	}
	getAuthenticatedJSON(t, publicClient, publicBaseURL+"/v1/runs/"+runID+"/attempts/"+completed.StageExecutionID+"/runtime-diagnostics", &diagnostics)
	if diagnostics.Status != "available" || len(diagnostics.Items) != expectedWorkers {
		t.Fatalf("public Runtime diagnostics: %+v", diagnostics)
	}
	var builder telemetry.RuntimeDiagnostic
	for _, item := range diagnostics.Items {
		if item.LogicalAgent == "builder" {
			builder = item
		}
	}
	progress, phases := builder.Compaction, builder.ModelPhases
	if progress == nil || progress.Generation != 2 || progress.Attempts != 2 || progress.Failures != 0 || progress.Phase != "applied" ||
		progress.BeforePrompt == nil || progress.BeforePrompt.Generation != 1 ||
		progress.AfterPrompt == nil || progress.AfterPrompt.Generation != 2 || progress.AfterPrompt.PromptTokens == nil ||
		phases == nil || phases.Main.ModelCalls != 6 || phases.Compaction.ModelCalls != 2 || phases.Finalization.ModelCalls != 1 || phases.Summarization.ModelCalls != 0 {
		t.Fatalf("public compaction observation: progress=%+v phases=%+v", progress, phases)
	}
	if phases.Main.TotalTokens+phases.Compaction.TotalTokens+phases.Finalization.TotalTokens != uint64(*report.Report.Worker.Metrics.TotalTokens) {
		t.Fatal("public model phase usage was duplicated or lost")
	}

	// The immutable Stage snapshot carries the exact compact@1 pins.
	var stage resolved.ResolvedStage
	if err := json.Unmarshal(completed.StageSpecSnapshot, &stage); err != nil {
		t.Fatalf("decode Stage snapshot: %v", err)
	}
	selection := stage.Agents["builder"].ContextManagement
	consumer := stage.ExecutionConfig.Agents["builder"]
	if selection.Strategy != contracts.ContextStrategyCompact || selection.WorkerCompaction == nil ||
		consumer.CompactorModelPolicy == nil || consumer.CompactorModelPolicy.Model != "worker-compactor-model" ||
		consumer.Origins.CompactorModelPolicy != expectedOrigin || !selection.IncludeReasoning ||
		selection.KeepRecentGroups != 1 || selection.MaxCompactions != 4 ||
		selection.ContextWindowRatio != 0.5 ||
		!strings.HasPrefix(selection.Instructions.Text, "Keep decisions, completed steps") {
		t.Fatalf("Stage snapshot compact@1 selection = %+v", selection)
	}

	if observerProcess == nil {
		waitForRuntimeReleased(t, ctx, runtimeProcess, controlClient, runtimeBaseURL, report.AllocationID, workRoot)
	} else {
		waitForRuntimeIdle(t, ctx, runtimeProcess, controlClient, runtimeBaseURL, workRoot)
		waitForRuntimeIdle(t, ctx, observerProcess, controlClient, observerURL, observerWork)
		assertCompactionRetentionSafe(t, ctx, pool, runID, server, observerProcess)
	}
	assertCompactionRetentionSafe(t, ctx, pool, runID, server, runtimeProcess)
}

func assertCompactionRetentionSafe(
	t *testing.T,
	ctx context.Context,
	pool *pgxpool.Pool,
	runID string,
	server, runtimeProcess *childProcess,
) {
	t.Helper()
	queries := []string{
		`SELECT COALESCE(string_agg(data::text, E'\n'), '') FROM workflow_run_events WHERE run_id = $1`,
		`SELECT COALESCE(string_agg(report.report::text, E'\n'), '') FROM allocation_execution_reports AS report JOIN stage_executions AS execution ON execution.stage_execution_id = report.stage_execution_id WHERE execution.run_id = $1`,
		`SELECT COALESCE(string_agg(metrics.metrics::text || metrics.summary::text, E'\n'), '') FROM stage_metrics AS metrics JOIN stage_executions AS execution ON execution.stage_execution_id = metrics.stage_execution_id WHERE execution.run_id = $1`,
	}
	forbidden := []string{
		llmGatewayToken, compactionReasoningCanary,
		"Contractor Worker context compaction input", "Wrote notes a to c",
	}
	for _, query := range queries {
		var retained string
		if err := pool.QueryRow(ctx, query, runID).Scan(&retained); err != nil {
			t.Fatalf("read retained compaction surface: %v", err)
		}
		for _, value := range forbidden {
			if strings.Contains(retained, value) {
				t.Fatalf("retained compaction surface exposed checkpoint or private data")
			}
		}
	}
	for _, process := range []*childProcess{server, runtimeProcess} {
		logs := process.logs.redacted()
		for _, value := range forbidden {
			if strings.Contains(logs, value) {
				t.Fatalf("%s logs exposed checkpoint or private data", process.name)
			}
		}
	}
}

// compactionGateway scripts one isolated Worker invocation that crosses the
// compaction boundary twice. It checks each request in flight and keeps only
// counters and failures.
type compactionGateway struct {
	server *httptest.Server
	token  string

	mu            sync.Mutex
	failFirstMain bool
	main          int
	compactor     int
	finalizer     int
	planner       int
	totalTokens   int64
	pending       map[string]string
	failures      []string
}

func newCompactionGateway(token string) *compactionGateway {
	gateway := &compactionGateway{token: token, pending: map[string]string{}}
	gateway.server = httptest.NewServer(http.HandlerFunc(gateway.serveHTTP))
	return gateway
}

func (g *compactionGateway) URL() string { return g.server.URL + "/v1" }

func (g *compactionGateway) close() { g.server.Close() }

func (g *compactionGateway) Failures() []string {
	g.mu.Lock()
	defer g.mu.Unlock()
	return append([]string(nil), g.failures...)
}

func (g *compactionGateway) Counts() (main, compactor, finalizer int) {
	g.mu.Lock()
	defer g.mu.Unlock()
	return g.main, g.compactor, g.finalizer
}

func (g *compactionGateway) TotalTokens() int64 {
	g.mu.Lock()
	defer g.mu.Unlock()
	return g.totalTokens
}

func (g *compactionGateway) fail(w http.ResponseWriter, message string) {
	g.mu.Lock()
	g.failures = append(g.failures, message)
	g.mu.Unlock()
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusBadRequest)
	_ = json.NewEncoder(w).Encode(map[string]any{
		"error": map[string]any{"message": message, "type": "invalid_request_error"},
	})
}

func (g *compactionGateway) complete(
	w http.ResponseWriter, model string, message map[string]any, finish string, prompt, completion int,
) {
	g.mu.Lock()
	g.totalTokens += int64(prompt + completion)
	g.mu.Unlock()
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]any{
		"id": "chatcmpl-worker-compaction", "object": "chat.completion",
		"created": time.Now().Unix(), "model": model,
		"choices": []any{map[string]any{"index": 0, "message": message, "finish_reason": finish}},
		"usage":   summarizerUsage(prompt, completion, prompt+completion),
	})
}

func (g *compactionGateway) serveHTTP(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost || !strings.HasSuffix(r.URL.Path, "/chat/completions") ||
		r.Header.Get("Authorization") != "Bearer "+g.token {
		g.fail(w, "unsupported compaction Gateway request")
		return
	}
	defer r.Body.Close()
	decoder := json.NewDecoder(http.MaxBytesReader(w, r.Body, 4<<20))
	decoder.UseNumber()
	var request map[string]any
	if err := decoder.Decode(&request); err != nil {
		g.fail(w, "invalid compaction Gateway request")
		return
	}
	model, _ := request["model"].(string)
	model = strings.TrimPrefix(model, "openai/")
	encoded, err := json.Marshal(request)
	if err != nil {
		g.fail(w, "compaction Gateway request could not be inspected")
		return
	}
	text := string(encoded)
	if strings.Contains(text, g.token) {
		g.fail(w, "a model request carried the Gateway credential")
		return
	}
	finalizerInput, isFinalizer, err := decodeWorkerResultFinalizerRequest(request)
	if err != nil {
		g.fail(w, err.Error())
		return
	}
	switch {
	case isFinalizer:
		g.serveFinalizer(w, request, model, finalizerInput)
	case model == "worker-compactor-model":
		g.serveCompactor(w, request, model, text)
	case model == "planner-model":
		g.serveRouter(w, request, model)
	case model == "worker-model":
		g.serveMain(w, request, model, text)
	default:
		g.fail(w, "unexpected compaction Gateway model alias")
	}
}

func (g *compactionGateway) serveRouter(w http.ResponseWriter, request map[string]any, model string) {
	g.mu.Lock()
	g.planner++
	step := g.planner
	g.mu.Unlock()
	var message map[string]any
	switch step {
	case 1:
		message = toolCallMessage("compact-router-add", "add_subtask", map[string]any{"objective": "Complete the source analysis", "instructions": "Use the exact source and produce builder/result"})
	case 2:
		message = toolCallMessage("compact-router-execute", "execute_current_subtask", map[string]any{"subtask_id": "0", "worker_name": "builder"})
	case 3:
		artifact, ok := exactArtifactValue(request, "builder", "result")
		if !ok {
			g.fail(w, "Router did not observe the compacted Worker's exact result")
			return
		}
		message = toolCallMessage("compact-router-finish", "finish", map[string]any{"outcome": "succeeded", "summary": "Compacted Worker finished", "artifacts": map[string]any{"result": artifact}})
	default:
		g.fail(w, "Router exceeded its three-call script")
		return
	}
	g.complete(w, model, message, "tool_calls", 100, 10)
}

func (g *compactionGateway) serveCompactor(w http.ResponseWriter, request map[string]any, model, text string) {
	g.mu.Lock()
	g.compactor++
	call := g.compactor
	g.mu.Unlock()
	if !requestHasNoModelTools(request) || request["response_format"] == nil ||
		requestInteger(request, "max_completion_tokens", "max_tokens") != 2048 {
		g.fail(w, "compactor surface was not tool-free and pinned")
		return
	}
	if !strings.Contains(text, "Contractor Worker context compaction input") ||
		!strings.Contains(text, "Keep decisions, completed steps") {
		g.fail(w, "compactor did not receive its Runtime input and pinned instructions")
		return
	}
	switch call {
	case 1:
		// Group 1 is compacted; group 2 stays protected. Selected reasoning
		// arrives as a tagged, redacted ordinary part.
		if !strings.Contains(text, "note-a") || strings.Contains(text, "note-b") ||
			!strings.Contains(text, `\"reasoning\":\"`+compactionReasoningCanary+` [REDACTED]`) {
			g.fail(w, "first compactor input did not cover exactly group 1")
			return
		}
		g.complete(w, model, compactorSummary("Wrote note-a."), "stop", 400, 20)
	case 2:
		if !strings.Contains(text, "Wrote note-a.") || !strings.Contains(text, "note-b") ||
			!strings.Contains(text, "note-c") || strings.Contains(text, "note-d") {
			g.fail(w, "second compactor input did not replace memory with groups 2 and 3")
			return
		}
		g.complete(w, model, compactorSummary("Wrote notes a to c."), "stop", 500, 20)
	default:
		g.fail(w, "compactor was invoked more than twice")
	}
}

func compactorSummary(summary string) map[string]any {
	encoded, _ := json.Marshal(map[string]string{"summary": summary})
	return map[string]any{"role": "assistant", "content": string(encoded)}
}

func (g *compactionGateway) serveMain(w http.ResponseWriter, request map[string]any, model, text string) {
	tools, err := optionalRequestToolNames(request)
	if err != nil || !equalStringSlices(tools, []string{"read_context_history", "write_text_artifact"}) {
		g.fail(w, fmt.Sprintf("compacted Worker exposed tools %v", tools))
		return
	}
	g.mu.Lock()
	initialFailure := g.failFirstMain
	g.failFirstMain = false
	g.mu.Unlock()
	if initialFailure {
		g.complete(w, model, map[string]any{"role": "assistant", "content": "controlled initial failure"}, "content_filter", 0, 0)
		return
	}
	g.mu.Lock()
	g.main++
	call := g.main
	g.mu.Unlock()
	// After a checkpoint the request carries the memory and only the
	// retained groups; reasoning is never replayed to the Worker model.
	checks := map[int]func() bool{
		3: func() bool {
			return strings.Contains(text, "Wrote note-a.") && !strings.Contains(text, "compaction-write-a") &&
				strings.Contains(text, "compaction-write-b")
		},
		4: func() bool {
			return strings.Contains(text, "compaction-write-c") && !strings.Contains(text, "compaction-write-a")
		},
		5: func() bool {
			return strings.Contains(text, "Wrote notes a to c.") && !strings.Contains(text, "Wrote note-a.") &&
				!strings.Contains(text, "compaction-write-c") && strings.Contains(text, "compaction-write-d")
		},
	}
	if check, ok := checks[call]; ok && (!check() || strings.Contains(text, compactionReasoningCanary)) {
		g.fail(w, fmt.Sprintf("main request %d did not carry the exact checkpoint projection", call))
		return
	}
	write := func(id, name, body string) map[string]any {
		return toolCallMessage(id, "write_text_artifact", map[string]any{
			"name": name, "text": body, "media_type": "text/plain", "expected_revision": nil,
		})
	}
	switch call {
	case 1:
		message := write("compaction-write-a", "note-a", "first note")
		message["reasoning_content"] = compactionReasoningCanary + " " + g.token
		g.complete(w, model, message, "tool_calls", 1000, 10)
	case 2:
		// Measured usage at the boundary: min(8192*0.5, 8192-1024) = 4096.
		g.complete(w, model, write("compaction-write-b", "note-b", "second note"), "tool_calls", 5000, 10)
	case 3:
		g.complete(w, model, write("compaction-write-c", "note-c", "third note"), "tool_calls", 1200, 10)
	case 4:
		g.complete(w, model, write("compaction-write-d", "note-d", "fourth note"), "tool_calls", 5200, 10)
	case 5:
		g.complete(w, model, write("compaction-write-result", "result", "artifact produced after compaction"),
			"tool_calls", 1300, 10)
	case 6:
		subtaskID, err := workerRequestSubtaskID(request)
		if err != nil {
			g.fail(w, err.Error())
			return
		}
		result := "Compacted Worker result"
		g.mu.Lock()
		g.pending[subtaskID] = result
		g.mu.Unlock()
		g.complete(w, model, map[string]any{"role": "assistant", "content": result}, "stop", 1400, 10)
	default:
		g.fail(w, "unexpected additional main Worker call")
	}
}

func (g *compactionGateway) serveFinalizer(
	w http.ResponseWriter, request map[string]any, model string, input workerResultFinalizerInput,
) {
	if model != "worker-model" || !requestHasNoModelTools(request) || request["response_format"] == nil {
		g.fail(w, "invalid Worker result-finalizer surface")
		return
	}
	g.mu.Lock()
	expected, ok := g.pending[input.SubtaskID]
	delete(g.pending, input.SubtaskID)
	g.finalizer++
	g.mu.Unlock()
	if !ok || expected != input.ResultText {
		g.fail(w, "Worker result finalizer changed or invented a candidate")
		return
	}
	message, err := workerResultFinalizerMessage(input)
	if err != nil {
		g.fail(w, err.Error())
		return
	}
	g.complete(w, model, message, "stop", 7, 3)
}
