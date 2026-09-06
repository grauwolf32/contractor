//go:build e2e

package e2e

import (
	"bytes"
	"context"
	cryptorand "crypto/rand"
	"crypto/tls"
	"crypto/x509"
	"encoding/hex"
	"encoding/json"
	"errors"
	"io"
	"net"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/auth"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/localpki"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/telemetry"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

const (
	e2eInput        = "contractor-e2e-input\n"
	e2eMediaType    = "text/plain"
	publicToken     = "contractor-e2e-public-token"
	llmGatewayToken = "contractor-e2e-gateway-token"
)

type artifactRef struct {
	Namespace string  `json:"namespace"`
	Name      string  `json:"name"`
	Revision  *string `json:"revision,omitempty"`
}

type runCreateResponse struct {
	RunID                string            `json:"runId"`
	ProjectID            *string           `json:"projectId,omitempty"`
	State                string            `json:"state"`
	RuntimeLabels        []string          `json:"runtimeLabels"`
	Labels               map[string]string `json:"labels"`
	RuntimeConfiguration json.RawMessage   `json:"runtimeConfiguration"`
	ProjectHTTPTarget    json.RawMessage   `json:"projectHttpTarget,omitempty"`
}

type runStatus struct {
	RunID                  string                 `json:"runId"`
	ProjectID              *string                `json:"projectId,omitempty"`
	Workflow               string                 `json:"workflow"`
	State                  string                 `json:"state"`
	Deletable              bool                   `json:"deletable"`
	RuntimeLabels          []string               `json:"runtimeLabels"`
	Labels                 map[string]string      `json:"labels"`
	RuntimeConfiguration   json.RawMessage        `json:"runtimeConfiguration"`
	ProjectHTTPTarget      json.RawMessage        `json:"projectHttpTarget,omitempty"`
	Cancellation           json.RawMessage        `json:"cancellation,omitempty"`
	Parameters             map[string]string      `json:"parameters,omitempty"`
	Inputs                 map[string]artifactRef `json:"inputs,omitempty"`
	Attempts               []runAttempt           `json:"attempts"`
	Transitions            []json.RawMessage      `json:"transitions"`
	Outputs                map[string]artifactRef `json:"outputs"`
	OutputPublications     []runOutputPublication `json:"outputPublications"`
	EventCursor            json.RawMessage        `json:"eventCursor,omitempty"`
	ActiveStageExecutionID *string                `json:"activeStageExecutionId,omitempty"`
	CreatedAt              time.Time              `json:"createdAt,omitempty"`
	UpdatedAt              time.Time              `json:"updatedAt,omitempty"`
	StartedAt              *time.Time             `json:"startedAt,omitempty"`
	FinishedAt             *time.Time             `json:"finishedAt,omitempty"`
}

type runOutputPublication struct {
	Output       string       `json:"output"`
	Status       string       `json:"status"`
	Source       artifactRef  `json:"source"`
	Target       *artifactRef `json:"target,omitempty"`
	ErrorCode    string       `json:"errorCode,omitempty"`
	ErrorMessage string       `json:"errorMessage,omitempty"`
	CreatedAt    time.Time    `json:"createdAt"`
}

type runAttempt struct {
	StageExecutionID     string             `json:"stageExecutionId"`
	Stage                string             `json:"stage"`
	Objective            string             `json:"objective,omitempty"`
	Attempt              int                `json:"attempt"`
	PreviousExecutionID  *string            `json:"previousExecutionId,omitempty"`
	ExecutionConfig      json.RawMessage    `json:"executionConfig"`
	RuntimeConfiguration json.RawMessage    `json:"runtimeConfiguration,omitempty"`
	State                string             `json:"state"`
	Result               json.RawMessage    `json:"result,omitempty"`
	Termination          json.RawMessage    `json:"termination,omitempty"`
	Metrics              *telemetry.Summary `json:"metrics,omitempty"`
	Diagnostics          json.RawMessage    `json:"diagnostics,omitempty"`
	Plan                 json.RawMessage    `json:"plan,omitempty"`
	CreatedAt            time.Time          `json:"createdAt,omitempty"`
	UpdatedAt            time.Time          `json:"updatedAt,omitempty"`
	PlannerStartedAt     *time.Time         `json:"plannerStartedAt,omitempty"`
	TerminalAt           *time.Time         `json:"terminalAt,omitempty"`
}

func TestLocalGoToPythonArtifactCopy(t *testing.T) {
	if testing.Short() {
		t.Skip("end-to-end process test")
	}
	databaseURL := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if databaseURL == "" {
		t.Fatal("CONTRACTOR_TEST_DATABASE_URL is required")
	}
	repositoryRoot := repoRoot(t)
	temporaryRoot := t.TempDir()
	ctx, cancel := context.WithTimeout(context.Background(), 110*time.Second)
	defer cancel()

	isolateURL := isolatedDatabase(t, ctx, databaseURL)
	serverBinary := filepath.Join(temporaryRoot, "contractor-server")
	runChecked(t, repositoryRoot, nil, "go", "build", "-o", serverBinary, "./cmd/contractor-server")
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
		LeafOptions: leaf, URI: "urn:contractor:control-plane:e2e",
	})
	if err != nil {
		t.Fatalf("issue Control Plane certificate: %v", err)
	}
	agentPaths, err := generator.IssueAgent(pkiRoot, "e2e-agent", leaf)
	if err != nil {
		t.Fatalf("issue Runtime Agent certificate: %v", err)
	}

	gateway := newFakeGateway(llmGatewayToken)
	t.Cleanup(gateway.close)
	configRoot := stageE2EConfiguration(
		t, filepath.Join(repositoryRoot, "configs", "e2e"),
		filepath.Join(temporaryRoot, "configs"), gateway.URL(),
	)
	publicAddress := freeAddress(t)
	privateAddress := freeAddress(t)
	runtimeAddress := freeAddress(t)
	publicBaseURL := "http://" + publicAddress
	privateBaseURL := "https://" + privateAddress
	runtimeBaseURL := "https://" + runtimeAddress
	userID := "e2e-user-" + randomHex(t, 8)
	localAuthFile := writeE2ELocalAuth(t, temporaryRoot, userID)
	server := startProcess(t, "Go Server", repositoryRoot, map[string]string{
		"CONTRACTOR_DATABASE_URL":            isolateURL,
		"CONTRACTOR_CONFIG_ROOT":             configRoot,
		"CONTRACTOR_PUBLIC_LISTEN":           publicAddress,
		"CONTRACTOR_PRIVATE_LISTEN":          privateAddress,
		"CONTRACTOR_PRIVATE_URL":             privateBaseURL,
		"CONTRACTOR_CA_FILE":                 caPaths.Certificate,
		"CONTRACTOR_CONTROL_PLANE_CERT_FILE": controlPlanePaths.Certificate,
		"CONTRACTOR_CONTROL_PLANE_KEY_FILE":  controlPlanePaths.PrivateKey,
		"CONTRACTOR_LLM_GATEWAY_TOKEN":       llmGatewayToken,
		"CONTRACTOR_PUBLIC_USER_ID":          userID,
		"CONTRACTOR_PUBLIC_BEARER_TOKEN":     publicToken,
		"CONTRACTOR_LOCAL_AUTH_FILE":         localAuthFile,
		"CONTRACTOR_BROWSER_ORIGINS":         "https://ui.contractor.invalid",
	}, serverBinary, "serve")

	publicClient := &http.Client{Timeout: 5 * time.Second}
	waitForHTTP(t, ctx, server, publicClient, publicBaseURL+"/readyz", http.StatusOK)
	assertPrivateTLSRejectsUnauthenticated(t, caPaths.Certificate, privateBaseURL+"/private/v1/agents/register")

	python := filepath.Join(repositoryRoot, "runtime", ".venv", "bin", "python")
	if info, statErr := os.Stat(python); statErr != nil || info.IsDir() {
		t.Fatalf("Python Runtime environment is missing at %s; run 'cd runtime && uv sync --locked'", python)
	}
	workRoot := filepath.Join(temporaryRoot, "runtime-work")
	runtimeProcess := startProcess(
		t, "Python Runtime Agent", filepath.Join(repositoryRoot, "runtime"),
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
	assertPrivateTLSRejectsUnauthenticated(t, caPaths.Certificate, runtimeBaseURL+"/healthz")
	waitForProcessLog(t, ctx, runtimeProcess, "runtime agent registered")

	uploaded := uploadInput(t, publicClient, publicBaseURL)
	runID := createRun(t, publicClient, publicBaseURL, uploaded)
	completed := waitForRun(t, ctx, server, runtimeProcess, gateway, publicClient, publicBaseURL, runID)
	if len(completed.Attempts) != 1 || completed.Attempts[0].Stage != "copy" ||
		completed.Attempts[0].Attempt != 1 || completed.Attempts[0].State != "succeeded" {
		t.Fatalf("unexpected Stage attempts: %+v", completed.Attempts)
	}
	if completed.Attempts[0].Metrics == nil ||
		completed.Attempts[0].Metrics.ModelCalls != 4 ||
		completed.Attempts[0].Metrics.ToolCalls != 3 ||
		!completed.Attempts[0].Metrics.ReportsComplete {
		t.Fatalf("unexpected public metrics summary: %+v", completed.Attempts[0].Metrics)
	}
	output, ok := completed.Outputs["result"]
	if !ok || output.Revision == nil {
		t.Fatalf("Run has no exact result output: %+v", completed.Outputs)
	}

	data, mediaType := download(t, publicClient, publicBaseURL+"/v1/runs/"+url.PathEscape(runID)+"/outputs/result")
	if string(data) != e2eInput || mediaType != e2eMediaType {
		t.Fatalf("output = (%q, %q), want (%q, %q)", data, mediaType, e2eInput, e2eMediaType)
	}
	originalURL := publicBaseURL + "/v1/artifacts/projects/source?revision=" + url.QueryEscape(*uploaded.Revision)
	original, originalMediaType := download(t, publicClient, originalURL)
	if string(original) != e2eInput || originalMediaType != e2eMediaType {
		t.Fatalf("original User artifact changed: (%q, %q)", original, originalMediaType)
	}

	pool, err := pgxpool.New(ctx, isolateURL)
	if err != nil {
		t.Fatalf("open assertion database: %v", err)
	}
	t.Cleanup(pool.Close)
	allocationID := assertDurableExecution(t, ctx, pool, runID, output)
	waitForRuntimeReleased(t, ctx, runtimeProcess, controlClient, runtimeBaseURL, allocationID, workRoot)

	budgetRunID := createWorkflowRun(
		t, publicClient, publicBaseURL, "budget-loop@1", "e2e-budget-loop", uploaded,
	)
	budgetStatus := waitForRunState(
		t, ctx, server, runtimeProcess, gateway, publicClient, publicBaseURL, budgetRunID, "failed",
	)
	if len(budgetStatus.Attempts) != 1 || budgetStatus.Attempts[0].Stage != "analyze" ||
		budgetStatus.Attempts[0].State != "failed" || budgetStatus.Attempts[0].Metrics == nil ||
		budgetStatus.Attempts[0].Metrics.ModelCalls != 3 ||
		budgetStatus.Attempts[0].Metrics.ToolCalls != 3 ||
		!budgetStatus.Attempts[0].Metrics.ReportsComplete {
		t.Fatalf(
			"unexpected bounded Stage attempt: %+v metrics=%+v",
			budgetStatus.Attempts, budgetStatus.Attempts[0].Metrics,
		)
	}
	var budgetResult contracts.StageContentResult
	if err := json.Unmarshal(budgetStatus.Attempts[0].Result, &budgetResult); err != nil ||
		budgetResult.Error == nil || budgetResult.Error.Code != "worker_budget_exhausted" ||
		!budgetResult.Error.Retryable {
		t.Fatalf("bounded Worker result = (%+v, %v)", budgetResult, err)
	}
	budgetAllocationID := assertBudgetExecution(t, ctx, pool, budgetRunID)
	waitForRuntimeReleased(
		t, ctx, runtimeProcess, controlClient, runtimeBaseURL, budgetAllocationID, workRoot,
	)
	if gateway.Calls() != 6 || len(gateway.Failures()) != 0 {
		t.Fatalf("fake gateway calls/failures = %d/%v, want 6/none", gateway.Calls(), gateway.Failures())
	}

	for _, secret := range []string{publicToken, llmGatewayToken} {
		if strings.Contains(server.logs.redacted(), secret) || strings.Contains(runtimeProcess.logs.redacted(), secret) {
			t.Fatalf("process logs contain a configured secret")
		}
	}
}

func uploadInput(t *testing.T, client *http.Client, baseURL string) artifactRef {
	t.Helper()
	request, err := http.NewRequest(http.MethodPut, baseURL+"/v1/artifacts/projects/source", strings.NewReader(e2eInput))
	if err != nil {
		t.Fatal(err)
	}
	request.Header.Set("Authorization", "Bearer "+publicToken)
	request.Header.Set("Content-Type", e2eMediaType)
	request.Header.Set("If-None-Match", "*")
	response := do(t, client, request, http.StatusCreated)
	defer response.Body.Close()
	var payload struct {
		Artifact  artifactRef `json:"artifact"`
		MediaType string      `json:"mediaType"`
		Size      int64       `json:"size"`
	}
	decodeResponse(t, response, &payload)
	if payload.Artifact.Namespace != "projects" || payload.Artifact.Name != "source" ||
		payload.Artifact.Revision == nil || payload.MediaType != e2eMediaType || payload.Size != int64(len(e2eInput)) {
		t.Fatalf("upload returned invalid ArtifactRef: %+v", payload.Artifact)
	}
	return payload.Artifact
}

func createRun(t *testing.T, client *http.Client, baseURL string, input artifactRef) string {
	return createWorkflowRun(t, client, baseURL, "artifact-copy@1", "e2e-create-run", input)
}

func createWorkflowRun(
	t *testing.T,
	client *http.Client,
	baseURL, workflow, idempotencyKey string,
	input artifactRef,
) string {
	return createWorkflowRunWithParameters(
		t, client, baseURL, workflow, idempotencyKey, input, map[string]string{},
	)
}

func createWorkflowRunWithParameters(
	t *testing.T,
	client *http.Client,
	baseURL, workflow, idempotencyKey string,
	input artifactRef,
	parameters map[string]string,
) string {
	t.Helper()
	body, err := json.Marshal(map[string]any{
		"workflow": workflow, "parameters": parameters,
		"artifacts": map[string]artifactRef{"source": input},
	})
	if err != nil {
		t.Fatal(err)
	}
	request, err := http.NewRequest(http.MethodPost, baseURL+"/v1/runs", bytes.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	request.Header.Set("Authorization", "Bearer "+publicToken)
	request.Header.Set("Content-Type", "application/json")
	request.Header.Set("Idempotency-Key", idempotencyKey)
	response := do(t, client, request, http.StatusAccepted)
	defer response.Body.Close()
	var payload runCreateResponse
	decodeResponse(t, response, &payload)
	if payload.RunID == "" || payload.State != "running" {
		t.Fatalf("create Run response = %+v", payload)
	}
	return payload.RunID
}

func waitForRun(
	t *testing.T,
	ctx context.Context,
	server, runtimeProcess *childProcess,
	gateway interface{ Failures() []string },
	client *http.Client,
	baseURL, runID string,
) runStatus {
	return waitForRunState(
		t, ctx, server, runtimeProcess, gateway, client, baseURL, runID, "succeeded",
	)
}

func waitForRunState(
	t *testing.T,
	ctx context.Context,
	server, runtimeProcess *childProcess,
	gateway interface{ Failures() []string },
	client *http.Client,
	baseURL, runID, expectedState string,
) runStatus {
	t.Helper()
	ticker := time.NewTicker(200 * time.Millisecond)
	defer ticker.Stop()
	for {
		request, _ := http.NewRequestWithContext(ctx, http.MethodGet, baseURL+"/v1/runs/"+url.PathEscape(runID), nil)
		request.Header.Set("Authorization", "Bearer "+publicToken)
		response, err := client.Do(request)
		if err == nil {
			var status runStatus
			if response.StatusCode == http.StatusOK {
				decodeResponse(t, response, &status)
			} else {
				response.Body.Close()
				t.Fatalf("poll Run returned HTTP %d", response.StatusCode)
			}
			response.Body.Close()
			switch status.State {
			case expectedState:
				return status
			case "succeeded", "failed", "cancelled":
				t.Fatalf("Run reached %s: %+v\nserver:\n%s\nruntime:\n%s\ngateway: %v",
					status.State, status,
					server.logs.redacted(publicToken, llmGatewayToken),
					runtimeProcess.logs.redacted(publicToken, llmGatewayToken),
					gateway.Failures())
			}
		}
		for _, process := range []*childProcess{server, runtimeProcess} {
			if exited, processErr := process.exited(); exited {
				t.Fatalf("%s exited while Run was active: %v\n%s", process.name, processErr,
					process.logs.redacted(publicToken, llmGatewayToken))
			}
		}
		select {
		case <-ctx.Done():
			t.Fatalf("wait for Run: %v\nserver:\n%s\nruntime:\n%s\ngateway: %v", ctx.Err(),
				server.logs.redacted(publicToken, llmGatewayToken),
				runtimeProcess.logs.redacted(publicToken, llmGatewayToken), gateway.Failures())
		case <-ticker.C:
		}
	}
}

func assertDurableExecution(
	t *testing.T,
	ctx context.Context,
	pool *pgxpool.Pool,
	runID string,
	output artifactRef,
) string {
	t.Helper()
	store := runstore.NewPostgresStore(pool)
	executions, err := store.ListStageExecutions(ctx, runID)
	if err != nil || len(executions) != 1 {
		t.Fatalf("StageExecutions = (%+v, %v), want one", executions, err)
	}
	execution := executions[0]
	if execution.State != runstore.StageSucceeded || execution.PlannerSessionID == nil ||
		execution.PlannerInvocationID == nil || execution.CandidateResult == nil ||
		execution.AcceptedResult == nil || execution.FinalizationID == nil ||
		execution.PlannerStartedAt == nil || execution.TerminalAt == nil {
		t.Fatalf("incomplete terminal StageExecution: %+v", execution)
	}
	acceptedCopied := execution.AcceptedResult.Artifacts["copied"]
	if execution.AcceptedResult.Outcome != contracts.StageSucceeded || acceptedCopied.Revision == nil {
		t.Fatalf("accepted result/output mismatch: %+v / %+v", execution.AcceptedResult, output)
	}
	events, err := store.ListPlannerEvents(ctx, *execution.PlannerSessionID, 0)
	if err != nil || len(events) != 3 {
		t.Fatalf("Planner events = (%+v, %v), want start, request, and completion", events, err)
	}
	var requestEvent struct {
		Kind string `json:"kind"`
	}
	var completionEvent struct {
		Kind string `json:"kind"`
	}
	if json.Unmarshal(events[1].Event, &requestEvent) != nil || requestEvent.Kind != "worker_request" ||
		json.Unmarshal(events[2].Event, &completionEvent) != nil || completionEvent.Kind != "planner_completed" {
		t.Fatalf("unexpected Planner event sequence: %s / %s / %s", events[0].Event, events[1].Event, events[2].Event)
	}
	allocations, err := store.ListStageAllocations(ctx, execution.StageExecutionID)
	if err != nil || len(allocations) != 1 || allocations[0].RuntimeAgentInstanceID == "" {
		t.Fatalf("Stage allocations = (%+v, %v), want one registered Runtime Agent", allocations, err)
	}
	reports, err := store.ListStageExecutionReports(ctx, execution.StageExecutionID)
	if err != nil || len(reports) != 1 {
		t.Fatalf("execution reports = (%+v, %v), want one", reports, err)
	}
	report := reports[0]
	if report.AllocationID != allocations[0].AllocationID || report.LogicalAgentName != "builder" ||
		!report.Report.Worker.Complete || !report.Report.Runtime.Complete {
		t.Fatalf("invalid trusted report envelope: %+v", report)
	}
	workerMetrics := report.Report.Worker.Metrics
	wantCounters := map[string]struct {
		got  *int64
		want int64
	}{
		"llm_calls":      {workerMetrics.ModelCalls, 4},
		"input_tokens":   {workerMetrics.InputTokens, 28},
		"output_tokens":  {workerMetrics.OutputTokens, 12},
		"total_tokens":   {workerMetrics.TotalTokens, 40},
		"read_artifact":  {workerMetrics.Tools["read_artifact"].Calls, 1},
		"write_artifact": {workerMetrics.Tools["write_artifact"].Calls, 1},
	}
	for key, counter := range wantCounters {
		if counter.got == nil || *counter.got != counter.want {
			t.Fatalf("report counter %s = %v, want %d; all=%+v", key, counter.got, counter.want, workerMetrics)
		}
	}
	budget := workerMetrics.WorkerBudget
	if budget == nil || budget.MaxModelCalls != 8 || budget.MaxToolCalls != 16 ||
		budget.MaxTotalTokens != 32768 || budget.ObservedModelCalls != 4 ||
		budget.ObservedToolCalls != 2 || budget.ObservedTotalTokens != 40 ||
		budget.TokenUsageUnavailable != 0 || budget.Exhausted != nil {
		t.Fatalf("successful Worker budget = %+v", budget)
	}

	var frozen bool
	var currentRevision string
	if err := pool.QueryRow(ctx, `
SELECT frozen, current_revision
FROM artifact_bindings
WHERE scope_kind = 'run' AND scope_id = $1 AND namespace = 'outputs' AND name = 'result'`, runID,
	).Scan(&frozen, &currentRevision); err != nil || !frozen || currentRevision != *output.Revision {
		t.Fatalf("output binding = (frozen=%v revision=%q err=%v), want frozen exact output", frozen, currentRevision, err)
	}
	lineage := make(map[string]int)
	rows, err := pool.Query(ctx, `
SELECT lineage_kind, count(*)
FROM artifact_lineage
WHERE target_scope_kind = 'run' AND target_scope_id = $1
GROUP BY lineage_kind`, runID)
	if err != nil {
		t.Fatal(err)
	}
	for rows.Next() {
		var kind string
		var count int
		if err := rows.Scan(&kind, &count); err != nil {
			rows.Close()
			t.Fatal(err)
		}
		lineage[kind] = count
	}
	rows.Close()
	if lineage["input_fork"] != 1 || lineage["output_bind"] != 1 {
		t.Fatalf("artifact lineage = %v, want one input fork and output bind", lineage)
	}
	var sourceNamespace, sourceName, sourceRevision string
	if err := pool.QueryRow(ctx, `
SELECT source_namespace, source_name, source_revision
FROM artifact_lineage
WHERE target_scope_kind = 'run' AND target_scope_id = $1
  AND target_namespace = 'outputs' AND target_name = 'result'
  AND target_revision = $2 AND lineage_kind = 'output_bind'`, runID, *output.Revision,
	).Scan(&sourceNamespace, &sourceName, &sourceRevision); err != nil ||
		sourceNamespace != acceptedCopied.Namespace || sourceName != acceptedCopied.Name ||
		sourceRevision != *acceptedCopied.Revision {
		t.Fatalf("output lineage source = %s/%s@%s (err=%v), want accepted %v",
			sourceNamespace, sourceName, sourceRevision, err, acceptedCopied)
	}
	return allocations[0].AllocationID
}

func assertBudgetExecution(
	t *testing.T,
	ctx context.Context,
	pool *pgxpool.Pool,
	runID string,
) string {
	t.Helper()
	store := runstore.NewPostgresStore(pool)
	executions, err := store.ListStageExecutions(ctx, runID)
	if err != nil || len(executions) != 1 {
		t.Fatalf("bounded StageExecutions = (%+v, %v), want one", executions, err)
	}
	execution := executions[0]
	if execution.State != runstore.StageFailed || execution.AcceptedResult == nil ||
		execution.AcceptedResult.Outcome != contracts.StageFailed ||
		execution.AcceptedResult.Error == nil ||
		execution.AcceptedResult.Error.Code != "worker_budget_exhausted" ||
		!execution.AcceptedResult.Error.Retryable || execution.Termination != nil {
		t.Fatalf("bounded Stage terminal contract = %+v", execution)
	}
	allocations, err := store.ListStageAllocations(ctx, execution.StageExecutionID)
	if err != nil || len(allocations) != 1 {
		t.Fatalf("bounded Stage allocations = (%+v, %v), want one", allocations, err)
	}
	reports, err := store.ListStageExecutionReports(ctx, execution.StageExecutionID)
	if err != nil || len(reports) != 1 {
		t.Fatalf("bounded execution reports = (%+v, %v), want one", reports, err)
	}
	report := reports[0].Report.Worker
	budget := report.Metrics.WorkerBudget
	if !report.Complete || !reports[0].Report.Runtime.Complete || budget == nil ||
		budget.MaxModelCalls != 8 || budget.MaxToolCalls != 2 || budget.MaxTotalTokens != 32768 ||
		budget.ObservedModelCalls != 3 || budget.ObservedToolCalls != 2 ||
		budget.ObservedTotalTokens != 30 || budget.TokenUsageUnavailable != 0 ||
		budget.Exhausted == nil || *budget.Exhausted != "tool_calls" {
		t.Fatalf("bounded Worker report = %+v", report)
	}
	found := false
	for _, executionError := range report.Errors {
		if executionError.Code == "worker_budget_exhausted" && executionError.Retryable != nil &&
			*executionError.Retryable {
			found = true
		}
	}
	if !found {
		t.Fatalf("bounded Worker report has no retryable exhaustion error: %+v", report.Errors)
	}
	return allocations[0].AllocationID
}

func waitForRuntimeReleased(
	t *testing.T,
	ctx context.Context,
	process *childProcess,
	client *http.Client,
	baseURL, allocationID, workRoot string,
) {
	t.Helper()
	deadline := time.Now().Add(10 * time.Second)
	for time.Now().Before(deadline) {
		request, _ := http.NewRequestWithContext(ctx, http.MethodGet, baseURL+"/readyz", nil)
		response, err := client.Do(request)
		if err == nil {
			var payload struct {
				State string `json:"state"`
			}
			if response.StatusCode == http.StatusOK {
				_ = json.NewDecoder(response.Body).Decode(&payload)
			}
			response.Body.Close()
			if payload.State == "idle" && workRootEmpty(workRoot) {
				cardURL := baseURL + "/private/v1/allocations/" + url.PathEscape(allocationID) + "/a2a/.well-known/agent-card.json"
				cardRequest, _ := http.NewRequestWithContext(ctx, http.MethodGet, cardURL, nil)
				cardResponse, cardErr := client.Do(cardRequest)
				cardReleased := false
				if cardErr == nil {
					cardReleased = cardResponse.StatusCode == http.StatusConflict
					cardResponse.Body.Close()
				}
				stateURL := baseURL + "/private/v1/allocations/" + url.PathEscape(allocationID) + "/agent-state"
				stateRequest, _ := http.NewRequestWithContext(ctx, http.MethodGet, stateURL, nil)
				stateResponse, stateErr := client.Do(stateRequest)
				stateReleased := false
				if stateErr == nil {
					stateReleased = stateResponse.StatusCode == http.StatusConflict ||
						stateResponse.StatusCode == http.StatusNotFound
					stateResponse.Body.Close()
				}
				if cardReleased && stateReleased {
					return
				}
			}
		}
		if exited, processErr := process.exited(); exited {
			t.Fatalf("Runtime Agent exited before release verification: %v\n%s", processErr,
				process.logs.redacted(publicToken, llmGatewayToken))
		}
		time.Sleep(100 * time.Millisecond)
	}
	t.Fatalf("Runtime allocation was not fully released; work root empty=%v\n%s",
		workRootEmpty(workRoot), process.logs.redacted(publicToken, llmGatewayToken))
}

func workRootEmpty(root string) bool {
	entries, err := os.ReadDir(root)
	return errors.Is(err, os.ErrNotExist) || err == nil && len(entries) == 0
}

func download(t *testing.T, client *http.Client, target string) ([]byte, string) {
	t.Helper()
	request, _ := http.NewRequest(http.MethodGet, target, nil)
	request.Header.Set("Authorization", "Bearer "+publicToken)
	response := do(t, client, request, http.StatusOK)
	defer response.Body.Close()
	data, err := io.ReadAll(response.Body)
	if err != nil {
		t.Fatal(err)
	}
	return data, response.Header.Get("Content-Type")
}

func do(t *testing.T, client *http.Client, request *http.Request, expected int) *http.Response {
	t.Helper()
	response, err := client.Do(request)
	if err != nil {
		t.Fatalf("%s %s: %v", request.Method, request.URL, err)
	}
	if response.StatusCode != expected {
		body := copyBounded(response.Body)
		response.Body.Close()
		t.Fatalf("%s %s returned HTTP %d, want %d: %s", request.Method, request.URL, response.StatusCode, expected, body)
	}
	return response
}

func decodeResponse(t *testing.T, response *http.Response, target any) {
	t.Helper()
	decoder := json.NewDecoder(response.Body)
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(target); err != nil {
		t.Fatalf("decode HTTP %d response: %v", response.StatusCode, err)
	}
}

func waitForHTTP(
	t *testing.T,
	ctx context.Context,
	process *childProcess,
	client *http.Client,
	target string,
	expected int,
) {
	t.Helper()
	for {
		request, _ := http.NewRequestWithContext(ctx, http.MethodGet, target, nil)
		response, err := client.Do(request)
		if err == nil {
			response.Body.Close()
			if response.StatusCode == expected {
				return
			}
		}
		if exited, processErr := process.exited(); exited {
			t.Fatalf("%s exited during readiness: %v\n%s", process.name, processErr,
				process.logs.redacted(publicToken, llmGatewayToken))
		}
		select {
		case <-ctx.Done():
			t.Fatalf("wait for %s readiness: %v\n%s", process.name, ctx.Err(),
				process.logs.redacted(publicToken, llmGatewayToken))
		case <-time.After(50 * time.Millisecond):
		}
	}
}

func waitForProcessLog(t *testing.T, ctx context.Context, process *childProcess, expected string) {
	t.Helper()
	for {
		logs := process.logs.redacted(publicToken, llmGatewayToken)
		if strings.Contains(logs, expected) {
			return
		}
		if exited, processErr := process.exited(); exited {
			t.Fatalf("%s exited while waiting for log %q: %v\n%s", process.name, expected, processErr, logs)
		}
		select {
		case <-ctx.Done():
			t.Fatalf("wait for %s log %q: %v\n%s", process.name, expected, ctx.Err(), logs)
		case <-time.After(50 * time.Millisecond):
		}
	}
}

func assertPrivateTLSRejectsUnauthenticated(t *testing.T, caFile, target string) {
	t.Helper()
	roots := certificatePool(t, caFile)
	client := &http.Client{
		Transport: &http.Transport{TLSClientConfig: &tls.Config{
			MinVersion: tls.VersionTLS13, RootCAs: roots,
		}},
		Timeout: 3 * time.Second,
	}
	response, err := client.Get(target)
	if response != nil {
		response.Body.Close()
	}
	if err == nil {
		t.Fatalf("unauthenticated private TLS request unexpectedly reached HTTP: %s", target)
	}
}

func newMTLSClient(t *testing.T, caFile string, identity localpki.Paths) *http.Client {
	t.Helper()
	certificate, err := tls.LoadX509KeyPair(identity.Certificate, identity.PrivateKey)
	if err != nil {
		t.Fatal(err)
	}
	return &http.Client{
		Transport: &http.Transport{TLSClientConfig: &tls.Config{
			MinVersion: tls.VersionTLS13, RootCAs: certificatePool(t, caFile),
			Certificates: []tls.Certificate{certificate},
		}},
		Timeout: 5 * time.Second,
	}
}

func certificatePool(t *testing.T, caFile string) *x509.CertPool {
	t.Helper()
	data, err := os.ReadFile(caFile)
	if err != nil {
		t.Fatal(err)
	}
	pool := x509.NewCertPool()
	if !pool.AppendCertsFromPEM(data) {
		t.Fatal("test CA file contains no certificate")
	}
	return pool
}

func freeAddress(t *testing.T) string {
	t.Helper()
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	address := listener.Addr().String()
	if err := listener.Close(); err != nil {
		t.Fatal(err)
	}
	return address
}

func isolatedDatabase(t *testing.T, ctx context.Context, databaseURL string) string {
	t.Helper()
	parsed, err := url.Parse(databaseURL)
	if err != nil || parsed.Scheme != "postgres" && parsed.Scheme != "postgresql" {
		t.Fatalf("CONTRACTOR_TEST_DATABASE_URL must be a PostgreSQL URL")
	}
	admin, err := pgxpool.New(ctx, databaseURL)
	if err != nil {
		t.Fatalf("open PostgreSQL: %v", err)
	}
	if err := admin.Ping(ctx); err != nil {
		admin.Close()
		t.Fatalf("ping PostgreSQL: %v", err)
	}
	schema := "contractor_e2e_" + randomHex(t, 8)
	identifier := pgx.Identifier{schema}.Sanitize()
	if _, err := admin.Exec(ctx, "CREATE SCHEMA "+identifier); err != nil {
		admin.Close()
		t.Fatalf("create E2E schema: %v", err)
	}
	t.Cleanup(func() {
		cleanup, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		if _, err := admin.Exec(cleanup, "DROP SCHEMA "+identifier+" CASCADE"); err != nil {
			t.Logf("drop E2E schema: %v", err)
		}
		admin.Close()
	})
	query := parsed.Query()
	query.Set("search_path", schema)
	parsed.RawQuery = query.Encode()
	return parsed.String()
}

func randomHex(t *testing.T, size int) string {
	t.Helper()
	data := make([]byte, size)
	if _, err := cryptorand.Read(data); err != nil {
		t.Fatal(err)
	}
	return hex.EncodeToString(data)
}

func writeE2ELocalAuth(t *testing.T, root, userID string) string {
	t.Helper()
	hash, err := auth.HashPassword([]byte("contractor e2e local password"))
	if err != nil {
		t.Fatal(err)
	}
	document, err := auth.BootstrapYAML(userID, "admin", hash)
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(root, "local-auth-"+userID+".yaml")
	if err := os.WriteFile(path, document, 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}

func repoRoot(t *testing.T) string {
	t.Helper()
	_, source, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("locate E2E source")
	}
	root, err := filepath.Abs(filepath.Join(filepath.Dir(source), "..", ".."))
	if err != nil {
		t.Fatal(err)
	}
	return root
}
