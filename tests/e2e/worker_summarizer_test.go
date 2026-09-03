//go:build e2e

package e2e

import (
	"bytes"
	"context"
	"encoding/json"
	"net"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/localpki"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/jackc/pgx/v5/pgxpool"
)

type summarizerProcessExpectation struct {
	workflow              string
	mode                  string
	state                 string
	summarized            bool
	normalCalls           int
	normalTokens          int64
	tokenUsageUnavailable int64
	summaryFailure        string
}

func TestWorkerSummarizerProductionBoundaries(t *testing.T) {
	if testing.Short() {
		t.Skip("end-to-end process test")
	}
	databaseURL := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if databaseURL == "" {
		t.Fatal("CONTRACTOR_TEST_DATABASE_URL is required")
	}
	repositoryRoot := repoRoot(t)
	temporaryRoot := t.TempDir()
	ctx, cancel := context.WithTimeout(context.Background(), 420*time.Second)
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
		LeafOptions: leaf, URI: "urn:contractor:control-plane:worker-summarizer-e2e",
	})
	if err != nil {
		t.Fatalf("issue Control Plane certificate: %v", err)
	}
	agentPaths, err := generator.IssueAgent(pkiRoot, "worker-summarizer-e2e-agent", leaf)
	if err != nil {
		t.Fatalf("issue Runtime Agent certificate: %v", err)
	}

	gateway := newSummarizerGateway(llmGatewayToken)
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
	userID := "summarizer-e2e-user-" + randomHex(t, 8)
	localAuthFile := writeE2ELocalAuth(t, temporaryRoot, userID)
	server := startProcess(t, "Worker summarizer Go Server", repositoryRoot, map[string]string{
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
	python := filepath.Join(repositoryRoot, "runtime", ".venv", "bin", "python")
	if info, statErr := os.Stat(python); statErr != nil || info.IsDir() {
		t.Fatalf("Python Runtime environment is missing at %s; run 'cd runtime && uv sync --locked'", python)
	}
	workRoot := filepath.Join(temporaryRoot, "runtime-work")
	runtimeProcess := startProcess(
		t, "Worker summarizer Python Runtime", filepath.Join(repositoryRoot, "runtime"),
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

	pool, err := pgxpool.New(ctx, isolateURL)
	if err != nil {
		t.Fatalf("open assertion database: %v", err)
	}
	t.Cleanup(pool.Close)
	input := uploadInput(t, publicClient, publicBaseURL)
	runIDs := make([]string, 0, 10)

	for _, expectation := range []summarizerProcessExpectation{
		{workflow: "worker-summarizer@1", mode: "cumulative", state: "succeeded", summarized: true, normalCalls: 1, normalTokens: 20_000},
		{workflow: "worker-summarizer-context@1", mode: "context-window", state: "succeeded", summarized: true, normalCalls: 1, normalTokens: 7171},
		{workflow: "worker-summarizer@1", mode: "normal-final", state: "succeeded", normalCalls: 2, normalTokens: 20_000},
		// LiteLLM normalizes an omitted OpenAI usage member to zero-valued
		// metadata. Runtime recognizes that representation as unavailable and
		// never guesses it into either soft rule.
		{workflow: "worker-summarizer@1", mode: "missing-usage", state: "succeeded", normalCalls: 2, normalTokens: 10, tokenUsageUnavailable: 1},
		{workflow: "worker-summarizer-disabled@1", mode: "disabled", state: "succeeded", normalCalls: 2, normalTokens: 25_010},
	} {
		runID, allocationID := runSummarizerProcessScenario(
			t, ctx, server, runtimeProcess, gateway, publicClient, controlClient,
			pool, publicBaseURL, runtimeBaseURL, workRoot, input, expectation,
		)
		runIDs = append(runIDs, runID)
		waitForRuntimeReleased(
			t, ctx, runtimeProcess, controlClient, runtimeBaseURL, allocationID, workRoot,
		)
	}

	for _, expectation := range []summarizerProcessExpectation{
		{workflow: "worker-summarizer@1", mode: "invalid-summary", state: "failed", summarized: true, normalCalls: 1, normalTokens: 20_000, summaryFailure: "result_invalid"},
		{workflow: "worker-summarizer@1", mode: "provider-timeout", state: "failed", summarized: true, normalCalls: 1, normalTokens: 20_000, summaryFailure: "gateway_unavailable"},
		{workflow: "worker-summarizer@1", mode: "reuse-after-failure", state: "succeeded", normalCalls: 2, normalTokens: 20_000},
	} {
		runID, allocationID := runSummarizerProcessScenario(
			t, ctx, server, runtimeProcess, gateway, publicClient, controlClient,
			pool, publicBaseURL, runtimeBaseURL, workRoot, input, expectation,
		)
		runIDs = append(runIDs, runID)
		waitForRuntimeReleased(
			t, ctx, runtimeProcess, controlClient, runtimeBaseURL, allocationID, workRoot,
		)
	}

	cancelRunID := createWorkflowRunWithParameters(
		t, publicClient, publicBaseURL, "worker-summarizer@1",
		"worker-summarizer-cancel", input, map[string]string{"mode": "cancel-summary"},
	)
	runIDs = append(runIDs, cancelRunID)
	if err := gateway.WaitForSummary(ctx, "cancel-summary"); err != nil {
		t.Fatalf("wait for active terminal summarizer: %v", err)
	}
	cancelAllocationID := waitForActiveAllocationID(t, ctx, pool, cancelRunID)
	assertRequestedSummarizerState(
		t, ctx, controlClient, runtimeBaseURL, cancelAllocationID,
	)
	cancelWorkflowRun(t, publicClient, publicBaseURL, cancelRunID)
	cancelled := waitForRunState(
		t, ctx, server, runtimeProcess, gateway, publicClient, publicBaseURL,
		cancelRunID, "cancelled",
	)
	if len(cancelled.Attempts) != 1 || cancelled.Attempts[0].State != "cancelled" {
		t.Fatalf("cancelled summarizer attempt = %+v", cancelled.Attempts)
	}
	cancelReport := loadSummarizerProcessReport(t, ctx, pool, cancelRunID)
	assertSummarizerReport(t, cancelReport.Report.Worker, summarizerProcessExpectation{
		mode: "cancel-summary", summarized: true, normalCalls: 1,
		normalTokens: 20_000, summaryFailure: "cancelled",
	})
	if normal, summary := gateway.Counts("cancel-summary"); normal != 1 || summary != 1 {
		t.Fatalf("cancel-summary Gateway calls = %d/%d, want 1/1", normal, summary)
	}
	waitForRuntimeReleased(
		t, ctx, runtimeProcess, controlClient, runtimeBaseURL, cancelAllocationID, workRoot,
	)

	reuseRunID, reuseAllocationID := runSummarizerProcessScenario(
		t, ctx, server, runtimeProcess, gateway, publicClient, controlClient,
		pool, publicBaseURL, runtimeBaseURL, workRoot, input,
		summarizerProcessExpectation{
			workflow: "worker-summarizer@1", mode: "reuse-after-cancel",
			state: "succeeded", normalCalls: 2, normalTokens: 20_000,
		},
	)
	runIDs = append(runIDs, reuseRunID)
	waitForRuntimeReleased(
		t, ctx, runtimeProcess, controlClient, runtimeBaseURL, reuseAllocationID, workRoot,
	)

	assertSummarizerGatewayCaptures(t, gateway)
	assertSummarizerRetentionSafe(t, ctx, pool, runIDs, server, runtimeProcess, gateway)
	if failures := gateway.Failures(); len(failures) != 0 {
		t.Fatalf("summarizer Gateway fixture failures: %v", failures)
	}
}

func runSummarizerProcessScenario(
	t *testing.T,
	ctx context.Context,
	server, runtimeProcess *childProcess,
	gateway *summarizerGateway,
	publicClient, controlClient *http.Client,
	pool *pgxpool.Pool,
	publicBaseURL, runtimeBaseURL, workRoot string,
	input artifactRef,
	expectation summarizerProcessExpectation,
) (string, string) {
	t.Helper()
	runID := createWorkflowRunWithParameters(
		t, publicClient, publicBaseURL, expectation.workflow,
		"worker-summarizer-"+expectation.mode, input,
		map[string]string{"mode": expectation.mode},
	)
	status := waitForRunState(
		t, ctx, server, runtimeProcess, gateway, publicClient, publicBaseURL,
		runID, expectation.state,
	)
	if len(status.Attempts) != 1 || status.Attempts[0].State != expectation.state {
		t.Fatalf("%s attempt = %+v", expectation.mode, status.Attempts)
	}
	attempt := status.Attempts[0]
	if attempt.Metrics == nil || !attempt.Metrics.ReportsComplete ||
		attempt.Metrics.ModelCalls != int64(expectation.normalCalls) ||
		attempt.Metrics.TotalTokens != expectation.normalTokens {
		t.Fatalf("%s public normal-loop metrics = %+v", expectation.mode, attempt.Metrics)
	}
	var stageResult contracts.StageContentResult
	if err := json.Unmarshal(attempt.Result, &stageResult); err != nil {
		t.Fatalf("%s Stage result: %v (%s)", expectation.mode, err, attempt.Result)
	}
	if expectation.state == "succeeded" {
		if stageResult.Outcome != contracts.StageSucceeded || stageResult.Error != nil {
			t.Fatalf("%s successful Stage result = %+v", expectation.mode, stageResult)
		}
		output, ok := status.Outputs["result"]
		selected := stageResult.Artifacts["result"]
		if !ok || output.Revision == nil || selected.Namespace != "builder" ||
			selected.Name != "result" || selected.Revision == nil {
			t.Fatalf("%s exact output/result = %+v / %+v", expectation.mode, status.Outputs, stageResult)
		}
		body, mediaType := download(
			t, publicClient, publicBaseURL+"/v1/runs/"+url.PathEscape(runID)+"/outputs/result",
		)
		want := "artifact produced for " + expectation.mode
		if string(body) != want || mediaType != "text/plain" {
			t.Fatalf("%s output = (%q, %q), want (%q, text/plain)", expectation.mode, body, mediaType, want)
		}
	} else {
		if stageResult.Outcome != contracts.StageFailed || stageResult.Error == nil ||
			stageResult.Error.Code != "worker_summarization_failed" || !stageResult.Error.Retryable {
			t.Fatalf("%s failure Stage result = %+v", expectation.mode, stageResult)
		}
	}
	report := loadSummarizerProcessReport(t, ctx, pool, runID)
	assertSummarizerReport(t, report.Report.Worker, expectation)
	if !report.Report.Runtime.Complete {
		t.Fatalf("%s Runtime report is incomplete: %+v", expectation.mode, report.Report.Runtime)
	}
	wantSummaryCalls := 0
	if expectation.summarized {
		wantSummaryCalls = 1
	}
	if normal, summary := gateway.Counts(expectation.mode); normal != expectation.normalCalls ||
		summary != wantSummaryCalls {
		t.Fatalf("%s Gateway calls = %d/%d, want %d/%d", expectation.mode,
			normal, summary, expectation.normalCalls, wantSummaryCalls)
	}
	return runID, report.AllocationID
}

func loadSummarizerProcessReport(
	t *testing.T,
	ctx context.Context,
	pool *pgxpool.Pool,
	runID string,
) runstore.StageExecutionReport {
	t.Helper()
	store := runstore.NewPostgresStore(pool)
	executions, err := store.ListStageExecutions(ctx, runID)
	if err != nil || len(executions) != 1 {
		t.Fatalf("%s StageExecutions = (%+v, %v), want one", runID, executions, err)
	}
	reports, err := store.ListStageExecutionReports(ctx, executions[0].StageExecutionID)
	if err != nil || len(reports) != 1 {
		t.Fatalf("%s execution reports = (%+v, %v), want one", runID, reports, err)
	}
	return reports[0]
}

func assertSummarizerReport(
	t *testing.T,
	report contracts.ExecutionReport,
	expectation summarizerProcessExpectation,
) {
	t.Helper()
	metrics := report.Metrics
	if !report.Complete || metrics.ModelCalls == nil ||
		*metrics.ModelCalls != int64(expectation.normalCalls) ||
		metrics.TotalTokens == nil || *metrics.TotalTokens != expectation.normalTokens ||
		metrics.WorkerBudget == nil ||
		metrics.WorkerBudget.ObservedModelCalls != int64(expectation.normalCalls) ||
		metrics.WorkerBudget.ObservedToolCalls != 1 ||
		metrics.WorkerBudget.ObservedTotalTokens != expectation.normalTokens ||
		metrics.WorkerBudget.TokenUsageUnavailable != expectation.tokenUsageUnavailable {
		t.Fatalf("%s trusted Worker metrics = %+v; modelCalls=%v totalTokens=%v budget=%+v",
			expectation.mode, metrics, optionalInt64(metrics.ModelCalls),
			optionalInt64(metrics.TotalTokens), metrics.WorkerBudget)
	}
	tool := metrics.Tools["write_text_artifact"]
	if tool.Calls == nil || *tool.Calls != 1 || tool.Succeeded == nil || *tool.Succeeded != 1 {
		t.Fatalf("%s write_text_artifact metrics = %+v", expectation.mode, tool)
	}
	if !expectation.summarized {
		if metrics.Summarizer != nil {
			t.Fatalf("%s unexpectedly retained summarizer metrics: %+v", expectation.mode, metrics.Summarizer)
		}
		return
	}
	summary := metrics.Summarizer
	if summary == nil || summary.Attempts != 1 || summary.ModelCalls != 1 {
		t.Fatalf("%s summarizer metrics = %+v", expectation.mode, summary)
	}
	if expectation.summaryFailure == "" {
		if summary.Succeeded != 1 || summary.Failed != 0 || summary.TotalTokens != 18 ||
			summary.TokenUsageUnavailable != 0 || len(summary.FailureCodes) != 0 {
			t.Fatalf("%s successful summarizer metrics = %+v", expectation.mode, summary)
		}
		return
	}
	if summary.Succeeded != 0 || summary.Failed != 1 ||
		summary.FailureCodes[expectation.summaryFailure] != 1 {
		t.Fatalf("%s failed summarizer metrics = %+v", expectation.mode, summary)
	}
	if expectation.summaryFailure == "gateway_unavailable" &&
		(summary.TotalTokens != 0 || summary.TokenUsageUnavailable != 1) {
		t.Fatalf("%s provider failure usage = %+v", expectation.mode, summary)
	}
}

func optionalInt64(value *int64) any {
	if value == nil {
		return nil
	}
	return *value
}

func waitForActiveAllocationID(
	t *testing.T,
	ctx context.Context,
	pool *pgxpool.Pool,
	runID string,
) string {
	t.Helper()
	ticker := time.NewTicker(50 * time.Millisecond)
	defer ticker.Stop()
	for {
		var allocationID string
		err := pool.QueryRow(ctx, `
SELECT allocation.allocation_id
FROM stage_allocations AS allocation
JOIN stage_executions AS execution
  ON execution.stage_execution_id = allocation.stage_execution_id
WHERE execution.run_id = $1
ORDER BY allocation.created_at DESC
LIMIT 1`, runID).Scan(&allocationID)
		if err == nil && allocationID != "" {
			return allocationID
		}
		select {
		case <-ctx.Done():
			t.Fatalf("wait for active allocation: %v", ctx.Err())
		case <-ticker.C:
		}
	}
}

func assertRequestedSummarizerState(
	t *testing.T,
	ctx context.Context,
	client *http.Client,
	runtimeBaseURL, allocationID string,
) {
	t.Helper()
	target := runtimeBaseURL + "/private/v1/allocations/" +
		url.PathEscape(allocationID) + "/agent-state"
	request, _ := http.NewRequestWithContext(ctx, http.MethodGet, target, nil)
	response, err := client.Do(request)
	if err != nil {
		t.Fatalf("read active summarizer State: %v", err)
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusOK {
		t.Fatalf("read active summarizer State = HTTP %d", response.StatusCode)
	}
	var snapshot contracts.AgentStateSnapshot
	body := copyBounded(response.Body)
	if err := json.Unmarshal([]byte(body), &snapshot); err != nil || snapshot.Validate() != nil {
		t.Fatalf("decode active summarizer State = (%v, %s)", err, body)
	}
	if snapshot.State.CurrentInvocation == nil ||
		snapshot.State.CurrentInvocation.Summarizer.Phase != "requested" ||
		snapshot.State.CurrentInvocation.Summarizer.RequestStateRevision == nil ||
		snapshot.State.CurrentInvocation.Summarizer.ModelCalls != 0 {
		t.Fatalf("active summarizer State = %+v", snapshot.State.CurrentInvocation)
	}
	for _, forbidden := range []string{llmGatewayToken, summarizerTranscriptCanary} {
		if strings.Contains(body, forbidden) {
			t.Fatalf("active Worker State retained forbidden summarizer data")
		}
	}
}

func cancelWorkflowRun(t *testing.T, client *http.Client, baseURL, runID string) {
	t.Helper()
	request, err := http.NewRequest(
		http.MethodPost, baseURL+"/v1/runs/"+url.PathEscape(runID)+"/cancel",
		strings.NewReader(`{"reason":"summarizer cancellation boundary"}`),
	)
	if err != nil {
		t.Fatal(err)
	}
	request.Header.Set("Authorization", "Bearer "+publicToken)
	request.Header.Set("Content-Type", "application/json")
	response := do(t, client, request, http.StatusAccepted)
	response.Body.Close()
}

func assertSummarizerGatewayCaptures(t *testing.T, gateway *summarizerGateway) {
	t.Helper()
	captures := gateway.Captures()
	summaryCount := 0
	for _, capture := range captures {
		if capture.Model != "worker-summarizer-model" {
			continue
		}
		summaryCount++
		if len(capture.Tools) != 0 || !capture.ResponseSchema ||
			capture.MaxOutputTokens != 2048 || !capture.TranscriptPresent ||
			!capture.SecretRedacted {
			t.Fatalf("unsafe or unpinned summarizer capture: %+v", capture)
		}
	}
	if summaryCount != 5 {
		t.Fatalf("terminal summarizer requests = %d, want 5", summaryCount)
	}
	encoded, err := json.Marshal(captures)
	if err != nil {
		t.Fatal(err)
	}
	for _, forbidden := range []string{llmGatewayToken, summarizerTranscriptCanary} {
		if bytes.Contains(encoded, []byte(forbidden)) {
			t.Fatalf("Gateway retained forbidden request material")
		}
	}
}

func assertSummarizerRetentionSafe(
	t *testing.T,
	ctx context.Context,
	pool *pgxpool.Pool,
	runIDs []string,
	server, runtimeProcess *childProcess,
	gateway *summarizerGateway,
) {
	t.Helper()
	queries := []string{
		`SELECT COALESCE(runtime_config_snapshot::text || state_reason_code || state_reason_message, '') FROM workflow_runs WHERE run_id = $1`,
		`SELECT COALESCE(string_agg(event::text, E'\n'), '') FROM planner_events WHERE run_id = $1`,
		`SELECT COALESCE(string_agg(data::text, E'\n'), '') FROM workflow_run_events WHERE run_id = $1`,
		`SELECT COALESCE(string_agg(session.state::text, E'\n'), '') FROM planner_sessions AS session JOIN stage_executions AS execution ON execution.stage_execution_id = session.stage_execution_id WHERE execution.run_id = $1`,
		`SELECT COALESCE(string_agg(report.report::text, E'\n'), '') FROM planner_execution_reports AS report JOIN stage_executions AS execution ON execution.stage_execution_id = report.stage_execution_id WHERE execution.run_id = $1`,
		`SELECT COALESCE(string_agg(report.report::text, E'\n'), '') FROM allocation_execution_reports AS report JOIN stage_executions AS execution ON execution.stage_execution_id = report.stage_execution_id WHERE execution.run_id = $1`,
		`SELECT COALESCE(string_agg(metrics.metrics::text || metrics.summary::text, E'\n'), '') FROM stage_metrics AS metrics JOIN stage_executions AS execution ON execution.stage_execution_id = metrics.stage_execution_id WHERE execution.run_id = $1`,
	}
	for _, runID := range runIDs {
		for _, query := range queries {
			var retained string
			if err := pool.QueryRow(ctx, query, runID).Scan(&retained); err != nil {
				t.Fatalf("read retained summarizer surface for %s: %v", runID, err)
			}
			for _, forbidden := range []string{
				publicToken, llmGatewayToken, summarizerTranscriptCanary,
				"Contractor terminal Worker summary input",
			} {
				if strings.Contains(retained, forbidden) {
					t.Fatalf("retained summarizer surface for %s exposed forbidden data", runID)
				}
			}
		}
	}
	for _, process := range []*childProcess{server, runtimeProcess} {
		logs := process.logs.redacted()
		for _, forbidden := range []string{publicToken, llmGatewayToken, summarizerTranscriptCanary} {
			if strings.Contains(logs, forbidden) {
				t.Fatalf("%s logs exposed summarizer credential/transcript data", process.name)
			}
		}
	}
	if failures := gateway.Failures(); len(failures) != 0 {
		t.Fatalf("Gateway fixture recorded failures before retention check: %v", failures)
	}
}
