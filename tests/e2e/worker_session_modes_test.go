//go:build e2e

package e2e

import (
	"context"
	"encoding/json"
	"fmt"
	"net"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/localpki"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/stageresult"
	"github.com/grauwolf32/contractor/internal/telemetry"
	"github.com/jackc/pgx/v5/pgxpool"
)

func TestWorkerSessionModesAcrossProductionProcesses(t *testing.T) {
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
		LeafOptions: leaf, URI: "urn:contractor:control-plane:worker-session-e2e",
	})
	if err != nil {
		t.Fatalf("issue Control Plane certificate: %v", err)
	}
	agentPaths := make([]localpki.Paths, 2)
	for index := range agentPaths {
		agentPaths[index], err = generator.IssueAgent(
			pkiRoot, fmt.Sprintf("worker-session-e2e-agent-%d", index+1), leaf,
		)
		if err != nil {
			t.Fatalf("issue Runtime Agent %d certificate: %v", index+1, err)
		}
	}

	gateway := newWorkerSessionGateway(llmGatewayToken)
	t.Cleanup(gateway.close)
	configRoot := stageE2EConfiguration(
		t, filepath.Join(repositoryRoot, "configs", "e2e"),
		filepath.Join(temporaryRoot, "configs"), gateway.URL(),
	)
	stageWorkerSessionADKFixtures(t, repositoryRoot, configRoot)
	publicAddress := freeAddress(t)
	privateAddress := freeAddress(t)
	publicBaseURL := "http://" + publicAddress
	privateBaseURL := "https://" + privateAddress
	userID := "worker-session-e2e-user-" + randomHex(t, 8)
	localAuthFile := writeE2ELocalAuth(t, temporaryRoot, userID)
	server := startProcess(t, "Go Server", repositoryRoot, map[string]string{
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
	controlClient := newMTLSClient(t, caPaths.Certificate, controlPlanePaths)
	runtimes := make([]*childProcess, 0, 2)
	runtimeURLs := make([]string, 0, 2)
	workRoots := make([]string, 0, 2)
	for index := range agentPaths {
		runtimeAddress := freeAddress(t)
		runtimeBaseURL := "https://" + runtimeAddress
		workRoot := filepath.Join(temporaryRoot, fmt.Sprintf("runtime-work-%d", index+1))
		process := startProcess(
			t, fmt.Sprintf("Python Runtime Agent %d", index+1), filepath.Join(repositoryRoot, "runtime"),
			map[string]string{"PYTHONUNBUFFERED": "1"},
			python, "-m", "contractor_runtime",
			"--control-plane-url", privateBaseURL,
			"--advertised-control-url", runtimeBaseURL,
			"--advertised-a2a-url", runtimeBaseURL,
			"--ca-file", caPaths.Certificate,
			"--certificate-file", agentPaths[index].Certificate,
			"--private-key-file", agentPaths[index].PrivateKey,
			"--listen", runtimeAddress,
			"--work-root", workRoot,
			"--request-timeout-seconds", "10",
			"--shutdown-grace-seconds", "5",
		)
		waitForHTTP(t, ctx, process, controlClient, runtimeBaseURL+"/healthz", http.StatusOK)
		waitForProcessLog(t, ctx, process, "runtime agent registered")
		runtimes = append(runtimes, process)
		runtimeURLs = append(runtimeURLs, runtimeBaseURL)
		workRoots = append(workRoots, workRoot)
	}
	waitForObservedRuntimeAgents(
		t, ctx, server, runtimes, publicClient, publicBaseURL,
		func(items []observedRuntimeAgent) bool { return len(items) == 2 },
	)

	pool, err := pgxpool.New(ctx, isolateURL)
	if err != nil {
		t.Fatalf("open assertion database: %v", err)
	}
	t.Cleanup(pool.Close)
	store := runstore.NewPostgresStore(pool)
	telemetryReports := telemetry.NewRepository(pool)

	streamlineRunID := createEmptyWorkflowRun(
		t, publicClient, publicBaseURL, "worker-session-streamline@1", "worker-session-streamline",
	)
	waitForWorkerSessionRun(
		t, ctx, server, runtimes, gateway, publicClient, publicBaseURL, streamlineRunID,
	)
	assertWorkerSessionExecutions(
		t, ctx, store, telemetryReports, streamlineRunID,
		map[string]workerSessionStageExpectation{
			"isolated": {mode: contracts.WorkerSessionIsolated, workers: []string{"builder"}},
			"after":    {mode: contracts.WorkerSessionIsolated, workers: []string{"builder"}},
		},
	)

	routerRunID := createEmptyWorkflowRun(
		t, publicClient, publicBaseURL, "worker-session-router@1", "worker-session-router",
	)
	waitForWorkerSessionRun(
		t, ctx, server, runtimes, gateway, publicClient, publicBaseURL, routerRunID,
	)
	routerAllocations := assertWorkerSessionExecutions(
		t, ctx, store, telemetryReports, routerRunID,
		map[string]workerSessionStageExpectation{
			"shared": {mode: contracts.WorkerSessionShared, workers: []string{"builder", "reviewer"}},
			"after":  {mode: contracts.WorkerSessionIsolated, workers: []string{"builder"}},
		},
	)
	sharedInstances := map[string]bool{}
	for _, allocation := range routerAllocations["shared"] {
		sharedInstances[allocation.RuntimeAgentInstanceID] = true
	}
	for _, allocation := range routerAllocations["after"] {
		if !sharedInstances[allocation.RuntimeAgentInstanceID] {
			t.Fatalf("Router follow-up Stage used a non-reused Runtime slot: %+v", allocation)
		}
	}

	runWorkerSessionADKProbes(t, workerSessionProbeHarness{
		ctx: ctx, server: server, runtimes: runtimes, gateway: gateway,
		publicClient: publicClient, controlClient: controlClient, publicURL: publicBaseURL,
		runtimeURLs: runtimeURLs, workRoots: workRoots, pool: pool, store: store,
		knownInstances: sharedInstances,
	})
	gateway.assertComplete(t)
	if failures := gateway.Failures(); len(failures) != 0 {
		t.Fatalf("worker-session Gateway failures: %v", failures)
	}
	for index := range runtimes {
		waitForRuntimeIdle(t, ctx, runtimes[index], controlClient, runtimeURLs[index], workRoots[index])
	}
	for _, secret := range []string{publicToken, llmGatewayToken} {
		if strings.Contains(server.logs.redacted(), secret) {
			t.Fatal("Server logs contain a configured secret")
		}
		for _, process := range runtimes {
			if strings.Contains(process.logs.redacted(), secret) {
				t.Fatalf("%s logs contain a configured secret", process.name)
			}
		}
	}
}

type workerSessionStageExpectation struct {
	mode    contracts.WorkerSessionMode
	workers []string
}

func assertWorkerSessionExecutions(
	t *testing.T,
	ctx context.Context,
	store *runstore.PostgresStore,
	telemetryReports *telemetry.Repository,
	runID string,
	expected map[string]workerSessionStageExpectation,
) map[string][]runstore.StageAllocation {
	t.Helper()
	executions := requireExecutions(t, ctx, store, runID, len(expected))
	allocations := make(map[string][]runstore.StageAllocation, len(expected))
	for _, execution := range executions {
		want, ok := expected[execution.StageName]
		if !ok {
			t.Fatalf("Run %s has unexpected StageExecution %+v", runID, execution)
		}
		assertSucceededExecution(t, execution, 1)
		stage, err := resolved.DecodeResolvedStageSnapshot(execution.StageSpecSnapshot)
		if err != nil {
			t.Fatalf("decode Stage %s snapshot: %v", execution.StageName, err)
		}
		if stage.Session != want.mode {
			t.Fatalf("Stage %s persisted session mode = %q, want %q", execution.StageName, stage.Session, want.mode)
		}
		allocations[execution.StageName] = assertCompleteAllocations(
			t, ctx, store, telemetryReports, execution, len(want.workers), want.workers...,
		)
	}
	return allocations
}

func waitForWorkerSessionRun(
	t *testing.T,
	ctx context.Context,
	server *childProcess,
	runtimes []*childProcess,
	gateway *workerSessionGateway,
	client *http.Client,
	baseURL, runID string,
) runStatus {
	t.Helper()
	ticker := time.NewTicker(100 * time.Millisecond)
	defer ticker.Stop()
	for {
		request, _ := http.NewRequestWithContext(
			ctx, http.MethodGet, baseURL+"/v1/runs/"+url.PathEscape(runID), nil,
		)
		request.Header.Set("Authorization", "Bearer "+publicToken)
		response, err := client.Do(request)
		if err == nil {
			var status runStatus
			if response.StatusCode == http.StatusOK {
				decodeResponse(t, response, &status)
			}
			response.Body.Close()
			if status.State == "succeeded" {
				return status
			}
			if status.State == "failed" || status.State == "cancelled" {
				t.Fatalf("Run %s reached %s: %+v\n%s", runID, status.State, status,
					workerSessionDiagnostics(server, runtimes, gateway))
			}
		}
		for _, process := range append([]*childProcess{server}, runtimes...) {
			if exited, processErr := process.exited(); exited {
				t.Fatalf("%s exited while Run was active: %v\n%s", process.name, processErr,
					workerSessionDiagnostics(server, runtimes, gateway))
			}
		}
		if failures := gateway.Failures(); len(failures) != 0 {
			t.Fatalf("worker-session Gateway failures: %v\n%s", failures,
				workerSessionDiagnostics(server, runtimes, gateway))
		}
		select {
		case <-ctx.Done():
			t.Fatalf("wait for Run %s: %v\n%s", runID, ctx.Err(),
				workerSessionDiagnostics(server, runtimes, gateway))
		case <-ticker.C:
		}
	}
}

func workerSessionDiagnostics(
	server *childProcess, runtimes []*childProcess, gateway *workerSessionGateway,
) string {
	var result strings.Builder
	result.WriteString("server:\n")
	result.WriteString(server.logs.redacted(publicToken, llmGatewayToken))
	for _, process := range runtimes {
		result.WriteString("\n" + process.name + ":\n")
		result.WriteString(process.logs.redacted(publicToken, llmGatewayToken))
	}
	result.WriteString(fmt.Sprintf("\ngateway observations=%v failures=%v",
		gateway.Observations(), gateway.Failures()))
	return result.String()
}

func stageWorkerSessionADKFixtures(t *testing.T, repositoryRoot, target string) {
	t.Helper()
	for _, relative := range []string{
		"agent-templates/session_tools_worker.yaml", "model-policies/session_tools_worker.yaml",
		"instructions/session-tools-worker.md", "workflows/worker_session_adk.yaml",
	} {
		data, err := os.ReadFile(filepath.Join(repositoryRoot, "tests/e2e/testdata/adk213", relative))
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(target, relative), data, 0600); err != nil {
			t.Fatal(err)
		}
		if relative == "workflows/worker_session_adk.yaml" {
			shared := strings.ReplaceAll(string(data), "worker-session-adk-isolated", "worker-session-adk-shared")
			shared = strings.Replace(shared, "session: isolated", "session: shared", 1)
			if err := os.WriteFile(filepath.Join(target, "workflows/worker_session_adk_shared.yaml"), []byte(shared), 0600); err != nil {
				t.Fatal(err)
			}
		}
	}
}

type workerSessionProbeHarness struct {
	ctx                         context.Context
	server                      *childProcess
	runtimes                    []*childProcess
	gateway                     *workerSessionGateway
	publicClient, controlClient *http.Client
	publicURL                   string
	runtimeURLs, workRoots      []string
	pool                        *pgxpool.Pool
	store                       *runstore.PostgresStore
	knownInstances              map[string]bool
}

type workerSessionProbeExpectation struct {
	kind, failure, tool        string
	modelCalls, spentResponses int64
	toolFailed                 bool
	cancel                     bool
}

func runWorkerSessionADKProbes(t *testing.T, h workerSessionProbeHarness) {
	t.Helper()
	input := uploadInput(t, h.publicClient, h.publicURL)
	seenAllocations := map[string]bool{}
	for _, session := range []string{"isolated", "shared"} {
		cases := []workerSessionProbeExpectation{
			{kind: "artifact-pair", tool: "write_text_artifact", modelCalls: 3, spentResponses: 3},
			{kind: "unknown", tool: "unknown_tool", toolFailed: true, modelCalls: 3, spentResponses: 3},
			{kind: "raw-arguments", tool: "read_artifact", toolFailed: true, modelCalls: 5, spentResponses: 5},
		}
		if session == "isolated" {
			cases = append(cases,
				workerSessionProbeExpectation{kind: "main-empty", failure: "worker_gateway_unavailable", modelCalls: 1, spentResponses: 1},
				workerSessionProbeExpectation{kind: "main-thought", failure: "worker_result_missing", modelCalls: 1, spentResponses: 1},
				workerSessionProbeExpectation{kind: "main-limit", failure: "worker_output_limit_exceeded", modelCalls: 1, spentResponses: 1},
				workerSessionProbeExpectation{kind: "finalizer-empty", failure: "worker_gateway_unavailable", modelCalls: 2, spentResponses: 2},
				workerSessionProbeExpectation{kind: "finalizer-thought", failure: "worker_result_missing", modelCalls: 2, spentResponses: 2},
				workerSessionProbeExpectation{kind: "finalizer-limit", failure: "worker_output_limit_exceeded", modelCalls: 2, spentResponses: 2},
				workerSessionProbeExpectation{kind: "main-cancel", cancel: true, modelCalls: 1},
				workerSessionProbeExpectation{kind: "reuse-after-main-cancel", modelCalls: 2, spentResponses: 2},
				workerSessionProbeExpectation{kind: "finalizer-cancel", cancel: true, modelCalls: 2, spentResponses: 1},
				workerSessionProbeExpectation{kind: "reuse-after-finalizer-cancel", modelCalls: 2, spentResponses: 2},
			)
		}
		for _, want := range cases {
			if !t.Run("adk213/"+session+"/"+want.kind, func(t *testing.T) {
				mode := "adk213-" + session + "-" + want.kind
				if want.cancel {
					h.gateway.armADKProbe(mode)
				}
				runID := createWorkflowRunWithParameters(t, h.publicClient, h.publicURL,
					"worker-session-adk-"+session+"@1", mode, input, map[string]string{"mode": mode})
				state := "succeeded"
				if want.failure != "" {
					state = "failed"
				}
				if want.cancel {
					if err := h.gateway.waitADKProbe(h.ctx, mode, false); err != nil {
						t.Fatalf("provider admission: %v", err)
					}
					cancelWorkflowRun(t, h.publicClient, h.publicURL, runID)
					if err := h.gateway.waitADKProbe(h.ctx, mode, true); err != nil {
						t.Fatalf("provider cancellation join: %v", err)
					}
					state = "cancelled"
				}
				status := waitForRunStateAcross(t, h.ctx, h.server, h.runtimes, h.publicClient, h.publicURL, runID, state)
				if failures := h.gateway.Failures(); len(failures) != 0 {
					t.Fatalf("Gateway fixture failures: %v", failures)
				}
				if len(status.Attempts) != 1 || status.Attempts[0].State != state {
					t.Fatalf("authoritative attempt = %+v", status.Attempts)
				}
				attempt := status.Attempts[0]
				if !want.cancel {
					var result stageresult.StageContentResult
					if err := json.Unmarshal(attempt.Result, &result); err != nil {
						t.Fatal(err)
					}
					if want.failure == "" {
						if result.Outcome != stageresult.StageSucceeded || result.Error != nil || result.Summary != adkProbeResultPrefix+mode {
							t.Fatalf("successful result = %+v", result)
						}
					} else if result.Outcome != stageresult.StageFailed || result.Error == nil || result.Error.Code != want.failure || !result.Error.Retryable {
						t.Fatalf("failed result = %+v, want %s", result, want.failure)
					}
				}
				report := loadSummarizerProcessReport(t, h.ctx, h.pool, runID)
				metrics := report.Report.Worker.Metrics
				if !report.Report.Worker.Complete || !report.Report.Runtime.Complete || metrics.ModelCalls == nil || *metrics.ModelCalls != want.modelCalls || metrics.WorkerBudget == nil || metrics.WorkerBudget.ObservedModelCalls != want.modelCalls {
					t.Fatalf("trusted reports incomplete or wrong attempt counts: %+v", report)
				}
				tokens := int64(0)
				if metrics.TotalTokens != nil {
					tokens = *metrics.TotalTokens
				}
				if tokens != want.spentResponses*13 || metrics.WorkerBudget.ObservedTotalTokens != tokens {
					t.Fatalf("spent usage = %+v, want %d", metrics, want.spentResponses*13)
				}
				if attempt.Metrics == nil || !attempt.Metrics.ReportsComplete || attempt.Metrics.ModelCalls != want.modelCalls || attempt.Metrics.TotalTokens != tokens {
					t.Fatalf("public metrics differ from persisted report: %+v", attempt.Metrics)
				}
				wantTools := int64(1)
				if want.kind == "raw-arguments" {
					wantTools = 3
				}
				if want.tool != "" {
					tool := metrics.Tools[want.tool]
					if tool.Calls == nil || *tool.Calls != wantTools || metrics.WorkerBudget.ObservedToolCalls != wantTools || len(report.Report.Worker.ToolCalls) != int(wantTools) {
						t.Fatalf("tool accounting = %+v / %+v", metrics, report.Report.Worker.ToolCalls)
					}
					if want.toolFailed && (tool.Failed == nil || *tool.Failed != wantTools) {
						t.Fatalf("rejection counted as success: %+v", tool)
					}
					if !want.toolFailed && (tool.Succeeded == nil || *tool.Succeeded != wantTools) {
						t.Fatalf("completed tool lost: %+v", tool)
					}
				} else if metrics.WorkerBudget.ObservedToolCalls != 0 || len(report.Report.Worker.ToolCalls) != 0 {
					t.Fatal("terminal scenario dispatched a tool")
				}
				encoded, _ := json.Marshal(report)
				for _, forbidden := range []string{llmGatewayToken, publicToken, adkProbeUnknownName, "ADK213 private thought only"} {
					if strings.Contains(string(encoded), forbidden) {
						t.Fatal("private/provider content reached the stored report")
					}
				}
				executions := requireExecutions(t, h.ctx, h.store, runID, 1)
				allocations, err := h.store.ListStageAllocations(h.ctx, executions[0].StageExecutionID)
				if err != nil || len(allocations) != 1 {
					t.Fatalf("allocations = (%+v,%v)", allocations, err)
				}
				allocation := allocations[0]
				if seenAllocations[allocation.AllocationID] || !h.knownInstances[allocation.RuntimeAgentInstanceID] {
					t.Fatalf("slot reuse or release identity invalid: %+v", allocation)
				}
				seenAllocations[allocation.AllocationID] = true
				stage, err := resolved.DecodeResolvedStageSnapshot(executions[0].StageSpecSnapshot)
				if err != nil {
					t.Fatal(err)
				}
				if stage.Session != contracts.WorkerSessionMode(session) || allocation.AgentTemplateRef != stage.Agents["builder"].Template.Ref || allocation.RuntimeConfiguration == nil || allocation.RuntimeConfiguration.ModelPolicy != stage.Agents["builder"].Template.ModelPolicy.Ref {
					t.Fatal("resolved template/model/session pin changed between snapshot and allocation")
				}
				for index := range h.runtimes {
					waitForRuntimeReleased(t, h.ctx, h.runtimes[index], h.controlClient, h.runtimeURLs[index], allocation.AllocationID, h.workRoots[index])
				}
				allocations, err = h.store.ListStageAllocations(h.ctx, executions[0].StageExecutionID)
				if err != nil || len(allocations) != 1 || allocations[0].ReleaseCompletedAt == nil {
					t.Fatalf("release is not durably acknowledged: (%+v,%v)", allocations, err)
				}
				normal, finalizer, responses := h.gateway.adkProbeCounts(mode)
				if int64(normal+finalizer) != want.modelCalls || (want.tool != "" && int64(responses) != wantTools) {
					t.Fatalf("provider/receipt counts = %d/%d/%d", normal, finalizer, responses)
				}
				if want.kind == "artifact-pair" {
					ref, ok := h.gateway.adkProbeReceipt(mode)
					if !ok || ref.Revision == nil {
						t.Fatal("completed tool receipt is absent")
					}
					// Read the exact tool-returned revision after release; a latest
					// binding read could hide a corrupted or replaced receipt.
					body, mediaType := download(t, h.publicClient, h.publicURL+"/v1/runs/"+runID+"/artifacts/builder/spine?revision="+url.QueryEscape(*ref.Revision))
					if string(body) != adkProbeArtifactBody || mediaType != "text/plain" {
						t.Fatal("completed tool artifact was not durably readable after release")
					}
				}
			}) {
				t.FailNow()
			}
		}
	}
}
