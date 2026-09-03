//go:build e2e

package e2e

import (
	"archive/zip"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"

	workflowconfig "github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/localpki"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/jackc/pgx/v5/pgxpool"
)

func TestRoutingAndEscalationProductionBoundaries(t *testing.T) {
	if testing.Short() {
		t.Skip("end-to-end process test")
	}
	databaseURL := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if databaseURL == "" {
		t.Fatal("CONTRACTOR_TEST_DATABASE_URL is required")
	}
	repositoryRoot := repoRoot(t)
	temporaryRoot := t.TempDir()
	ctx, cancel := context.WithTimeout(context.Background(), 300*time.Second)
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
		LeafOptions: leaf, URI: "urn:contractor:control-plane:routing-e2e",
	})
	if err != nil {
		t.Fatalf("issue Control Plane certificate: %v", err)
	}
	agentPaths := make([]localpki.Paths, 2)
	for index := range agentPaths {
		agentPaths[index], err = generator.IssueAgent(
			pkiRoot, fmt.Sprintf("routing-e2e-agent-%d", index+1), leaf,
		)
		if err != nil {
			t.Fatalf("issue Runtime Agent %d certificate: %v", index+1, err)
		}
	}

	gateway := newRoutingGateway(llmGatewayToken)
	t.Cleanup(gateway.close)
	configRoot := stageE2EConfiguration(
		t, filepath.Join(repositoryRoot, "configs", "e2e"),
		filepath.Join(temporaryRoot, "configs"), gateway.URL(),
	)
	publicAddress := freeAddress(t)
	privateAddress := freeAddress(t)
	publicBaseURL := "http://" + publicAddress
	privateBaseURL := "https://" + privateAddress
	userID := "routing-e2e-user-" + randomHex(t, 8)
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

	python := filepath.Join(repositoryRoot, "runtime", ".venv", "bin", "python")
	if info, statErr := os.Stat(python); statErr != nil || info.IsDir() {
		t.Fatalf("Python Runtime environment is missing at %s; run 'cd runtime && uv sync --locked'", python)
	}
	controlClient := newMTLSClient(t, caPaths.Certificate, controlPlanePaths)
	runtimes := make([]*childProcess, 0, 2)
	runtimeURLs := make([]string, 0, 2)
	runtimeInstanceIDs := make([]string, 0, 2)
	knownRuntimeIDs := map[string]bool{}
	workRoots := make([]string, 0, 2)
	workspaceRoots := make([]string, 0, 2)
	for index := range agentPaths {
		runtimeAddress := freeAddress(t)
		runtimeBaseURL := "https://" + runtimeAddress
		workRoot := filepath.Join(temporaryRoot, fmt.Sprintf("runtime-work-%d", index+1))
		workspaceRoot := filepath.Join(temporaryRoot, fmt.Sprintf("runtime-workspace-%d", index+1))
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
			"--workspace-storage", "local",
			"--workspace-work-root", workspaceRoot,
			"--request-timeout-seconds", "10",
			"--shutdown-grace-seconds", "5",
			"--heartbeat-interval-seconds", "1",
			"--confirmed-lease-seconds", "12",
		)
		waitForHTTP(t, ctx, process, controlClient, runtimeBaseURL+"/healthz", http.StatusOK)
		waitForProcessLog(t, ctx, process, "runtime agent registered")
		runtimes = append(runtimes, process)
		runtimeURLs = append(runtimeURLs, runtimeBaseURL)
		workRoots = append(workRoots, workRoot)
		workspaceRoots = append(workspaceRoots, workspaceRoot)
		agents, _ := waitForObservedRuntimeAgents(
			t, ctx, server, runtimes, publicClient, publicBaseURL,
			func(items []observedRuntimeAgent) bool { return len(items) == index+1 },
		)
		instanceID := ""
		for _, agent := range agents {
			if !knownRuntimeIDs[agent.InstanceID] {
				instanceID = agent.InstanceID
				break
			}
		}
		if instanceID == "" {
			t.Fatalf("cannot identify newly registered Runtime %d: %+v", index+1, agents)
		}
		knownRuntimeIDs[instanceID] = true
		runtimeInstanceIDs = append(runtimeInstanceIDs, instanceID)
	}

	pool, err := pgxpool.New(ctx, isolateURL)
	if err != nil {
		t.Fatalf("open assertion database: %v", err)
	}
	t.Cleanup(pool.Close)
	store := runstore.NewPostgresStore(pool)
	uploaded := uploadInput(t, publicClient, publicBaseURL)

	streamlineRunID := createWorkflowRunWithParameters(
		t, publicClient, publicBaseURL, "streamline-copy@1", "routing-e2e-streamline",
		uploaded, map[string]string{"mode": "streamline-strict"},
	)
	streamline := waitForRoutingRun(
		t, ctx, server, runtimes, gateway, publicClient, publicBaseURL, streamlineRunID,
	)
	assertPublicAttemptMetrics(t, streamline, [][2]int64{{6, 5}})
	assertRoutingOutput(t, ctx, pool, publicClient, publicBaseURL, streamlineRunID, streamline)
	streamlineExecutions := requireExecutions(t, ctx, store, streamlineRunID, 1)
	assertSucceededExecution(t, streamlineExecutions[0], 1)
	assertCompleteAllocations(t, ctx, store, streamlineExecutions[0], 1, "builder")
	assertModeledPlanEvents(t, ctx, store, streamlineExecutions[0], "builder", nil, runtimeURLs)

	builderCallsBeforeRouter := gateway.ModelCalls("worker-model")
	routerRunID := createWorkflowRunWithParameters(
		t, publicClient, publicBaseURL, "router-review@1", "routing-e2e-router",
		uploaded, map[string]string{"mode": "router-strict"},
	)
	routerStatus := waitForRoutingRun(
		t, ctx, server, runtimes, gateway, publicClient, publicBaseURL, routerRunID,
	)
	assertPublicAttemptMetrics(t, routerStatus, [][2]int64{{6, 5}})
	assertRoutingOutput(t, ctx, pool, publicClient, publicBaseURL, routerRunID, routerStatus)
	routerExecutions := requireExecutions(t, ctx, store, routerRunID, 1)
	assertSucceededExecution(t, routerExecutions[0], 1)
	routerAllocations := assertCompleteAllocations(
		t, ctx, store, routerExecutions[0], 2, "builder", "reviewer",
	)
	if gateway.ModelCalls("worker-model") != builderCallsBeforeRouter ||
		gateway.ModelCalls("reviewer-worker-model") != 3 {
		t.Fatalf("Router invoked wrong model: builder=%d->%d reviewer=%d", builderCallsBeforeRouter,
			gateway.ModelCalls("worker-model"), gateway.ModelCalls("reviewer-worker-model"))
	}
	assertModeledPlanEvents(
		t, ctx, store, routerExecutions[0], "reviewer", routerAllocations, runtimeURLs,
	)

	escalationRunID := createWorkflowRunWithParameters(
		t, publicClient, publicBaseURL, "escalating-copy@1", "routing-e2e-escalation",
		uploaded, map[string]string{"mode": "escalation-strict"},
	)
	escalatedStatus := waitForRoutingRun(
		t, ctx, server, runtimes, gateway, publicClient, publicBaseURL, escalationRunID,
	)
	assertPublicAttemptMetrics(t, escalatedStatus, [][2]int64{{1, 1}, {6, 5}})
	assertRoutingOutput(t, ctx, pool, publicClient, publicBaseURL, escalationRunID, escalatedStatus)
	assertEscalatedExecution(t, ctx, store, escalationRunID)

	wantModelCalls := map[string]int{
		"planner-model": 7, "strong-planner-model": 3, "worker-model": 3,
		"reviewer-worker-model": 3, "strong-worker-model": 3,
	}
	for modelName, want := range wantModelCalls {
		if got := gateway.ModelCalls(modelName); got != want {
			t.Fatalf("Gateway calls for %s = %d, want %d; observations=%+v failures=%v",
				modelName, got, want, gateway.Observations(), gateway.Failures())
		}
	}
	runWorkerObservationsProcessGate(
		t, ctx, pool, store, gateway, server, runtimes, controlClient,
		publicClient, publicBaseURL, runtimeURLs, runtimeInstanceIDs, workRoots, workspaceRoots,
	)
	if failures := gateway.Failures(); len(failures) != 0 {
		t.Fatalf("routing Gateway validation failures: %v", failures)
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

func runWorkerObservationsProcessGate(
	t *testing.T,
	ctx context.Context,
	pool *pgxpool.Pool,
	store *runstore.PostgresStore,
	gateway *routingGateway,
	server *childProcess,
	runtimes []*childProcess,
	controlClient, publicClient *http.Client,
	publicBaseURL string,
	runtimeURLs, runtimeInstanceIDs, workRoots, workspaceRoots []string,
) {
	t.Helper()
	initialFixture := fixtureForObservationScenario("observation-router-initial-planner")
	initialArchive := workerObservationArchive(t, map[string]string{
		initialFixture.BuilderPath:  initialFixture.Before + "\n",
		initialFixture.ReviewerPath: "review the initial fixture\n",
		initialFixture.UnreadPath:   "intentionally unread\n",
	})
	initialSource := uploadProjectArtifact(
		t, publicClient, publicBaseURL, "worker-observations-initial", "application/zip", initialArchive,
	)

	streamlineRunID := createWorkflowRunWithParameters(
		t, publicClient, publicBaseURL, "worker-observations-streamline@1",
		"worker-observations-streamline", initialSource,
		map[string]string{"scenario": "observations-streamline"},
	)
	streamlineStatus := waitForRoutingRun(
		t, ctx, server, runtimes, gateway, publicClient, publicBaseURL, streamlineRunID,
	)
	streamlineAllocations := assertWorkerObservationExecution(
		t, ctx, store, gateway, publicClient, publicBaseURL, streamlineStatus,
		"observation-streamline-initial-planner", map[string]string{
			"report": initialFixture.BuilderReport,
		},
	)
	waitForObservationAllocationsReleased(
		t, ctx, streamlineAllocations, runtimes, controlClient,
		runtimeURLs, runtimeInstanceIDs, workRoots, workspaceRoots,
	)

	routerRunID := createWorkflowRunWithParameters(
		t, publicClient, publicBaseURL, "worker-observations-router@1",
		"worker-observations-router-initial", initialSource,
		map[string]string{"scenario": "observations-initial"},
	)
	routerStatus := waitForRoutingRun(
		t, ctx, server, runtimes, gateway, publicClient, publicBaseURL, routerRunID,
	)
	routerAllocations := assertWorkerObservationExecution(
		t, ctx, store, gateway, publicClient, publicBaseURL, routerStatus,
		"observation-router-initial-planner", map[string]string{
			"builder_report":  initialFixture.BuilderReport,
			"reviewer_report": initialFixture.ReviewerReport,
		},
	)
	waitForObservationAllocationsReleased(
		t, ctx, routerAllocations, runtimes, controlClient,
		runtimeURLs, runtimeInstanceIDs, workRoots, workspaceRoots,
	)
	initialInstances := allocationInstanceSet(routerAllocations)
	if len(initialInstances) != 2 {
		t.Fatalf("observation Router used %d Runtime instances, want two: %+v", len(initialInstances), routerAllocations)
	}

	reuseFixture := fixtureForObservationScenario("observation-router-reuse-planner")
	reuseArchive := workerObservationArchive(t, map[string]string{
		reuseFixture.BuilderPath:  reuseFixture.Before + "\n",
		reuseFixture.ReviewerPath: "review the fresh fixture\n",
		reuseFixture.UnreadPath:   "fresh intentionally unread\n",
	})
	reuseSource := uploadProjectArtifact(
		t, publicClient, publicBaseURL, "worker-observations-reuse", "application/zip", reuseArchive,
	)
	reuseRunID := createWorkflowRunWithParameters(
		t, publicClient, publicBaseURL, "worker-observations-router@1",
		"worker-observations-router-reuse", reuseSource,
		map[string]string{"scenario": "observations-reuse"},
	)
	reuseStatus := waitForRoutingRun(
		t, ctx, server, runtimes, gateway, publicClient, publicBaseURL, reuseRunID,
	)
	reuseAllocations := assertWorkerObservationExecution(
		t, ctx, store, gateway, publicClient, publicBaseURL, reuseStatus,
		"observation-router-reuse-planner", map[string]string{
			"builder_report":  reuseFixture.BuilderReport,
			"reviewer_report": reuseFixture.ReviewerReport,
		},
	)
	if got := allocationInstanceSet(reuseAllocations); !equalStringSets(got, initialInstances) {
		t.Fatalf("fresh observation Router did not reuse both Runtime slots: got=%v want=%v", got, initialInstances)
	}
	waitForObservationAllocationsReleased(
		t, ctx, reuseAllocations, runtimes, controlClient,
		runtimeURLs, runtimeInstanceIDs, workRoots, workspaceRoots,
	)

	assertWorkerObservationPlannerRetention(
		t, ctx, pool, []string{streamlineRunID, routerRunID, reuseRunID},
		[]string{
			initialFixture.BuilderPath, initialFixture.ReviewerPath, initialFixture.UnreadPath,
			reuseFixture.BuilderPath, reuseFixture.ReviewerPath, reuseFixture.UnreadPath,
		},
	)
	_, operationsBody := observedRuntimeAgents(t, publicClient, publicBaseURL)
	publicStatuses, err := json.Marshal([]runStatus{streamlineStatus, routerStatus, reuseStatus})
	if err != nil {
		t.Fatal(err)
	}
	retainedSurfaces := map[string]string{
		"Server logs":       server.logs.redacted(publicToken, llmGatewayToken),
		"public Runs":       string(publicStatuses),
		"public Operations": operationsBody,
	}
	for _, runtime := range runtimes {
		retainedSurfaces[runtime.name+" logs"] = runtime.logs.redacted(publicToken, llmGatewayToken)
	}
	for _, forbidden := range []string{
		initialFixture.BuilderPath, initialFixture.ReviewerPath, initialFixture.UnreadPath,
		reuseFixture.BuilderPath, reuseFixture.ReviewerPath, reuseFixture.UnreadPath,
	} {
		for surface, retained := range retainedSurfaces {
			if strings.Contains(retained, forbidden) {
				t.Fatalf("%s retained Worker State path %q", surface, forbidden)
			}
		}
	}
}

func workerObservationArchive(t *testing.T, files map[string]string) []byte {
	t.Helper()
	var output bytes.Buffer
	writer := zip.NewWriter(&output)
	paths := make([]string, 0, len(files))
	for path := range files {
		paths = append(paths, path)
	}
	slices.Sort(paths)
	for _, path := range paths {
		header := &zip.FileHeader{Name: path, Method: zip.Store}
		header.SetMode(0o600)
		header.SetModTime(time.Date(2026, 9, 2, 0, 0, 0, 0, time.UTC))
		entry, err := writer.CreateHeader(header)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := entry.Write([]byte(files[path])); err != nil {
			t.Fatal(err)
		}
	}
	if err := writer.Close(); err != nil {
		t.Fatal(err)
	}
	return output.Bytes()
}

func assertWorkerObservationExecution(
	t *testing.T,
	ctx context.Context,
	store *runstore.PostgresStore,
	gateway *routingGateway,
	client *http.Client,
	baseURL string,
	status runStatus,
	scenario string,
	wantOutputs map[string]string,
) []runstore.StageAllocation {
	t.Helper()
	if status.State != "succeeded" || len(status.Attempts) != 1 ||
		status.Attempts[0].State != "succeeded" || status.Attempts[0].Metrics == nil ||
		!status.Attempts[0].Metrics.ReportsComplete {
		t.Fatalf("Worker observation Run is incomplete: %+v", status)
	}
	for slot, expected := range wantOutputs {
		ref, ok := status.Outputs[slot]
		if !ok || ref.Revision == nil {
			t.Fatalf("Worker observation Run omitted exact output %q: %+v", slot, status.Outputs)
		}
		data, mediaType := download(
			t, client, baseURL+"/v1/runs/"+url.PathEscape(status.RunID)+"/outputs/"+url.PathEscape(slot),
		)
		if string(data) != expected || mediaType != "text/markdown" {
			t.Fatalf("Worker observation output %q = (%q, %q), want (%q, text/markdown)",
				slot, data, mediaType, expected)
		}
	}
	executions := requireExecutions(t, ctx, store, status.RunID, 1)
	assertSucceededExecution(t, executions[0], 1)
	allocations, err := store.ListStageAllocations(ctx, executions[0].StageExecutionID)
	if err != nil || len(allocations) != len(wantOutputs) {
		t.Fatalf("Worker observation allocations = (%+v, %v), want %d", allocations, err, len(wantOutputs))
	}
	reports, err := store.ListStageExecutionReports(ctx, executions[0].StageExecutionID)
	if err != nil || len(reports) != len(allocations) {
		t.Fatalf("Worker observation reports = (%+v, %v), want %d", reports, err, len(allocations))
	}
	stateUsages := gateway.StateUsages()
	for _, report := range reports {
		usage, ok := stateUsages[scenario+"/"+report.LogicalAgentName]
		if !ok {
			t.Fatalf("Gateway did not observe live State usage for %s/%s: %+v",
				scenario, report.LogicalAgentName, stateUsages)
		}
		assertWorkerReportMatchesLiveUsage(t, report, usage)
	}
	return allocations
}

func assertWorkerReportMatchesLiveUsage(
	t *testing.T,
	report runstore.StageExecutionReport,
	usage workerObservationUsage,
) {
	t.Helper()
	worker := report.Report.Worker
	metrics := worker.Metrics
	if !worker.Complete || !report.Report.Runtime.Complete || metrics.ModelCalls == nil ||
		metrics.TotalTokens == nil || *metrics.ModelCalls != usage.ModelCalls ||
		*metrics.TotalTokens != usage.TotalTokens || int64(len(worker.ToolCalls)) != usage.ToolCalls {
		t.Fatalf("durable report disagrees with live Worker State usage: report=%+v usage=%+v", worker, usage)
	}
	for name, calls := range usage.Tools {
		value, ok := metrics.Tools[name]
		if !ok || value.Calls == nil || *value.Calls != calls || value.Failed == nil || *value.Failed != 0 {
			t.Fatalf("durable tool metrics %q = %+v, want calls=%d failed=0", name, value, calls)
		}
	}
}

func waitForObservationAllocationsReleased(
	t *testing.T,
	ctx context.Context,
	allocations []runstore.StageAllocation,
	runtimes []*childProcess,
	client *http.Client,
	runtimeURLs, runtimeInstanceIDs, workRoots, workspaceRoots []string,
) {
	t.Helper()
	for _, allocation := range allocations {
		index := runtimeProcessIndex(t, runtimeInstanceIDs, allocation.RuntimeAgentInstanceID)
		waitForRuntimeReleased(
			t, ctx, runtimes[index], client, runtimeURLs[index], allocation.AllocationID, workRoots[index],
		)
		deadline := time.Now().Add(10 * time.Second)
		for !workRootEmpty(workspaceRoots[index]) && time.Now().Before(deadline) {
			time.Sleep(50 * time.Millisecond)
		}
		if !workRootEmpty(workspaceRoots[index]) {
			t.Fatalf("Runtime %s retained its allocation-private project workspace", allocation.RuntimeAgentInstanceID)
		}
	}
}

func runtimeProcessIndex(t *testing.T, runtimeInstanceIDs []string, instanceID string) int {
	t.Helper()
	for index, candidate := range runtimeInstanceIDs {
		if candidate == instanceID {
			return index
		}
	}
	t.Fatalf("cannot map Runtime instance %s to its process", instanceID)
	return -1
}

func allocationInstanceSet(allocations []runstore.StageAllocation) map[string]bool {
	result := make(map[string]bool, len(allocations))
	for _, allocation := range allocations {
		result[allocation.RuntimeAgentInstanceID] = true
	}
	return result
}

func equalStringSets(left, right map[string]bool) bool {
	if len(left) != len(right) {
		return false
	}
	for value := range left {
		if !right[value] {
			return false
		}
	}
	return true
}

func assertWorkerObservationPlannerRetention(
	t *testing.T,
	ctx context.Context,
	pool *pgxpool.Pool,
	runIDs, forbidden []string,
) {
	t.Helper()
	for _, runID := range runIDs {
		queries := []string{
			`SELECT COALESCE(string_agg(event::text, E'\n'), '') FROM planner_events WHERE run_id = $1`,
			`SELECT COALESCE(string_agg(report.report::text, E'\n'), '')
FROM planner_execution_reports AS report JOIN stage_executions AS execution
ON execution.stage_execution_id = report.stage_execution_id WHERE execution.run_id = $1`,
		}
		for _, query := range queries {
			var retained string
			if err := pool.QueryRow(ctx, query, runID).Scan(&retained); err != nil {
				t.Fatalf("read retained Planner observation surface for %s: %v", runID, err)
			}
			for _, value := range forbidden {
				if strings.Contains(retained, value) {
					t.Fatalf("durable Planner surface for %s retained State path %q", runID, value)
				}
			}
		}
	}
}

func assertPublicAttemptMetrics(t *testing.T, status runStatus, want [][2]int64) {
	t.Helper()
	if len(status.Attempts) != len(want) {
		t.Fatalf("Run %s attempts = %d, want %d", status.RunID, len(status.Attempts), len(want))
	}
	for index, expected := range want {
		metrics := status.Attempts[index].Metrics
		if metrics == nil || !metrics.ReportsComplete || metrics.ModelCalls != expected[0] ||
			metrics.ToolCalls != expected[1] {
			t.Fatalf("Run %s attempt %d metrics = %+v, want model/tool=%d/%d complete",
				status.RunID, index+1, metrics, expected[0], expected[1])
		}
	}
}

func waitForRoutingRun(
	t *testing.T,
	ctx context.Context,
	server *childProcess,
	runtimes []*childProcess,
	gateway *routingGateway,
	client *http.Client,
	baseURL, runID string,
) runStatus {
	t.Helper()
	ticker := time.NewTicker(200 * time.Millisecond)
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
					routingDiagnostics(server, runtimes, gateway))
			}
		}
		for _, process := range append([]*childProcess{server}, runtimes...) {
			if exited, processErr := process.exited(); exited {
				t.Fatalf("%s exited while Run was active: %v\n%s", process.name, processErr,
					routingDiagnostics(server, runtimes, gateway))
			}
		}
		select {
		case <-ctx.Done():
			t.Fatalf("wait for Run %s: %v\n%s", runID, ctx.Err(),
				routingDiagnostics(server, runtimes, gateway))
		case <-ticker.C:
		}
	}
}

func routingDiagnostics(
	server *childProcess, runtimes []*childProcess, gateway *routingGateway,
) string {
	var result strings.Builder
	result.WriteString("server:\n")
	result.WriteString(server.logs.redacted(publicToken, llmGatewayToken))
	for _, process := range runtimes {
		result.WriteString("\n" + process.name + ":\n")
		result.WriteString(process.logs.redacted(publicToken, llmGatewayToken))
	}
	result.WriteString(fmt.Sprintf("\ngateway observations=%+v failures=%v", gateway.Observations(), gateway.Failures()))
	return result.String()
}

func requireExecutions(
	t *testing.T, ctx context.Context, store *runstore.PostgresStore, runID string, want int,
) []runstore.StageExecution {
	t.Helper()
	executions, err := store.ListStageExecutions(ctx, runID)
	if err != nil || len(executions) != want {
		t.Fatalf("Run %s StageExecutions = (%+v, %v), want %d", runID, executions, err, want)
	}
	return executions
}

func assertSucceededExecution(t *testing.T, execution runstore.StageExecution, attempt int) {
	t.Helper()
	if execution.Attempt != attempt || execution.State != runstore.StageSucceeded ||
		execution.PlannerSessionID == nil || execution.PlannerInvocationID == nil ||
		execution.CandidateResult == nil || execution.AcceptedResult == nil ||
		execution.AcceptedResult.Outcome != contracts.StageSucceeded ||
		execution.FinalizationID == nil || execution.TerminalAt == nil {
		t.Fatalf("incomplete successful StageExecution: %+v", execution)
	}
}

func assertCompleteAllocations(
	t *testing.T,
	ctx context.Context,
	store *runstore.PostgresStore,
	execution runstore.StageExecution,
	want int,
	logicalNames ...string,
) []runstore.StageAllocation {
	t.Helper()
	allocations, err := store.ListStageAllocations(ctx, execution.StageExecutionID)
	if err != nil || len(allocations) != want {
		t.Fatalf("Stage %s allocations = (%+v, %v), want %d", execution.StageExecutionID, allocations, err, want)
	}
	reports, err := store.ListStageExecutionReports(ctx, execution.StageExecutionID)
	if err != nil || len(reports) != want {
		t.Fatalf("Stage %s reports = (%+v, %v), want %d", execution.StageExecutionID, reports, err, want)
	}
	wantNames := make(map[string]bool, len(logicalNames))
	for _, name := range logicalNames {
		wantNames[name] = true
	}
	for _, allocation := range allocations {
		if !wantNames[allocation.LogicalAgentName] || allocation.RuntimeAgentInstanceID == "" {
			t.Fatalf("unexpected durable allocation: %+v", allocation)
		}
	}
	for _, report := range reports {
		if !wantNames[report.LogicalAgentName] || !report.Report.Worker.Complete ||
			!report.Report.Runtime.Complete {
			t.Fatalf("incomplete trusted allocation report: %+v", report)
		}
	}
	return allocations
}

func assertModeledPlanEvents(
	t *testing.T,
	ctx context.Context,
	store *runstore.PostgresStore,
	execution runstore.StageExecution,
	wantWorker string,
	allocations []runstore.StageAllocation,
	runtimeURLs []string,
) {
	t.Helper()
	if execution.PlannerSessionID == nil {
		t.Fatal("modeled Planner has no durable session")
	}
	plannerEvents, err := store.ListPlannerEvents(ctx, *execution.PlannerSessionID, 0)
	if err != nil || len(plannerEvents) < 8 {
		t.Fatalf("Planner events = (%d, %v), want a complete modeled trace", len(plannerEvents), err)
	}
	runEvents, err := store.ListRunEvents(ctx, execution.RunID, 0, 1000)
	if err != nil {
		t.Fatalf("list Run events: %v", err)
	}
	runEventsBySequence := make(map[int64]runstore.WorkflowRunEvent, len(runEvents))
	for _, event := range runEvents {
		runEventsBySequence[event.SequenceNumber] = event
	}
	selected := 0
	for index, plannerEvent := range plannerEvents {
		if plannerEvent.RunEventSequence == nil {
			t.Fatalf("Planner/Run event %d is not atomically linked", index)
		}
		event, ok := runEventsBySequence[*plannerEvent.RunEventSequence]
		if !ok || event.EventID != plannerEvent.EventID {
			t.Fatalf("Planner/Run event %d points at a different journal record", index)
		}
		payload := string(event.Data)
		for _, allocation := range allocations {
			if strings.Contains(payload, allocation.AllocationID) ||
				strings.Contains(payload, allocation.RuntimeAgentInstanceID) {
				t.Fatalf("Run event leaked physical allocation identity: %s", payload)
			}
		}
		for _, runtimeURL := range runtimeURLs {
			if strings.Contains(payload, runtimeURL) {
				t.Fatalf("Run event leaked Runtime placement URL: %s", payload)
			}
		}
		if event.Kind == runstore.RunEventPlannerDispatchSelected {
			var data struct {
				WorkerName string `json:"workerName"`
			}
			if err := json.Unmarshal(event.Data, &data); err != nil || data.WorkerName != wantWorker {
				t.Fatalf("dispatch-selected = (%s, %v), want logical Worker %q", event.Data, err, wantWorker)
			}
			selected++
		}
	}
	if selected != 1 {
		t.Fatalf("dispatch-selected count = %d, want one", selected)
	}
}

func assertRoutingOutput(
	t *testing.T,
	ctx context.Context,
	pool *pgxpool.Pool,
	client *http.Client,
	baseURL, runID string,
	status runStatus,
) {
	t.Helper()
	output, ok := status.Outputs["result"]
	if !ok || output.Revision == nil {
		t.Fatalf("Run %s has no exact output: %+v", runID, status.Outputs)
	}
	data, mediaType := download(
		t, client, baseURL+"/v1/runs/"+url.PathEscape(runID)+"/outputs/result",
	)
	if string(data) != e2eInput || mediaType != e2eMediaType {
		t.Fatalf("Run %s output = (%q, %q), want exact input", runID, data, mediaType)
	}
	var frozen bool
	var currentRevision string
	if err := pool.QueryRow(ctx, `
SELECT frozen, current_revision
FROM artifact_bindings
WHERE scope_kind = 'run' AND scope_id = $1 AND namespace = 'outputs' AND name = 'result'`, runID,
	).Scan(&frozen, &currentRevision); err != nil || !frozen || currentRevision != *output.Revision {
		t.Fatalf("Run %s output binding = (%t, %q, %v), want frozen %q",
			runID, frozen, currentRevision, err, *output.Revision)
	}
}

func assertEscalatedExecution(
	t *testing.T, ctx context.Context, store *runstore.PostgresStore, runID string,
) {
	t.Helper()
	executions := requireExecutions(t, ctx, store, runID, 2)
	base, strong := executions[0], executions[1]
	if base.Attempt != 1 || base.State != runstore.StageFailed || base.AcceptedResult == nil ||
		base.AcceptedResult.Error == nil || base.AcceptedResult.Error.Code != "base_policy_rejected" ||
		base.AcceptedResult.Error.Retryable || base.ExecutionConfigVariant != runstore.StageExecutionConfigBase {
		t.Fatalf("base escalation execution = %+v", base)
	}
	assertSucceededExecution(t, strong, 2)
	if strong.PreviousExecutionID == nil || *strong.PreviousExecutionID != base.StageExecutionID ||
		strong.ExecutionConfigVariant != runstore.StageExecutionConfigFailedEscalation ||
		strong.EscalationOrdinal == nil || *strong.EscalationOrdinal != 1 {
		t.Fatalf("strong escalation linkage = %+v", strong)
	}
	var stage workflowconfig.ResolvedStage
	if err := json.Unmarshal(strong.StageSpecSnapshot, &stage); err != nil {
		t.Fatalf("decode escalated Stage snapshot: %v", err)
	}
	if stage.ExecutionConfig.Planner == nil ||
		stage.ExecutionConfig.Planner.ModelPolicy.Ref.PolicyID != "strong_planner" ||
		stage.ExecutionConfig.Agents["builder"].ModelPolicy.Ref.PolicyID != "strong_worker" ||
		stage.ExecutionConfig.Planner.LLMGateway.Ref.GatewayID != "local-litellm" ||
		stage.ExecutionConfig.Agents["builder"].LLMGateway.Ref.GatewayID != "local-litellm" {
		t.Fatalf("escalated Stage did not pin exact stronger refs: %+v", stage.ExecutionConfig)
	}
	decisions, err := store.ListStageTransitionDecisions(ctx, runID)
	if err != nil || len(decisions) != 2 ||
		decisions[0].Action != runstore.StageTransitionEscalate ||
		decisions[0].TargetExecutionID == nil || *decisions[0].TargetExecutionID != strong.StageExecutionID ||
		decisions[0].EscalationOrdinal == nil || *decisions[0].EscalationOrdinal != 1 ||
		decisions[0].EscalationExhausted || decisions[1].Action != runstore.StageTransitionSucceed {
		t.Fatalf("escalation decisions = (%+v, %v)", decisions, err)
	}
	assertCompleteAllocations(t, ctx, store, base, 1, "builder")
	allocations := assertCompleteAllocations(t, ctx, store, strong, 1, "builder")
	assertModeledPlanEvents(t, ctx, store, strong, "builder", allocations, nil)
}

func waitForRuntimeIdle(
	t *testing.T,
	ctx context.Context,
	process *childProcess,
	client *http.Client,
	baseURL, workRoot string,
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
				return
			}
		}
		if exited, processErr := process.exited(); exited {
			t.Fatalf("%s exited before slot reuse check: %v", process.name, processErr)
		}
		time.Sleep(100 * time.Millisecond)
	}
	t.Fatalf("%s did not return to an idle reusable slot; work root empty=%t\n%s",
		process.name, workRootEmpty(workRoot), process.logs.redacted(publicToken, llmGatewayToken))
}
