//go:build e2e

package e2e

import (
	"bytes"
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

	"github.com/grauwolf32/contractor/internal/localpki"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/jackc/pgx/v5/pgxpool"
)

func TestSchedulerConcurrencyAcrossProductionProcesses(t *testing.T) {
	if testing.Short() {
		t.Skip("end-to-end process test")
	}
	databaseURL := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if databaseURL == "" {
		t.Fatal("CONTRACTOR_TEST_DATABASE_URL is required")
	}
	repositoryRoot := repoRoot(t)
	temporaryRoot := t.TempDir()
	ctx, cancel := context.WithTimeout(context.Background(), 210*time.Second)
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
		LeafOptions: leaf, URI: "urn:contractor:control-plane:scheduler-concurrency-e2e",
	})
	if err != nil {
		t.Fatalf("issue Control Plane certificate: %v", err)
	}
	agentPaths := make([]localpki.Paths, 2)
	for index := range agentPaths {
		agentPaths[index], err = generator.IssueAgent(
			pkiRoot, fmt.Sprintf("scheduler-concurrency-agent-%d", index+1), leaf,
		)
		if err != nil {
			t.Fatalf("issue Runtime Agent %d certificate: %v", index+1, err)
		}
	}

	gateway := newSchedulerConcurrencyGateway(llmGatewayToken)
	t.Cleanup(gateway.close)
	configRoot := stageE2EConfiguration(
		t, filepath.Join(repositoryRoot, "configs", "e2e"),
		filepath.Join(temporaryRoot, "configs"), gateway.URL(),
	)
	publicAddress := freeAddress(t)
	privateAddress := freeAddress(t)
	publicBaseURL := "http://" + publicAddress
	privateBaseURL := "https://" + privateAddress
	userID := "scheduler-concurrency-user-" + randomHex(t, 8)
	serverEnvironment := map[string]string{
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
		"CONTRACTOR_LOCAL_AUTH_FILE":         writeE2ELocalAuth(t, temporaryRoot, userID),
		"CONTRACTOR_BROWSER_ORIGINS":         "https://ui.contractor.invalid",
	}
	server := startProcess(
		t, "Scheduler-concurrency Go Server", repositoryRoot, serverEnvironment,
		serverBinary, "serve",
	)
	publicClient := &http.Client{Timeout: 5 * time.Second}
	waitForHTTP(t, ctx, server, publicClient, publicBaseURL+"/readyz", http.StatusOK)

	initialSettings, initialETag := getSchedulerSettings(t, publicClient, publicBaseURL)
	if initialSettings.Maximum != 1 || initialSettings.Revision != "1" || initialETag != `"1"` {
		t.Fatalf("fresh Scheduler settings = (%+v, %q), want default one", initialSettings, initialETag)
	}

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
			t, fmt.Sprintf("Scheduler-concurrency Python Runtime %d", index+1),
			filepath.Join(repositoryRoot, "runtime"), map[string]string{"PYTHONUNBUFFERED": "1"},
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
			"--heartbeat-interval-seconds", "1",
			"--confirmed-lease-seconds", "12",
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

	uploaded := uploadInput(t, publicClient, publicBaseURL)
	serialRuns := []string{
		createWorkflowRun(t, publicClient, publicBaseURL, "artifact-copy@1", "scheduler-default-1", uploaded),
		createWorkflowRun(t, publicClient, publicBaseURL, "artifact-copy@1", "scheduler-default-2", uploaded),
	}
	waitForGatewayInitialCalls(t, ctx, server, runtimes, gateway, 1)
	select {
	case <-time.After(300 * time.Millisecond):
	case <-ctx.Done():
		t.Fatal(ctx.Err())
	}
	if initial, blocked, maximum := gateway.snapshot(); initial != 1 || blocked != 1 || maximum != 1 {
		t.Fatalf("default-one Gateway concurrency = initial:%d blocked:%d max:%d", initial, blocked, maximum)
	}
	gateway.releaseBarrier()
	waitForSchedulerRuns(t, ctx, server, runtimes, gateway, publicClient, publicBaseURL, serialRuns)
	for index := range runtimes {
		waitForRuntimeIdle(t, ctx, runtimes[index], controlClient, runtimeURLs[index], workRoots[index])
	}
	if err := gateway.resetBarrier(); err != nil {
		t.Fatal(err)
	}

	updated := replaceSchedulerSettings(t, publicClient, publicBaseURL, initialETag, 2)
	if updated.Maximum != 2 || updated.Revision != "2" {
		t.Fatalf("updated Scheduler settings = %+v", updated)
	}

	concurrentRuns := []string{
		createWorkflowRun(t, publicClient, publicBaseURL, "artifact-copy@1", "scheduler-concurrent-1", uploaded),
		createWorkflowRun(t, publicClient, publicBaseURL, "artifact-copy@1", "scheduler-concurrent-2", uploaded),
		createWorkflowRun(t, publicClient, publicBaseURL, "artifact-copy@1", "scheduler-concurrent-3", uploaded),
	}
	waitForGatewayInitialCalls(t, ctx, server, runtimes, gateway, 2)

	pool, err := pgxpool.New(ctx, isolateURL)
	if err != nil {
		t.Fatalf("open Scheduler-concurrency assertion database: %v", err)
	}
	t.Cleanup(pool.Close)
	store := runstore.NewPostgresStore(pool)
	activeRuns, instances := activeAllocatedRuns(t, ctx, store, concurrentRuns)
	if len(activeRuns) != 2 || len(instances) != 2 {
		t.Fatalf("active production lanes = runs:%v Runtime instances:%v, want two distinct of each", activeRuns, instances)
	}
	for _, runID := range concurrentRuns {
		if activeRuns[runID] {
			continue
		}
		executions, listErr := store.ListStageExecutions(ctx, runID)
		if listErr != nil || len(executions) > 1 {
			t.Fatalf("waiting Run %s has invalid executions=%+v error=%v", runID, executions, listErr)
		}
		if len(executions) == 1 {
			execution := executions[0]
			allocations, allocationErr := store.ListStageAllocations(ctx, execution.StageExecutionID)
			if allocationErr != nil || len(allocations) != 0 || execution.State != runstore.StagePreparing ||
				execution.PlannerSessionID != nil || execution.PlannerStartedAt != nil {
				t.Fatalf("third Run %s crossed the Runtime-capacity wait boundary: execution=%+v allocations=%+v error=%v",
					runID, execution, allocations, allocationErr)
			}
		}
	}
	if initial, blocked, maximum := gateway.snapshot(); initial != 2 || blocked != 2 || maximum != 2 {
		t.Fatalf("limit-two Gateway concurrency = initial:%d blocked:%d max:%d", initial, blocked, maximum)
	}
	gateway.releaseBarrier()
	waitForSchedulerRuns(t, ctx, server, runtimes, gateway, publicClient, publicBaseURL, concurrentRuns)
	for index := range runtimes {
		waitForRuntimeIdle(t, ctx, runtimes[index], controlClient, runtimeURLs[index], workRoots[index])
	}
	waitForSchedulerConcurrencyReleased(t, ctx, store, concurrentRuns)

	for _, process := range runtimes {
		process.stop(t)
	}
	server.stop(t)
	server = startProcess(
		t, "restarted Scheduler-concurrency Go Server", repositoryRoot, serverEnvironment,
		serverBinary, "serve",
	)
	waitForHTTP(t, ctx, server, publicClient, publicBaseURL+"/readyz", http.StatusOK)
	restarted, restartedETag := getSchedulerSettings(t, publicClient, publicBaseURL)
	if restarted.Maximum != 2 || restarted.Revision != "2" || restartedETag != `"2"` {
		t.Fatalf("restarted Scheduler settings = (%+v, %q), want durable two", restarted, restartedETag)
	}

	if failures := gateway.Failures(); len(failures) != 0 {
		t.Fatalf("Scheduler-concurrency Gateway failures: %v", failures)
	}
	for _, secret := range []string{publicToken, llmGatewayToken} {
		if strings.Contains(server.logs.redacted(), secret) {
			t.Fatal("restarted Server logs contain a configured secret")
		}
		for _, process := range runtimes {
			if strings.Contains(process.logs.redacted(), secret) {
				t.Fatalf("%s logs contain a configured secret", process.name)
			}
		}
	}
}

type schedulerSettingsPayload struct {
	Maximum   int       `json:"maxConcurrentRuns"`
	Revision  string    `json:"revision"`
	UpdatedAt time.Time `json:"updatedAt"`
}

func getSchedulerSettings(
	t *testing.T,
	client *http.Client,
	baseURL string,
) (schedulerSettingsPayload, string) {
	t.Helper()
	request, err := http.NewRequest(http.MethodGet, baseURL+"/v1/operations/settings/scheduler", nil)
	if err != nil {
		t.Fatal(err)
	}
	request.Header.Set("Authorization", "Bearer "+publicToken)
	response := do(t, client, request, http.StatusOK)
	defer response.Body.Close()
	var payload schedulerSettingsPayload
	decodeResponse(t, response, &payload)
	if response.Header.Get("Cache-Control") != "no-store" {
		t.Fatalf("Scheduler settings Cache-Control = %q", response.Header.Get("Cache-Control"))
	}
	return payload, response.Header.Get("ETag")
}

func replaceSchedulerSettings(
	t *testing.T,
	client *http.Client,
	baseURL, etag string,
	maximum int,
) schedulerSettingsPayload {
	t.Helper()
	body, err := json.Marshal(map[string]int{"maxConcurrentRuns": maximum})
	if err != nil {
		t.Fatal(err)
	}
	request, err := http.NewRequest(
		http.MethodPut, baseURL+"/v1/operations/settings/scheduler", bytes.NewReader(body),
	)
	if err != nil {
		t.Fatal(err)
	}
	request.Header.Set("Authorization", "Bearer "+publicToken)
	request.Header.Set("Content-Type", "application/json")
	request.Header.Set("If-Match", etag)
	response := do(t, client, request, http.StatusOK)
	defer response.Body.Close()
	var payload schedulerSettingsPayload
	decodeResponse(t, response, &payload)
	return payload
}

func waitForGatewayInitialCalls(
	t *testing.T,
	ctx context.Context,
	server *childProcess,
	runtimes []*childProcess,
	gateway *schedulerConcurrencyGateway,
	want int,
) {
	t.Helper()
	ticker := time.NewTicker(50 * time.Millisecond)
	defer ticker.Stop()
	for {
		initial, _, _ := gateway.snapshot()
		if initial >= want {
			return
		}
		for _, process := range append([]*childProcess{server}, runtimes...) {
			if exited, processErr := process.exited(); exited {
				t.Fatalf("%s exited while waiting for %d model calls: %v\n%s", process.name, want,
					processErr, schedulerConcurrencyDiagnostics(server, runtimes, gateway))
			}
		}
		if failures := gateway.Failures(); len(failures) != 0 {
			t.Fatalf("Scheduler-concurrency Gateway failures: %v", failures)
		}
		select {
		case <-ctx.Done():
			t.Fatalf("wait for %d initial Gateway calls: %v\n%s", want, ctx.Err(),
				schedulerConcurrencyDiagnostics(server, runtimes, gateway))
		case <-ticker.C:
		}
	}
}

func waitForSchedulerRuns(
	t *testing.T,
	ctx context.Context,
	server *childProcess,
	runtimes []*childProcess,
	gateway *schedulerConcurrencyGateway,
	client *http.Client,
	baseURL string,
	runIDs []string,
) {
	t.Helper()
	remaining := make(map[string]bool, len(runIDs))
	for _, runID := range runIDs {
		remaining[runID] = true
	}
	ticker := time.NewTicker(75 * time.Millisecond)
	defer ticker.Stop()
	for len(remaining) > 0 {
		for runID := range remaining {
			request, _ := http.NewRequestWithContext(
				ctx, http.MethodGet, baseURL+"/v1/runs/"+url.PathEscape(runID), nil,
			)
			request.Header.Set("Authorization", "Bearer "+publicToken)
			response, err := client.Do(request)
			if err != nil {
				continue
			}
			var status runStatus
			if response.StatusCode == http.StatusOK {
				decodeResponse(t, response, &status)
			}
			response.Body.Close()
			switch status.State {
			case "succeeded":
				delete(remaining, runID)
			case "failed", "cancelled":
				t.Fatalf("Run %s reached %s: %+v\n%s", runID, status.State, status,
					schedulerConcurrencyDiagnostics(server, runtimes, gateway))
			}
		}
		for _, process := range append([]*childProcess{server}, runtimes...) {
			if exited, processErr := process.exited(); exited {
				t.Fatalf("%s exited while Runs were active: %v\n%s", process.name, processErr,
					schedulerConcurrencyDiagnostics(server, runtimes, gateway))
			}
		}
		if failures := gateway.Failures(); len(failures) != 0 {
			t.Fatalf("Scheduler-concurrency Gateway failures: %v\n%s", failures,
				schedulerConcurrencyDiagnostics(server, runtimes, gateway))
		}
		select {
		case <-ctx.Done():
			t.Fatalf("wait for Scheduler-concurrency Runs: %v; remaining=%v\n%s", ctx.Err(), remaining,
				schedulerConcurrencyDiagnostics(server, runtimes, gateway))
		case <-ticker.C:
		}
	}
}

func activeAllocatedRuns(
	t *testing.T,
	ctx context.Context,
	store *runstore.PostgresStore,
	runIDs []string,
) (map[string]bool, map[string]bool) {
	t.Helper()
	runs := make(map[string]bool)
	instances := make(map[string]bool)
	for _, runID := range runIDs {
		executions, err := store.ListStageExecutions(ctx, runID)
		if err != nil {
			t.Fatal(err)
		}
		for _, execution := range executions {
			allocations, err := store.ListStageAllocations(ctx, execution.StageExecutionID)
			if err != nil {
				t.Fatal(err)
			}
			for _, allocation := range allocations {
				if allocation.ReleaseCompletedAt == nil {
					runs[runID] = true
					instances[allocation.RuntimeAgentInstanceID] = true
				}
			}
		}
	}
	return runs, instances
}

func waitForSchedulerConcurrencyReleased(
	t *testing.T,
	ctx context.Context,
	store *runstore.PostgresStore,
	runIDs []string,
) {
	t.Helper()
	ticker := time.NewTicker(50 * time.Millisecond)
	defer ticker.Stop()
	for {
		complete := true
		for _, runID := range runIDs {
			run, err := store.GetRun(ctx, runID)
			if err != nil || run.State != runstore.RunSucceeded {
				t.Fatalf("terminal Run %s = (%+v, %v)", runID, run, err)
			}
			if run.SchedulerClaim != nil {
				complete = false
				continue
			}
			executions, err := store.ListStageExecutions(ctx, runID)
			if err != nil || len(executions) != 1 {
				t.Fatalf("Run %s executions = (%+v, %v), want one", runID, executions, err)
			}
			allocations, err := store.ListStageAllocations(ctx, executions[0].StageExecutionID)
			if err != nil || len(allocations) != 1 {
				t.Fatalf("Run %s allocations = (%+v, %v), want one", runID, allocations, err)
			}
			if allocations[0].ReleaseCompletedAt == nil {
				complete = false
			}
		}
		if complete {
			return
		}
		select {
		case <-ctx.Done():
			t.Fatalf("wait for Scheduler claims and allocations to release: %v", ctx.Err())
		case <-ticker.C:
		}
	}
}

func schedulerConcurrencyDiagnostics(
	server *childProcess,
	runtimes []*childProcess,
	gateway *schedulerConcurrencyGateway,
) string {
	var result strings.Builder
	result.WriteString("server:\n")
	result.WriteString(server.logs.redacted(publicToken, llmGatewayToken))
	for _, process := range runtimes {
		result.WriteString("\n" + process.name + ":\n")
		result.WriteString(process.logs.redacted(publicToken, llmGatewayToken))
	}
	initial, blocked, maximum := gateway.snapshot()
	result.WriteString(fmt.Sprintf(
		"\ngateway initial=%d blocked=%d maximum=%d failures=%v",
		initial, blocked, maximum, gateway.Failures(),
	))
	return result.String()
}
