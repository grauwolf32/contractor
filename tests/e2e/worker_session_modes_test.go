//go:build e2e

package e2e

import (
	"context"
	"fmt"
	"net"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	workflowconfig "github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/localpki"
	"github.com/grauwolf32/contractor/internal/runstore"
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
	ctx, cancel := context.WithTimeout(context.Background(), 240*time.Second)
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
	publicAddress := freeAddress(t)
	privateAddress := freeAddress(t)
	publicBaseURL := "http://" + publicAddress
	privateBaseURL := "https://" + privateAddress
	userID := "worker-session-e2e-user-" + randomHex(t, 8)
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

	pool, err := pgxpool.New(ctx, isolateURL)
	if err != nil {
		t.Fatalf("open assertion database: %v", err)
	}
	t.Cleanup(pool.Close)
	store := runstore.NewPostgresStore(pool)

	streamlineRunID := createEmptyWorkflowRun(
		t, publicClient, publicBaseURL, "worker-session-streamline@1", "worker-session-streamline",
	)
	waitForWorkerSessionRun(
		t, ctx, server, runtimes, gateway, publicClient, publicBaseURL, streamlineRunID,
	)
	assertWorkerSessionExecutions(
		t, ctx, store, streamlineRunID,
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
		t, ctx, store, routerRunID,
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
		stage, err := workflowconfig.DecodeResolvedStageSnapshot(execution.StageSpecSnapshot)
		if err != nil {
			t.Fatalf("decode Stage %s snapshot: %v", execution.StageName, err)
		}
		if stage.Session != want.mode {
			t.Fatalf("Stage %s persisted session mode = %q, want %q", execution.StageName, stage.Session, want.mode)
		}
		allocations[execution.StageName] = assertCompleteAllocations(
			t, ctx, store, execution, len(want.workers), want.workers...,
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
