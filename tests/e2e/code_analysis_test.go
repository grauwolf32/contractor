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

	"github.com/grauwolf32/contractor/internal/localpki"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/jackc/pgx/v5/pgxpool"
)

const codeAnalysisSourceCanary = "CODE_ANALYSIS_SOURCE_CONTENT_CANARY"

var completeCodeAnalysisTools = []string{
	"attack_surface", "complexity_hotspots", "entrypoint_paths_to", "find_callees",
	"find_callers", "find_symbol", "functions_that_raise", "graph_summary",
	"list_symbols", "paths_between", "search_def",
}

func TestCodeAnalysisAcrossHeterogeneousRuntimeProcesses(t *testing.T) {
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
		t.Fatalf("initialize code-analysis test CA: %v", err)
	}
	leaf := localpki.LeafOptions{IPAddresses: []net.IP{net.ParseIP("127.0.0.1")}}
	controlPlanePaths, err := generator.IssueControlPlane(pkiRoot, localpki.ControlPlaneOptions{
		LeafOptions: leaf, URI: "urn:contractor:control-plane:code-analysis-e2e",
	})
	if err != nil {
		t.Fatalf("issue code-analysis Control Plane certificate: %v", err)
	}
	memoryIdentity, err := generator.IssueAgent(pkiRoot, "code-analysis-memory", leaf)
	if err != nil {
		t.Fatalf("issue memory Runtime certificate: %v", err)
	}
	localIdentity, err := generator.IssueAgent(pkiRoot, "code-analysis-local", leaf)
	if err != nil {
		t.Fatalf("issue local Runtime certificate: %v", err)
	}

	gateway := newCodeAnalysisGateway(llmGatewayToken)
	t.Cleanup(gateway.close)
	configRoot := stageE2EConfiguration(
		t, filepath.Join(repositoryRoot, "configs"),
		filepath.Join(temporaryRoot, "configs"), gateway.URL(),
	)
	writeCodeAnalysisE2EConfiguration(t, configRoot)

	publicAddress, privateAddress := freeAddress(t), freeAddress(t)
	publicBaseURL := "http://" + publicAddress
	privateBaseURL := "https://" + privateAddress
	userID := "code-analysis-e2e-user-" + randomHex(t, 8)
	server := startProcess(t, "Go Server code analysis", repositoryRoot, map[string]string{
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
	}, serverBinary, "serve", "--runtime-request-timeout", "90s")
	publicClient := &http.Client{Timeout: 8 * time.Second}
	waitForHTTP(t, ctx, server, publicClient, publicBaseURL+"/readyz", http.StatusOK)

	python := filepath.Join(repositoryRoot, "runtime", ".venv", "bin", "python")
	if info, statErr := os.Stat(python); statErr != nil || info.IsDir() {
		t.Fatalf("Python Runtime environment is missing at %s; run 'cd runtime && uv sync --locked'", python)
	}
	memoryAddress := freeAddress(t)
	memoryBaseURL := "https://" + memoryAddress
	memoryWorkRoot := filepath.Join(temporaryRoot, "memory-runtime-work")
	memoryRuntime := startCodeAnalysisRuntime(
		t, "Python memory code-analysis Runtime", repositoryRoot, python,
		privateBaseURL, memoryBaseURL, memoryAddress, memoryWorkRoot, "memory", "",
		caPaths.Certificate, memoryIdentity,
	)
	t.Cleanup(func() {
		if !t.Failed() {
			return
		}
		for _, process := range []*childProcess{server, memoryRuntime} {
			t.Logf("%s logs:\n%s", process.name, process.logs.redacted(
				publicToken, llmGatewayToken, codeAnalysisSourceCanary,
				codeAnalysisDuplicateSymbol, codeAnalysisEditedPath,
			))
		}
		t.Logf("code-analysis Gateway: calls=%d scenarios=%d failures=%v",
			gateway.Calls(), gateway.CompletedScenarios(), gateway.Failures())
	})
	controlClient := newMTLSClient(t, caPaths.Certificate, controlPlanePaths)
	waitForHTTP(t, ctx, memoryRuntime, controlClient, memoryBaseURL+"/healthz", http.StatusOK)
	waitForProcessLog(t, ctx, memoryRuntime, "runtime agent registered")

	initialAgents, initialSnapshot := waitForObservedRuntimeAgents(
		t, ctx, server, []*childProcess{memoryRuntime}, publicClient, publicBaseURL,
		func(items []observedRuntimeAgent) bool {
			return len(items) == 1 && items[0].SlotState == "idle" &&
				items[0].ConfirmedLeaseUntil != nil
		},
	)
	memoryAgent := initialAgents[0]
	if got := runtimeToolset(memoryAgent, "code-analysis@1").Tools; !slices.Equal(got, []string{"list_symbols", "search_def"}) {
		t.Fatalf("memory Runtime code-analysis capability = %v, want exact shallow subset", got)
	}

	sourceBytes := codeAnalysisSourceArchive(t)
	source := uploadProjectArtifact(
		t, publicClient, publicBaseURL, "code-analysis-source", "application/zip", sourceBytes,
	)
	pool, err := pgxpool.New(ctx, isolateURL)
	if err != nil {
		t.Fatalf("open code-analysis assertion database: %v", err)
	}
	t.Cleanup(pool.Close)
	store := runstore.NewPostgresStore(pool)

	graphInitialID := createCodeAnalysisRun(
		t, publicClient, publicBaseURL, "code-analysis-graph-e2e@1",
		"code-analysis-graph-wait", source,
	)
	waitingExecution := waitForUnallocatedPreparingExecution(t, ctx, store, graphInitialID)
	if gateway.Calls() != 0 {
		t.Fatalf("graph Run reached the model Gateway without graph capacity: %d calls", gateway.Calls())
	}

	shallowInitialID := createCodeAnalysisRun(
		t, publicClient, publicBaseURL, "code-analysis-shallow-e2e@1",
		"code-analysis-shallow-initial", source,
	)
	shallowInitial := waitForRunAcross(
		t, ctx, server, []*childProcess{memoryRuntime}, gateway,
		publicClient, publicBaseURL, shallowInitialID,
	)
	shallowInitialAllocation := assertCodeAnalysisRun(
		t, ctx, store, publicClient, publicBaseURL, shallowInitial,
		memoryAgent.InstanceID, 1,
	)
	waitForRuntimeReleased(
		t, ctx, memoryRuntime, controlClient, memoryBaseURL,
		shallowInitialAllocation.AllocationID, memoryWorkRoot,
	)
	assertSameUnallocatedPreparingExecution(t, ctx, store, waitingExecution)

	localRuntimeAddress := freeAddress(t)
	localAdvertisedAddress := freeAddress(t)
	localRuntimeBaseURL := "https://" + localRuntimeAddress
	localAdvertisedBaseURL := "https://" + localAdvertisedAddress
	localWorkRoot := filepath.Join(temporaryRoot, "local-runtime-work")
	localWorkspaceRoot := filepath.Join(temporaryRoot, "local-workspaces")
	releaseProxy := newRuntimeReleaseLossProxy(
		t, localAdvertisedAddress, localRuntimeBaseURL, caPaths.Certificate,
		localIdentity, controlPlanePaths,
	)
	localRuntime := startCodeAnalysisRuntime(
		t, "Python local graph Runtime", repositoryRoot, python,
		privateBaseURL, localAdvertisedBaseURL, localRuntimeAddress,
		localWorkRoot, "local", localWorkspaceRoot, caPaths.Certificate, localIdentity,
	)
	t.Cleanup(func() {
		if t.Failed() {
			t.Logf("%s logs:\n%s", localRuntime.name, localRuntime.logs.redacted(
				publicToken, llmGatewayToken, codeAnalysisSourceCanary,
				codeAnalysisDuplicateSymbol, codeAnalysisEditedPath,
			))
		}
	})
	runtimes := []*childProcess{memoryRuntime, localRuntime}
	waitForHTTP(t, ctx, localRuntime, controlClient, localAdvertisedBaseURL+"/healthz", http.StatusOK)
	waitForProcessLog(t, ctx, localRuntime, "runtime agent registered")
	registeredAgents, registeredSnapshot := waitForObservedRuntimeAgents(
		t, ctx, server, runtimes, publicClient, publicBaseURL,
		func(items []observedRuntimeAgent) bool {
			if len(items) != 2 {
				return false
			}
			for _, item := range items {
				if len(runtimeToolset(item, "code-analysis@1").Tools) == len(completeCodeAnalysisTools) {
					return item.SlotState == "idle" && item.ConfirmedLeaseUntil != nil
				}
			}
			return false
		},
	)
	localAgent, ok := findRuntimeWithTool(registeredAgents, "code-analysis@1", "find_symbol")
	if !ok || localAgent.InstanceID == memoryAgent.InstanceID {
		t.Fatalf("local graph Runtime is absent from Operations: %+v", registeredAgents)
	}
	if got := runtimeToolset(localAgent, "code-analysis@1").Tools; !slices.Equal(got, completeCodeAnalysisTools) {
		t.Fatalf("local Runtime code-analysis capability = %v, want %v", got, completeCodeAnalysisTools)
	}

	graphInitial := waitForRunAcross(
		t, ctx, server, runtimes, gateway, publicClient, publicBaseURL, graphInitialID,
	)
	assertCodeAnalysisRun(
		t, ctx, store, publicClient, publicBaseURL, graphInitial,
		localAgent.InstanceID, 1,
	)
	if graphInitial.Attempts[0].StageExecutionID != waitingExecution.StageExecutionID ||
		graphInitial.Attempts[0].Attempt != 1 {
		t.Fatalf("capacity wait consumed graph attempt: %+v", graphInitial.Attempts)
	}
	select {
	case <-releaseProxy.dropped:
	case <-ctx.Done():
		t.Fatalf("wait for lost graph release acknowledgement: %v", ctx.Err())
	}

	// The graph slot remains authoritatively fenced while release acknowledgements
	// are lost. The independent shallow slot must continue to make progress.
	shallowUnrelatedID := createCodeAnalysisRun(
		t, publicClient, publicBaseURL, "code-analysis-shallow-e2e@1",
		"code-analysis-shallow-unrelated", source,
	)
	shallowUnrelated := waitForRunAcross(
		t, ctx, server, runtimes, gateway, publicClient, publicBaseURL, shallowUnrelatedID,
	)
	shallowUnrelatedAllocation := assertCodeAnalysisRun(
		t, ctx, store, publicClient, publicBaseURL, shallowUnrelated,
		memoryAgent.InstanceID, 1,
	)
	waitForRuntimeReleased(
		t, ctx, memoryRuntime, controlClient, memoryBaseURL,
		shallowUnrelatedAllocation.AllocationID, memoryWorkRoot,
	)

	releaseProxy.allowReleaseAcknowledgement()
	waitForObservedRuntimeAgents(
		t, ctx, server, runtimes, publicClient, publicBaseURL,
		func(items []observedRuntimeAgent) bool {
			if len(items) != 2 {
				return false
			}
			for _, item := range items {
				if item.SlotState != "idle" || item.ObservedState != "idle" ||
					item.CurrentAllocationID != nil || item.AuthoritativeAllocationID != nil {
					return false
				}
			}
			return true
		},
	)

	graphReuseID := createCodeAnalysisRun(
		t, publicClient, publicBaseURL, "code-analysis-graph-e2e@1",
		"code-analysis-graph-reuse", source,
	)
	graphReuse := waitForRunAcross(
		t, ctx, server, runtimes, gateway, publicClient, publicBaseURL, graphReuseID,
	)
	graphReuseAllocation := assertCodeAnalysisRun(
		t, ctx, store, publicClient, publicBaseURL, graphReuse,
		localAgent.InstanceID, 1,
	)
	waitForRuntimeReleased(
		t, ctx, localRuntime, controlClient, localAdvertisedBaseURL,
		graphReuseAllocation.AllocationID, localWorkRoot,
	)

	finalAgents, finalSnapshot := waitForObservedRuntimeAgents(
		t, ctx, server, runtimes, publicClient, publicBaseURL,
		func(items []observedRuntimeAgent) bool {
			if len(items) != 2 {
				return false
			}
			for _, item := range items {
				if item.SlotState != "idle" || item.ObservedState != "idle" ||
					item.CurrentAllocationID != nil || item.AuthoritativeAllocationID != nil {
					return false
				}
			}
			return true
		},
	)
	if len(finalAgents) != 2 || !workRootEmpty(memoryWorkRoot) ||
		!workRootEmpty(localWorkRoot) || !workRootEmpty(localWorkspaceRoot) {
		t.Fatalf("code-analysis slots retained private state: agents=%+v memory=%t local=%t workspace=%t",
			finalAgents, workRootEmpty(memoryWorkRoot), workRootEmpty(localWorkRoot),
			workRootEmpty(localWorkspaceRoot))
	}
	if drops, failures := releaseProxy.snapshot(); drops == 0 || len(failures) != 0 {
		t.Fatalf("code-analysis release-loss proxy = drops:%d failures:%v", drops, failures)
	}
	if gateway.CompletedScenarios() != 4 || len(gateway.Failures()) != 0 {
		t.Fatalf("code-analysis Gateway = scenarios:%d calls:%d failures:%v",
			gateway.CompletedScenarios(), gateway.Calls(), gateway.Failures())
	}

	operationsEvidence := []string{initialSnapshot, registeredSnapshot, finalSnapshot}
	processes := append([]*childProcess{server}, runtimes...)
	for _, runID := range []string{
		graphInitialID, shallowInitialID, shallowUnrelatedID, graphReuseID,
	} {
		assertCodeAnalysisRetainedExecutionSafe(t, ctx, pool, runID)
	}
	assertCodeAnalysisOperationalSurfacesSafe(
		t, operationsEvidence, processes, temporaryRoot, memoryWorkRoot,
		localWorkRoot, localWorkspaceRoot,
	)
	assertUserArtifactUnchanged(
		t, publicClient, publicBaseURL, source, sourceBytes, "application/zip",
	)
}

func startCodeAnalysisRuntime(
	t *testing.T,
	name, repositoryRoot, python, controlPlaneURL, advertisedURL, listenAddress,
	workRoot, storage, workspaceRoot, caFile string,
	identity localpki.Paths,
) *childProcess {
	t.Helper()
	args := []string{
		"-m", "contractor_runtime",
		"--control-plane-url", controlPlaneURL,
		"--advertised-control-url", advertisedURL,
		"--advertised-a2a-url", advertisedURL,
		"--ca-file", caFile,
		"--certificate-file", identity.Certificate,
		"--private-key-file", identity.PrivateKey,
		"--listen", listenAddress,
		"--work-root", workRoot,
		"--workspace-storage", storage,
		"--request-timeout-seconds", "30",
		"--shutdown-grace-seconds", "8",
		"--heartbeat-interval-seconds", "1",
		"--confirmed-lease-seconds", "12",
	}
	if workspaceRoot != "" {
		args = append(args, "--workspace-work-root", workspaceRoot)
	}
	return startProcess(
		t, name, filepath.Join(repositoryRoot, "runtime"),
		map[string]string{"PYTHONUNBUFFERED": "1"}, python, args...,
	)
}

func createCodeAnalysisRun(
	t *testing.T,
	client *http.Client,
	baseURL, workflow, idempotencyKey string,
	source artifactRef,
) string {
	t.Helper()
	body, err := json.Marshal(map[string]any{
		"workflow":   workflow,
		"parameters": map[string]string{},
		"artifacts":  map[string]artifactRef{"source": source},
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
	var payload runStatus
	decodeResponse(t, response, &payload)
	if payload.RunID == "" || payload.State != "running" {
		t.Fatalf("create code-analysis Run response = %+v", payload)
	}
	return payload.RunID
}

func assertCodeAnalysisRun(
	t *testing.T,
	ctx context.Context,
	store runstore.Repository,
	client *http.Client,
	baseURL string,
	status runStatus,
	wantRuntimeInstanceID string,
	wantAttempt int,
) runstore.StageAllocation {
	t.Helper()
	if status.State != "succeeded" || len(status.Attempts) != 1 ||
		status.Attempts[0].State != "succeeded" || status.Attempts[0].Attempt != wantAttempt ||
		status.Outputs["report"].Revision == nil {
		t.Fatalf("code-analysis terminal Run = %+v", status)
	}
	report, mediaType := download(
		t, client, baseURL+"/v1/runs/"+url.PathEscape(status.RunID)+"/outputs/report",
	)
	if string(report) != codeAnalysisReport || mediaType != "text/markdown" {
		t.Fatalf("code-analysis report = (%q, %q)", report, mediaType)
	}
	allocation := onlyRunAllocation(t, ctx, store, status.RunID)
	if allocation.RuntimeAgentInstanceID != wantRuntimeInstanceID {
		t.Fatalf("code-analysis Run %s used Runtime %s, want %s",
			status.RunID, allocation.RuntimeAgentInstanceID, wantRuntimeInstanceID)
	}
	return allocation
}

func codeAnalysisSourceArchive(t *testing.T) []byte {
	t.Helper()
	files := make([]struct{ name, content string }, 0, 48)
	files = append(files, struct{ name, content string }{
		name: "app.py",
		content: "MARKER = \"" + codeAnalysisSourceCanary + "\"\n\n" +
			"def sink():\n    return MARKER\n\n" +
			"def branch_a():\n    return sink()\n\n" +
			"def branch_b():\n    return sink()\n\n" +
			"def branch_c():\n    return sink()\n\n" +
			"def branch_d():\n    return sink()\n\n" +
			"def entry():\n    branch_a()\n    branch_b()\n    branch_c()\n    return branch_d()\n",
	})
	for index := 0; index < 40; index++ {
		files = append(files, struct{ name, content string }{
			name: fmt.Sprintf("private-path-canary/dup_%03d.py", index),
			content: fmt.Sprintf(
				"def %s():\n    return %d\n", codeAnalysisDuplicateSymbol, index,
			),
		})
	}
	var output bytes.Buffer
	writer := zip.NewWriter(&output)
	for _, file := range files {
		header := &zip.FileHeader{Name: file.name, Method: zip.Store}
		header.SetMode(0o600)
		header.SetModTime(time.Date(2026, 9, 2, 0, 0, 0, 0, time.UTC))
		entry, err := writer.CreateHeader(header)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := entry.Write([]byte(file.content)); err != nil {
			t.Fatal(err)
		}
	}
	if err := writer.Close(); err != nil {
		t.Fatal(err)
	}
	return output.Bytes()
}

func writeCodeAnalysisE2EConfiguration(t *testing.T, root string) {
	t.Helper()
	files := map[string]string{
		"instructions/code-analysis-e2e-planner.md": "Delegate the bounded analysis objective once and return its report.\n",
		"instructions/code-analysis-e2e-worker.md":  "Inspect the current project with the selected structural tools and publish analysis/report as Markdown.\n",
		"agent-templates/code_analysis_shallow_e2e.yaml": `apiVersion: contractor/v1alpha1
kind: AgentTemplate
metadata: {name: code_analysis_shallow_e2e, version: "1"}
spec:
  description: Exercises portable structural workspace analysis
  runtime: adk@1
  instructions: {ref: instructions/code-analysis-e2e-worker.md}
  modelPolicy: domain_worker@1
  toolsets:
    - {ref: code-analysis@1, tools: [list_symbols, search_def]}
    - {ref: text-artifacts@1, tools: [write_text_artifact]}
  sandboxProfile: local-workdir@1
`,
		"agent-templates/code_analysis_graph_e2e.yaml": `apiVersion: contractor/v1alpha1
kind: AgentTemplate
metadata: {name: code_analysis_graph_e2e, version: "1"}
spec:
  description: Exercises the complete local graph surface and overlay invalidation
  runtime: adk@1
  instructions: {ref: instructions/code-analysis-e2e-worker.md}
  modelPolicy: domain_worker@1
  toolsets:
    - ref: code-analysis@1
      tools: [attack_surface, complexity_hotspots, entrypoint_paths_to, find_callees, find_callers, find_symbol, functions_that_raise, graph_summary, list_symbols, paths_between, search_def]
    - {ref: edit-files@1, tools: [edit]}
    - {ref: text-artifacts@1, tools: [write_text_artifact]}
  sandboxProfile: local-workdir@1
`,
		"workflows/code_analysis_shallow_e2e.yaml": codeAnalysisWorkflowYAML(
			"code-analysis-shallow-e2e", "code_analysis_shallow_e2e@1",
		),
		"workflows/code_analysis_graph_e2e.yaml": codeAnalysisWorkflowYAML(
			"code-analysis-graph-e2e", "code_analysis_graph_e2e@1",
		),
	}
	for relative, content := range files {
		path := filepath.Join(root, filepath.FromSlash(relative))
		if err := os.WriteFile(path, []byte(content), 0o600); err != nil {
			t.Fatalf("write code-analysis E2E config %s: %v", relative, err)
		}
	}
}

func codeAnalysisWorkflowYAML(name, template string) string {
	return fmt.Sprintf(`apiVersion: contractor/v1alpha1
kind: Workflow
metadata: {name: %s, version: "1"}
spec:
  parameters: {}
  inputs:
    source: {required: true, mediaTypes: [application/zip]}
  outputs:
    report: {required: true, mediaTypes: [text/markdown]}
  executionConfig:
    workers: {llmGateway: local-litellm@1, credential: development-worker}
  entryStage: analyze
  stages:
    analyze:
      objective: Inspect the exact effective workspace and publish a bounded report
      instructions: {ref: instructions/code-analysis-e2e-planner.md}
      planner: passthrough@1
      agents:
        analyst: {template: %s, namespace: analysis}
      context:
        artifacts:
          source: {namespace: inputs, name: source, required: true}
        workspace:
          mode: overlay
          sources:
            - {artifact: source, target: ""}
      result:
        artifacts:
          report:
            required: true
            mediaTypes: [text/markdown]
            from: {namespace: analysis, name: report}
      workflowOutputs: {report: report}
      on:
        succeeded: {succeed: {}}
        failed: {fail: {}}
        interrupted: {fail: {}}
`, name, template)
}

func assertCodeAnalysisRetainedExecutionSafe(
	t *testing.T,
	ctx context.Context,
	pool *pgxpool.Pool,
	runID string,
) {
	t.Helper()
	queries := []string{
		`SELECT COALESCE(runtime_config_snapshot::text || state_reason_code || state_reason_message, '')
FROM workflow_runs WHERE run_id = $1`,
		`SELECT COALESCE(string_agg(event::text, E'\n'), '') FROM planner_events WHERE run_id = $1`,
		`SELECT COALESCE(string_agg(data::text, E'\n'), '') FROM workflow_run_events WHERE run_id = $1`,
		`SELECT COALESCE(string_agg(session.state::text, E'\n'), '')
FROM planner_sessions AS session JOIN stage_executions AS execution
ON execution.stage_execution_id = session.stage_execution_id WHERE execution.run_id = $1`,
		`SELECT COALESCE(string_agg(report.report::text, E'\n'), '')
FROM planner_execution_reports AS report JOIN stage_executions AS execution
ON execution.stage_execution_id = report.stage_execution_id WHERE execution.run_id = $1`,
		`SELECT COALESCE(string_agg(report.report::text, E'\n'), '')
FROM allocation_execution_reports AS report JOIN stage_executions AS execution
ON execution.stage_execution_id = report.stage_execution_id WHERE execution.run_id = $1`,
		`SELECT COALESCE(string_agg(metrics.metrics::text || metrics.summary::text, E'\n'), '')
FROM stage_metrics AS metrics JOIN stage_executions AS execution
ON execution.stage_execution_id = metrics.stage_execution_id WHERE execution.run_id = $1`,
	}
	for _, query := range queries {
		var retained string
		if err := pool.QueryRow(ctx, query, runID).Scan(&retained); err != nil {
			t.Fatalf("read retained code-analysis execution surface: %v", err)
		}
		for _, canary := range []string{
			codeAnalysisSourceCanary, codeAnalysisDuplicateSymbol,
			codeAnalysisReplacementSymbol, codeAnalysisEditedPath,
			publicToken, llmGatewayToken,
		} {
			if strings.Contains(retained, canary) {
				t.Fatalf("retained execution surface for %s exposed code-analysis canary", runID)
			}
		}
	}
}

func assertCodeAnalysisOperationalSurfacesSafe(
	t *testing.T,
	operations []string,
	processes []*childProcess,
	privatePaths ...string,
) {
	t.Helper()
	canaries := []string{
		codeAnalysisSourceCanary, codeAnalysisDuplicateSymbol,
		codeAnalysisReplacementSymbol, codeAnalysisEditedPath,
		publicToken, llmGatewayToken,
	}
	canaries = append(canaries, privatePaths...)
	for _, canary := range canaries {
		for _, snapshot := range operations {
			if strings.Contains(snapshot, canary) {
				t.Fatalf("Operations snapshot exposed code-analysis canary")
			}
		}
		for _, process := range processes {
			if strings.Contains(process.logs.redacted(), canary) {
				t.Fatalf("%s logs exposed code-analysis canary", process.name)
			}
		}
	}
}
