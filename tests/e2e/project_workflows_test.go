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
	"sort"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/localpki"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/jackc/pgx/v5/pgxpool"
	"go.yaml.in/yaml/v4"
)

const (
	projectSource = `from fastapi import Depends, FastAPI, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
import httpx

app = FastAPI(title="Widget Service")
bearer = HTTPBearer()

@app.get("/widgets/{widget_id}")
async def get_widget(
    widget_id: str,
    credentials: HTTPAuthorizationCredentials = Depends(bearer),
) -> dict[str, object]:
    if credentials.credentials != "fixture-token":
        raise HTTPException(status_code=401, detail="invalid token")
    async with httpx.AsyncClient(base_url="https://inventory.example") as client:
        response = await client.get(f"/v1/items/{widget_id}")
    return {"id": widget_id, "available": response.status_code == 200}
`
	openAPISeed = `openapi: 3.0.3
info:
  title: Widget Service
  version: 1.0.0
paths: {}
`
	likeC4Seed = `specification {
  element actor
  element system
  element container
  element external
  relationship calls
  tag public
  tag external
}
`
)

type projectRunEvidence struct {
	lastAllocationID  string
	runtimeInstanceID string
}

type persistedBinding struct {
	revision  string
	mediaType string
	frozen    bool
}

func TestProjectWorkflowsFromWorkspace(t *testing.T) {
	if testing.Short() {
		t.Skip("end-to-end process test")
	}
	databaseURL := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if databaseURL == "" {
		t.Fatal("CONTRACTOR_TEST_DATABASE_URL is required")
	}
	repositoryRoot := repoRoot(t)
	temporaryRoot := t.TempDir()
	ctx, cancel := context.WithTimeout(context.Background(), 230*time.Second)
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
		LeafOptions: leaf, URI: "urn:contractor:control-plane:project-workflows-e2e",
	})
	if err != nil {
		t.Fatalf("issue Control Plane certificate: %v", err)
	}
	agentPaths, err := generator.IssueAgent(pkiRoot, "project-workflows-e2e-agent", leaf)
	if err != nil {
		t.Fatalf("issue Runtime Agent certificate: %v", err)
	}

	validatorBin, validatorLog := installDomainValidators(t, temporaryRoot)
	gateway := newDomainGateway(llmGatewayToken)
	t.Cleanup(gateway.close)
	configRoot := stageE2EConfiguration(
		t, filepath.Join(repositoryRoot, "configs"),
		filepath.Join(temporaryRoot, "configs"), gateway.URL(),
	)
	publicAddress := freeAddress(t)
	privateAddress := freeAddress(t)
	runtimeAddress := freeAddress(t)
	publicBaseURL := "http://" + publicAddress
	privateBaseURL := "https://" + privateAddress
	runtimeBaseURL := "https://" + runtimeAddress
	userID := "project-e2e-user-" + randomHex(t, 8)
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

	publicClient := &http.Client{Timeout: 8 * time.Second}
	waitForHTTP(t, ctx, server, publicClient, publicBaseURL+"/readyz", http.StatusOK)
	python := filepath.Join(repositoryRoot, "runtime", ".venv", "bin", "python")
	if info, statErr := os.Stat(python); statErr != nil || info.IsDir() {
		t.Fatalf("Python Runtime environment is missing at %s; run 'cd runtime && uv sync --locked'", python)
	}
	workRoot := filepath.Join(temporaryRoot, "runtime-work")
	workspaceRoot := filepath.Join(temporaryRoot, "runtime-project-workspaces")
	runtimeProcess := startProcess(
		t, "Python Runtime Agent", filepath.Join(repositoryRoot, "runtime"),
		map[string]string{
			"PATH":             validatorBin + string(os.PathListSeparator) + os.Getenv("PATH"),
			"PYTHONUNBUFFERED": "1",
		},
		python, "-m", "contractor_runtime",
		"--control-plane-url", privateBaseURL,
		"--advertised-control-url", runtimeBaseURL,
		"--advertised-a2a-url", runtimeBaseURL,
		"--ca-file", caPaths.Certificate,
		"--certificate-file", agentPaths.Certificate,
		"--private-key-file", agentPaths.PrivateKey,
		"--listen", runtimeAddress,
		"--work-root", workRoot,
		"--workspace-storage", "local",
		"--workspace-work-root", workspaceRoot,
		"--request-timeout-seconds", "12",
		"--shutdown-grace-seconds", "5",
	)
	controlClient := newMTLSClient(t, caPaths.Certificate, controlPlanePaths)
	waitForHTTP(t, ctx, runtimeProcess, controlClient, runtimeBaseURL+"/healthz", http.StatusOK)

	sourceBytes := projectSourceArchive(t)
	source := uploadProjectArtifact(
		t, publicClient, publicBaseURL, "project-source", "application/zip", sourceBytes,
	)
	openAPISeedRef := uploadProjectArtifact(
		t, publicClient, publicBaseURL, "project-openapi-seed", "application/yaml", []byte(openAPISeed),
	)
	likeC4SeedRef := uploadProjectArtifact(
		t, publicClient, publicBaseURL, "project-likec4-seed", "text/plain", []byte(likeC4Seed),
	)

	pool, err := pgxpool.New(ctx, isolateURL)
	if err != nil {
		t.Fatalf("open assertion database: %v", err)
	}
	t.Cleanup(pool.Close)

	openAPIRunID := createProjectRun(t, publicClient, publicBaseURL, "openapi-from-workspace@3", map[string]artifactRef{
		"source": source, "existing_openapi": openAPISeedRef,
	})
	openAPIStatus := waitForDomainRun(
		t, ctx, server, runtimeProcess, gateway, publicClient, publicBaseURL, openAPIRunID,
	)
	assertProjectRunStatus(
		t, openAPIStatus,
		[]string{"dependency_discovery", "project_discovery", "openapi_build", "openapi_validate"},
		[]int64{5, 5, 9, 9},
	)
	openAPIBytes, openAPIMediaType := download(
		t, publicClient, publicBaseURL+"/v1/runs/"+url.PathEscape(openAPIRunID)+"/outputs/openapi",
	)
	assertOpenAPIOutput(t, openAPIBytes, openAPIMediaType)
	openAPIReport, reportMediaType := download(
		t, publicClient,
		publicBaseURL+"/v1/runs/"+url.PathEscape(openAPIRunID)+"/outputs/validation_report",
	)
	if reportMediaType != "text/markdown" || !strings.Contains(string(openAPIReport), "Final valid: true") {
		t.Fatalf("unexpected OpenAPI validation report (%q): %s", reportMediaType, openAPIReport)
	}
	openAPIEvidence := assertProjectRunDurable(
		t, ctx, pool, openAPIRunID,
		[]string{"dependency_discovery", "project_discovery", "openapi_build", "openapi_validate"},
		[]int64{5, 5, 9, 9},
		map[string]string{
			"inputs/source": "application/zip", "inputs/existing_openapi": "application/yaml",
			"analysis/dependencies": "text/markdown", "analysis/project": "text/markdown",
			"analysis/workspace_state": "application/vnd.contractor.workspace-overlay+json",
			"analysis/workspace_diff":  "text/x-diff",
			"openapi/openapi":          "application/yaml", "openapi/validation-report": "text/markdown",
			"openapi/workspace_state": "application/vnd.contractor.workspace-overlay+json",
			"openapi/workspace_diff":  "text/x-diff",
			"outputs/openapi":         "application/yaml", "outputs/validation_report": "text/markdown",
			"outputs/workspace_state": "application/vnd.contractor.workspace-overlay+json",
			"outputs/workspace_diff":  "text/x-diff",
		},
		map[string]string{
			"openapi": "openapi", "validation_report": "validation_report",
			"workspace_state": "workspace_state", "workspace_diff": "workspace_diff",
		},
	)
	waitForRuntimeReleased(
		t, ctx, runtimeProcess, controlClient, runtimeBaseURL,
		openAPIEvidence.lastAllocationID, workRoot,
	)

	likeC4RunID := createProjectRun(t, publicClient, publicBaseURL, "likec4-from-workspace@3", map[string]artifactRef{
		"source": source, "existing_likec4": likeC4SeedRef,
	})
	likeC4Status := waitForDomainRun(
		t, ctx, server, runtimeProcess, gateway, publicClient, publicBaseURL, likeC4RunID,
	)
	assertProjectRunStatus(
		t, likeC4Status,
		[]string{"dependency_discovery", "project_discovery", "likec4_build", "likec4_validate"},
		[]int64{5, 5, 11, 9},
	)
	likeC4Bytes, likeC4MediaType := download(
		t, publicClient, publicBaseURL+"/v1/runs/"+url.PathEscape(likeC4RunID)+"/outputs/architecture",
	)
	assertLikeC4Output(t, likeC4Bytes, likeC4MediaType)
	likeC4Report, reportMediaType := download(
		t, publicClient,
		publicBaseURL+"/v1/runs/"+url.PathEscape(likeC4RunID)+"/outputs/validation_report",
	)
	if reportMediaType != "text/markdown" || !strings.Contains(string(likeC4Report), "Final valid: true") {
		t.Fatalf("unexpected LikeC4 validation report (%q): %s", reportMediaType, likeC4Report)
	}
	likeC4Evidence := assertProjectRunDurable(
		t, ctx, pool, likeC4RunID,
		[]string{"dependency_discovery", "project_discovery", "likec4_build", "likec4_validate"},
		[]int64{5, 5, 11, 9},
		map[string]string{
			"inputs/source": "application/zip", "inputs/existing_likec4": "text/plain",
			"analysis/dependencies": "text/markdown", "analysis/project": "text/markdown",
			"analysis/workspace_state": "application/vnd.contractor.workspace-overlay+json",
			"analysis/workspace_diff":  "text/x-diff",
			"likec4/architecture":      "text/vnd.likec4", "likec4/validation-report": "text/markdown",
			"likec4/workspace_state": "application/vnd.contractor.workspace-overlay+json",
			"likec4/workspace_diff":  "text/x-diff",
			"skills/likec4":          "application/vnd.contractor.agent-skill+zip",
			"outputs/architecture":   "text/vnd.likec4", "outputs/validation_report": "text/markdown",
			"outputs/workspace_state": "application/vnd.contractor.workspace-overlay+json",
			"outputs/workspace_diff":  "text/x-diff",
		},
		map[string]string{
			"architecture": "architecture", "validation_report": "validation_report",
			"workspace_state": "workspace_state", "workspace_diff": "workspace_diff",
		},
	)
	waitForRuntimeReleased(
		t, ctx, runtimeProcess, controlClient, runtimeBaseURL,
		likeC4Evidence.lastAllocationID, workRoot,
	)
	if !workRootEmpty(workspaceRoot) {
		t.Fatal("project Workflow Runtime retained an allocation-private workspace")
	}
	if openAPIEvidence.runtimeInstanceID != likeC4Evidence.runtimeInstanceID {
		t.Fatalf("workflows used different Runtime slots: %q != %q",
			openAPIEvidence.runtimeInstanceID, likeC4Evidence.runtimeInstanceID)
	}

	assertUserArtifactUnchanged(t, publicClient, publicBaseURL, source, sourceBytes, "application/zip")
	assertUserArtifactUnchanged(
		t, publicClient, publicBaseURL, openAPISeedRef, []byte(openAPISeed), "application/yaml",
	)
	assertUserArtifactUnchanged(
		t, publicClient, publicBaseURL, likeC4SeedRef, []byte(likeC4Seed), "text/plain",
	)
	assertValidatorInvocations(t, validatorLog, 3, 5)
	if gateway.CompletedStages() != 8 || gateway.Calls() != 58 || len(gateway.Failures()) != 0 {
		t.Fatalf("domain gateway stages/calls/failures = %d/%d/%v, want 8/58/none",
			gateway.CompletedStages(), gateway.Calls(), gateway.Failures())
	}
	if observations := gateway.Observations(); len(observations) != 58 {
		t.Fatalf("gateway observations = %d, want 58", len(observations))
	}

	for _, secret := range []string{publicToken, llmGatewayToken} {
		if strings.Contains(server.logs.redacted(), secret) ||
			strings.Contains(runtimeProcess.logs.redacted(), secret) {
			t.Fatal("process logs contain a configured secret")
		}
	}
}

func projectSourceArchive(t *testing.T) []byte {
	t.Helper()
	files := map[string]string{
		"app.py": projectSource,
		"pyproject.toml": `[project]
name = "widget-service"
version = "1.0.0"
dependencies = [
  "fastapi>=0.116",
  "httpx>=0.28",
]
`,
		"README.md": "# Widget Service\n\nRun the FastAPI application with an ASGI server.\n",
	}
	names := make([]string, 0, len(files))
	for name := range files {
		names = append(names, name)
	}
	sort.Strings(names)
	var output bytes.Buffer
	writer := zip.NewWriter(&output)
	for _, name := range names {
		header := &zip.FileHeader{Name: name, Method: zip.Store}
		header.SetMode(0o600)
		header.SetModTime(time.Date(2026, 8, 30, 0, 0, 0, 0, time.UTC))
		entry, err := writer.CreateHeader(header)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := entry.Write([]byte(files[name])); err != nil {
			t.Fatal(err)
		}
	}
	if err := writer.Close(); err != nil {
		t.Fatal(err)
	}
	return output.Bytes()
}

func installDomainValidators(t *testing.T, root string) (string, string) {
	t.Helper()
	bin := filepath.Join(root, "validator-bin")
	if err := os.MkdirAll(bin, 0o700); err != nil {
		t.Fatal(err)
	}
	logPath := filepath.Join(root, "validator-invocations.log")
	quotedLog := shellSingleQuote(logPath)
	vacuum := `#!/bin/sh
if [ "$1" = "version" ]; then
  printf 'vacuum fixture version\n'
  exit 0
fi
if [ "$1" != "spectral-report" ] || [ "$2" != "-i" ] || [ "$3" != "-o" ]; then
  exit 2
fi
payload=$(cat)
case "$payload" in
  *'/widgets/{widget_id}'*) ;;
  *) exit 2 ;;
esac
printf 'vacuum\n' >> ` + quotedLog + `
printf '[]\n'
`
	likeC4 := `#!/bin/sh
if [ "$1" = "version" ]; then
  printf 'likec4 fixture version\n'
  exit 0
fi
if [ "$1" != "validate" ] || [ "$2" != "--json" ] || [ "$3" != "--no-layout" ] || [ "$4" != "--file" ]; then
  exit 2
fi
grep -Fq 'specification {' "$5" || exit 2
printf 'likec4\n' >> ` + quotedLog + `
printf '{"valid":true,"errors":[],"stats":{"fixture":true}}\n'
`
	for name, content := range map[string]string{"vacuum": vacuum, "likec4": likeC4} {
		path := filepath.Join(bin, name)
		if err := os.WriteFile(path, []byte(content), 0o700); err != nil {
			t.Fatal(err)
		}
	}
	return bin, logPath
}

func shellSingleQuote(value string) string {
	return "'" + strings.ReplaceAll(value, "'", "'\\''") + "'"
}

func uploadProjectArtifact(
	t *testing.T,
	client *http.Client,
	baseURL, name, mediaType string,
	data []byte,
) artifactRef {
	t.Helper()
	target := baseURL + "/v1/artifacts/projects/" + url.PathEscape(name)
	request, err := http.NewRequest(http.MethodPut, target, bytes.NewReader(data))
	if err != nil {
		t.Fatal(err)
	}
	request.Header.Set("Authorization", "Bearer "+publicToken)
	request.Header.Set("Content-Type", mediaType)
	request.Header.Set("If-None-Match", "*")
	response := do(t, client, request, http.StatusCreated)
	defer response.Body.Close()
	var payload struct {
		Artifact  artifactRef `json:"artifact"`
		MediaType string      `json:"mediaType"`
		Size      int64       `json:"size"`
	}
	decodeResponse(t, response, &payload)
	if payload.Artifact.Namespace != "projects" || payload.Artifact.Name != name ||
		payload.Artifact.Revision == nil || payload.MediaType != mediaType ||
		payload.Size != int64(len(data)) {
		t.Fatalf("upload %q returned %+v", name, payload)
	}
	return payload.Artifact
}

func createProjectRun(
	t *testing.T,
	client *http.Client,
	baseURL, workflow string,
	inputs map[string]artifactRef,
) string {
	t.Helper()
	body, err := json.Marshal(map[string]any{
		"workflow": workflow,
		"parameters": map[string]string{
			"objective": "Model the implemented API, architecture, and trust boundaries",
		},
		"artifacts": inputs,
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
	request.Header.Set("Idempotency-Key", "project-e2e-"+strings.ReplaceAll(workflow, "@", "-v"))
	response := do(t, client, request, http.StatusAccepted)
	defer response.Body.Close()
	var payload runCreateResponse
	decodeResponse(t, response, &payload)
	if payload.RunID == "" || payload.State != "initializing" && payload.State != "running" {
		t.Fatalf("create %s Run response = %+v", workflow, payload)
	}
	return payload.RunID
}

func waitForDomainRun(
	t *testing.T,
	ctx context.Context,
	server, runtimeProcess *childProcess,
	gateway *domainGateway,
	client *http.Client,
	baseURL, runID string,
) runStatus {
	t.Helper()
	ticker := time.NewTicker(150 * time.Millisecond)
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
			} else {
				response.Body.Close()
				t.Fatalf("poll Run returned HTTP %d", response.StatusCode)
			}
			response.Body.Close()
			switch status.State {
			case "succeeded":
				return status
			case "failed", "cancelled":
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

func assertProjectRunStatus(
	t *testing.T,
	status runStatus,
	stages []string,
	modelCalls []int64,
) {
	t.Helper()
	if status.State != "succeeded" || len(status.Attempts) != len(stages) {
		t.Fatalf("unexpected terminal Run: %+v", status)
	}
	for index, attempt := range status.Attempts {
		if attempt.Stage != stages[index] || attempt.Attempt != 1 || attempt.State != "succeeded" {
			t.Fatalf("attempt %d = %+v, want %s attempt 1 succeeded", index, attempt, stages[index])
		}
		if attempt.Metrics == nil || attempt.Metrics.ModelCalls != modelCalls[index] ||
			attempt.Metrics.ToolCalls != modelCalls[index]-1 || !attempt.Metrics.ReportsComplete {
			t.Fatalf("attempt %s metrics = %+v, want model/tool=%d/%d and complete reports",
				attempt.Stage, attempt.Metrics, modelCalls[index], modelCalls[index]-1)
		}
	}
}

func assertProjectRunDurable(
	t *testing.T,
	ctx context.Context,
	pool *pgxpool.Pool,
	runID string,
	stages []string,
	modelCalls []int64,
	expectedBindings map[string]string,
	outputSlots map[string]string,
) projectRunEvidence {
	t.Helper()
	store := runstore.NewPostgresStore(pool)
	executions, err := store.ListStageExecutions(ctx, runID)
	if err != nil || len(executions) != len(stages) {
		t.Fatalf("StageExecutions = (%+v, %v), want %d", executions, err, len(stages))
	}
	instanceID := ""
	lastAllocationID := ""
	for index, execution := range executions {
		if execution.StageName != stages[index] || execution.Attempt != 1 ||
			execution.State != runstore.StageSucceeded || execution.CandidateResult == nil ||
			execution.AcceptedResult == nil || execution.PlannerSessionID == nil ||
			execution.PlannerInvocationID == nil || execution.FinalizationID == nil ||
			execution.TerminalAt == nil {
			t.Fatalf("execution %d is incomplete: %+v", index, execution)
		}
		allocations, allocationErr := store.ListStageAllocations(ctx, execution.StageExecutionID)
		if allocationErr != nil || len(allocations) != 1 {
			t.Fatalf("allocations for %s = (%+v, %v)", execution.StageName, allocations, allocationErr)
		}
		allocation := allocations[0]
		if instanceID == "" {
			instanceID = allocation.RuntimeAgentInstanceID
		}
		if allocation.RuntimeAgentInstanceID != instanceID {
			t.Fatalf("Stage %s changed Runtime slot: %q != %q",
				execution.StageName, allocation.RuntimeAgentInstanceID, instanceID)
		}
		lastAllocationID = allocation.AllocationID
		reports, reportErr := store.ListStageExecutionReports(ctx, execution.StageExecutionID)
		if reportErr != nil || len(reports) != 1 ||
			!reports[0].Report.Worker.Complete || !reports[0].Report.Runtime.Complete {
			t.Fatalf("reports for %s = (%+v, %v), want one complete envelope",
				execution.StageName, reports, reportErr)
		}
		metrics := reports[0].Report.Worker.Metrics
		if metrics.ModelCalls == nil || *metrics.ModelCalls != modelCalls[index] {
			t.Fatalf("Worker model calls for %s = %v, want %d",
				execution.StageName, metrics.ModelCalls, modelCalls[index])
		}
		var workerToolCalls int64
		for _, tool := range metrics.Tools {
			if tool.Calls != nil {
				workerToolCalls += *tool.Calls
			}
		}
		if workerToolCalls != modelCalls[index]-2 {
			t.Fatalf("Worker tool calls for %s = %d, want %d; metrics=%+v",
				execution.StageName, workerToolCalls, modelCalls[index]-2, metrics.Tools)
		}
		budget := metrics.WorkerBudget
		if budget == nil || budget.MaxModelCalls != 24 || budget.MaxToolCalls != 96 ||
			budget.MaxTotalTokens != 250000 || budget.ObservedModelCalls != modelCalls[index] ||
			budget.ObservedToolCalls != workerToolCalls ||
			budget.ObservedTotalTokens != modelCalls[index]*16 ||
			budget.TokenUsageUnavailable != 0 || budget.Exhausted != nil {
			t.Fatalf("Worker budget for %s = %+v", execution.StageName, budget)
		}
	}
	decisions, err := store.ListStageTransitionDecisions(ctx, runID)
	if err != nil || len(decisions) != len(stages) {
		t.Fatalf("transition decisions = (%+v, %v), want %d", decisions, err, len(stages))
	}
	for index, decision := range decisions {
		want := runstore.StageTransitionNext
		if index == len(decisions)-1 {
			want = runstore.StageTransitionSucceed
		}
		if decision.SourceExecutionID != executions[index].StageExecutionID || decision.Action != want {
			t.Fatalf("transition %d = %+v, want action %s", index, decision, want)
		}
	}

	bindings := loadPersistedBindings(t, ctx, pool, runID)
	if len(bindings) != len(expectedBindings) {
		t.Fatalf("Run bindings = %+v, want exactly %+v", bindings, expectedBindings)
	}
	for key, mediaType := range expectedBindings {
		binding, ok := bindings[key]
		if !ok || binding.revision == "" || binding.mediaType != mediaType {
			t.Fatalf("binding %s = %+v, want media type %s", key, binding, mediaType)
		}
		wantFrozen := strings.HasPrefix(key, "outputs/")
		if binding.frozen != wantFrozen {
			t.Fatalf("binding %s frozen=%v, want %v", key, binding.frozen, wantFrozen)
		}
	}
	for _, name := range []string{"dependencies", "project"} {
		var revisions int
		if err := pool.QueryRow(ctx, `
SELECT count(*)
FROM artifact_binding_revisions
WHERE scope_kind = 'run' AND scope_id = $1 AND namespace = 'analysis' AND name = $2`,
			runID, name,
		).Scan(&revisions); err != nil || revisions != 1 {
			t.Fatalf("analysis/%s revisions = %d (err=%v), want one", name, revisions, err)
		}
	}
	var inputForks, outputBinds int
	expectedInputForks := 0
	for key := range expectedBindings {
		if strings.HasPrefix(key, "inputs/") || strings.HasPrefix(key, "skills/") {
			expectedInputForks++
		}
	}
	if err := pool.QueryRow(ctx, `
SELECT count(*) FILTER (WHERE lineage_kind = 'input_fork'),
       count(*) FILTER (WHERE lineage_kind = 'output_bind')
FROM artifact_lineage
WHERE target_scope_kind = 'run' AND target_scope_id = $1`, runID,
	).Scan(&inputForks, &outputBinds); err != nil ||
		inputForks != expectedInputForks || outputBinds != len(outputSlots) {
		t.Fatalf("lineage input/output = %d/%d (err=%v), want %d/%d",
			inputForks, outputBinds, err, expectedInputForks, len(outputSlots))
	}

	final := executions[len(executions)-1].AcceptedResult
	for output, resultSlot := range outputSlots {
		outputRef, ok := final.Artifacts[resultSlot]
		if !ok || outputRef.Revision == nil {
			t.Fatalf("accepted final result has no exact %s: %+v", resultSlot, final)
		}
		binding := bindings["outputs/"+output]
		if binding.revision == *outputRef.Revision {
			t.Fatalf("output %s reused the source binding revision %s instead of creating a frozen output revision",
				output, binding.revision)
		}
		var sourceNamespace, sourceName, sourceRevision string
		if err := pool.QueryRow(ctx, `
SELECT source_namespace, source_name, source_revision
FROM artifact_lineage
WHERE target_scope_kind = 'run' AND target_scope_id = $1
  AND target_namespace = 'outputs' AND target_name = $2
  AND lineage_kind = 'output_bind'`, runID, output,
		).Scan(&sourceNamespace, &sourceName, &sourceRevision); err != nil ||
			sourceNamespace != outputRef.Namespace || sourceName != outputRef.Name ||
			sourceRevision != *outputRef.Revision {
			t.Fatalf("output %s lineage = %s/%s@%s (err=%v), want %+v",
				output, sourceNamespace, sourceName, sourceRevision, err, outputRef)
		}
	}
	if instanceID == "" || lastAllocationID == "" {
		t.Fatal("durable executions did not identify the Runtime allocation")
	}
	return projectRunEvidence{lastAllocationID: lastAllocationID, runtimeInstanceID: instanceID}
}

func loadPersistedBindings(
	t *testing.T,
	ctx context.Context,
	pool *pgxpool.Pool,
	runID string,
) map[string]persistedBinding {
	t.Helper()
	rows, err := pool.Query(ctx, `
SELECT b.namespace, b.name, b.current_revision, b.frozen, v.media_type
FROM artifact_bindings AS b
JOIN artifact_binding_revisions AS r
  ON r.scope_kind = b.scope_kind AND r.scope_id = b.scope_id
 AND r.namespace = b.namespace AND r.name = b.name AND r.revision = b.current_revision
JOIN artifact_versions AS v ON v.version_id = r.version_id
WHERE b.scope_kind = 'run' AND b.scope_id = $1
ORDER BY b.namespace, b.name`, runID)
	if err != nil {
		t.Fatal(err)
	}
	defer rows.Close()
	result := make(map[string]persistedBinding)
	for rows.Next() {
		var namespace, name string
		var binding persistedBinding
		if err := rows.Scan(
			&namespace, &name, &binding.revision, &binding.frozen, &binding.mediaType,
		); err != nil {
			t.Fatal(err)
		}
		result[namespace+"/"+name] = binding
	}
	if err := rows.Err(); err != nil {
		t.Fatal(err)
	}
	return result
}

func assertOpenAPIOutput(t *testing.T, data []byte, mediaType string) {
	t.Helper()
	if mediaType != "application/yaml" || bytes.Equal(data, []byte(openAPISeed)) {
		t.Fatalf("OpenAPI output media/content was not advanced: %q", mediaType)
	}
	var document map[string]any
	if err := yaml.Unmarshal(data, &document); err != nil {
		t.Fatalf("parse OpenAPI output: %v\n%s", err, data)
	}
	paths, ok := document["paths"].(map[string]any)
	if !ok {
		t.Fatalf("OpenAPI paths are absent: %+v", document)
	}
	path, ok := paths["/widgets/{widget_id}"].(map[string]any)
	if !ok {
		t.Fatalf("fixture path is absent: %+v", paths)
	}
	method, ok := path["get"].(map[string]any)
	if !ok {
		t.Fatalf("fixture GET operation is absent: %+v", path)
	}
	responses, ok := method["responses"].(map[string]any)
	if !ok || responses["200"] == nil || responses["401"] == nil {
		t.Fatalf("fixture responses are absent: %+v", method)
	}
	evidence, ok := path["x-path-files"].([]any)
	if !ok || len(evidence) != 1 || evidence[0] != "app.py" {
		t.Fatalf("fixture provenance = %+v, want app.py", path["x-path-files"])
	}
}

func assertLikeC4Output(t *testing.T, data []byte, mediaType string) {
	t.Helper()
	if mediaType != "text/vnd.likec4" || bytes.Equal(data, []byte(likeC4Seed)) {
		t.Fatalf("LikeC4 output media/content was not advanced: %q", mediaType)
	}
	content := string(data)
	for _, fragment := range []string{
		"API Client", "Widget Service", "Inventory Service", "widgets.api -[calls]-> inventory",
		"Evidence: app.py:8-17", "views {", "view index", "autoLayout LeftRight",
	} {
		if !strings.Contains(content, fragment) {
			t.Fatalf("LikeC4 output does not contain %q:\n%s", fragment, content)
		}
	}
}

func assertUserArtifactUnchanged(
	t *testing.T,
	client *http.Client,
	baseURL string,
	ref artifactRef,
	want []byte,
	wantMediaType string,
) {
	t.Helper()
	if ref.Revision == nil {
		t.Fatal("assert exact User artifact requires a revision")
	}
	target := fmt.Sprintf(
		"%s/v1/artifacts/%s/%s?revision=%s", baseURL,
		url.PathEscape(ref.Namespace), url.PathEscape(ref.Name), url.QueryEscape(*ref.Revision),
	)
	data, mediaType := download(t, client, target)
	if !bytes.Equal(data, want) || mediaType != wantMediaType {
		t.Fatalf("User artifact %s/%s changed: media=%q bytes_equal=%v",
			ref.Namespace, ref.Name, mediaType, bytes.Equal(data, want))
	}
}

func assertValidatorInvocations(t *testing.T, path string, vacuum, likeC4 int) {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	counts := map[string]int{}
	for _, line := range strings.Split(strings.TrimSpace(string(data)), "\n") {
		counts[line]++
	}
	if counts["vacuum"] != vacuum || counts["likec4"] != likeC4 || len(counts) != 2 {
		t.Fatalf("validator invocations = %v, want vacuum=%d likec4=%d", counts, vacuum, likeC4)
	}
}
