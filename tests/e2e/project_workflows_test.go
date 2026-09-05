//go:build e2e

package e2e

import (
	"archive/zip"
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/localpki"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/jackc/pgx/v5/pgxpool"
	"go.yaml.in/yaml/v4"
)

const (
	projectOriginSecret = "PROJECT_ORIGIN_SECRET_MUST_NEVER_REACH_MODEL_OR_SAFE_STATE"
	projectSource       = `from fastapi import Depends, FastAPI, HTTPException
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
	lastAllocationID       string
	runtimeInstanceByStage map[string]string
}

type projectResourceResponse struct {
	ProjectID   string    `json:"projectId"`
	Kind        string    `json:"kind"`
	Name        string    `json:"name"`
	Description string    `json:"description"`
	Lifecycle   string    `json:"lifecycle"`
	Revision    string    `json:"revision"`
	CreatedAt   time.Time `json:"createdAt"`
	UpdatedAt   time.Time `json:"updatedAt"`
	HTTPTarget  *struct {
		URL        string `json:"url"`
		Credential *struct {
			CredentialID string `json:"credentialId"`
			Kind         string `json:"kind"`
		} `json:"credential,omitempty"`
	} `json:"httpTarget,omitempty"`
}

type persistedBinding struct {
	revision  string
	mediaType string
	frozen    bool
}

func TestProjectWorkspaceLifecycleAcrossProductionProcesses(t *testing.T) {
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
		LeafOptions: leaf, URI: "urn:contractor:control-plane:project-workflows-e2e",
	})
	if err != nil {
		t.Fatalf("issue Control Plane certificate: %v", err)
	}
	openAPIIdentity, err := generator.IssueAgent(pkiRoot, "project-openapi-agent", leaf)
	if err != nil {
		t.Fatalf("issue OpenAPI Runtime Agent certificate: %v", err)
	}
	likeC4Identity, err := generator.IssueAgent(pkiRoot, "project-likec4-agent", leaf)
	if err != nil {
		t.Fatalf("issue LikeC4 Runtime Agent certificate: %v", err)
	}

	openAPIBin, likeC4Bin, validatorLog := installSplitDomainValidators(t, temporaryRoot)
	gateway := newBlockedDomainGateway(llmGatewayToken, domainGatewayStages())
	gateway.forbid(projectOriginSecret)
	t.Cleanup(gateway.close)
	configRoot := stageE2EConfiguration(
		t, filepath.Join(repositoryRoot, "configs"),
		filepath.Join(temporaryRoot, "configs"), gateway.URL(),
	)
	publicAddress := freeAddress(t)
	privateAddress := freeAddress(t)
	openAPIRuntimeAddress, likeC4RuntimeAddress := freeAddress(t), freeAddress(t)
	publicBaseURL := "http://" + publicAddress
	privateBaseURL := "https://" + privateAddress
	openAPIRuntimeBaseURL := "https://" + openAPIRuntimeAddress
	likeC4RuntimeBaseURL := "https://" + likeC4RuntimeAddress
	userID := "project-e2e-user-" + randomHex(t, 8)
	localAuthFile := writeE2ELocalAuth(t, temporaryRoot, userID)
	masterKeyFile := filepath.Join(temporaryRoot, "credential-master-key")
	masterKey := base64.StdEncoding.EncodeToString(bytes.Repeat([]byte{0x37}, 32))
	if err := os.WriteFile(masterKeyFile, []byte(masterKey), 0o600); err != nil {
		t.Fatal(err)
	}
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
		"CONTRACTOR_LOCAL_AUTH_FILE":         localAuthFile,
		"CONTRACTOR_BROWSER_ORIGINS":         "https://ui.contractor.invalid",
	}
	startServer := func(name string) *childProcess {
		return startProcess(
			t, name, repositoryRoot, serverEnvironment,
			serverBinary, "serve", "--credential-master-key-file", masterKeyFile,
		)
	}
	server := startServer("Go Server")
	initialServer := server

	publicClient := &http.Client{Timeout: 8 * time.Second}
	waitForHTTP(t, ctx, server, publicClient, publicBaseURL+"/readyz", http.StatusOK)
	python := filepath.Join(repositoryRoot, "runtime", ".venv", "bin", "python")
	if info, statErr := os.Stat(python); statErr != nil || info.IsDir() {
		t.Fatalf("Python Runtime environment is missing at %s; run 'cd runtime && uv sync --locked'", python)
	}
	openAPIWorkRoot := filepath.Join(temporaryRoot, "runtime-openapi-work")
	openAPIWorkspaceRoot := filepath.Join(temporaryRoot, "runtime-openapi-workspaces")
	openAPIRuntime := startProjectRuntime(
		t, "Python Runtime Agent OpenAPI", repositoryRoot, python, privateBaseURL,
		openAPIRuntimeAddress, openAPIWorkRoot, openAPIWorkspaceRoot, openAPIBin,
		caPaths.Certificate, openAPIIdentity,
	)
	likeC4WorkRoot := filepath.Join(temporaryRoot, "runtime-likec4-work")
	likeC4WorkspaceRoot := filepath.Join(temporaryRoot, "runtime-likec4-workspaces")
	likeC4Runtime := startProjectRuntime(
		t, "Python Runtime Agent LikeC4", repositoryRoot, python, privateBaseURL,
		likeC4RuntimeAddress, likeC4WorkRoot, likeC4WorkspaceRoot, likeC4Bin,
		caPaths.Certificate, likeC4Identity,
	)
	controlClient := newMTLSClient(t, caPaths.Certificate, controlPlanePaths)
	waitForHTTP(t, ctx, openAPIRuntime, controlClient, openAPIRuntimeBaseURL+"/healthz", http.StatusOK)
	waitForHTTP(t, ctx, likeC4Runtime, controlClient, likeC4RuntimeBaseURL+"/healthz", http.StatusOK)
	waitForProcessLog(t, ctx, openAPIRuntime, "runtime agent registered")
	waitForProcessLog(t, ctx, likeC4Runtime, "runtime agent registered")
	registered, _ := waitForObservedRuntimeAgents(
		t, ctx, server, []*childProcess{openAPIRuntime, likeC4Runtime}, publicClient, publicBaseURL,
		func(agents []observedRuntimeAgent) bool {
			_, hasOpenAPI := findRuntimeWithTool(agents, "openapi@1", "validate_openapi")
			_, hasLikeC4 := findRuntimeWithTool(agents, "likec4@1", "validate_likec4")
			return len(agents) == 2 && hasOpenAPI && hasLikeC4
		},
	)
	openAPIAgent, hasOpenAPI := findRuntimeWithTool(registered, "openapi@1", "validate_openapi")
	likeC4Agent, hasLikeC4 := findRuntimeWithTool(registered, "likec4@1", "validate_likec4")
	if !hasOpenAPI || !hasLikeC4 || openAPIAgent.InstanceID == likeC4Agent.InstanceID {
		t.Fatalf("specialist Runtime capabilities are not distinct: %+v", registered)
	}

	project := createProjectResource(t, publicClient, publicBaseURL)
	createProjectOriginCredential(t, publicClient, publicBaseURL)
	project = updateProjectTarget(
		t, publicClient, publicBaseURL, project.ProjectID, project.Revision,
		map[string]any{"url": "https://service.example.test/api", "credential": map[string]string{
			"credentialId": "project-origin", "kind": "http-origin-bearer@1",
		}}, http.StatusOK,
	)
	if project.Revision != "2" || project.HTTPTarget == nil ||
		project.HTTPTarget.Credential == nil ||
		project.HTTPTarget.Credential.CredentialID != "project-origin" {
		t.Fatalf("attached Project target = %+v", project)
	}
	updateProjectTarget(
		t, publicClient, publicBaseURL, project.ProjectID, "1",
		map[string]any{"url": "https://stale.example.test"}, http.StatusPreconditionFailed,
	)

	sourceBytes := projectSourceArchive(t)
	source := uploadProjectScopeArtifact(
		t, publicClient, publicBaseURL, project.ProjectID,
		"sources", "service", "application/zip", sourceBytes,
	)
	openAPISeedRef := uploadProjectScopeArtifact(
		t, publicClient, publicBaseURL, project.ProjectID,
		"openapi", "seed", "application/yaml", []byte(openAPISeed),
	)
	likeC4SeedRef := uploadProjectScopeArtifact(
		t, publicClient, publicBaseURL, project.ProjectID,
		"likec4", "seed", "text/plain", []byte(likeC4Seed),
	)

	pool, err := pgxpool.New(ctx, isolateURL)
	if err != nil {
		t.Fatalf("open assertion database: %v", err)
	}
	t.Cleanup(pool.Close)

	openAPIRunRequest := projectRunRequest("openapi-from-workspace@4", map[string]artifactRef{
		"source": source, "existing_openapi": openAPISeedRef,
	})
	openAPIRunID := postProjectRun(
		t, publicClient, publicBaseURL, project.ProjectID,
		"project-e2e-openapi-v4", openAPIRunRequest, false,
	)
	select {
	case <-gateway.blockedRequest():
	case <-time.After(30 * time.Second):
		t.Fatalf("OpenAPI Run did not reach the deterministic model gateway\nserver:\n%s", server.logs.redacted())
	}
	assertProjectQueueMembership(t, publicClient, publicBaseURL, project, openAPIRunID)
	replayedOpenAPIRunID := postProjectRun(
		t, publicClient, publicBaseURL, project.ProjectID,
		"project-e2e-openapi-v4", openAPIRunRequest, true,
	)
	if replayedOpenAPIRunID != openAPIRunID {
		t.Fatalf("Project Run response-loss replay returned %q, want %q", replayedOpenAPIRunID, openAPIRunID)
	}
	gateway.releaseBlockedRequest()
	openAPIStatus := waitForDomainRun(
		t, ctx, server, openAPIRuntime, gateway, publicClient, publicBaseURL, openAPIRunID,
		likeC4Runtime,
	)
	assertRunProjectAndPublications(
		t, openAPIStatus, project.ProjectID,
		map[string]string{
			"openapi": "published", "openapi_validation_report": "published",
			"workspace_state": "published", "workspace_diff": "published",
		},
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
		publicBaseURL+"/v1/runs/"+url.PathEscape(openAPIRunID)+"/outputs/openapi_validation_report",
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
			"outputs/openapi":         "application/yaml", "outputs/openapi_validation_report": "text/markdown",
			"outputs/workspace_state": "application/vnd.contractor.workspace-overlay+json",
			"outputs/workspace_diff":  "text/x-diff",
		},
		map[string]string{
			"openapi": "openapi", "openapi_validation_report": "validation_report",
			"workspace_state": "workspace_state", "workspace_diff": "workspace_diff",
		},
	)
	waitForRuntimeReleased(
		t, ctx, openAPIRuntime, controlClient, openAPIRuntimeBaseURL,
		openAPIEvidence.lastAllocationID, openAPIWorkRoot,
	)
	assertSpecialistPlacement(
		t, openAPIEvidence, []string{"openapi_build", "openapi_validate"}, openAPIAgent.InstanceID,
	)
	assertProjectInputLineage(t, ctx, pool, project.ProjectID, openAPIRunID, "source", source)
	assertProjectPublishedOutputLineage(t, ctx, pool, project.ProjectID, openAPIRunID, openAPIStatus)
	assertProjectArtifactBytes(
		t, publicClient, publicBaseURL, project.ProjectID,
		artifactRef{Namespace: "outputs", Name: "openapi"}, openAPIBytes, "application/yaml",
	)

	openAPIRuntime.stop(t)
	server.stop(t)
	server = startServer("Go Server after durable restart")
	waitForHTTP(t, ctx, server, publicClient, publicBaseURL+"/readyz", http.StatusOK)
	waitForObservedRuntimeAgents(
		t, ctx, server, []*childProcess{likeC4Runtime}, publicClient, publicBaseURL,
		func(agents []observedRuntimeAgent) bool {
			agent, found := findRuntimeWithTool(agents, "likec4@1", "validate_likec4")
			return found && agent.InstanceID == likeC4Agent.InstanceID && agent.ConfirmedLeaseUntil != nil
		},
	)

	likeC4RunRequest := projectRunRequest("likec4-from-workspace@4", map[string]artifactRef{
		"source": source, "existing_likec4": likeC4SeedRef,
	})
	likeC4RunID := postProjectRun(
		t, publicClient, publicBaseURL, project.ProjectID,
		"project-e2e-likec4-v4", likeC4RunRequest, false,
	)
	likeC4Status := waitForDomainRun(
		t, ctx, server, likeC4Runtime, gateway, publicClient, publicBaseURL, likeC4RunID,
	)
	assertRunProjectAndPublications(
		t, likeC4Status, project.ProjectID,
		map[string]string{
			"likec4": "published", "likec4_validation_report": "published",
			"workspace_state": "already_present", "workspace_diff": "already_present",
		},
	)
	assertProjectRunStatus(
		t, likeC4Status,
		[]string{"dependency_discovery", "project_discovery", "likec4_build", "likec4_validate"},
		[]int64{5, 5, 11, 9},
	)
	likeC4Bytes, likeC4MediaType := download(
		t, publicClient, publicBaseURL+"/v1/runs/"+url.PathEscape(likeC4RunID)+"/outputs/likec4",
	)
	assertLikeC4Output(t, likeC4Bytes, likeC4MediaType)
	likeC4Report, reportMediaType := download(
		t, publicClient,
		publicBaseURL+"/v1/runs/"+url.PathEscape(likeC4RunID)+"/outputs/likec4_validation_report",
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
			"outputs/likec4":         "text/vnd.likec4", "outputs/likec4_validation_report": "text/markdown",
			"outputs/workspace_state": "application/vnd.contractor.workspace-overlay+json",
			"outputs/workspace_diff":  "text/x-diff",
		},
		map[string]string{
			"likec4": "architecture", "likec4_validation_report": "validation_report",
			"workspace_state": "workspace_state", "workspace_diff": "workspace_diff",
		},
	)
	waitForRuntimeReleased(
		t, ctx, likeC4Runtime, controlClient, likeC4RuntimeBaseURL,
		likeC4Evidence.lastAllocationID, likeC4WorkRoot,
	)
	assertSpecialistPlacement(
		t, likeC4Evidence, []string{"likec4_build", "likec4_validate"}, likeC4Agent.InstanceID,
	)
	if openAPIAgent.InstanceID == likeC4Agent.InstanceID {
		t.Fatal("OpenAPI and LikeC4 specialist work used one Runtime identity")
	}
	if !workRootEmpty(openAPIWorkspaceRoot) || !workRootEmpty(likeC4WorkspaceRoot) {
		t.Fatal("Project Workflow Runtime retained an allocation-private workspace")
	}
	assertProjectInputLineage(t, ctx, pool, project.ProjectID, likeC4RunID, "source", source)
	assertProjectPublishedOutputLineage(t, ctx, pool, project.ProjectID, likeC4RunID, likeC4Status)
	var projectRunCount int
	if err := pool.QueryRow(
		ctx, `SELECT count(*) FROM workflow_runs WHERE project_id = $1`, project.ProjectID,
	).Scan(&projectRunCount); err != nil || projectRunCount != 2 {
		t.Fatalf("Project Run count after response-loss replay = %d (err=%v), want 2", projectRunCount, err)
	}
	assertProjectArtifactBytes(
		t, publicClient, publicBaseURL, project.ProjectID,
		artifactRef{Namespace: "outputs", Name: "likec4"}, likeC4Bytes, "text/vnd.likec4",
	)

	assertProjectArtifactBytes(
		t, publicClient, publicBaseURL, project.ProjectID, source, sourceBytes, "application/zip",
	)
	assertProjectArtifactBytes(
		t, publicClient, publicBaseURL, project.ProjectID,
		openAPISeedRef, []byte(openAPISeed), "application/yaml",
	)
	assertProjectArtifactBytes(
		t, publicClient, publicBaseURL, project.ProjectID,
		likeC4SeedRef, []byte(likeC4Seed), "text/plain",
	)
	assertValidatorInvocations(t, validatorLog, 3, 5)
	if gateway.CompletedStages() != 8 || gateway.Calls() != 58 || len(gateway.Failures()) != 0 {
		t.Fatalf("domain gateway stages/calls/failures = %d/%d/%v, want 8/58/none",
			gateway.CompletedStages(), gateway.Calls(), gateway.Failures())
	}
	if observations := gateway.Observations(); len(observations) != 58 {
		t.Fatalf("gateway observations = %d, want 58", len(observations))
	}

	assertRuntimeCredentialInUse(t, publicClient, publicBaseURL, project.ProjectID)
	project = updateProjectTarget(
		t, publicClient, publicBaseURL, project.ProjectID, project.Revision, nil, http.StatusOK,
	)
	if project.HTTPTarget != nil || project.Revision != "3" {
		t.Fatalf("detached Project target = %+v", project)
	}
	deleteRuntimeCredential(t, publicClient, publicBaseURL, http.StatusNoContent)
	assertProjectSecretNotRetained(
		t, ctx, pool, []*childProcess{initialServer, server, openAPIRuntime, likeC4Runtime},
	)

	for _, secret := range []string{publicToken, llmGatewayToken, projectOriginSecret} {
		if strings.Contains(server.logs.redacted(), secret) ||
			strings.Contains(openAPIRuntime.logs.redacted(), secret) ||
			strings.Contains(likeC4Runtime.logs.redacted(), secret) {
			t.Fatal("process logs contain a configured secret")
		}
	}
}

func startProjectRuntime(
	t *testing.T,
	name, repositoryRoot, python, privateBaseURL, address, workRoot, workspaceRoot, validatorBin, caFile string,
	identity localpki.Paths,
) *childProcess {
	t.Helper()
	baseURL := "https://" + address
	return startProcess(
		t, name, filepath.Join(repositoryRoot, "runtime"),
		map[string]string{
			// Deliberately exclude the host's /usr/local and user npm bins so
			// each process advertises exactly one external validator.
			"PATH":             validatorBin + string(os.PathListSeparator) + "/usr/bin:/bin",
			"PYTHONUNBUFFERED": "1",
		},
		python, "-m", "contractor_runtime",
		"--control-plane-url", privateBaseURL,
		"--advertised-control-url", baseURL,
		"--advertised-a2a-url", baseURL,
		"--ca-file", caFile,
		"--certificate-file", identity.Certificate,
		"--private-key-file", identity.PrivateKey,
		"--listen", address,
		"--work-root", workRoot,
		"--workspace-storage", "local",
		"--workspace-work-root", workspaceRoot,
		"--request-timeout-seconds", "12",
		"--shutdown-grace-seconds", "5",
	)
}

func createProjectResource(t *testing.T, client *http.Client, baseURL string) projectResourceResponse {
	t.Helper()
	body := []byte(`{"kind":"project","name":"Widget service","description":"OpenAPI and LikeC4 workspace"}`)
	request, err := http.NewRequest(http.MethodPost, baseURL+"/v1/projects", bytes.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	request.Header.Set("Authorization", "Bearer "+publicToken)
	request.Header.Set("Content-Type", "application/json")
	request.Header.Set("Idempotency-Key", "project-workspace-e2e")
	response := do(t, client, request, http.StatusCreated)
	defer response.Body.Close()
	var project projectResourceResponse
	decodeResponse(t, response, &project)
	if project.ProjectID == "" || project.Name != "Widget service" || project.Lifecycle != "active" ||
		project.Revision != "1" ||
		response.Header.Get("ETag") != `"1"` {
		t.Fatalf("create Project response = %+v headers=%v", project, response.Header)
	}
	return project
}

func createProjectOriginCredential(t *testing.T, client *http.Client, baseURL string) {
	t.Helper()
	body, err := json.Marshal(map[string]any{
		"credentialId": "project-origin", "kind": "http-origin-bearer@1",
		"material": map[string]string{"token": projectOriginSecret},
	})
	if err != nil {
		t.Fatal(err)
	}
	request, err := http.NewRequest(
		http.MethodPost, baseURL+"/v1/operations/runtime-credentials", bytes.NewReader(body),
	)
	if err != nil {
		t.Fatal(err)
	}
	request.Header.Set("Authorization", "Bearer "+publicToken)
	request.Header.Set("Content-Type", "application/json")
	request.Header.Set("Idempotency-Key", "project-origin-create")
	response := do(t, client, request, http.StatusCreated)
	defer response.Body.Close()
	encoded, err := io.ReadAll(io.LimitReader(response.Body, 1<<20))
	if err != nil {
		t.Fatal(err)
	}
	if bytes.Contains(encoded, []byte(projectOriginSecret)) || bytes.Contains(encoded, []byte(`"material"`)) {
		t.Fatalf("Runtime credential response exposed write-only material: %s", encoded)
	}
	var metadata struct {
		CredentialID string `json:"credentialId"`
		Kind         string `json:"kind"`
	}
	if err := json.Unmarshal(encoded, &metadata); err != nil || metadata.CredentialID != "project-origin" ||
		metadata.Kind != "http-origin-bearer@1" {
		t.Fatalf("Runtime credential metadata = (%+v, %v)", metadata, err)
	}
}

func updateProjectTarget(
	t *testing.T,
	client *http.Client,
	baseURL, projectID, revision string,
	target any,
	expectedStatus int,
) projectResourceResponse {
	t.Helper()
	body, err := json.Marshal(map[string]any{"httpTarget": target})
	if err != nil {
		t.Fatal(err)
	}
	request, err := http.NewRequest(
		http.MethodPatch, baseURL+"/v1/projects/"+url.PathEscape(projectID), bytes.NewReader(body),
	)
	if err != nil {
		t.Fatal(err)
	}
	request.Header.Set("Authorization", "Bearer "+publicToken)
	request.Header.Set("Content-Type", "application/json")
	request.Header.Set("If-Match", strconv.Quote(revision))
	response := do(t, client, request, expectedStatus)
	defer response.Body.Close()
	if expectedStatus != http.StatusOK {
		_, _ = io.Copy(io.Discard, io.LimitReader(response.Body, 1<<20))
		return projectResourceResponse{}
	}
	var project projectResourceResponse
	decodeResponse(t, response, &project)
	if response.Header.Get("ETag") != strconv.Quote(project.Revision) {
		t.Fatalf("Project ETag/body revision mismatch: headers=%v project=%+v", response.Header, project)
	}
	return project
}

func uploadProjectScopeArtifact(
	t *testing.T,
	client *http.Client,
	baseURL, projectID, namespace, name, mediaType string,
	data []byte,
) artifactRef {
	t.Helper()
	target := fmt.Sprintf(
		"%s/v1/projects/%s/artifacts/%s/%s", baseURL, url.PathEscape(projectID),
		url.PathEscape(namespace), url.PathEscape(name),
	)
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
	if payload.Artifact.Namespace != namespace || payload.Artifact.Name != name ||
		payload.Artifact.Revision == nil || payload.MediaType != mediaType ||
		payload.Size != int64(len(data)) {
		t.Fatalf("Project artifact upload %s/%s returned %+v", namespace, name, payload)
	}
	return payload.Artifact
}

func projectRunRequest(workflow string, inputs map[string]artifactRef) []byte {
	body, err := json.Marshal(map[string]any{
		"workflow": workflow,
		"parameters": map[string]string{
			"objective": "Model the implemented API, architecture, and trust boundaries",
		},
		"artifacts": inputs,
	})
	if err != nil {
		panic(err)
	}
	return body
}

func postProjectRun(
	t *testing.T,
	client *http.Client,
	baseURL, projectID, idempotencyKey string,
	body []byte,
	wantReplay bool,
) string {
	t.Helper()
	request, err := http.NewRequest(
		http.MethodPost, baseURL+"/v1/projects/"+url.PathEscape(projectID)+"/runs", bytes.NewReader(body),
	)
	if err != nil {
		t.Fatal(err)
	}
	request.Header.Set("Authorization", "Bearer "+publicToken)
	request.Header.Set("Content-Type", "application/json")
	request.Header.Set("Idempotency-Key", idempotencyKey)
	response := do(t, client, request, http.StatusAccepted)
	defer response.Body.Close()
	if got := response.Header.Get("Idempotency-Replayed"); (got == "true") != wantReplay {
		t.Fatalf("Project Run replay header = %q, want replay=%t", got, wantReplay)
	}
	var payload runCreateResponse
	decodeResponse(t, response, &payload)
	if payload.RunID == "" || payload.ProjectID == nil || *payload.ProjectID != projectID ||
		payload.State != "initializing" && payload.State != "running" {
		t.Fatalf("create Project Run response = %+v", payload)
	}
	return payload.RunID
}

func assertProjectQueueMembership(
	t *testing.T,
	client *http.Client,
	baseURL string,
	project projectResourceResponse,
	runID string,
) {
	t.Helper()
	request, err := http.NewRequest(http.MethodGet, baseURL+"/v1/queue?membership=project&limit=100", nil)
	if err != nil {
		t.Fatal(err)
	}
	request.Header.Set("Authorization", "Bearer "+publicToken)
	response := do(t, client, request, http.StatusOK)
	defer response.Body.Close()
	var page struct {
		Items []struct {
			RunID   string `json:"runId"`
			Project *struct {
				ProjectID string `json:"projectId"`
				Name      string `json:"name"`
			} `json:"project"`
		} `json:"items"`
	}
	body, err := io.ReadAll(io.LimitReader(response.Body, 1<<20))
	if err != nil || json.Unmarshal(body, &page) != nil {
		t.Fatalf("decode Project Queue response: %v body=%s", err, body)
	}
	for _, item := range page.Items {
		if item.RunID == runID && item.Project != nil &&
			item.Project.ProjectID == project.ProjectID && item.Project.Name == project.Name {
			return
		}
	}
	t.Fatalf("Project Run %s is absent from Project Queue: %+v", runID, page.Items)
}

func assertRunProjectAndPublications(
	t *testing.T,
	status runStatus,
	projectID string,
	want map[string]string,
) {
	t.Helper()
	if status.ProjectID == nil || *status.ProjectID != projectID {
		t.Fatalf("Run Project = %v, want %s", status.ProjectID, projectID)
	}
	if len(status.OutputPublications) != len(want) {
		t.Fatalf("output publications = %+v, want %v", status.OutputPublications, want)
	}
	seen := make(map[string]bool, len(status.OutputPublications))
	for _, publication := range status.OutputPublications {
		wantStatus, expected := want[publication.Output]
		if !expected || publication.Status != wantStatus || publication.Source.Revision == nil ||
			publication.ErrorCode != "" || publication.ErrorMessage != "" {
			t.Fatalf("unexpected output publication: %+v (want %v)", publication, want)
		}
		if publication.Status == "published" {
			if publication.Target == nil || publication.Target.Namespace != "outputs" ||
				publication.Target.Name != publication.Output || publication.Target.Revision == nil {
				t.Fatalf("published output has no exact Project target: %+v", publication)
			}
		} else if publication.Target != nil {
			t.Fatalf("already-present output unexpectedly claims a target: %+v", publication)
		}
		seen[publication.Output] = true
	}
	for output := range want {
		if !seen[output] {
			t.Fatalf("output %q has no publication result", output)
		}
	}
}

func assertSpecialistPlacement(
	t *testing.T,
	evidence projectRunEvidence,
	stages []string,
	wantInstanceID string,
) {
	t.Helper()
	for _, stage := range stages {
		if got := evidence.runtimeInstanceByStage[stage]; got != wantInstanceID {
			t.Fatalf("Stage %s used Runtime %q, want specialist %q", stage, got, wantInstanceID)
		}
	}
}

func assertProjectInputLineage(
	t *testing.T,
	ctx context.Context,
	pool *pgxpool.Pool,
	projectID, runID, slot string,
	source artifactRef,
) {
	t.Helper()
	if source.Revision == nil {
		t.Fatal("Project input lineage assertion requires an exact source")
	}
	var sourceScopeKind, sourceScopeID, sourceNamespace, sourceName, sourceRevision string
	err := pool.QueryRow(ctx, `
SELECT source_scope_kind, source_scope_id, source_namespace, source_name, source_revision
FROM artifact_lineage
WHERE target_scope_kind = 'run' AND target_scope_id = $1
  AND target_namespace = 'inputs' AND target_name = $2
  AND lineage_kind = 'input_fork'`, runID, slot).Scan(
		&sourceScopeKind, &sourceScopeID, &sourceNamespace, &sourceName, &sourceRevision,
	)
	if err != nil || sourceScopeKind != "project" || sourceScopeID != projectID ||
		sourceNamespace != source.Namespace || sourceName != source.Name || sourceRevision != *source.Revision {
		t.Fatalf("Project input lineage = %s/%s:%s/%s@%s (err=%v), want %s/%s@%s",
			sourceScopeKind, sourceScopeID, sourceNamespace, sourceName, sourceRevision, err,
			source.Namespace, source.Name, *source.Revision)
	}
}

func assertProjectPublishedOutputLineage(
	t *testing.T,
	ctx context.Context,
	pool *pgxpool.Pool,
	projectID, runID string,
	status runStatus,
) {
	t.Helper()
	for _, publication := range status.OutputPublications {
		if publication.Status != "published" {
			continue
		}
		if publication.Target == nil || publication.Target.Revision == nil || publication.Source.Revision == nil {
			t.Fatalf("publication is not exact: %+v", publication)
		}
		var sourceScopeKind, sourceScopeID, sourceNamespace, sourceName, sourceRevision string
		err := pool.QueryRow(ctx, `
SELECT source_scope_kind, source_scope_id, source_namespace, source_name, source_revision
FROM artifact_lineage
WHERE target_scope_kind = 'project' AND target_scope_id = $1
  AND target_namespace = 'outputs' AND target_name = $2 AND target_revision = $3
  AND lineage_kind = 'project_output_publish'`,
			projectID, publication.Output, *publication.Target.Revision,
		).Scan(&sourceScopeKind, &sourceScopeID, &sourceNamespace, &sourceName, &sourceRevision)
		if err != nil || sourceScopeKind != "run" || sourceScopeID != runID ||
			sourceNamespace != publication.Source.Namespace || sourceName != publication.Source.Name ||
			sourceRevision != *publication.Source.Revision {
			t.Fatalf("Project output lineage for %s = %s/%s:%s/%s@%s (err=%v), want %+v",
				publication.Output, sourceScopeKind, sourceScopeID, sourceNamespace, sourceName,
				sourceRevision, err, publication.Source)
		}
	}
}

func assertProjectArtifactBytes(
	t *testing.T,
	client *http.Client,
	baseURL, projectID string,
	ref artifactRef,
	want []byte,
	wantMediaType string,
) {
	t.Helper()
	target := fmt.Sprintf(
		"%s/v1/projects/%s/artifacts/%s/%s", baseURL, url.PathEscape(projectID),
		url.PathEscape(ref.Namespace), url.PathEscape(ref.Name),
	)
	if ref.Revision != nil {
		target += "?revision=" + url.QueryEscape(*ref.Revision)
	}
	data, mediaType := download(t, client, target)
	if !bytes.Equal(data, want) || mediaType != wantMediaType {
		t.Fatalf("Project artifact %s/%s changed: media=%q bytes_equal=%v",
			ref.Namespace, ref.Name, mediaType, bytes.Equal(data, want))
	}
}

func assertRuntimeCredentialInUse(
	t *testing.T,
	client *http.Client,
	baseURL, projectID string,
) {
	t.Helper()
	request, err := http.NewRequest(
		http.MethodDelete, baseURL+"/v1/operations/runtime-credentials/project-origin", nil,
	)
	if err != nil {
		t.Fatal(err)
	}
	request.Header.Set("Authorization", "Bearer "+publicToken)
	request.Header.Set("Idempotency-Key", "project-origin-delete-attached")
	response := do(t, client, request, http.StatusConflict)
	defer response.Body.Close()
	body, err := io.ReadAll(io.LimitReader(response.Body, 1<<20))
	if err != nil || !bytes.Contains(body, []byte(projectID)) || bytes.Contains(body, []byte(projectOriginSecret)) {
		t.Fatalf("attached credential delete response = %s (err=%v)", body, err)
	}
}

func deleteRuntimeCredential(t *testing.T, client *http.Client, baseURL string, expectedStatus int) {
	t.Helper()
	request, err := http.NewRequest(
		http.MethodDelete, baseURL+"/v1/operations/runtime-credentials/project-origin", nil,
	)
	if err != nil {
		t.Fatal(err)
	}
	request.Header.Set("Authorization", "Bearer "+publicToken)
	request.Header.Set("Idempotency-Key", "project-origin-delete-detached")
	response := do(t, client, request, expectedStatus)
	response.Body.Close()
}

func assertProjectSecretNotRetained(
	t *testing.T,
	ctx context.Context,
	pool *pgxpool.Pool,
	processes []*childProcess,
) {
	t.Helper()
	var ciphertext []byte
	if err := pool.QueryRow(ctx, `
SELECT ciphertext FROM runtime_credentials WHERE credential_id = 'project-origin'`,
	).Scan(&ciphertext); err != nil {
		t.Fatal(err)
	}
	if bytes.Contains(ciphertext, []byte(projectOriginSecret)) {
		t.Fatal("encrypted Runtime credential row contains plaintext")
	}
	for name, query := range map[string]string{
		"Project": `SELECT coalesce(string_agg(row_to_json(p)::text, ''), '') FROM projects p`,
		"Run":     `SELECT coalesce(string_agg(row_to_json(r)::text, ''), '') FROM workflow_runs r`,
		"allocation": `SELECT coalesce(string_agg(runtime_configuration::text, ''), '')
FROM stage_allocations`,
	} {
		var retained string
		if err := pool.QueryRow(ctx, query).Scan(&retained); err != nil {
			t.Fatalf("scan %s safe state: %v", name, err)
		}
		if strings.Contains(retained, projectOriginSecret) {
			t.Fatalf("%s safe state retained Project origin plaintext", name)
		}
	}
	rows, err := pool.Query(ctx, `SELECT payload FROM artifact_blobs`)
	if err != nil {
		t.Fatal(err)
	}
	defer rows.Close()
	for rows.Next() {
		var payload []byte
		if err := rows.Scan(&payload); err != nil {
			t.Fatal(err)
		}
		if bytes.Contains(payload, []byte(projectOriginSecret)) {
			t.Fatal("artifact payload retained Project origin plaintext")
		}
	}
	if err := rows.Err(); err != nil {
		t.Fatal(err)
	}
	for _, process := range processes {
		if strings.Contains(process.logs.redacted(), projectOriginSecret) {
			t.Fatalf("%s logs retained Project origin plaintext", process.name)
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

func installSplitDomainValidators(t *testing.T, root string) (string, string, string) {
	t.Helper()
	combined, logPath := installDomainValidators(t, root)
	openAPIBin := filepath.Join(root, "openapi-validator-bin")
	likeC4Bin := filepath.Join(root, "likec4-validator-bin")
	for _, directory := range []string{openAPIBin, likeC4Bin} {
		if err := os.Mkdir(directory, 0o700); err != nil {
			t.Fatal(err)
		}
	}
	for source, target := range map[string]string{
		filepath.Join(combined, "vacuum"): filepath.Join(openAPIBin, "vacuum"),
		filepath.Join(combined, "likec4"): filepath.Join(likeC4Bin, "likec4"),
	} {
		content, err := os.ReadFile(source)
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(target, content, 0o700); err != nil {
			t.Fatal(err)
		}
	}
	return openAPIBin, likeC4Bin, logPath
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
	additionalRuntimes ...*childProcess,
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
		processes := append([]*childProcess{server, runtimeProcess}, additionalRuntimes...)
		for _, process := range processes {
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
	runtimeInstanceByStage := make(map[string]string, len(executions))
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
		if allocation.RuntimeAgentInstanceID == "" {
			t.Fatalf("Stage %s allocation has no Runtime identity", execution.StageName)
		}
		runtimeInstanceByStage[execution.StageName] = allocation.RuntimeAgentInstanceID
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
	if len(runtimeInstanceByStage) != len(stages) || lastAllocationID == "" {
		t.Fatal("durable executions did not identify the Runtime allocation")
	}
	return projectRunEvidence{
		lastAllocationID: lastAllocationID, runtimeInstanceByStage: runtimeInstanceByStage,
	}
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
