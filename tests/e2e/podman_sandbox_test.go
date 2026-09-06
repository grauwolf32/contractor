//go:build e2e

package e2e

import (
	"archive/zip"
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/localpki"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/jackc/pgx/v5/pgxpool"
)

func TestPodmanSandboxAcrossProductionProcesses(t *testing.T) {
	databaseURL, image := os.Getenv("CONTRACTOR_TEST_DATABASE_URL"), os.Getenv("CONTRACTOR_TEST_PODMAN_IMAGE")
	if databaseURL == "" || image == "" || !strings.Contains(image, "@sha256:") {
		t.Fatal("CONTRACTOR_TEST_DATABASE_URL and preinstalled digest-pinned CONTRACTOR_TEST_PODMAN_IMAGE are required")
	}
	root, temporary := repoRoot(t), t.TempDir()
	ctx, cancel := context.WithTimeout(context.Background(), 180*time.Second)
	defer cancel()
	database := isolatedDatabase(t, ctx, databaseURL)
	binary := filepath.Join(temporary, "contractor-server")
	runChecked(t, root, nil, "go", "build", "-o", binary, "./cmd/contractor-server")
	runChecked(t, root, map[string]string{"CONTRACTOR_DATABASE_URL": database}, binary, "migrate")
	generator := localpki.Generator{}
	pkiRoot := filepath.Join(temporary, "pki")
	ca, err := generator.InitCA(pkiRoot, false)
	if err != nil {
		t.Fatal(err)
	}
	leaf := localpki.LeafOptions{IPAddresses: []net.IP{net.ParseIP("127.0.0.1")}}
	control, err := generator.IssueControlPlane(pkiRoot, localpki.ControlPlaneOptions{LeafOptions: leaf, URI: "urn:contractor:control-plane:podman-e2e"})
	if err != nil {
		t.Fatal(err)
	}
	agent, err := generator.IssueAgent(pkiRoot, "podman-e2e", leaf)
	if err != nil {
		t.Fatal(err)
	}
	report := []byte("{\"passed\": 3, \"status\": \"ok\"}\n")
	gateway := newBlockedDomainGateway(llmGatewayToken, []domainGatewayStage{{
		name: "check", tools: []string{"edit", "exec_command", "read_file", "write_artifact"},
		steps: []domainGatewayStep{
			toolGatewayStep("read_file", fixedArguments(map[string]any{"path": "calculator.py"})),
			toolGatewayStep("edit", fixedArguments(map[string]any{"path": "calculator.py", "old": "return a - b", "new": "return a + b"})),
			toolGatewayStep("exec_command", fixedArguments(map[string]any{"command": "python3 -B check.py", "cwd": "", "timeout_seconds": 30})),
			toolGatewayStep("read_file", fixedArguments(map[string]any{"path": "report.json"})),
			toolGatewayStep("write_artifact", fixedArguments(map[string]any{"namespace": "builder", "name": "check_report", "media_type": "application/json", "data_base64": base64.StdEncoding.EncodeToString(report), "expected_revision": nil})),
			finalGatewayStep("Offline check passed", map[string]domainArtifactBinding{"report": {namespace: "builder", name: "check_report"}}),
		},
	}})
	t.Cleanup(gateway.close)
	config := stageE2EConfiguration(t, filepath.Join(root, "configs"), filepath.Join(temporary, "configs"), gateway.URL())
	publicAddress, privateAddress, runtimeAddress := freeAddress(t), freeAddress(t), freeAddress(t)
	publicURL, privateURL, runtimeURL := "http://"+publicAddress, "https://"+privateAddress, "https://"+runtimeAddress
	user := "podman-e2e-" + randomHex(t, 8)
	server := startProcess(t, "Podman Go Server", root, map[string]string{
		"CONTRACTOR_DATABASE_URL": database, "CONTRACTOR_CONFIG_ROOT": config,
		"CONTRACTOR_PUBLIC_LISTEN": publicAddress, "CONTRACTOR_PRIVATE_LISTEN": privateAddress,
		"CONTRACTOR_PRIVATE_URL": privateURL, "CONTRACTOR_CA_FILE": ca.Certificate,
		"CONTRACTOR_CONTROL_PLANE_CERT_FILE": control.Certificate, "CONTRACTOR_CONTROL_PLANE_KEY_FILE": control.PrivateKey,
		"CONTRACTOR_LLM_GATEWAY_TOKEN": llmGatewayToken, "CONTRACTOR_PUBLIC_USER_ID": user,
		"CONTRACTOR_PUBLIC_BEARER_TOKEN": publicToken, "CONTRACTOR_LOCAL_AUTH_FILE": writeE2ELocalAuth(t, temporary, user),
		"CONTRACTOR_BROWSER_ORIGINS": "https://ui.contractor.invalid",
	}, binary, "serve")
	client := &http.Client{Timeout: 5 * time.Second}
	waitForHTTP(t, ctx, server, client, publicURL+"/readyz", http.StatusOK)
	owner := "e2e-" + randomHex(t, 8)
	scratch, projectRoot := filepath.Join(temporary, "scratch"), filepath.Join(temporary, "project")
	runtime := startProcess(t, "Podman Python Runtime", filepath.Join(root, "runtime"), map[string]string{
		"PYTHONUNBUFFERED": "1", "CONTRACTOR_PODMAN_ENABLED": "true", "CONTRACTOR_PODMAN_IMAGE": image, "CONTRACTOR_PODMAN_OWNER": owner,
	}, filepath.Join(root, "runtime/.venv/bin/python"), "-m", "contractor_runtime",
		"--control-plane-url", privateURL, "--advertised-control-url", runtimeURL, "--advertised-a2a-url", runtimeURL,
		"--ca-file", ca.Certificate, "--certificate-file", agent.Certificate, "--private-key-file", agent.PrivateKey,
		"--listen", runtimeAddress, "--work-root", scratch, "--workspace-storage", "local", "--workspace-work-root", projectRoot,
		"--request-timeout-seconds", "12", "--shutdown-grace-seconds", "30")
	waitForProcessLog(t, ctx, runtime, "runtime agent registered")
	observed, _ := waitForObservedRuntimeAgents(t, ctx, server, []*childProcess{runtime}, client, publicURL,
		func(agents []observedRuntimeAgent) bool {
			_, ok := findRuntimeWithTool(agents, "code-execution@1", "exec_command")
			return ok
		})
	compatible, ok := findRuntimeWithTool(observed, "code-execution@1", "exec_command")
	if !ok {
		t.Fatal("verified Podman capacity missing")
	}
	project := createProjectResource(t, client, publicURL)
	var source bytes.Buffer
	bundle := zip.NewWriter(&source)
	for _, name := range []string{"calculator.py", "check.py"} {
		data, err := os.ReadFile(filepath.Join(root, "configs/fixtures/podman-python-check", name))
		if err != nil {
			t.Fatal(err)
		}
		member, err := bundle.Create(name)
		if err != nil {
			t.Fatal(err)
		}
		if _, err = member.Write(data); err != nil {
			t.Fatal(err)
		}
	}
	if err := bundle.Close(); err != nil {
		t.Fatal(err)
	}
	input := uploadProjectScopeArtifact(t, client, publicURL, project.ProjectID, "sources", "python", "application/zip", source.Bytes())
	body, err := json.Marshal(map[string]any{"workflow": "podman-python-check@1", "parameters": map[string]string{}, "artifacts": map[string]artifactRef{"source": input}})
	if err != nil {
		t.Fatal(err)
	}
	runID := postProjectRun(t, client, publicURL, project.ProjectID, "podman-sample", body, false)
	select {
	case <-gateway.blockedRequest():
	case <-ctx.Done():
		t.Fatal("Worker did not reach deterministic gateway")
	}
	// Model is paused: a single hydrated allocation container is already owned.
	containers := podmanOwnedIDs(t, ctx, owner)
	if len(containers) != 1 {
		t.Fatalf("allocation containers = %v", containers)
	}
	gateway.releaseBlockedRequest()
	status := waitForRun(t, ctx, server, runtime, gateway, client, publicURL, runID)
	if len(status.Outputs) != 1 || status.Outputs["report"].Revision == nil {
		t.Fatalf("ordinary output missing: %+v", status.Outputs)
	}
	pool, err := pgxpool.New(ctx, database)
	if err != nil {
		t.Fatal(err)
	}
	defer pool.Close()
	store := runstore.NewPostgresStore(pool)
	executions, err := store.ListStageExecutions(ctx, runID)
	if err != nil || len(executions) != 1 {
		t.Fatalf("executions: %v %v", executions, err)
	}
	allocations, err := store.ListStageAllocations(ctx, executions[0].StageExecutionID)
	if err != nil || len(allocations) != 1 || allocations[0].RuntimeAgentInstanceID != compatible.InstanceID {
		t.Fatalf("placement: %v %v", allocations, err)
	}
	assertProjectInputLineage(t, ctx, pool, project.ProjectID, runID, "source", input)
	assertRunProjectAndPublications(t, status, project.ProjectID, map[string]string{"report": "published"})
	assertProjectArtifactBytes(t, client, publicURL, project.ProjectID, artifactRef{Namespace: "outputs", Name: "report"}, report, "application/json")
	waitForObservedRuntimeAgents(t, ctx, server, []*childProcess{runtime}, client, publicURL,
		func(agents []observedRuntimeAgent) bool {
			for _, candidate := range agents {
				if candidate.InstanceID == compatible.InstanceID && candidate.SlotState == "idle" && candidate.AuthoritativeAllocationID == nil && candidate.CurrentAllocationID == nil {
					return true
				}
			}
			return false
		})
	entries, err := os.ReadDir(scratch)
	if err != nil {
		t.Fatal(err)
	}
	for _, entry := range entries {
		if entry.Name() != ".contractor-podman-owner-v1" {
			t.Fatalf("scratch not cleaned: %s", entry.Name())
		}
	}
	if ids := podmanOwnedIDs(t, ctx, owner); len(ids) != 0 {
		t.Fatalf("owned containers after release: %v", ids)
	}
	if paths, _ := filepath.Glob(filepath.Join(projectRoot, "workspace-*")); len(paths) != 0 {
		t.Fatalf("retained bind directories: %v", paths)
	}
	if failures := gateway.Failures(); len(failures) != 0 {
		t.Fatalf("gateway failures: %v", failures)
	}
}

func podmanOwnedIDs(t *testing.T, ctx context.Context, owner string) []string {
	t.Helper()
	command := exec.CommandContext(ctx, "podman", "--remote=false", "ps", "-a", "--filter", "label=io.contractor.sandbox.managed=1", "--filter", "label=io.contractor.sandbox.owner="+owner, "--format", "{{.ID}}")
	output, err := command.Output()
	if err != nil {
		t.Fatalf("inspect test-owned containers: %v", err)
	}
	return strings.Fields(string(output))
}
