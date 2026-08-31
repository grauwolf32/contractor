//go:build e2e

package e2e

import (
	"context"
	"encoding/json"
	"io"
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

const (
	capabilityEnvironmentCanary = "CAPABILITY_ENVIRONMENT_SECRET_CANARY"
	capabilityProbeOutputCanary = "CAPABILITY_PROBE_OUTPUT_SECRET_CANARY"
	capabilityPathCanary        = "capability-private-path-canary"
)

type observedRuntimeCapability struct {
	Ref   string   `json:"ref"`
	Tools []string `json:"tools"`
}

type observedRuntimeAgent struct {
	InstanceID                string                      `json:"instanceId"`
	SupportedRuntimes         []string                    `json:"supportedRuntimes"`
	SupportedToolsets         []observedRuntimeCapability `json:"supportedToolsets"`
	SupportedSandboxProfiles  []string                    `json:"supportedSandboxProfiles"`
	ObservedState             string                      `json:"observedState"`
	SlotState                 string                      `json:"slotState"`
	ConfirmedLeaseUntil       *time.Time                  `json:"confirmedLeaseUntil,omitempty"`
	CurrentAllocationID       *string                     `json:"currentAllocationId,omitempty"`
	AuthoritativeAllocationID *string                     `json:"authoritativeAllocationId,omitempty"`
}

func TestHeterogeneousRuntimeCapabilityPlacement(t *testing.T) {
	if testing.Short() {
		t.Skip("end-to-end process test")
	}
	databaseURL := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if databaseURL == "" {
		t.Fatal("CONTRACTOR_TEST_DATABASE_URL is required")
	}
	repositoryRoot := repoRoot(t)
	temporaryRoot := t.TempDir()
	ctx, cancel := context.WithTimeout(context.Background(), 130*time.Second)
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
		LeafOptions: leaf, URI: "urn:contractor:control-plane:capability-e2e",
	})
	if err != nil {
		t.Fatalf("issue Control Plane certificate: %v", err)
	}
	incompatibleIdentity, err := generator.IssueAgent(pkiRoot, "capability-incompatible", leaf)
	if err != nil {
		t.Fatalf("issue incompatible Runtime Agent certificate: %v", err)
	}
	compatibleIdentity, err := generator.IssueAgent(pkiRoot, "capability-compatible", leaf)
	if err != nil {
		t.Fatalf("issue compatible Runtime Agent certificate: %v", err)
	}

	gateway := newCapabilityGateway(llmGatewayToken)
	t.Cleanup(gateway.close)
	configRoot := stageE2EConfiguration(
		t, filepath.Join(repositoryRoot, "configs", "e2e"),
		filepath.Join(temporaryRoot, "configs"), gateway.URL(),
	)
	publicAddress := freeAddress(t)
	privateAddress := freeAddress(t)
	publicBaseURL := "http://" + publicAddress
	privateBaseURL := "https://" + privateAddress
	userID := "capability-e2e-user-" + randomHex(t, 8)
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
		"CONTRACTOR_LOCAL_AUTH_FILE":         writeE2ELocalAuth(t, temporaryRoot, userID),
		"CONTRACTOR_BROWSER_ORIGINS":         "https://ui.contractor.invalid",
	}, serverBinary, "serve")
	publicClient := &http.Client{Timeout: 5 * time.Second}
	waitForHTTP(t, ctx, server, publicClient, publicBaseURL+"/readyz", http.StatusOK)

	python := filepath.Join(repositoryRoot, "runtime", ".venv", "bin", "python")
	if info, statErr := os.Stat(python); statErr != nil || info.IsDir() {
		t.Fatalf("Python Runtime environment is missing at %s; run 'cd runtime && uv sync --locked'", python)
	}
	emptyBin := filepath.Join(temporaryRoot, "isolated-empty-bin")
	if err := os.Mkdir(emptyBin, 0o700); err != nil {
		t.Fatal(err)
	}
	incompatibleAddress := freeAddress(t)
	incompatibleBaseURL := "https://" + incompatibleAddress
	incompatibleWorkRoot := filepath.Join(temporaryRoot, "runtime-incompatible-work")
	incompatibleProcess := startCapabilityRuntime(
		t, "Python Runtime Agent without LikeC4", repositoryRoot, python,
		privateBaseURL, incompatibleBaseURL, incompatibleAddress, incompatibleWorkRoot,
		caPaths.Certificate, incompatibleIdentity, emptyBin,
	)

	controlClient := newMTLSClient(t, caPaths.Certificate, controlPlanePaths)
	waitForHTTP(
		t, ctx, incompatibleProcess, controlClient,
		incompatibleBaseURL+"/healthz", http.StatusOK,
	)
	initialAgents, initialBody := waitForObservedRuntimeAgents(
		t, ctx, server, []*childProcess{incompatibleProcess}, publicClient, publicBaseURL,
		func(agents []observedRuntimeAgent) bool {
			return len(agents) == 1 && agents[0].SlotState == "idle" &&
				agents[0].ConfirmedLeaseUntil != nil
		},
	)
	incompatibleAgent := initialAgents[0]
	likeC4WithoutValidator := runtimeToolset(incompatibleAgent, "likec4@1")
	if containsString(likeC4WithoutValidator.Tools, "validate_likec4") ||
		!containsString(likeC4WithoutValidator.Tools, "write_likec4") ||
		!containsString(incompatibleAgent.SupportedRuntimes, "adk@1") ||
		!containsString(incompatibleAgent.SupportedSandboxProfiles, "local-workdir@1") {
		t.Fatalf("incompatible Runtime capability snapshot = %+v", incompatibleAgent)
	}
	incompatibleFingerprint := runtimeCapabilityFingerprint(t, incompatibleAgent)

	uploaded := uploadInput(t, publicClient, publicBaseURL)
	runID := createWorkflowRun(
		t, publicClient, publicBaseURL, "capability-copy@1", "capability-placement-run", uploaded,
	)
	pool, err := pgxpool.New(ctx, isolateURL)
	if err != nil {
		t.Fatalf("open assertion database: %v", err)
	}
	t.Cleanup(pool.Close)
	store := runstore.NewPostgresStore(pool)
	waitingExecution := waitForUnallocatedPreparingExecution(t, ctx, store, runID)
	time.Sleep(750 * time.Millisecond)
	assertSameUnallocatedPreparingExecution(t, ctx, store, waitingExecution)
	if gateway.Calls() != 0 {
		t.Fatalf("Planner or Worker reached the Gateway while capacity was incompatible: %d calls", gateway.Calls())
	}
	waitingAgents, waitingBody := observedRuntimeAgents(t, publicClient, publicBaseURL)
	if len(waitingAgents) != 1 || waitingAgents[0].InstanceID != incompatibleAgent.InstanceID ||
		waitingAgents[0].SlotState != "idle" || waitingAgents[0].AuthoritativeAllocationID != nil {
		t.Fatalf("incompatible slot was reserved while Stage waited: %+v", waitingAgents)
	}

	fakeBin, validatorLog := installCapabilityLikeC4(t, temporaryRoot)
	compatibleAddress := freeAddress(t)
	compatibleBaseURL := "https://" + compatibleAddress
	compatibleWorkRoot := filepath.Join(temporaryRoot, "runtime-compatible-work")
	compatibleProcess := startCapabilityRuntime(
		t, "Python Runtime Agent with LikeC4", repositoryRoot, python,
		privateBaseURL, compatibleBaseURL, compatibleAddress, compatibleWorkRoot,
		caPaths.Certificate, compatibleIdentity, fakeBin,
	)
	waitForHTTP(
		t, ctx, compatibleProcess, controlClient,
		compatibleBaseURL+"/healthz", http.StatusOK,
	)
	registeredAgents, registeredBody := waitForObservedRuntimeAgents(
		t, ctx, server, []*childProcess{incompatibleProcess, compatibleProcess},
		publicClient, publicBaseURL,
		func(agents []observedRuntimeAgent) bool {
			if len(agents) != 2 {
				return false
			}
			for _, agent := range agents {
				if agent.ConfirmedLeaseUntil == nil {
					return false
				}
				if containsString(runtimeToolset(agent, "likec4@1").Tools, "validate_likec4") {
					return true
				}
			}
			return false
		},
	)
	compatibleAgent, ok := findRuntimeWithTool(
		registeredAgents, "likec4@1", "validate_likec4",
	)
	if !ok || compatibleAgent.InstanceID == incompatibleAgent.InstanceID {
		t.Fatalf("compatible Runtime Agent was not distinct: %+v", registeredAgents)
	}
	compatibleFingerprint := runtimeCapabilityFingerprint(t, compatibleAgent)

	completed := waitForRun(
		t, ctx, server, compatibleProcess, gateway, publicClient, publicBaseURL, runID,
	)
	if len(completed.Attempts) != 1 || completed.Attempts[0].Attempt != 1 ||
		completed.Attempts[0].StageExecutionID != waitingExecution.StageExecutionID ||
		completed.Attempts[0].State != "succeeded" {
		t.Fatalf("capacity wait consumed or replaced the Stage attempt: %+v", completed.Attempts)
	}
	output := completed.Outputs["result"]
	if output.Revision == nil {
		t.Fatalf("capability Run output is not exact: %+v", completed.Outputs)
	}
	outputBytes, outputMediaType := download(
		t, publicClient, publicBaseURL+"/v1/runs/"+url.PathEscape(runID)+"/outputs/result",
	)
	if string(outputBytes) != e2eInput || outputMediaType != e2eMediaType {
		t.Fatalf("capability Run output = (%q, %q)", outputBytes, outputMediaType)
	}
	allocationID := assertDurableExecution(t, ctx, pool, runID, output)
	allocations, err := store.ListStageAllocations(ctx, waitingExecution.StageExecutionID)
	if err != nil || len(allocations) != 1 ||
		allocations[0].RuntimeAgentInstanceID != compatibleAgent.InstanceID ||
		allocations[0].RuntimeAgentInstanceID == incompatibleAgent.InstanceID {
		t.Fatalf("capability allocation placement = (%+v, %v)", allocations, err)
	}
	waitForRuntimeReleased(
		t, ctx, compatibleProcess, controlClient, compatibleBaseURL, allocationID, compatibleWorkRoot,
	)

	finalAgents, finalBody := waitForObservedRuntimeAgents(
		t, ctx, server, []*childProcess{incompatibleProcess, compatibleProcess},
		publicClient, publicBaseURL,
		func(agents []observedRuntimeAgent) bool {
			if len(agents) != 2 {
				return false
			}
			for _, agent := range agents {
				if agent.SlotState != "idle" || agent.ObservedState != "idle" ||
					agent.CurrentAllocationID != nil || agent.AuthoritativeAllocationID != nil {
					return false
				}
			}
			return true
		},
	)
	finalIncompatible, found := findRuntimeByID(finalAgents, incompatibleAgent.InstanceID)
	if !found || runtimeCapabilityFingerprint(t, finalIncompatible) != incompatibleFingerprint {
		t.Fatalf("incompatible Runtime snapshot changed: %+v", finalIncompatible)
	}
	finalCompatible, found := findRuntimeByID(finalAgents, compatibleAgent.InstanceID)
	if !found || runtimeCapabilityFingerprint(t, finalCompatible) != compatibleFingerprint {
		t.Fatalf("compatible Runtime snapshot changed: %+v", finalCompatible)
	}
	if !workRootEmpty(incompatibleWorkRoot) || !workRootEmpty(compatibleWorkRoot) {
		t.Fatalf("Runtime work roots were not cleaned: incompatible=%t compatible=%t",
			workRootEmpty(incompatibleWorkRoot), workRootEmpty(compatibleWorkRoot))
	}
	validatorInvocations, err := os.ReadFile(validatorLog)
	if err != nil || strings.TrimSpace(string(validatorInvocations)) != "version" {
		t.Fatalf("LikeC4 startup probe invocations = %q, %v", validatorInvocations, err)
	}

	publicBodies := strings.Join([]string{initialBody, waitingBody, registeredBody, finalBody}, "\n")
	logs := strings.Join([]string{
		server.logs.redacted(), incompatibleProcess.logs.redacted(), compatibleProcess.logs.redacted(),
	}, "\n")
	for _, detail := range []struct {
		name  string
		value string
	}{
		{"public token", publicToken}, {"Gateway token", llmGatewayToken},
		{"environment canary", capabilityEnvironmentCanary},
		{"probe-output canary", capabilityProbeOutputCanary},
		{"path canary", capabilityPathCanary}, {"executable directory", fakeBin},
		{"Control Plane private URL", privateBaseURL},
		{"incompatible Runtime URL", incompatibleBaseURL},
		{"compatible Runtime URL", compatibleBaseURL},
		{"Control Plane private address", privateAddress},
		{"incompatible Runtime address", incompatibleAddress},
		{"compatible Runtime address", compatibleAddress},
	} {
		if strings.Contains(publicBodies, detail.value) {
			t.Fatalf("public Operations response exposed %s", detail.name)
		}
		if strings.Contains(logs, detail.value) {
			t.Fatalf("process logs exposed %s", detail.name)
		}
	}
}

func startCapabilityRuntime(
	t *testing.T,
	name, repositoryRoot, python, controlPlaneURL, runtimeBaseURL, runtimeAddress,
	workRoot, caFile string,
	identity localpki.Paths,
	path string,
) *childProcess {
	t.Helper()
	return startProcess(
		t, name, filepath.Join(repositoryRoot, "runtime"),
		map[string]string{
			"PATH":                       path,
			"PYTHONUNBUFFERED":           "1",
			"CONTRACTOR_TEST_ENV_CANARY": capabilityEnvironmentCanary,
		},
		python, "-m", "contractor_runtime",
		"--control-plane-url", controlPlaneURL,
		"--advertised-control-url", runtimeBaseURL,
		"--advertised-a2a-url", runtimeBaseURL,
		"--ca-file", caFile,
		"--certificate-file", identity.Certificate,
		"--private-key-file", identity.PrivateKey,
		"--listen", runtimeAddress,
		"--work-root", workRoot,
		"--request-timeout-seconds", "10",
		"--shutdown-grace-seconds", "5",
	)
}

func installCapabilityLikeC4(t *testing.T, root string) (string, string) {
	t.Helper()
	bin := filepath.Join(root, capabilityPathCanary)
	if err := os.Mkdir(bin, 0o700); err != nil {
		t.Fatal(err)
	}
	logPath := filepath.Join(root, "capability-likec4-invocations.log")
	script := `#!/bin/sh
if [ "$1" = "version" ]; then
  printf 'version\n' >> ` + shellSingleQuote(logPath) + `
  printf '` + capabilityProbeOutputCanary + `\n'
  exit 0
fi
if [ "$1" = "validate" ] && [ "$2" = "--json" ] && [ "$3" = "--no-layout" ] && [ "$4" = "--file" ]; then
  printf '{"valid":true,"errors":[]}\n'
  exit 0
fi
exit 2
`
	if err := os.WriteFile(filepath.Join(bin, "likec4"), []byte(script), 0o700); err != nil {
		t.Fatal(err)
	}
	return bin, logPath
}

func waitForUnallocatedPreparingExecution(
	t *testing.T,
	ctx context.Context,
	store runstore.Repository,
	runID string,
) runstore.StageExecution {
	t.Helper()
	ticker := time.NewTicker(100 * time.Millisecond)
	defer ticker.Stop()
	for {
		executions, err := store.ListStageExecutions(ctx, runID)
		if err == nil && len(executions) == 1 && executions[0].State == runstore.StagePreparing {
			allocations, allocationErr := store.ListStageAllocations(ctx, executions[0].StageExecutionID)
			if allocationErr == nil && len(allocations) == 0 {
				if executions[0].Attempt != 1 || executions[0].PlannerSessionID != nil ||
					executions[0].PlannerInvocationID != nil || executions[0].PlannerStartedAt != nil {
					t.Fatalf("preparing execution started Planner without capacity: %+v", executions[0])
				}
				return executions[0]
			}
		}
		select {
		case <-ctx.Done():
			t.Fatalf("wait for unallocated preparing StageExecution: %v", ctx.Err())
		case <-ticker.C:
		}
	}
}

func assertSameUnallocatedPreparingExecution(
	t *testing.T,
	ctx context.Context,
	store runstore.Repository,
	want runstore.StageExecution,
) {
	t.Helper()
	executions, err := store.ListStageExecutions(ctx, want.RunID)
	if err != nil || len(executions) != 1 {
		t.Fatalf("waiting StageExecutions = (%+v, %v)", executions, err)
	}
	got := executions[0]
	allocations, allocationErr := store.ListStageAllocations(ctx, got.StageExecutionID)
	if got.StageExecutionID != want.StageExecutionID || got.Attempt != 1 ||
		got.State != runstore.StagePreparing || got.PlannerSessionID != nil ||
		got.PlannerInvocationID != nil || got.PlannerStartedAt != nil ||
		allocationErr != nil || len(allocations) != 0 {
		t.Fatalf("capacity wait changed attempt or execution authority: execution=%+v allocations=%+v error=%v",
			got, allocations, allocationErr)
	}
}

func waitForObservedRuntimeAgents(
	t *testing.T,
	ctx context.Context,
	server *childProcess,
	runtimes []*childProcess,
	client *http.Client,
	baseURL string,
	ready func([]observedRuntimeAgent) bool,
) ([]observedRuntimeAgent, string) {
	t.Helper()
	ticker := time.NewTicker(100 * time.Millisecond)
	defer ticker.Stop()
	for {
		agents, body := observedRuntimeAgents(t, client, baseURL)
		if ready(agents) {
			return agents, body
		}
		for _, process := range append([]*childProcess{server}, runtimes...) {
			if exited, processErr := process.exited(); exited {
				t.Fatalf("%s exited while observing capabilities: %v\n%s",
					process.name, processErr, process.logs.redacted(publicToken, llmGatewayToken))
			}
		}
		select {
		case <-ctx.Done():
			t.Fatalf("wait for Runtime Agent Operations snapshot: %v", ctx.Err())
		case <-ticker.C:
		}
	}
}

func observedRuntimeAgents(
	t *testing.T,
	client *http.Client,
	baseURL string,
) ([]observedRuntimeAgent, string) {
	t.Helper()
	request, err := http.NewRequest(http.MethodGet, baseURL+"/v1/operations/snapshot", nil)
	if err != nil {
		t.Fatal(err)
	}
	request.Header.Set("Authorization", "Bearer "+publicToken)
	response := do(t, client, request, http.StatusOK)
	defer response.Body.Close()
	body, err := io.ReadAll(io.LimitReader(response.Body, 1<<20))
	if err != nil {
		t.Fatal(err)
	}
	var payload struct {
		RuntimeAgents []observedRuntimeAgent `json:"runtimeAgents"`
	}
	if err := json.Unmarshal(body, &payload); err != nil {
		t.Fatalf("decode Operations snapshot: %v", err)
	}
	return payload.RuntimeAgents, string(body)
}

func runtimeToolset(agent observedRuntimeAgent, ref string) observedRuntimeCapability {
	for _, capability := range agent.SupportedToolsets {
		if capability.Ref == ref {
			return capability
		}
	}
	return observedRuntimeCapability{}
}

func findRuntimeWithTool(
	agents []observedRuntimeAgent,
	toolsetRef, tool string,
) (observedRuntimeAgent, bool) {
	for _, agent := range agents {
		if containsString(runtimeToolset(agent, toolsetRef).Tools, tool) {
			return agent, true
		}
	}
	return observedRuntimeAgent{}, false
}

func findRuntimeByID(
	agents []observedRuntimeAgent,
	instanceID string,
) (observedRuntimeAgent, bool) {
	for _, agent := range agents {
		if agent.InstanceID == instanceID {
			return agent, true
		}
	}
	return observedRuntimeAgent{}, false
}

func runtimeCapabilityFingerprint(t *testing.T, agent observedRuntimeAgent) string {
	t.Helper()
	encoded, err := json.Marshal(struct {
		Runtimes  []string                    `json:"runtimes"`
		Toolsets  []observedRuntimeCapability `json:"toolsets"`
		Sandboxes []string                    `json:"sandboxes"`
	}{agent.SupportedRuntimes, agent.SupportedToolsets, agent.SupportedSandboxProfiles})
	if err != nil {
		t.Fatal(err)
	}
	return string(encoded)
}

func containsString(values []string, want string) bool {
	for _, value := range values {
		if value == want {
			return true
		}
	}
	return false
}
