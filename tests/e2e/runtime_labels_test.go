//go:build e2e

package e2e

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/httpapi/privateartifacts"
	"github.com/grauwolf32/contractor/internal/localpki"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
	"github.com/jackc/pgx/v5/pgxpool"
)

const (
	runtimeOTLPRunCanary   = "RUNTIME_LABEL_OTLP_RUN_SECRET_CANARY"
	runtimeOTLPAgentCanary = "RUNTIME_LABEL_OTLP_AGENT_SECRET_CANARY"
	runtimeProxyCanary     = "RUNTIME_LABEL_PROXY_SECRET_CANARY"
	runtimeProxyUsername   = "runtime-label-proxy-user"
)

type runtimeOperations struct {
	t        *testing.T
	client   *http.Client
	baseURL  string
	evidence [][]byte
}

type runtimeOperationResponse struct {
	body   []byte
	header http.Header
}

type observedRuntimePrincipal struct {
	RuntimeAgentID string                `json:"runtimeAgentId"`
	Labels         []string              `json:"labels"`
	Revision       string                `json:"revision"`
	Availability   string                `json:"availability"`
	Live           *observedRuntimeAgent `json:"live,omitempty"`
}

type pinnedRuntimeConfig struct {
	Label           string            `json:"label"`
	BindingRevision string            `json:"bindingRevision"`
	Config          runtimeconfig.Ref `json:"config"`
}

type createdLabeledRun struct {
	RunID                string   `json:"runId"`
	State                string   `json:"state"`
	RuntimeLabels        []string `json:"runtimeLabels"`
	RuntimeConfiguration struct {
		Default pinnedRuntimeConfig   `json:"default"`
		Labels  []pinnedRuntimeConfig `json:"labels"`
	} `json:"runtimeConfiguration"`
}

func TestLabelDrivenRuntimeConfigurationAcrossProcesses(t *testing.T) {
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
		LeafOptions: leaf, URI: "urn:contractor:control-plane:runtime-label-e2e",
	})
	if err != nil {
		t.Fatalf("issue Control Plane certificate: %v", err)
	}
	proxyOnlyIdentity, err := generator.IssueAgent(pkiRoot, "runtime-label-proxy-only", leaf)
	if err != nil {
		t.Fatalf("issue proxy-only Runtime Agent certificate: %v", err)
	}
	telemetryIdentity, err := generator.IssueAgent(pkiRoot, "runtime-label-telemetry", leaf)
	if err != nil {
		t.Fatalf("issue telemetry Runtime Agent certificate: %v", err)
	}

	gateway := newFakeGateway(llmGatewayToken)
	t.Cleanup(gateway.close)
	runCollector := newFakeOTLPCollector(runtimeOTLPRunCanary)
	t.Cleanup(runCollector.close)
	agentCollector := newFakeOTLPCollector(runtimeOTLPAgentCanary)
	t.Cleanup(agentCollector.close)
	oldProxy := newAuthenticatedForwardProxy(gateway.server.URL, runtimeProxyCanary, true)
	t.Cleanup(oldProxy.close)
	newProxy := newAuthenticatedForwardProxy(gateway.server.URL, runtimeProxyCanary, false)
	t.Cleanup(newProxy.close)
	rejectingProxy := newRejectingForwardProxy()
	t.Cleanup(rejectingProxy.close)

	configRoot := stageE2EConfiguration(
		t, filepath.Join(repositoryRoot, "configs", "e2e"),
		filepath.Join(temporaryRoot, "configs"), gateway.URL(),
	)
	publicAddress, privateAddress := freeAddress(t), freeAddress(t)
	publicBaseURL := "http://" + publicAddress
	privateBaseURL := "https://" + privateAddress
	userID := "runtime-label-e2e-user-" + randomHex(t, 8)
	masterKeyFile := filepath.Join(temporaryRoot, "credential-master-key")
	masterKey := base64.StdEncoding.EncodeToString(bytes.Repeat([]byte{0x5a}, 32))
	if err := os.WriteFile(masterKeyFile, []byte(masterKey), 0o600); err != nil {
		t.Fatal(err)
	}
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
	}, serverBinary, "serve", "--credential-master-key-file", masterKeyFile)
	publicClient := &http.Client{Timeout: 5 * time.Second}
	waitForHTTP(t, ctx, server, publicClient, publicBaseURL+"/readyz", http.StatusOK)

	python := filepath.Join(repositoryRoot, "runtime", ".venv", "bin", "python")
	if info, statErr := os.Stat(python); statErr != nil || info.IsDir() {
		t.Fatalf("Python Runtime environment is missing at %s; run 'cd runtime && uv sync --locked'", python)
	}
	proxyOnlyAddress, telemetryAddress := freeAddress(t), freeAddress(t)
	proxyOnlyWork := filepath.Join(temporaryRoot, "runtime-proxy-work")
	telemetryWork := filepath.Join(temporaryRoot, "runtime-telemetry-work")
	proxyOnlyRuntime := startRuntimeWithAdapters(
		t, "Python Runtime Agent proxy-only", repositoryRoot, python,
		privateBaseURL, proxyOnlyAddress, proxyOnlyWork, caPaths.Certificate,
		proxyOnlyIdentity, []string{"http-proxy@1"},
	)
	telemetryRuntime := startRuntimeWithAdapters(
		t, "Python Runtime Agent telemetry", repositoryRoot, python,
		privateBaseURL, telemetryAddress, telemetryWork, caPaths.Certificate,
		telemetryIdentity, []string{"otlp-http@1"},
	)
	controlClient := newMTLSClient(t, caPaths.Certificate, controlPlanePaths)
	waitForHTTP(t, ctx, proxyOnlyRuntime, controlClient, "https://"+proxyOnlyAddress+"/healthz", http.StatusOK)
	waitForHTTP(t, ctx, telemetryRuntime, controlClient, "https://"+telemetryAddress+"/healthz", http.StatusOK)
	runtimes := []*childProcess{proxyOnlyRuntime, telemetryRuntime}
	agents, _ := waitForObservedRuntimeAgents(
		t, ctx, server, runtimes, publicClient, publicBaseURL,
		func(items []observedRuntimeAgent) bool {
			if len(items) != 2 {
				return false
			}
			foundProxyOnly, foundTelemetry := false, false
			for _, item := range items {
				if item.SlotState != "idle" || item.ConfirmedLeaseUntil == nil {
					return false
				}
				switch strings.Join(item.SupportedRuntimeAdapters, ",") {
				case "http-proxy@1":
					foundProxyOnly = true
				case "otlp-http@1":
					foundTelemetry = true
				}
			}
			return foundProxyOnly && foundTelemetry
		},
	)
	if len(agents) != 2 {
		t.Fatalf("Runtime capability observations = %+v", agents)
	}

	pool, err := pgxpool.New(ctx, isolateURL)
	if err != nil {
		t.Fatalf("open assertion database: %v", err)
	}
	t.Cleanup(pool.Close)
	store := runstore.NewPostgresStore(pool)
	operations := &runtimeOperations{t: t, client: publicClient, baseURL: publicBaseURL}
	uploaded := uploadInput(t, publicClient, publicBaseURL)

	// No labels resolves only the built-in empty patch and therefore preserves
	// the authored Workflow executionConfig and direct Gateway behavior.
	baseline := operations.createRun("artifact-copy@1", "runtime-label-baseline", uploaded, nil)
	baselineStatus := waitForRunAcross(
		t, ctx, server, runtimes, gateway, publicClient, publicBaseURL, baseline.RunID,
	)
	if baseline.RuntimeConfiguration.Default.Config.Name != "contractor-empty" ||
		len(baseline.RuntimeConfiguration.Labels) != 0 || baselineStatus.State != "succeeded" {
		t.Fatalf("unlabeled baseline did not preserve empty default: create=%+v status=%+v", baseline, baselineStatus)
	}
	baselineAllocation := onlyRunAllocation(t, ctx, store, baseline.RunID)
	gateway.ResetScenario()

	operations.createRuntimeCredential("otel-run", "otlp-headers@1", map[string]any{
		"headers": map[string]string{"x-contractor-token": runtimeOTLPRunCanary},
	})
	operations.createRuntimeCredential("otel-agent", "otlp-headers@1", map[string]any{
		"headers": map[string]string{"x-contractor-token": runtimeOTLPAgentCanary},
	})
	runDebug := operations.publishRuntimeConfig("debug-run", "1", map[string]any{
		"planner": map[string]any{"telemetry": map[string]any{
			"adapter": "otlp-http@1", "endpoint": runCollector.URL(),
			"credential": "otel-run", "captureContent": false, "flushTimeoutSeconds": 2,
		}},
		"worker": map[string]any{"telemetry": map[string]any{
			"adapter": "otlp-http@1", "endpoint": runCollector.URL(),
			"credential": "otel-run", "captureContent": false, "flushTimeoutSeconds": 2,
		}},
	})
	agentDebug := operations.publishRuntimeConfig("debug-agent", "1", map[string]any{
		"worker": map[string]any{"telemetry": map[string]any{
			"adapter": "otlp-http@1", "endpoint": agentCollector.URL(),
			"credential": "otel-agent", "captureContent": false, "flushTimeoutSeconds": 2,
		}},
	})
	operations.createRuntimeLabel("debug", runDebug)
	operations.createRuntimeLabel("agent-debug", agentDebug)
	principals := operations.runtimeAgentPrincipals()
	telemetryPrincipal := principalWithAdapters(t, principals, "otlp-http@1")
	assertDuplicatePrincipalRegistrationRejected(
		t, operations, caPaths.Certificate, telemetryIdentity, privateBaseURL, telemetryPrincipal,
	)
	updatedPrincipal := operations.replacePrincipalLabels(telemetryPrincipal, []string{"agent-debug"})

	debug := operations.createRun("artifact-copy@1", "runtime-label-debug-agent", uploaded, []string{"debug"})
	debugStatus := waitForRunAcross(
		t, ctx, server, runtimes, gateway, publicClient, publicBaseURL, debug.RunID,
	)
	debugAllocation := onlyRunAllocation(t, ctx, store, debug.RunID)
	if debugStatus.State != "succeeded" || debugAllocation.RuntimeAgentID != telemetryPrincipal.RuntimeAgentID ||
		baselineAllocation.RuntimeConfiguration == nil || debugAllocation.RuntimeConfiguration == nil ||
		baselineAllocation.RuntimeConfiguration.ModelPolicy != debugAllocation.RuntimeConfiguration.ModelPolicy ||
		agentCollector.requests() == 0 || runCollector.requests() == 0 {
		t.Fatalf("independent Planner/Agent telemetry routing was not applied: principal=%s allocation=%+v agentOTLP=%d runOTLP=%d",
			telemetryPrincipal.RuntimeAgentID, debugAllocation, agentCollector.requests(), runCollector.requests())
	}
	assertOTLPPayloadSafe(t, agentCollector.payloads(), runtimeOTLPAgentCanary, runtimeOTLPRunCanary, runtimeProxyCanary, runtimeProxyUsername)
	assertOTLPPayloadsContainSafeScope(t, runCollector.payloads(), "contractor.planner.", runtimeOTLPAgentCanary, runtimeOTLPRunCanary, runtimeProxyCanary, runtimeProxyUsername)
	gateway.ResetScenario()
	operations.replacePrincipalLabels(updatedPrincipal, []string{})

	// Delivery failure remains telemetry-only and the same physical slot is
	// released for later work.
	runCollector.setReject(true)
	failedExport := operations.createRun("artifact-copy@1", "runtime-label-debug-failed-export", uploaded, []string{"debug"})
	failedExportStatus := waitForRunAcross(
		t, ctx, server, runtimes, gateway, publicClient, publicBaseURL, failedExport.RunID,
	)
	if failedExportStatus.State != "succeeded" || runCollector.requests() == 0 {
		t.Fatalf("OTLP delivery failure changed semantic outcome: status=%+v requests=%d", failedExportStatus, runCollector.requests())
	}
	failedExportMetrics := onlyRuntimeAdapterMetrics(t, ctx, store, failedExport.RunID, "otlp-http@1")
	if failedExportMetrics.FailedOperations == 0 || failedExportMetrics.FlushAttempted == nil ||
		!*failedExportMetrics.FlushAttempted || failedExportMetrics.FlushSucceeded == nil ||
		*failedExportMetrics.FlushSucceeded || failedExportMetrics.LastErrorCode == nil ||
		*failedExportMetrics.LastErrorCode != "flush_failed" {
		t.Fatalf("failed OTLP delivery metrics = %+v", failedExportMetrics)
	}
	assertOTLPPayloadSafe(t, runCollector.payloads(), runtimeOTLPAgentCanary, runtimeOTLPRunCanary, runtimeProxyCanary, runtimeProxyUsername)
	gateway.ResetScenario()

	operations.createRuntimeCredential("caido-basic", "http-proxy-basic@1", map[string]any{
		"username": runtimeProxyUsername, "password": runtimeProxyCanary,
	})
	caidoOld := operations.publishRuntimeConfig("caido-old", "1", map[string]any{
		"worker": map[string]any{"httpProxy": map[string]any{
			"adapter": "http-proxy@1", "proxyUrl": oldProxy.URL(),
			"credential": "caido-basic", "targets": []string{"llm-gateway"},
		}},
	})
	caidoNew := operations.publishRuntimeConfig("caido-new", "1", map[string]any{
		"worker": map[string]any{"httpProxy": map[string]any{
			"adapter": "http-proxy@1", "proxyUrl": newProxy.URL(),
			"credential": "caido-basic", "targets": []string{"llm-gateway"},
		}},
	})
	operations.createRuntimeLabel("caido", caidoOld)
	pinnedOld := operations.createRun("artifact-copy@1", "runtime-label-caido-old", uploaded, []string{"caido"})
	oldProxy.waitForFirstRequest(t, ctx)
	oldAllocation := onlyRunAllocation(t, ctx, store, pinnedOld.RunID)
	foreignIdentity := telemetryIdentity
	if oldAllocation.RuntimeAgentID == telemetryPrincipal.RuntimeAgentID {
		foreignIdentity = proxyOnlyIdentity
	}
	assertForeignAgentCannotBorrowAllocation(
		t, operations, caPaths.Certificate, foreignIdentity, privateBaseURL, oldAllocation,
	)
	operations.rebindRuntimeLabel("caido", "1", caidoNew)
	oldProxy.releaseFirstRequest()
	oldStatus := waitForRunAcross(
		t, ctx, server, runtimes, gateway, publicClient, publicBaseURL, pinnedOld.RunID,
	)
	if oldStatus.State != "succeeded" || len(pinnedOld.RuntimeConfiguration.Labels) != 1 ||
		pinnedOld.RuntimeConfiguration.Labels[0].Config != caidoOld || oldProxy.requests() != 3 ||
		newProxy.requests() != 0 {
		t.Fatalf("active caido allocation was not pinned: create=%+v old/new requests=%d/%d",
			pinnedOld, oldProxy.requests(), newProxy.requests())
	}
	gateway.ResetScenario()

	pinnedNew := operations.createRun("artifact-copy@1", "runtime-label-caido-new", uploaded, []string{"caido"})
	newStatus := waitForRunAcross(
		t, ctx, server, runtimes, gateway, publicClient, publicBaseURL, pinnedNew.RunID,
	)
	if newStatus.State != "succeeded" || len(pinnedNew.RuntimeConfiguration.Labels) != 1 ||
		pinnedNew.RuntimeConfiguration.Labels[0].BindingRevision != "2" ||
		pinnedNew.RuntimeConfiguration.Labels[0].Config != caidoNew || newProxy.requests() != 3 {
		t.Fatalf("later caido allocation did not use rebound config: create=%+v requests=%d", pinnedNew, newProxy.requests())
	}
	if failures := append(oldProxy.failures(), newProxy.failures()...); len(failures) != 0 {
		t.Fatalf("proxy failures: %v", failures)
	}
	if failures := append(runCollector.failuresSnapshot(), agentCollector.failuresSnapshot()...); len(failures) != 0 {
		t.Fatalf("OTLP collector envelope failures: %v", failures)
	}

	// A required proxy failure is semantic rather than supplementary. It must
	// fail the Run through the normal Worker result path, release the selected
	// allocation and leave both specialist processes reusable.
	refused := operations.publishRuntimeConfig("caido-refused", "1", map[string]any{
		"worker": map[string]any{"httpProxy": map[string]any{
			"adapter": "http-proxy@1", "proxyUrl": rejectingProxy.URL(),
			"targets": []string{"llm-gateway"},
		}},
	})
	operations.createRuntimeLabel("caido-refused", refused)
	refusedRun := operations.createRun(
		"artifact-copy@1", "runtime-label-caido-refused", uploaded, []string{"caido-refused"},
	)
	refusedStatus := waitForRunTerminalAcross(
		t, ctx, server, runtimes, publicClient, publicBaseURL, refusedRun.RunID,
	)
	if refusedStatus.State != "failed" || rejectingProxy.requests() == 0 {
		t.Fatalf("required proxy failure was not semantic: status=%+v requests=%d",
			refusedStatus, rejectingProxy.requests())
	}
	gateway.ResetScenario()

	// The adapter sets are intentionally disjoint. These final probes therefore
	// prove reuse of both physical slots after all injected failures: debug can
	// run only on the OTLP specialist and caido only on the proxy specialist.
	runCollector.setReject(false)
	reuseTelemetry := operations.createRun(
		"artifact-copy@1", "runtime-label-reuse-telemetry", uploaded, []string{"debug"},
	)
	if status := waitForRunAcross(
		t, ctx, server, runtimes, gateway, publicClient, publicBaseURL, reuseTelemetry.RunID,
	); status.State != "succeeded" {
		t.Fatalf("telemetry specialist reuse status = %+v", status)
	}
	if allocation := onlyRunAllocation(t, ctx, store, reuseTelemetry.RunID); allocation.RuntimeAgentID != telemetryPrincipal.RuntimeAgentID {
		t.Fatalf("telemetry reuse selected principal %s, want %s", allocation.RuntimeAgentID, telemetryPrincipal.RuntimeAgentID)
	}
	gateway.ResetScenario()
	reuseProxy := operations.createRun(
		"artifact-copy@1", "runtime-label-reuse-proxy", uploaded, []string{"caido"},
	)
	if status := waitForRunAcross(
		t, ctx, server, runtimes, gateway, publicClient, publicBaseURL, reuseProxy.RunID,
	); status.State != "succeeded" {
		t.Fatalf("proxy specialist reuse status = %+v", status)
	}
	proxyPrincipal := principalWithAdapters(t, principals, "http-proxy@1")
	if allocation := onlyRunAllocation(t, ctx, store, reuseProxy.RunID); allocation.RuntimeAgentID != proxyPrincipal.RuntimeAgentID {
		t.Fatalf("proxy reuse selected principal %s, want %s", allocation.RuntimeAgentID, proxyPrincipal.RuntimeAgentID)
	}
	gateway.ResetScenario()

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
	if len(finalAgents) != 2 {
		t.Fatalf("Runtime Agents were not reusable: %+v", finalAgents)
	}
	operations.evidence = append(operations.evidence, []byte(finalSnapshot))
	assertRuntimeLabelSecretsAbsent(
		t, ctx, pool, operations.evidence,
		append([]*childProcess{server}, runtimes...),
		[]string{runtimeOTLPRunCanary, runtimeOTLPAgentCanary, runtimeProxyCanary, runtimeProxyUsername},
	)
}

func startRuntimeWithAdapters(
	t *testing.T,
	name, repositoryRoot, python, controlPlaneURL, runtimeAddress, workRoot, caFile string,
	identity localpki.Paths,
	adapters []string,
) *childProcess {
	t.Helper()
	runtimeBaseURL := "https://" + runtimeAddress
	args := []string{
		"-m", "contractor_runtime",
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
	}
	for _, adapter := range adapters {
		args = append(args, "--runtime-adapter", adapter)
	}
	return startProcess(
		t, name, filepath.Join(repositoryRoot, "runtime"),
		map[string]string{"PYTHONUNBUFFERED": "1"}, python, args...,
	)
}

func assertDuplicatePrincipalRegistrationRejected(
	t *testing.T,
	operations *runtimeOperations,
	caFile string,
	identity localpki.Paths,
	controlPlaneURL string,
	principal observedRuntimePrincipal,
) {
	t.Helper()
	if principal.Live == nil {
		t.Fatalf("Runtime Agent principal has no live process: %+v", principal)
	}
	toolsets := make([]contracts.ToolsetCapability, len(principal.Live.SupportedToolsets))
	for index, capability := range principal.Live.SupportedToolsets {
		toolsets[index] = contracts.ToolsetCapability{
			Ref: capability.Ref, Tools: append([]string(nil), capability.Tools...),
		}
	}
	adapters := make([]contracts.RuntimeAdapterRef, len(principal.Live.SupportedRuntimeAdapters))
	for index, adapter := range principal.Live.SupportedRuntimeAdapters {
		adapters[index] = contracts.RuntimeAdapterRef(adapter)
	}
	registration := contracts.AgentRegistrationV2{
		APIVersion: contracts.APIVersion, PrivateProtocolVersion: contracts.PrivateProtocolVersionV2,
		InstanceID: "duplicate-runtime-" + randomHex(t, 8), SoftwareVersion: "0.1.0",
		StartedAt: time.Now().UTC(), ControlURL: "https://127.0.0.1:1", A2AURL: "https://127.0.0.1:1",
		InitialLabels: []string{}, SupportedRuntimes: append([]string(nil), principal.Live.SupportedRuntimes...),
		SupportedToolsets:        toolsets,
		SupportedSandboxProfiles: append([]string(nil), principal.Live.SupportedSandboxProfiles...),
		SupportedRuntimeAdapters: adapters, ObservedState: contracts.AgentIdle,
	}
	if err := registration.Validate(); err != nil {
		t.Fatalf("duplicate registration fixture: %v", err)
	}
	body, err := json.Marshal(registration)
	if err != nil {
		t.Fatal(err)
	}
	request, err := http.NewRequest(
		http.MethodPost, controlPlaneURL+"/private/v1/agents/register", bytes.NewReader(body),
	)
	if err != nil {
		t.Fatal(err)
	}
	request.Header.Set("Content-Type", "application/json")
	response := do(t, newMTLSClient(t, caFile, identity), request, http.StatusConflict)
	defer response.Body.Close()
	responseBody, err := io.ReadAll(io.LimitReader(response.Body, 64<<10))
	if err != nil {
		t.Fatal(err)
	}
	operations.evidence = append(operations.evidence, responseBody)
}

func assertForeignAgentCannotBorrowAllocation(
	t *testing.T,
	operations *runtimeOperations,
	caFile string,
	foreignIdentity localpki.Paths,
	privateBaseURL string,
	allocation runstore.StageAllocation,
) {
	t.Helper()
	target := privateBaseURL + "/private/v1/allocations/" +
		url.PathEscape(allocation.AllocationID) + "/artifacts"
	request, err := http.NewRequest(http.MethodGet, target, nil)
	if err != nil {
		t.Fatal(err)
	}
	request.Header.Set(privateartifacts.RuntimeInstanceHeader, allocation.RuntimeAgentInstanceID)
	response := do(t, newMTLSClient(t, caFile, foreignIdentity), request, http.StatusNotFound)
	defer response.Body.Close()
	body, err := io.ReadAll(io.LimitReader(response.Body, 64<<10))
	if err != nil {
		t.Fatal(err)
	}
	operations.evidence = append(operations.evidence, body)
}

func (o *runtimeOperations) request(
	method, path string,
	body any,
	expected int,
	headers map[string]string,
) []byte {
	o.t.Helper()
	return o.requestResponse(method, path, body, expected, headers).body
}

func (o *runtimeOperations) requestResponse(
	method, path string,
	body any,
	expected int,
	headers map[string]string,
) runtimeOperationResponse {
	o.t.Helper()
	var reader io.Reader
	if body != nil {
		encoded, err := json.Marshal(body)
		if err != nil {
			o.t.Fatal(err)
		}
		reader = bytes.NewReader(encoded)
	}
	request, err := http.NewRequest(method, o.baseURL+path, reader)
	if err != nil {
		o.t.Fatal(err)
	}
	request.Header.Set("Authorization", "Bearer "+publicToken)
	if body != nil {
		request.Header.Set("Content-Type", "application/json")
	}
	for name, value := range headers {
		request.Header.Set(name, value)
	}
	response := do(o.t, o.client, request, expected)
	defer response.Body.Close()
	data, err := io.ReadAll(io.LimitReader(response.Body, 2<<20))
	if err != nil {
		o.t.Fatal(err)
	}
	o.evidence = append(o.evidence, append([]byte(nil), data...))
	return runtimeOperationResponse{body: data, header: response.Header.Clone()}
}

func (o *runtimeOperations) requireReplay(response runtimeOperationResponse, boundary string) {
	o.t.Helper()
	if response.header.Get("Idempotency-Replayed") != "true" {
		o.t.Fatalf("%s response-loss replay was not acknowledged: headers=%v body=%s",
			boundary, response.header, response.body)
	}
}

func (o *runtimeOperations) createRuntimeCredential(id, kind string, material any) {
	o.t.Helper()
	body := map[string]any{
		"credentialId": id, "kind": kind, "material": material,
	}
	headers := map[string]string{"Idempotency-Key": "e2e-create-" + id}
	data := o.request(http.MethodPost, "/v1/operations/runtime-credentials", body, http.StatusCreated, headers)
	if bytes.Contains(data, []byte("material")) {
		o.t.Fatalf("Runtime credential response exposed material: %s", data)
	}
	o.requireReplay(
		o.requestResponse(http.MethodPost, "/v1/operations/runtime-credentials", body, http.StatusCreated, headers),
		"Runtime credential create",
	)
}

func (o *runtimeOperations) publishRuntimeConfig(name, version string, spec any) runtimeconfig.Ref {
	o.t.Helper()
	body := map[string]any{
		"apiVersion": "contractor/v1alpha1", "kind": "RuntimeConfig",
		"metadata": map[string]string{"name": name, "version": version}, "spec": spec,
	}
	headers := map[string]string{"Idempotency-Key": "e2e-publish-" + name + "-" + version}
	data := o.request(http.MethodPost, "/v1/operations/runtime-configs", body, http.StatusCreated, headers)
	var resource struct {
		Ref runtimeconfig.Ref `json:"ref"`
	}
	if err := json.Unmarshal(data, &resource); err != nil || resource.Ref.Name != name {
		o.t.Fatalf("decode published RuntimeConfig: ref=%+v error=%v body=%s", resource.Ref, err, data)
	}
	o.requireReplay(
		o.requestResponse(http.MethodPost, "/v1/operations/runtime-configs", body, http.StatusCreated, headers),
		"RuntimeConfig publish",
	)
	return resource.Ref
}

func (o *runtimeOperations) createRuntimeLabel(label string, ref runtimeconfig.Ref) {
	o.t.Helper()
	path := "/v1/operations/runtime-labels/" + url.PathEscape(label)
	body := map[string]any{"config": ref}
	headers := map[string]string{"Idempotency-Key": "e2e-create-label-" + label, "If-None-Match": "*"}
	o.request(http.MethodPut, path, body, http.StatusCreated, headers)
	o.requireReplay(
		o.requestResponse(http.MethodPut, path, body, http.StatusCreated, headers),
		"Runtime label create",
	)
}

func (o *runtimeOperations) rebindRuntimeLabel(label, revision string, ref runtimeconfig.Ref) {
	o.t.Helper()
	path := "/v1/operations/runtime-labels/" + url.PathEscape(label)
	body := map[string]any{"config": ref}
	headers := map[string]string{
		"Idempotency-Key": "e2e-rebind-label-" + label + "-" + revision,
		"If-Match":        strconv.Quote(revision),
	}
	o.request(http.MethodPut, path, body, http.StatusOK, headers)
	o.requireReplay(
		o.requestResponse(http.MethodPut, path, body, http.StatusOK, headers),
		"Runtime label rebind",
	)
}

func (o *runtimeOperations) runtimeAgentPrincipals() []observedRuntimePrincipal {
	o.t.Helper()
	data := o.request(http.MethodGet, "/v1/operations/runtime-agent-principals?limit=20", nil, http.StatusOK, nil)
	var page struct {
		Items []observedRuntimePrincipal `json:"items"`
	}
	if err := json.Unmarshal(data, &page); err != nil {
		o.t.Fatalf("decode Runtime Agent principals: %v", err)
	}
	return page.Items
}

func (o *runtimeOperations) replacePrincipalLabels(
	principal observedRuntimePrincipal,
	labels []string,
) observedRuntimePrincipal {
	o.t.Helper()
	data := o.request(
		http.MethodPut,
		"/v1/operations/runtime-agent-principals/"+url.PathEscape(principal.RuntimeAgentID)+"/labels",
		map[string]any{"labels": labels}, http.StatusOK,
		map[string]string{
			"Idempotency-Key": "e2e-agent-labels-" + principal.RuntimeAgentID[:12] + "-" + principal.Revision,
			"If-Match":        strconv.Quote(principal.Revision),
		},
	)
	var result observedRuntimePrincipal
	if err := json.Unmarshal(data, &result); err != nil {
		o.t.Fatalf("decode Runtime Agent principal mutation: %v", err)
	}
	replay := o.requestResponse(
		http.MethodPut,
		"/v1/operations/runtime-agent-principals/"+url.PathEscape(principal.RuntimeAgentID)+"/labels",
		map[string]any{"labels": labels}, http.StatusOK,
		map[string]string{
			"Idempotency-Key": "e2e-agent-labels-" + principal.RuntimeAgentID[:12] + "-" + principal.Revision,
			"If-Match":        strconv.Quote(principal.Revision),
		},
	)
	o.requireReplay(replay, "Runtime Agent label replacement")
	return result
}

func (o *runtimeOperations) createRun(
	workflow, idempotencyKey string,
	input artifactRef,
	labels []string,
) createdLabeledRun {
	o.t.Helper()
	body := map[string]any{
		"workflow": workflow, "parameters": map[string]string{},
		"artifacts": map[string]artifactRef{"source": input},
	}
	if labels != nil {
		body["runtimeLabels"] = labels
	}
	headers := map[string]string{"Idempotency-Key": idempotencyKey}
	data := o.request(http.MethodPost, "/v1/runs", body, http.StatusAccepted, headers)
	var result createdLabeledRun
	if err := json.Unmarshal(data, &result); err != nil || result.RunID == "" || result.State != "running" {
		o.t.Fatalf("create labeled Run = (%+v, %v): %s", result, err, data)
	}
	replay := o.requestResponse(http.MethodPost, "/v1/runs", body, http.StatusAccepted, headers)
	o.requireReplay(replay, "Run create")
	var replayed createdLabeledRun
	if err := json.Unmarshal(replay.body, &replayed); err != nil || replayed.RunID != result.RunID ||
		!samePinnedRuntimeConfiguration(result, replayed) {
		o.t.Fatalf("Run replay changed identity or pinned Runtime configuration: first=%+v replay=%+v error=%v",
			result, replayed, err)
	}
	return result
}

func samePinnedRuntimeConfiguration(left, right createdLabeledRun) bool {
	leftEncoded, leftErr := json.Marshal(left.RuntimeConfiguration)
	rightEncoded, rightErr := json.Marshal(right.RuntimeConfiguration)
	return leftErr == nil && rightErr == nil && bytes.Equal(leftEncoded, rightEncoded)
}

func principalWithAdapters(
	t *testing.T,
	principals []observedRuntimePrincipal,
	adapters ...string,
) observedRuntimePrincipal {
	t.Helper()
	want := strings.Join(adapters, ",")
	for _, principal := range principals {
		if principal.Live != nil && strings.Join(principal.Live.SupportedRuntimeAdapters, ",") == want {
			return principal
		}
	}
	t.Fatalf("no Runtime Agent principal advertises %v: %+v", adapters, principals)
	return observedRuntimePrincipal{}
}

func waitForRunAcross(
	t *testing.T,
	ctx context.Context,
	server *childProcess,
	runtimes []*childProcess,
	gateway interface{ Failures() []string },
	client *http.Client,
	baseURL, runID string,
) runStatus {
	t.Helper()
	ticker := time.NewTicker(100 * time.Millisecond)
	defer ticker.Stop()
	for {
		request, _ := http.NewRequestWithContext(ctx, http.MethodGet, baseURL+"/v1/runs/"+url.PathEscape(runID), nil)
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
				t.Fatalf("Run %s reached %s: %+v", runID, status.State, status.Attempts)
			}
		}
		for _, process := range append([]*childProcess{server}, runtimes...) {
			if exited, processErr := process.exited(); exited {
				t.Fatalf("%s exited while waiting for Run: %v\n%s", process.name, processErr,
					process.logs.redacted(publicToken, llmGatewayToken, runtimeOTLPRunCanary, runtimeOTLPAgentCanary, runtimeProxyCanary))
			}
		}
		if failures := gateway.Failures(); len(failures) != 0 {
			t.Fatalf("fake Gateway failures: %v", failures)
		}
		select {
		case <-ctx.Done():
			t.Fatalf("wait for Run %s: %v", runID, ctx.Err())
		case <-ticker.C:
		}
	}
}

func waitForRunTerminalAcross(
	t *testing.T,
	ctx context.Context,
	server *childProcess,
	runtimes []*childProcess,
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
			switch status.State {
			case "succeeded", "failed", "cancelled":
				return status
			}
		}
		for _, process := range append([]*childProcess{server}, runtimes...) {
			if exited, processErr := process.exited(); exited {
				t.Fatalf("%s exited while waiting for terminal Run: %v\n%s", process.name, processErr,
					process.logs.redacted(publicToken, llmGatewayToken, runtimeOTLPRunCanary,
						runtimeOTLPAgentCanary, runtimeProxyCanary))
			}
		}
		select {
		case <-ctx.Done():
			t.Fatalf("wait for terminal Run %s: %v", runID, ctx.Err())
		case <-ticker.C:
		}
	}
}

func onlyRunAllocation(
	t *testing.T,
	ctx context.Context,
	store runstore.Repository,
	runID string,
) runstore.StageAllocation {
	t.Helper()
	executions, err := store.ListStageExecutions(ctx, runID)
	if err != nil || len(executions) != 1 {
		t.Fatalf("Run StageExecutions = (%+v, %v), want one", executions, err)
	}
	allocations, err := store.ListStageAllocations(ctx, executions[0].StageExecutionID)
	if err != nil || len(allocations) != 1 {
		t.Fatalf("Run Stage allocations = (%+v, %v), want one", allocations, err)
	}
	return allocations[0]
}

func onlyRuntimeAdapterMetrics(
	t *testing.T,
	ctx context.Context,
	store runstore.Repository,
	runID, adapter string,
) contracts.RuntimeAdapterMetricsV2 {
	t.Helper()
	executions, err := store.ListStageExecutions(ctx, runID)
	if err != nil || len(executions) != 1 {
		t.Fatalf("Run StageExecutions = (%+v, %v), want one", executions, err)
	}
	reports, err := store.ListStageExecutionReports(ctx, executions[0].StageExecutionID)
	if err != nil || len(reports) != 1 {
		t.Fatalf("Run execution reports = (%+v, %v), want one", reports, err)
	}
	metrics, ok := reports[0].Report.Runtime.Adapters[contracts.RuntimeAdapterRef(adapter)]
	if !ok {
		t.Fatalf("Run report has no %s adapter metrics: %+v", adapter, reports[0].Report.Runtime.Adapters)
	}
	return metrics
}

type fakeOTLPCollector struct {
	server *httptest.Server
	token  string

	mu       sync.Mutex
	reject   bool
	bodies   [][]byte
	failures []string
}

func newFakeOTLPCollector(token string) *fakeOTLPCollector {
	collector := &fakeOTLPCollector{token: token}
	collector.server = httptest.NewServer(http.HandlerFunc(collector.serveHTTP))
	return collector
}

func (c *fakeOTLPCollector) URL() string { return c.server.URL + "/v1/traces" }

func (c *fakeOTLPCollector) close() { c.server.Close() }

func (c *fakeOTLPCollector) setReject(value bool) {
	c.mu.Lock()
	c.reject = value
	c.mu.Unlock()
}

func (c *fakeOTLPCollector) requests() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return len(c.bodies)
}

func (c *fakeOTLPCollector) payloads() [][]byte {
	c.mu.Lock()
	defer c.mu.Unlock()
	result := make([][]byte, len(c.bodies))
	for index := range c.bodies {
		result[index] = append([]byte(nil), c.bodies[index]...)
	}
	return result
}

func (c *fakeOTLPCollector) failuresSnapshot() []string {
	c.mu.Lock()
	defer c.mu.Unlock()
	return append([]string(nil), c.failures...)
}

func (c *fakeOTLPCollector) serveHTTP(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost || r.URL.Path != "/v1/traces" ||
		r.Header.Get("Content-Type") != "application/x-protobuf" ||
		r.Header.Get("x-contractor-token") != c.token {
		c.mu.Lock()
		c.failures = append(c.failures, "invalid OTLP request envelope")
		c.mu.Unlock()
		http.Error(w, "invalid request", http.StatusBadRequest)
		return
	}
	body, err := io.ReadAll(io.LimitReader(r.Body, 3<<20))
	if err != nil || len(body) == 0 {
		http.Error(w, "invalid protobuf", http.StatusBadRequest)
		return
	}
	c.mu.Lock()
	c.bodies = append(c.bodies, append([]byte(nil), body...))
	reject := c.reject
	c.mu.Unlock()
	if reject {
		http.Error(w, "unavailable", http.StatusServiceUnavailable)
		return
	}
	w.Header().Set("Content-Type", "application/x-protobuf")
	w.WriteHeader(http.StatusOK)
}

func assertOTLPPayloadSafe(t *testing.T, payloads [][]byte, secrets ...string) {
	t.Helper()
	if len(payloads) == 0 {
		t.Fatal("fake OTLP collector received no protobuf requests")
	}
	for _, payload := range payloads {
		if !bytes.Contains(payload, []byte("contractor.runtime.")) {
			t.Fatal("OTLP protobuf payload has no Contractor safe provenance")
		}
		for _, secret := range secrets {
			if bytes.Contains(payload, []byte(secret)) {
				t.Fatalf("OTLP protobuf payload exposed secret canary")
			}
		}
	}
}

func assertOTLPPayloadsContainSafeScope(
	t *testing.T,
	payloads [][]byte,
	scopePrefix string,
	secrets ...string,
) {
	t.Helper()
	found := false
	for _, payload := range payloads {
		if bytes.Contains(payload, []byte(scopePrefix)) {
			found = true
		}
		for _, secret := range secrets {
			if bytes.Contains(payload, []byte(secret)) {
				t.Fatal("OTLP protobuf payload exposed secret canary")
			}
		}
	}
	if !found {
		t.Fatalf("OTLP protobuf payloads have no %q scope", scopePrefix)
	}
}

type authenticatedForwardProxy struct {
	server      *httptest.Server
	targetHost  string
	password    string
	transport   *http.Transport
	first       chan struct{}
	release     chan struct{}
	releaseOnce sync.Once
	blockFirst  bool

	mu           sync.Mutex
	calls        int
	failureCodes []string
}

type rejectingForwardProxy struct {
	server *httptest.Server

	mu    sync.Mutex
	calls int
}

func newRejectingForwardProxy() *rejectingForwardProxy {
	proxy := &rejectingForwardProxy{}
	proxy.server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		proxy.mu.Lock()
		proxy.calls++
		proxy.mu.Unlock()
		http.Error(w, "proxy upstream unavailable", http.StatusBadGateway)
	}))
	return proxy
}

func (p *rejectingForwardProxy) URL() string { return p.server.URL }

func (p *rejectingForwardProxy) close() { p.server.Close() }

func (p *rejectingForwardProxy) requests() int {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.calls
}

func newAuthenticatedForwardProxy(target, password string, blockFirst bool) *authenticatedForwardProxy {
	parsed, _ := url.Parse(target)
	proxy := &authenticatedForwardProxy{
		targetHost: parsed.Host, password: password,
		transport: &http.Transport{Proxy: nil},
		first:     make(chan struct{}, 1), release: make(chan struct{}), blockFirst: blockFirst,
	}
	proxy.server = httptest.NewServer(http.HandlerFunc(proxy.serveHTTP))
	return proxy
}

func (p *authenticatedForwardProxy) URL() string { return p.server.URL }

func (p *authenticatedForwardProxy) close() {
	p.releaseFirstRequest()
	p.transport.CloseIdleConnections()
	p.server.Close()
}

func (p *authenticatedForwardProxy) requests() int {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.calls
}

func (p *authenticatedForwardProxy) failures() []string {
	p.mu.Lock()
	defer p.mu.Unlock()
	return append([]string(nil), p.failureCodes...)
}

func (p *authenticatedForwardProxy) waitForFirstRequest(t *testing.T, ctx context.Context) {
	t.Helper()
	timer := time.NewTimer(20 * time.Second)
	defer timer.Stop()
	select {
	case <-p.first:
	case <-timer.C:
		t.Fatal("first proxy request did not arrive within 20s")
	case <-ctx.Done():
		t.Fatalf("wait for first proxy request: %v", ctx.Err())
	}
}

func (p *authenticatedForwardProxy) releaseFirstRequest() {
	p.releaseOnce.Do(func() { close(p.release) })
}

func (p *authenticatedForwardProxy) serveHTTP(w http.ResponseWriter, r *http.Request) {
	wantAuth := "Basic " + base64.StdEncoding.EncodeToString([]byte(runtimeProxyUsername+":"+p.password))
	if r.Header.Get("Proxy-Authorization") != wantAuth || r.URL.Host != p.targetHost ||
		r.Method != http.MethodPost || !strings.HasSuffix(r.URL.Path, "/chat/completions") {
		p.recordFailure("unexpected target or proxy authentication")
		http.Error(w, "proxy rejected request", http.StatusProxyAuthRequired)
		return
	}
	p.mu.Lock()
	p.calls++
	call := p.calls
	p.mu.Unlock()
	if p.blockFirst && call == 1 {
		select {
		case p.first <- struct{}{}:
		default:
		}
		select {
		case <-p.release:
		case <-r.Context().Done():
			p.recordFailure("blocked request context ended")
			return
		}
	}
	outgoing := r.Clone(r.Context())
	outgoing.RequestURI = ""
	outgoing.Header = r.Header.Clone()
	outgoing.Header.Del("Proxy-Authorization")
	response, err := p.transport.RoundTrip(outgoing)
	if err != nil {
		p.recordFailure("forward transport failed")
		http.Error(w, "proxy upstream unavailable", http.StatusBadGateway)
		return
	}
	defer response.Body.Close()
	for name, values := range response.Header {
		for _, value := range values {
			w.Header().Add(name, value)
		}
	}
	w.WriteHeader(response.StatusCode)
	_, _ = io.Copy(w, io.LimitReader(response.Body, 2<<20))
}

func (p *authenticatedForwardProxy) recordFailure(code string) {
	p.mu.Lock()
	p.failureCodes = append(p.failureCodes, code)
	p.mu.Unlock()
}

func assertRuntimeLabelSecretsAbsent(
	t *testing.T,
	ctx context.Context,
	pool *pgxpool.Pool,
	evidence [][]byte,
	processes []*childProcess,
	secrets []string,
) {
	t.Helper()
	for _, secret := range secrets {
		for _, body := range evidence {
			if bytes.Contains(body, []byte(secret)) {
				t.Fatalf("public/Operations response exposed Runtime credential canary")
			}
		}
		for _, process := range processes {
			if strings.Contains(process.logs.redacted(), secret) {
				t.Fatalf("%s logs exposed Runtime credential canary", process.name)
			}
		}
		var leaked bool
		err := pool.QueryRow(ctx, `
			SELECT EXISTS (
				SELECT 1 FROM runtime_config_versions WHERE position($1 in canonical_document) > 0
				UNION ALL SELECT 1 FROM workflow_runs WHERE position($1 in runtime_config_snapshot::text) > 0
				UNION ALL SELECT 1 FROM stage_allocations WHERE position($1 in coalesce(runtime_configuration::text, '')) > 0
				UNION ALL SELECT 1 FROM planner_events WHERE position($1 in event::text) > 0
				UNION ALL SELECT 1 FROM workflow_run_events WHERE position($1 in data::text) > 0
				UNION ALL SELECT 1 FROM runtime_credentials WHERE position(convert_to($1, 'UTF8') in ciphertext) > 0
			)
		`, secret).Scan(&leaked)
		if err != nil {
			t.Fatalf("scan retained database for credential canary: %v", err)
		}
		if leaked {
			t.Fatal("retained database exposed Runtime credential canary")
		}
	}
}
