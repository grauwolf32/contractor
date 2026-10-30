//go:build e2e

package e2e

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/localpki"
)

type engagementProcessHarness struct {
	*findingsHarness
	proxy            *engagementActionProxy
	controlClient    *http.Client
	httpObservations string
	runtimeURL       string
	startServer      func() *childProcess
	startRuntime     func() *childProcess
	secretsMu        sync.Mutex
	secrets          []string
}

func startEngagementProcessHarness(t *testing.T, targetCA, mode string) *engagementProcessHarness {
	return startEngagementProcessHarnessWithEnvironment(t, targetCA, mode, nil)
}

func startEngagementProcessHarnessWithEnvironment(t *testing.T, targetCA, mode string, environment map[string]string) *engagementProcessHarness {
	return startEngagementProcessHarnessOptions(t, targetCA, mode, environment, false)
}

func startEngagementProcessHarnessOptions(t *testing.T, targetCA, mode string, environment map[string]string, blockGateway bool, configure ...func(string)) *engagementProcessHarness {
	t.Helper()
	if testing.Short() {
		t.Fatal("findings process checks cannot run in short mode")
	}
	databaseURL := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if databaseURL == "" {
		t.Fatal("CONTRACTOR_TEST_DATABASE_URL is required")
	}
	repositoryRoot := repoRoot(t)
	temporaryRoot := t.TempDir()
	ctx, cancel := context.WithTimeout(context.Background(), 4*time.Minute)
	t.Cleanup(cancel)

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
		LeafOptions: leaf, URI: "urn:contractor:control-plane:findings-e2e",
	})
	if err != nil {
		t.Fatalf("issue Control Plane certificate: %v", err)
	}
	agentPaths, err := generator.IssueAgent(pkiRoot, "findings-agent", leaf)
	if err != nil {
		t.Fatalf("issue Runtime Agent certificate: %v", err)
	}

	gateway := newBlockedDomainGateway(llmGatewayToken, []domainGatewayStage{engagementFixtureStage()})
	t.Cleanup(gateway.close)
	configRoot := stageE2EConfiguration(
		t, filepath.Join(repositoryRoot, "configs"),
		filepath.Join(temporaryRoot, "configs"), gateway.URL(),
	)
	installEngagementFixture(t, configRoot)
	for _, edit := range configure {
		edit(configRoot)
	}
	publicAddress, privateAddress, runtimeAddress := freeAddress(t), freeAddress(t), freeAddress(t)
	publicBaseURL := "http://" + publicAddress
	privateBaseURL := "https://" + privateAddress
	runtimeBaseURL := "https://" + runtimeAddress
	proxy := newEngagementActionProxy(t, privateBaseURL, caPaths.Certificate, agentPaths, controlPlanePaths, mode)
	privateBaseURL = proxy.server.URL
	userID := "findings-user-" + randomHex(t, 8)
	localAuthFile := writeE2ELocalAuth(t, temporaryRoot, userID)
	masterKeyFile := filepath.Join(temporaryRoot, "credential-master-key")
	if err := os.WriteFile(masterKeyFile, []byte(base64.StdEncoding.EncodeToString(bytes.Repeat([]byte{0x5a}, 32))), 0o600); err != nil {
		t.Fatal(err)
	}
	serverEnvironment := map[string]string{
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
	}
	startServer := func() *childProcess {
		return startProcess(t, "Go Server", repositoryRoot, serverEnvironment, serverBinary, "serve", "--credential-master-key-file", masterKeyFile)
	}
	server := startServer()

	client := &http.Client{Timeout: 8 * time.Second}
	waitForHTTP(t, ctx, server, client, publicBaseURL+"/readyz", http.StatusOK)
	python := filepath.Join(repositoryRoot, "runtime", ".venv", "bin", "python")
	if info, statErr := os.Stat(python); statErr != nil || info.IsDir() {
		t.Fatalf("Python Runtime environment is missing at %s; run 'cd runtime && uv sync --locked'", python)
	}
	workRoot := filepath.Join(temporaryRoot, "runtime-work")
	httpObservations := filepath.Join(temporaryRoot, "http-observations.jsonl")
	runtimeEnvironment := map[string]string{"PYTHONUNBUFFERED": "1", "CONTRACTOR_TEST_TARGET_CA": targetCA, "CONTRACTOR_TEST_HTTP_OBSERVATIONS": httpObservations}
	for key, value := range environment {
		runtimeEnvironment[key] = value
	}
	startRuntime := func() *childProcess {
		return startProcess(
			t, "Python Runtime Agent", filepath.Join(repositoryRoot, "runtime"),
			runtimeEnvironment,
			python, filepath.Join(repositoryRoot, "tests", "e2e", "testdata", "engagement", "runtime.py"),
			"--control-plane-url", privateBaseURL,
			"--advertised-control-url", runtimeBaseURL,
			"--advertised-a2a-url", runtimeBaseURL,
			"--ca-file", caPaths.Certificate,
			"--certificate-file", agentPaths.Certificate,
			"--private-key-file", agentPaths.PrivateKey,
			"--listen", runtimeAddress,
			"--work-root", workRoot,
			"--workspace-storage", "local",
			"--workspace-work-root", filepath.Join(temporaryRoot, "workspace"),
			"--request-timeout-seconds", "12",
			"--shutdown-grace-seconds", "5",
		)
	}
	runtimeProcess := startRuntime()
	controlClient := newMTLSClient(t, caPaths.Certificate, controlPlanePaths)
	waitForHTTP(t, ctx, runtimeProcess, controlClient, runtimeBaseURL+"/healthz", http.StatusOK)
	waitForProcessLog(t, ctx, runtimeProcess, "runtime agent registered")
	h := &engagementProcessHarness{findingsHarness: &findingsHarness{ctx: ctx, server: server, runtimeProcess: runtimeProcess,
		gateway: gateway, client: client, baseURL: publicBaseURL, databaseURL: isolateURL, configRoot: configRoot}, proxy: proxy, httpObservations: httpObservations, controlClient: controlClient, runtimeURL: runtimeBaseURL, startServer: startServer, startRuntime: startRuntime}
	t.Cleanup(func() {
		if t.Failed() {
			h.secretsMu.Lock()
			secrets := append([]string{publicToken, llmGatewayToken}, h.secrets...)
			h.secretsMu.Unlock()
			t.Logf("engagement server:\n%s\nruntime:\n%s\ngateway: %v", h.server.logs.redacted(secrets...), h.runtimeProcess.logs.redacted(secrets...), gateway.Failures())
		}
	})

	if !blockGateway {
		gateway.releaseBlockedRequest()
	}
	return h
}

// Canary checks read raw observations. This list only protects failure logging.
func (h *engagementProcessHarness) protectSecrets(secrets ...string) {
	for _, secret := range secrets {
		digest := sha256.Sum256([]byte(secret))
		secrets = append(secrets, hex.EncodeToString(digest[:]))
	}
	h.secretsMu.Lock()
	h.secrets = append(h.secrets, secrets...)
	h.secretsMu.Unlock()
	h.server.logs.observeCanaries(secrets...)
	h.runtimeProcess.logs.observeCanaries(secrets...)
	h.gateway.forbid(secrets...)
}

func installEngagementFixture(t *testing.T, root string) {
	t.Helper()
	for source, target := range map[string]string{"model-policy.yaml": "model-policies/engagement_fixture.yaml", "agent.yaml": "agent-templates/engagement_fixture.yaml", "workflow.yaml": "workflows/engagement_fixture.yaml", "profile.yaml": "audit-profiles/engagement_fixture.yaml"} {
		body, err := os.ReadFile(filepath.Join(repoRoot(t), "tests", "e2e", "testdata", "engagement", source))
		if err != nil {
			t.Fatal(err)
		}
		if err = os.WriteFile(filepath.Join(root, target), body, 0600); err != nil {
			t.Fatal(err)
		}
	}
	if err := os.WriteFile(filepath.Join(root, "instructions", "engagement-fixture.md"), []byte("Read the assigned Audit task and submit its bounded result. Choose target requests explicitly using http_request and optional credential_id."), 0600); err != nil {
		t.Fatal(err)
	}
}
func engagementFixtureStage() domainGatewayStage {
	return engagementRequestStage(nil)
}

func engagementRequestStage(requests []map[string]any) domainGatewayStage {
	steps := []domainGatewayStep{toolGatewayStep("read_audit_task", fixedArguments(map[string]any{})), toolGatewayStep("http_credentials", fixedArguments(map[string]any{}))}
	for _, request := range requests {
		steps = append(steps, toolGatewayStep("http_request", fixedArguments(request)))
	}
	steps = append(steps, toolGatewayStep("submit_check_result", findingResultArguments(map[string]any{"assessment": "satisfied", "summary": "The explicit HTTP fixture completed; this model result makes no target vulnerability claim.", "completed": []string{}, "gaps": []string{}, "evidence": []any{}})), finalGatewayStep("Fixture result recorded", nil))
	return domainGatewayStage{name: "engagement/explicit-http", tools: []string{"read_audit_task", "submit_check_result", "http_request", "http_credentials", "http_credential_read"}, steps: steps}
}

func setEngagementRequests(h *engagementProcessHarness, requests []map[string]any) {
	h.gateway.mu.Lock()
	defer h.gateway.mu.Unlock()
	h.gateway.stages = []domainGatewayStage{engagementRequestStage(requests)}
	h.gateway.stageIndex, h.gateway.stepIndex = 0, 0
}

func setEngagementURLs(h *engagementProcessHarness, urls []string) {
	requests := make([]map[string]any, 0, len(urls))
	for _, url := range urls {
		requests = append(requests, map[string]any{"url": url})
	}
	setEngagementRequests(h, requests)
}

func engagementAuditPOST(t *testing.T, h *engagementProcessHarness, path string, body any, revision uint64, status int, result any) {
	t.Helper()
	data, err := json.Marshal(body)
	if err != nil {
		t.Fatal(err)
	}
	if body == nil {
		data = nil
	}
	request, err := http.NewRequestWithContext(h.ctx, http.MethodPost, h.baseURL+path, strings.NewReader(string(data)))
	if err != nil {
		t.Fatal(err)
	}
	request.Header.Set("Authorization", "Bearer "+publicToken)
	request.Header.Set("Content-Type", "application/json")
	request.Header.Set("Idempotency-Key", "engagement-"+randomHex(t, 8))
	if revision != 0 {
		request.Header.Set("If-Match", fmt.Sprintf(`"%d"`, revision))
	}
	response := do(t, h.client, request, status)
	defer response.Body.Close()
	if result != nil {
		decodeAuditProgramResponse(t, response, result)
	}
}
