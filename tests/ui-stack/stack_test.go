//go:build e2e

package uistack

import (
	"archive/zip"
	"bytes"
	"context"
	cryptorand "crypto/rand"
	"crypto/tls"
	"crypto/x509"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"log"
	"net"
	"net/http"
	"net/http/httputil"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/auth"
	workflowconfig "github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/localpki"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

const (
	uiStackPassword    = "contractor ui stack local password"
	publicBearerCanary = "contractor-ui-stack-public-bearer-CANARY"
	modelGatewayCanary = "sk-contractor-ui-stack-model-CANARY"
	managerAdminCanary = "sk-contractor-ui-stack-admin-CANARY"
	generatedKeyCanary = "sk-contractor-ui-stack-generated-CANARY"
	projectSource      = `from fastapi import Depends, FastAPI, HTTPException
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
)

type uiStack struct {
	t              *testing.T
	ctx            context.Context
	repositoryRoot string
	temporaryRoot  string
	pool           *pgxpool.Pool

	serverBinary      string
	serverEnv         []string
	uiEnv             []string
	serverInternalURL string
	uiInternalURL     string

	processMu sync.Mutex
	server    *childProcess
	ui        *childProcess
	runtime   *childProcess
	processes []*childProcess

	modelGateway       *modelGateway
	credentialManager  *credentialManagerFixture
	controlToken       string
	controlServer      *http.Server
	controlURL         string
	apiProxy           *http.Server
	uiProxy            *http.Server
	eventConnectionsMu sync.Mutex
	eventConnections   map[net.Conn]struct{}
	secrets            []string
}

func TestBrowserOperationsStack(t *testing.T) {
	if testing.Short() {
		t.Skip("browser end-to-end process test")
	}
	databaseURL := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if databaseURL == "" {
		t.Fatal("CONTRACTOR_TEST_DATABASE_URL is required")
	}
	repositoryRoot := repoRoot(t)
	temporaryRoot := t.TempDir()
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()

	isolateURL := isolatedDatabase(t, ctx, databaseURL)
	serverBinary := filepath.Join(temporaryRoot, "contractor-server")
	runChecked(t, repositoryRoot, inheritedEnvironment(nil), "go", "build", "-o", serverBinary, "./cmd/contractor-server")
	runChecked(t, repositoryRoot, inheritedEnvironment(map[string]string{
		"CONTRACTOR_DATABASE_URL": isolateURL,
	}), serverBinary, "migrate")

	modelGateway := newModelGateway(modelGatewayCanary)
	credentialManager := newCredentialManagerFixture(managerAdminCanary, generatedKeyCanary)
	t.Cleanup(modelGateway.close)
	t.Cleanup(credentialManager.close)

	configRoot := stageUIStackConfiguration(
		t, repositoryRoot, filepath.Join(temporaryRoot, "configs"),
		modelGateway.URL(), credentialManager.URL(),
	)
	managedConfigRoot := filepath.Join(temporaryRoot, "managed-configs")
	if err := os.MkdirAll(managedConfigRoot, 0o750); err != nil {
		t.Fatal(err)
	}
	snapshot, err := workflowconfig.Load(configRoot, workflowconfig.MVPDescriptors())
	if err != nil {
		t.Fatalf("load staged UI configuration: %v", err)
	}
	gateway, err := snapshot.LLMGateway("local-litellm@1")
	if err != nil {
		t.Fatal(err)
	}
	masterKeyFile := writeCredentialMasterKey(t, temporaryRoot)
	adminKeyFile := writeSecureFile(t, temporaryRoot, "litellm-admin-key", []byte(managerAdminCanary))
	bindings := fmt.Sprintf(`bindings:
  - llmGateway:
      gatewayId: %s
      version: %q
      digest: %s
    adminKeyFile: %q
`, gateway.Ref.GatewayID, gateway.Ref.Version, gateway.Ref.Digest, adminKeyFile)
	bindingsFile := writeSecureFile(t, temporaryRoot, "gateway-admin-bindings.yaml", []byte(bindings))

	pkiRoot := filepath.Join(temporaryRoot, "pki")
	generator := localpki.Generator{}
	caPaths, err := generator.InitCA(pkiRoot, false)
	if err != nil {
		t.Fatalf("initialize test CA: %v", err)
	}
	leaf := localpki.LeafOptions{IPAddresses: []net.IP{net.ParseIP("127.0.0.1")}}
	controlPlanePaths, err := generator.IssueControlPlane(pkiRoot, localpki.ControlPlaneOptions{
		LeafOptions: leaf, URI: "urn:contractor:control-plane:ui-stack-e2e",
	})
	if err != nil {
		t.Fatalf("issue Control Plane certificate: %v", err)
	}
	agentPaths, err := generator.IssueAgent(pkiRoot, "ui-stack-e2e-agent", leaf)
	if err != nil {
		t.Fatalf("issue Runtime Agent certificate: %v", err)
	}
	proxyPaths, err := generator.IssueAgent(pkiRoot, "ui-stack-browser-proxy", localpki.LeafOptions{
		DNSNames: []string{"api.contractor.test", "ui.contractor.test"},
	})
	if err != nil {
		t.Fatalf("issue browser proxy certificate: %v", err)
	}

	publicAddress := freeAddress(t, "127.0.0.1")
	privateAddress := freeAddress(t, "127.0.0.1")
	runtimeAddress := freeAddress(t, "127.0.0.1")
	uiAddress := freeAddress(t, "127.0.0.1")
	apiProxyAddress := freeAddress(t, "127.0.0.1")
	uiProxyAddress := freeAddress(t, "127.0.0.1")
	_, apiProxyPort, _ := net.SplitHostPort(apiProxyAddress)
	_, uiProxyPort, _ := net.SplitHostPort(uiProxyAddress)
	serverInternalURL := "http://" + publicAddress
	uiInternalURL := "http://" + uiAddress
	serverURL := "https://api.contractor.test:" + apiProxyPort
	apiDirectURL := "https://127.0.0.1:" + apiProxyPort
	privateURL := "https://" + privateAddress
	runtimeURL := "https://" + runtimeAddress
	uiURL := "https://ui.contractor.test:" + uiProxyPort
	uiDirectURL := "https://127.0.0.1:" + uiProxyPort
	_, uiPort, _ := net.SplitHostPort(uiAddress)
	userID := "ui-stack-user-" + randomHex(t, 8)
	localAuthFile := writeLocalAuth(t, temporaryRoot, userID)

	serverOverrides := map[string]string{
		"CONTRACTOR_DATABASE_URL":            isolateURL,
		"CONTRACTOR_OPERATOR_CONFIG_ROOT":    configRoot,
		"CONTRACTOR_MANAGED_CONFIG_ROOT":     managedConfigRoot,
		"CONTRACTOR_PUBLIC_LISTEN":           publicAddress,
		"CONTRACTOR_PRIVATE_LISTEN":          privateAddress,
		"CONTRACTOR_PRIVATE_URL":             privateURL,
		"CONTRACTOR_CA_FILE":                 caPaths.Certificate,
		"CONTRACTOR_CONTROL_PLANE_CERT_FILE": controlPlanePaths.Certificate,
		"CONTRACTOR_CONTROL_PLANE_KEY_FILE":  controlPlanePaths.PrivateKey,
		"CONTRACTOR_LLM_GATEWAY_TOKEN":       modelGatewayCanary,
		"CONTRACTOR_PUBLIC_USER_ID":          userID,
		"CONTRACTOR_PUBLIC_BEARER_TOKEN":     publicBearerCanary,
		"CONTRACTOR_LOCAL_AUTH_FILE":         localAuthFile,
		"CONTRACTOR_BROWSER_ORIGINS":         uiURL,
	}
	// Credential file options intentionally have no environment alias in the
	// production parser; startServerOrFatal passes their explicit flags.
	serverEnvironment := inheritedEnvironment(serverOverrides)
	uiEnvironment := cleanEnvironment(map[string]string{
		"NODE_ENV":                   "production",
		"CONTRACTOR_UI_API_BASE_URL": serverURL,
		"CONTRACTOR_UI_HOST":         "127.0.0.1",
		"CONTRACTOR_UI_PORT":         uiPort,
		"CONTRACTOR_UI_DIST_DIR":     filepath.Join(repositoryRoot, "ui", "dist"),
	}, "PATH", "LANG", "LC_ALL", "TZ")
	secrets := []string{publicBearerCanary, modelGatewayCanary, managerAdminCanary, generatedKeyCanary}
	assertEnvironmentHasNoSecrets(t, uiEnvironment, secrets)

	pool, err := pgxpool.New(ctx, isolateURL)
	if err != nil {
		t.Fatalf("open UI-stack assertion database: %v", err)
	}
	t.Cleanup(pool.Close)
	stack := &uiStack{
		t: t, ctx: ctx, repositoryRoot: repositoryRoot, temporaryRoot: temporaryRoot,
		pool: pool, serverBinary: serverBinary,
		serverEnv: serverEnvironment, uiEnv: uiEnvironment,
		serverInternalURL: serverInternalURL, uiInternalURL: uiInternalURL,
		modelGateway: modelGateway, credentialManager: credentialManager,
		controlToken: randomHex(t, 24), secrets: secrets,
	}
	t.Cleanup(stack.close)
	stack.startTLSProxies(proxyPaths, apiProxyAddress, uiProxyAddress)
	stack.startServerOrFatal(masterKeyFile, bindingsFile)

	python := filepath.Join(repositoryRoot, "runtime", ".venv", "bin", "python")
	if info, statErr := os.Stat(python); statErr != nil || info.IsDir() {
		t.Fatalf("Python Runtime environment is missing at %s; run 'cd runtime && uv sync --locked'", python)
	}
	validatorBin, _ := installValidators(t, temporaryRoot)
	workRoot := filepath.Join(temporaryRoot, "runtime-work")
	runtimeProcess := startProcess(
		t, "Python Runtime Agent", filepath.Join(repositoryRoot, "runtime"),
		inheritedEnvironment(map[string]string{
			"PATH":             validatorBin + string(os.PathListSeparator) + os.Getenv("PATH"),
			"PYTHONUNBUFFERED": "1",
		}),
		python, "-m", "contractor_runtime",
		"--control-plane-url", privateURL,
		"--advertised-control-url", runtimeURL,
		"--advertised-a2a-url", runtimeURL,
		"--ca-file", caPaths.Certificate,
		"--certificate-file", agentPaths.Certificate,
		"--private-key-file", agentPaths.PrivateKey,
		"--listen", runtimeAddress,
		"--work-root", workRoot,
		"--request-timeout-seconds", "90",
		"--shutdown-grace-seconds", "5",
	)
	stack.processMu.Lock()
	stack.runtime = runtimeProcess
	stack.processes = append(stack.processes, runtimeProcess)
	stack.processMu.Unlock()
	controlClient := newMTLSClient(t, caPaths.Certificate, controlPlanePaths)
	waitForHTTP(t, ctx, runtimeProcess, controlClient, runtimeURL+"/healthz", http.StatusOK, secrets)
	waitForProcessLog(t, ctx, runtimeProcess, "runtime agent registered", secrets)

	uiBuildEnvironment := cleanEnvironment(nil, "PATH", "HOME", "XDG_CACHE_HOME", "COREPACK_HOME", "TMPDIR", "LANG", "LC_ALL", "TZ")
	runChecked(t, filepath.Join(repositoryRoot, "ui"), uiBuildEnvironment, "corepack", "pnpm", "build")
	stack.startUIOrFatal()
	stack.startControlServer()

	sourceArchive := filepath.Join(temporaryRoot, "project-source.zip")
	if err := os.WriteFile(sourceArchive, projectSourceArchive(t), 0o600); err != nil {
		t.Fatal(err)
	}
	evidencePath := filepath.Join(temporaryRoot, "browser-evidence.json")
	screenshotPath := filepath.Join(temporaryRoot, "browser-state.png")
	playwrightOutput := filepath.Join(temporaryRoot, "playwright-output")
	playwrightEnvironment := cleanEnvironment(map[string]string{
		"CONTRACTOR_UI_STACK":               "1",
		"CONTRACTOR_UI_E2E_BASE_URL":        uiURL,
		"CONTRACTOR_UI_E2E_API_URL":         serverURL,
		"CONTRACTOR_UI_E2E_API_DIRECT_URL":  apiDirectURL,
		"CONTRACTOR_UI_E2E_UI_DIRECT_URL":   uiDirectURL,
		"CONTRACTOR_UI_E2E_CONTROL_URL":     stack.controlURL,
		"CONTRACTOR_UI_E2E_CONTROL_TOKEN":   stack.controlToken,
		"CONTRACTOR_UI_E2E_USERNAME":        "admin",
		"CONTRACTOR_UI_E2E_PASSWORD":        uiStackPassword,
		"CONTRACTOR_UI_E2E_SOURCE_ARCHIVE":  sourceArchive,
		"CONTRACTOR_UI_E2E_EVIDENCE_PATH":   evidencePath,
		"CONTRACTOR_UI_E2E_SCREENSHOT_PATH": screenshotPath,
		"CONTRACTOR_UI_E2E_OUTPUT_DIR":      playwrightOutput,
	}, "PATH", "HOME", "XDG_CACHE_HOME", "COREPACK_HOME", "TMPDIR", "LANG", "LC_ALL", "TZ")
	assertEnvironmentHasNoSecrets(t, playwrightEnvironment, secrets)
	runChecked(
		t, filepath.Join(repositoryRoot, "ui"), playwrightEnvironment,
		"corepack", "pnpm", "exec", "playwright", "test",
	)
	tracePath := singleFileNamed(t, playwrightOutput, "trace.zip")

	stack.assertCredentialLifecycle()
	stack.assertSecretBoundaries(evidencePath, tracePath, screenshotPath)
	calls, completedStages, gatewayFailures := modelGateway.snapshot()
	if calls != 34 || completedStages != 4 || len(gatewayFailures) != 0 {
		t.Fatalf("model Gateway calls/stages/failures = %d/%d/%v, want 34/4/none", calls, completedStages, gatewayFailures)
	}
	creates, deletes, managerFailures := credentialManager.snapshot()
	if creates != 1 || deletes != 1 || len(managerFailures) != 0 {
		t.Fatalf("credential manager creates/deletes/failures = %d/%d/%v, want 1/1/none", creates, deletes, managerFailures)
	}
}

func singleFileNamed(t *testing.T, root, name string) string {
	t.Helper()
	var matches []string
	if err := filepath.WalkDir(root, func(path string, entry fs.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		if !entry.IsDir() && entry.Name() == name {
			matches = append(matches, path)
		}
		return nil
	}); err != nil {
		t.Fatalf("find %s in %s: %v", name, root, err)
	}
	if len(matches) != 1 {
		t.Fatalf("found %d files named %s in %s, want one", len(matches), name, root)
	}
	return matches[0]
}

func (s *uiStack) serverCommand(masterKeyFile, bindingsFile string) (string, []string) {
	return s.serverBinary, []string{
		"serve", "--runtime-request-timeout", "90s", "--planner-timeout", "3m",
		"--credential-master-key-file", masterKeyFile,
		"--llm-gateway-admin-bindings-file", bindingsFile,
	}
}

func (s *uiStack) startServerOrFatal(masterKeyFile, bindingsFile string) {
	s.t.Helper()
	command, args := s.serverCommand(masterKeyFile, bindingsFile)
	process := startProcess(s.t, "Go Server", s.repositoryRoot, s.serverEnv, command, args...)
	s.processMu.Lock()
	s.server = process
	s.processes = append(s.processes, process)
	s.processMu.Unlock()
	waitForHTTP(s.t, s.ctx, process, &http.Client{Timeout: 5 * time.Second}, s.serverInternalURL+"/readyz", http.StatusOK, s.secrets)
}

func (s *uiStack) startUIOrFatal() {
	s.t.Helper()
	process := startProcess(
		s.t, "Node UI", filepath.Join(s.repositoryRoot, "ui"), s.uiEnv,
		"node", "server/index.mjs",
	)
	s.processMu.Lock()
	s.ui = process
	s.processes = append(s.processes, process)
	s.processMu.Unlock()
	waitForHTTP(s.t, s.ctx, process, &http.Client{Timeout: 5 * time.Second}, s.uiInternalURL+"/healthz", http.StatusOK, s.secrets)
}

func (s *uiStack) startTLSProxies(
	certificate localpki.Paths, apiAddress, uiAddress string,
) {
	s.t.Helper()
	start := func(name, address, target string, trackHijacked bool) *http.Server {
		upstream, err := url.Parse(target)
		if err != nil {
			s.t.Fatal(err)
		}
		proxy := httputil.NewSingleHostReverseProxy(upstream)
		proxy.ErrorHandler = func(w http.ResponseWriter, _ *http.Request, err error) {
			http.Error(w, name+" upstream unavailable", http.StatusBadGateway)
			_ = err
		}
		server := &http.Server{
			Handler: proxy, ReadHeaderTimeout: 5 * time.Second,
			TLSConfig: &tls.Config{MinVersion: tls.VersionTLS13},
			ErrorLog:  log.New(io.Discard, "", 0),
		}
		if trackHijacked {
			server.ConnState = func(connection net.Conn, state http.ConnState) {
				s.eventConnectionsMu.Lock()
				defer s.eventConnectionsMu.Unlock()
				switch state {
				case http.StateHijacked:
					if s.eventConnections == nil {
						s.eventConnections = make(map[net.Conn]struct{})
					}
					s.eventConnections[connection] = struct{}{}
				case http.StateClosed:
					delete(s.eventConnections, connection)
				}
			}
		}
		listener, err := net.Listen("tcp", address)
		if err != nil {
			s.t.Fatal(err)
		}
		go func() {
			if serveErr := server.ServeTLS(listener, certificate.Certificate, certificate.PrivateKey); serveErr != nil && !errors.Is(serveErr, http.ErrServerClosed) {
				s.t.Errorf("%s TLS proxy: %v", name, serveErr)
			}
		}()
		return server
	}
	s.apiProxy = start("API proxy", apiAddress, s.serverInternalURL, true)
	s.uiProxy = start("UI proxy", uiAddress, s.uiInternalURL, false)
}

func (s *uiStack) interruptEvents() error {
	s.eventConnectionsMu.Lock()
	connections := make([]net.Conn, 0, len(s.eventConnections))
	for connection := range s.eventConnections {
		connections = append(connections, connection)
		delete(s.eventConnections, connection)
	}
	s.eventConnectionsMu.Unlock()
	if len(connections) == 0 {
		return errors.New("no active event WebSocket connection")
	}
	for _, connection := range connections {
		if err := connection.Close(); err != nil && !errors.Is(err, net.ErrClosed) {
			return err
		}
	}
	return nil
}

func (s *uiStack) restartUI() error {
	s.processMu.Lock()
	defer s.processMu.Unlock()
	s.ui.stop(s.t)
	process, err := launchProcess(
		"Node UI", filepath.Join(s.repositoryRoot, "ui"), s.uiEnv,
		"node", "server/index.mjs",
	)
	if err != nil {
		return err
	}
	s.ui = process
	s.processes = append(s.processes, process)
	return waitForHTTPError(s.ctx, process, &http.Client{Timeout: 3 * time.Second}, s.uiInternalURL+"/healthz", http.StatusOK)
}

func (s *uiStack) restartServer() error {
	s.processMu.Lock()
	defer s.processMu.Unlock()
	s.server.stop(s.t)
	masterKeyFile := filepath.Join(s.temporaryRoot, "credential-master-key")
	bindingsFile := filepath.Join(s.temporaryRoot, "gateway-admin-bindings.yaml")
	command, args := s.serverCommand(masterKeyFile, bindingsFile)
	process, err := launchProcess("Go Server", s.repositoryRoot, s.serverEnv, command, args...)
	if err != nil {
		return err
	}
	s.server = process
	s.processes = append(s.processes, process)
	return waitForHTTPError(s.ctx, process, &http.Client{Timeout: 3 * time.Second}, s.serverInternalURL+"/readyz", http.StatusOK)
}

func (s *uiStack) startControlServer() {
	s.t.Helper()
	listener, err := net.Listen("tcp", freeAddress(s.t, "127.0.0.3"))
	if err != nil {
		s.t.Fatal(err)
	}
	s.controlURL = "http://" + listener.Addr().String()
	mux := http.NewServeMux()
	mux.HandleFunc("POST /{action}", func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "Bearer "+s.controlToken || r.ContentLength > 0 {
			http.Error(w, "forbidden", http.StatusForbidden)
			return
		}
		var actionErr error
		switch r.PathValue("action") {
		case "release-worker":
			s.modelGateway.release()
		case "restart-ui":
			actionErr = s.restartUI()
		case "restart-server":
			actionErr = s.restartServer()
		case "interrupt-events":
			actionErr = s.interruptEvents()
		case "check-credential-storage":
			actionErr = s.checkCredentialStorage(r.Context())
		default:
			http.NotFound(w, r)
			return
		}
		if actionErr != nil {
			http.Error(w, "control action failed: "+actionErr.Error(), http.StatusInternalServerError)
			return
		}
		w.WriteHeader(http.StatusNoContent)
	})
	s.controlServer = &http.Server{Handler: mux, ReadHeaderTimeout: 3 * time.Second}
	go func() {
		if serveErr := s.controlServer.Serve(listener); serveErr != nil && !errors.Is(serveErr, http.ErrServerClosed) {
			s.t.Errorf("UI-stack control server: %v", serveErr)
		}
	}()
}

func (s *uiStack) checkCredentialStorage(ctx context.Context) error {
	var rows int
	var plaintextInCiphertext bool
	var safeColumns string
	err := s.pool.QueryRow(ctx, `
SELECT count(*),
       COALESCE(bool_or(position(convert_to($1, 'UTF8') in ciphertext) > 0), false),
       COALESCE(string_agg(concat_ws('|', credential_id, llm_gateway_id,
         llm_gateway_version, llm_gateway_digest, remote_key_id,
         COALESCE(label, ''), gateway_policy::text, encryption_schema_version,
         key_id), E'\n'), '')
FROM llm_credentials
WHERE credential_id = 'ui-stack-key'`, generatedKeyCanary).Scan(&rows, &plaintextInCiphertext, &safeColumns)
	if err != nil {
		return err
	}
	if rows != 1 || plaintextInCiphertext || containsAny(safeColumns, s.secrets) {
		return errors.New("active credential storage exposed a plaintext secret")
	}
	return nil
}

func (s *uiStack) assertCredentialLifecycle() {
	s.t.Helper()
	var active, tombstones int
	if err := s.pool.QueryRow(s.ctx, `SELECT
  (SELECT count(*) FROM llm_credentials WHERE credential_id = 'ui-stack-key'),
  (SELECT count(*) FROM llm_credential_tombstones WHERE credential_id = 'ui-stack-key')`).Scan(&active, &tombstones); err != nil {
		s.t.Fatal(err)
	}
	if active != 0 || tombstones != 1 {
		s.t.Fatalf("credential active/tombstone rows = %d/%d, want 0/1", active, tombstones)
	}
	var safeRows string
	if err := s.pool.QueryRow(s.ctx, `SELECT concat_ws(E'\n',
  COALESCE((SELECT string_agg(concat_ws('|', kind, name, version, digest,
    request_digest, idempotency_key_digest, actor_id), E'\n') FROM configuration_publications), ''),
  COALESCE((SELECT string_agg(concat_ws('|', operation_id, idempotency_key,
    request_hash, credential_id, operation_kind, phase, request::text), E'\n')
    FROM credential_operations), ''),
  COALESCE((SELECT string_agg(concat_ws('|', credential_id, actor_id), E'\n')
    FROM llm_credential_tombstones), ''))`).Scan(&safeRows); err != nil {
		s.t.Fatal(err)
	}
	if containsAny(safeRows, s.secrets) {
		s.t.Fatal("database safe columns contain a seeded secret canary")
	}
}

func (s *uiStack) assertSecretBoundaries(evidencePath, tracePath, screenshotPath string) {
	s.t.Helper()
	s.processMu.Lock()
	processes := append([]*childProcess(nil), s.processes...)
	s.processMu.Unlock()
	for _, process := range processes {
		if containsAny(process.logs.contents(), s.secrets) {
			s.t.Fatalf("%s logs contain a seeded secret canary", process.name)
		}
	}
	assertEvidenceHasNoSecrets(s.t, evidencePath, s.secrets)
	assertZipHasNoSecrets(s.t, tracePath, s.secrets)
	if data, err := os.ReadFile(screenshotPath); err != nil || containsAny(string(data), s.secrets) {
		s.t.Fatalf("screenshot binary secret scan failed or matched: %v", err)
	}
	if tesseract, err := exec.LookPath("tesseract"); err == nil {
		command := exec.Command(tesseract, screenshotPath, "stdout")
		command.Dir = s.repositoryRoot
		command.Env = cleanEnvironment(nil, "PATH", "LANG", "LC_ALL", "TESSDATA_PREFIX")
		outputBytes, runErr := command.CombinedOutput()
		if runErr != nil {
			s.t.Logf("skip optional screenshot OCR: %v", runErr)
			return
		}
		output := string(outputBytes)
		if containsAny(output, s.secrets) {
			s.t.Fatal("screenshot OCR contains a seeded secret canary")
		}
	}
}

func (s *uiStack) close() {
	s.modelGateway.release()
	_ = s.interruptEvents()
	if s.controlServer != nil {
		ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
		_ = s.controlServer.Shutdown(ctx)
		cancel()
	}
	for _, proxy := range []*http.Server{s.uiProxy, s.apiProxy} {
		if proxy == nil {
			continue
		}
		ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
		_ = proxy.Shutdown(ctx)
		cancel()
	}
	s.processMu.Lock()
	processes := append([]*childProcess(nil), s.processes...)
	s.processMu.Unlock()
	for index := len(processes) - 1; index >= 0; index-- {
		processes[index].stop(s.t)
	}
}

func stageUIStackConfiguration(
	t *testing.T, repositoryRoot, target, gatewayURL, managementURL string,
) string {
	t.Helper()
	source := filepath.Join(repositoryRoot, "configs")
	encodedGatewayURL, _ := json.Marshal(gatewayURL)
	encodedManagementURL, _ := json.Marshal(managementURL)
	gatewayUpdated, managerUpdated, workflowUpdates := false, false, 0
	err := filepath.WalkDir(source, func(path string, entry fs.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		relative, err := filepath.Rel(source, path)
		if err != nil {
			return err
		}
		if relative == "examples" || strings.HasPrefix(relative, "examples"+string(filepath.Separator)) {
			if entry.IsDir() {
				return filepath.SkipDir
			}
			return nil
		}
		destination := filepath.Join(target, relative)
		if entry.Type()&os.ModeSymlink != 0 {
			return errors.New("configuration contains a symbolic link")
		}
		if entry.IsDir() {
			return os.MkdirAll(destination, 0o700)
		}
		if !entry.Type().IsRegular() {
			return errors.New("configuration contains a non-regular entry")
		}
		data, err := os.ReadFile(path)
		if err != nil {
			return err
		}
		text := string(data)
		if relative == filepath.Join("llm-gateways", "local_litellm.yaml") {
			lines := strings.Split(text, "\n")
			for index, line := range lines {
				switch {
				case strings.HasPrefix(line, "  url: "):
					lines[index] = "  url: " + string(encodedGatewayURL)
					gatewayUpdated = true
				case strings.HasPrefix(line, "    managementUrl: "):
					lines[index] = "    managementUrl: " + string(encodedManagementURL)
					managerUpdated = true
				}
			}
			text = strings.Join(lines, "\n")
		}
		if strings.HasPrefix(relative, "workflows"+string(filepath.Separator)) {
			text, workflowUpdates = addDevelopmentCredential(text, workflowUpdates)
		}
		return os.WriteFile(destination, []byte(text), 0o600)
	})
	if err != nil {
		t.Fatalf("stage UI-stack configuration: %v", err)
	}
	streamlineSource := filepath.Join(repositoryRoot, "configs", "e2e", "workflows", "streamline_copy.yaml")
	streamlineBytes, err := os.ReadFile(streamlineSource)
	if err != nil {
		t.Fatal(err)
	}
	streamline, nextUpdates := addDevelopmentCredential(string(streamlineBytes), workflowUpdates)
	workflowUpdates = nextUpdates
	if err := os.WriteFile(filepath.Join(target, "workflows", "streamline_copy.yaml"), []byte(streamline), 0o600); err != nil {
		t.Fatal(err)
	}
	if !gatewayUpdated || !managerUpdated || workflowUpdates == 0 {
		t.Fatalf("staged config updates = Gateway:%t manager:%t workflows:%d", gatewayUpdated, managerUpdated, workflowUpdates)
	}
	return target
}

func addDevelopmentCredential(text string, updates int) (string, int) {
	const gatewaySelection = "      llmGateway: local-litellm@1\n"
	const credentialSelection = gatewaySelection + "      credential: development-worker\n"
	count := strings.Count(text, gatewaySelection)
	return strings.ReplaceAll(text, gatewaySelection, credentialSelection), updates + count
}

func projectSourceArchive(t *testing.T) []byte {
	t.Helper()
	files := map[string]string{
		"app.py": projectSource,
		"pyproject.toml": `[project]
name = "widget-service"
version = "1.0.0"
dependencies = ["fastapi>=0.116", "httpx>=0.28"]
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
		header.SetModTime(time.Date(2026, 8, 31, 0, 0, 0, 0, time.UTC))
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

func installValidators(t *testing.T, root string) (string, string) {
	t.Helper()
	bin := filepath.Join(root, "validator-bin")
	if err := os.MkdirAll(bin, 0o700); err != nil {
		t.Fatal(err)
	}
	logPath := filepath.Join(root, "validator-invocations.log")
	quotedLog := shellSingleQuote(logPath)
	vacuum := `#!/bin/sh
if [ "$1" != "spectral-report" ] || [ "$2" != "-i" ] || [ "$3" != "-o" ]; then exit 2; fi
payload=$(cat)
case "$payload" in *'/widgets/{widget_id}'*) ;; *) exit 2 ;; esac
printf 'vacuum\n' >> ` + quotedLog + `
printf '[]\n'
`
	likeC4 := `#!/bin/sh
if [ "$1" != "validate" ] || [ "$2" != "--json" ] || [ "$3" != "--no-layout" ] || [ "$4" != "--file" ]; then exit 2; fi
grep -Fq 'specification {' "$5" || exit 2
printf 'likec4\n' >> ` + quotedLog + `
printf '{"valid":true,"errors":[],"stats":{"fixture":true}}\n'
`
	for name, content := range map[string]string{"vacuum": vacuum, "likec4": likeC4} {
		if err := os.WriteFile(filepath.Join(bin, name), []byte(content), 0o700); err != nil {
			t.Fatal(err)
		}
	}
	return bin, logPath
}

func shellSingleQuote(value string) string {
	return "'" + strings.ReplaceAll(value, "'", "'\\''") + "'"
}

func writeCredentialMasterKey(t *testing.T, root string) string {
	t.Helper()
	key := make([]byte, 32)
	if _, err := cryptorand.Read(key); err != nil {
		t.Fatal(err)
	}
	return writeSecureFile(t, root, "credential-master-key", []byte(base64.StdEncoding.EncodeToString(key)))
}

func writeSecureFile(t *testing.T, root, name string, data []byte) string {
	t.Helper()
	path := filepath.Join(root, name)
	if err := os.WriteFile(path, data, 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}

func writeLocalAuth(t *testing.T, root, userID string) string {
	t.Helper()
	hash, err := auth.HashPassword([]byte(uiStackPassword))
	if err != nil {
		t.Fatal(err)
	}
	document, err := auth.BootstrapYAML(userID, "admin", hash)
	if err != nil {
		t.Fatal(err)
	}
	return writeSecureFile(t, root, "local-auth.yaml", document)
}

func freeAddress(t *testing.T, host string) string {
	t.Helper()
	listener, err := net.Listen("tcp", net.JoinHostPort(host, "0"))
	if err != nil {
		t.Fatal(err)
	}
	address := listener.Addr().String()
	if err := listener.Close(); err != nil {
		t.Fatal(err)
	}
	return address
}

func waitForHTTP(
	t *testing.T, ctx context.Context, process *childProcess, client *http.Client,
	target string, expected int, secrets []string,
) {
	t.Helper()
	if err := waitForHTTPError(ctx, process, client, target, expected); err != nil {
		t.Fatalf("%v\n%s", err, process.logs.redacted(secrets...))
	}
}

func waitForHTTPError(
	ctx context.Context, process *childProcess, client *http.Client, target string, expected int,
) error {
	for {
		request, _ := http.NewRequestWithContext(ctx, http.MethodGet, target, nil)
		response, err := client.Do(request)
		if err == nil {
			response.Body.Close()
			if response.StatusCode == expected {
				return nil
			}
		}
		if exited, processErr := process.exited(); exited {
			return fmt.Errorf("%s exited during readiness: %v", process.name, processErr)
		}
		select {
		case <-ctx.Done():
			return fmt.Errorf("wait for %s readiness: %w", process.name, ctx.Err())
		case <-time.After(50 * time.Millisecond):
		}
	}
}

func waitForProcessLog(
	t *testing.T, ctx context.Context, process *childProcess, expected string, secrets []string,
) {
	t.Helper()
	for {
		logs := process.logs.redacted(secrets...)
		if strings.Contains(logs, expected) {
			return
		}
		if exited, processErr := process.exited(); exited {
			t.Fatalf("%s exited while waiting for log %q: %v\n%s", process.name, expected, processErr, logs)
		}
		select {
		case <-ctx.Done():
			t.Fatalf("wait for %s log %q: %v\n%s", process.name, expected, ctx.Err(), logs)
		case <-time.After(50 * time.Millisecond):
		}
	}
}

func newMTLSClient(t *testing.T, caFile string, identity localpki.Paths) *http.Client {
	t.Helper()
	certificate, err := tls.LoadX509KeyPair(identity.Certificate, identity.PrivateKey)
	if err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(caFile)
	if err != nil {
		t.Fatal(err)
	}
	roots := x509.NewCertPool()
	if !roots.AppendCertsFromPEM(data) {
		t.Fatal("test CA contains no certificate")
	}
	return &http.Client{Transport: &http.Transport{TLSClientConfig: &tls.Config{
		MinVersion: tls.VersionTLS13, RootCAs: roots, Certificates: []tls.Certificate{certificate},
	}}, Timeout: 5 * time.Second}
}

func isolatedDatabase(t *testing.T, ctx context.Context, databaseURL string) string {
	t.Helper()
	parsed, err := url.Parse(databaseURL)
	if err != nil || parsed.Scheme != "postgres" && parsed.Scheme != "postgresql" {
		t.Fatal("CONTRACTOR_TEST_DATABASE_URL must be a PostgreSQL URL")
	}
	admin, err := pgxpool.New(ctx, databaseURL)
	if err != nil {
		t.Fatalf("open PostgreSQL: %v", err)
	}
	if err := admin.Ping(ctx); err != nil {
		admin.Close()
		t.Fatalf("ping PostgreSQL: %v", err)
	}
	schema := "contractor_ui_stack_" + randomHex(t, 8)
	identifier := pgx.Identifier{schema}.Sanitize()
	if _, err := admin.Exec(ctx, "CREATE SCHEMA "+identifier); err != nil {
		admin.Close()
		t.Fatalf("create UI-stack schema: %v", err)
	}
	t.Cleanup(func() {
		cleanup, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		if _, err := admin.Exec(cleanup, "DROP SCHEMA "+identifier+" CASCADE"); err != nil {
			t.Logf("drop UI-stack schema: %v", err)
		}
		admin.Close()
	})
	query := parsed.Query()
	query.Set("search_path", schema)
	parsed.RawQuery = query.Encode()
	return parsed.String()
}

func randomHex(t *testing.T, size int) string {
	t.Helper()
	data := make([]byte, size)
	if _, err := cryptorand.Read(data); err != nil {
		t.Fatal(err)
	}
	return hex.EncodeToString(data)
}

func repoRoot(t *testing.T) string {
	t.Helper()
	_, source, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("locate UI-stack source")
	}
	root, err := filepath.Abs(filepath.Join(filepath.Dir(source), "..", ".."))
	if err != nil {
		t.Fatal(err)
	}
	return root
}

func assertEnvironmentHasNoSecrets(t *testing.T, environment, secrets []string) {
	t.Helper()
	if containsAny(strings.Join(environment, "\n"), secrets) {
		t.Fatal("secret-free child process environment contains a seeded secret canary")
	}
}

func containsAny(value string, candidates []string) bool {
	for _, candidate := range candidates {
		if candidate != "" && strings.Contains(value, candidate) {
			return true
		}
	}
	return false
}

func assertEvidenceHasNoSecrets(t *testing.T, path string, secrets []string) {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if containsAny(string(data), secrets) {
		t.Fatal("browser evidence contains a seeded secret canary")
	}
	var value any
	if err := json.Unmarshal(data, &value); err != nil {
		t.Fatal(err)
	}
	walkEvidence(value, func(candidate []byte) {
		if containsAny(string(candidate), secrets) {
			t.Fatal("decoded browser HTTP evidence contains a seeded secret canary")
		}
	})
}

func walkEvidence(value any, inspect func([]byte)) {
	switch current := value.(type) {
	case map[string]any:
		for key, child := range current {
			if key == "bodyBase64" {
				if encoded, ok := child.(string); ok {
					if decoded, err := base64.StdEncoding.DecodeString(encoded); err == nil {
						inspect(decoded)
					}
				}
			}
			walkEvidence(child, inspect)
		}
	case []any:
		for _, child := range current {
			walkEvidence(child, inspect)
		}
	}
}

func assertZipHasNoSecrets(t *testing.T, path string, secrets []string) {
	t.Helper()
	archive, err := zip.OpenReader(path)
	if err != nil {
		t.Fatal(err)
	}
	defer archive.Close()
	var totalSize uint64
	for _, file := range archive.File {
		if file.UncompressedSize64 > 64<<20 {
			t.Fatalf("trace entry %q exceeds inspection bound", file.Name)
		}
		totalSize += file.UncompressedSize64
		if totalSize > 256<<20 {
			t.Fatal("browser trace exceeds aggregate inspection bound")
		}
		reader, err := file.Open()
		if err != nil {
			t.Fatal(err)
		}
		data, readErr := io.ReadAll(io.LimitReader(reader, 64<<20+1))
		reader.Close()
		if readErr != nil {
			t.Fatal(readErr)
		}
		if containsAny(string(data), secrets) {
			t.Fatalf("browser trace entry %q contains a seeded secret canary", file.Name)
		}
	}
}
