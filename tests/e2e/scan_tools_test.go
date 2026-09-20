//go:build e2e

package e2e

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/localpki"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/jackc/pgx/v5/pgxpool"
)

type scanProcessHarness struct {
	ctx                                                context.Context
	repositoryRoot, temporaryRoot, databaseURL, userID string
	baseURL, browserBaseURL, targetURL                 string
	browserAPIURL, browserInternalURL                  string
	client                                             *http.Client
	server, runtime                                    *childProcess
	pool                                               *pgxpool.Pool
	fixture                                            *scanHTTPFixture
	restartServer                                      func()
}

type scanHTTPFixture struct {
	mu            sync.Mutex
	paths         []string
	sqlmapQueries []string
	requests      []scanHTTPRequest
	slowStarted   chan struct{}
	slowOnce      sync.Once
}

type scanHTTPRequest struct{ method, uri, authorization, body string }

func (f *scanHTTPFixture) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	body, _ := io.ReadAll(io.LimitReader(r.Body, 65536))
	f.mu.Lock()
	f.requests = append(f.requests, scanHTTPRequest{r.Method, r.URL.RequestURI(), r.Header.Get("Authorization"), string(body)})
	f.paths = append(f.paths, r.URL.Path)
	if r.URL.Path == "/query" {
		f.sqlmapQueries = append(f.sqlmapQueries, r.URL.RawQuery)
	}
	f.mu.Unlock()
	if strings.HasPrefix(r.URL.Path, "/slow/") {
		f.slowOnce.Do(func() { close(f.slowStarted) })
		select {
		case <-r.Context().Done():
			return
		case <-time.After(15 * time.Second):
		}
	}
	w.Header().Set("Content-Type", "text/plain")
	if r.URL.Path == "/missing" {
		w.WriteHeader(http.StatusNotFound)
	}
	_, _ = io.WriteString(w, "contractor-controlled-scan-fixture\n")
}

// This gate deliberately requires installed scanners. Fake executables would test
// dispatch, but would not establish the release's real scanner transport boundary.
func TestScanToolsAcrossProductionProcesses(t *testing.T) {
	if testing.Short() {
		t.Skip("end-to-end process test")
	}
	h := startScanStack(t)

	t.Run("nuclei_local_template", func(t *testing.T) {
		runID := h.createRun(t, "nuclei-target@1", map[string]string{"target": h.targetURL}, nil)
		status, report := h.completedReport(t, runID, "scan_nuclei")
		results, ok := report.Observation["results"].([]any)
		if !ok || len(results) != 1 || results[0].(map[string]any)["template-id"] != "contractor-local-fixture" {
			t.Fatalf("local nuclei template evidence = %+v", report.Observation)
		}
		h.assertExecution(t, status, "scan_nuclei", "")
	})
	t.Run("naabu_local_connect", func(t *testing.T) {
		runID := h.createRun(t, "naabu-host@1", map[string]string{"target": "127.0.0.1"}, nil)
		status, report := h.completedReport(t, runID, "scan_naabu")
		target, _ := url.Parse(h.targetURL)
		port, _ := strconv.Atoi(target.Port())
		results, ok := report.Observation["results"].([]any)
		if !ok || len(results) == 0 {
			t.Fatalf("naabu local listening port evidence = %+v", report.Observation)
		}
		// Some naabu versions emit the same open port during both discovery and
		// final output. Every record must still describe the selected local port.
		for _, item := range results {
			finding, ok := item.(map[string]any)
			if !ok || finding["port"] != float64(port) || finding["ip"] != "127.0.0.1" {
				t.Fatalf("unexpected naabu target evidence: %+v", item)
			}
		}
		h.assertExecution(t, status, "scan_naabu", "")
	})
	t.Run("sqlmap_exact_request", func(t *testing.T) {
		request := []byte(fmt.Sprintf(`{"schemaVersion":1,"method":"GET","url":%q,"headers":[{"name":"X-Fixture","value":"controlled-local"}],"body":"","testParameters":["id"]}`, h.targetURL+"/query?id=7"))
		input := h.upload(t, "request", "application/vnd.contractor.http-request+json", request, nil)
		runID := h.createRun(t, "sqlmap-request@1", nil, map[string]artifactRef{"request": input})
		status, report := h.completedReport(t, runID, "scan_sqlmap")
		h.assertScanInput(t, status, report, "request", input)
		if report.Observation["diagnosticsRedacted"] != true || report.Observation["stdout"] != "" || report.Observation["stderr"] != "" {
			t.Fatalf("request-mode scanner diagnostics are not redacted: %+v", report.Observation)
		}
		h.fixture.mu.Lock()
		queries := append([]string(nil), h.fixture.sqlmapQueries...)
		h.fixture.mu.Unlock()
		probed := false
		for _, query := range queries {
			if query != "id=7" {
				probed = true
			}
		}
		if len(queries) < 2 || queries[0] != "id=7" || !probed {
			t.Fatalf("sqlmap did not exercise selected local parameter: %v", queries)
		}
		h.assertExecution(t, status, "scan_sqlmap", "")
	})
	t.Run("ffuf_pins_uploaded_revision", func(t *testing.T) {
		input := h.upload(t, "wordlist", "text/vnd.contractor.wordlist", []byte("hit\r\nmissing\r\n"), nil)
		newer := h.upload(t, "wordlist", "text/vnd.contractor.wordlist", []byte("newer-revision-only\n"), input.Revision)
		if *newer.Revision == *input.Revision {
			t.Fatal("wordlist update did not create a new revision")
		}
		runID := h.createRun(t, "ffuf-wordlist@1", map[string]string{"target": h.targetURL + "/FUZZ"}, map[string]artifactRef{"wordlist": input})
		status, report := h.completedReport(t, runID, "scan_ffuf")
		h.assertScanInput(t, status, report, "wordlist", input)
		if report.Observation["wordlistEntries"] != float64(2) || report.Observation["payloadsAttempted"] != float64(2) || report.Observation["scanComplete"] != true || report.Observation["requestErrors"] != float64(0) {
			t.Fatalf("ffuf coverage = %+v", report.Observation)
		}
		results, ok := report.Observation["results"].([]any)
		if !ok || len(results) != 2 {
			t.Fatalf("ffuf result evidence = %+v", report.Observation)
		}
		payloads := map[string]bool{}
		for _, item := range results {
			payloads[item.(map[string]any)["input"].(map[string]any)["FUZZ"].(string)] = true
		}
		if !payloads["hit"] || !payloads["missing"] || payloads["newer-revision-only"] {
			t.Fatalf("ffuf used wrong revision: %v", payloads)
		}
		original, media := download(t, h.client, h.baseURL+"/v1/artifacts/scans/wordlist?revision="+url.QueryEscape(*input.Revision))
		if string(original) != "hit\r\nmissing\r\n" || media != "text/vnd.contractor.wordlist" {
			t.Fatal("original wordlist revision changed")
		}
		h.assertExecution(t, status, "scan_ffuf", "")
	})
	t.Run("invalid_input_is_failure", func(t *testing.T) {
		h.fixture.mu.Lock()
		before := len(h.fixture.paths)
		h.fixture.mu.Unlock()
		input := h.upload(t, "invalid-wordlist", "text/vnd.contractor.wordlist", []byte("bad\x00entry\n"), nil)
		runID := h.createRun(t, "ffuf-wordlist@1", map[string]string{"target": h.targetURL + "/FUZZ"}, map[string]artifactRef{"wordlist": input})
		status := h.terminal(t, runID, "failed")
		h.assertExecution(t, status, "scan_ffuf", "tool_input_invalid")
		if len(status.Outputs) != 0 {
			t.Fatal("invalid scan published a success output")
		}
		h.fixture.mu.Lock()
		after := len(h.fixture.paths)
		h.fixture.mu.Unlock()
		if after != before {
			t.Fatal("invalid wordlist reached the controlled HTTP target")
		}
	})
	t.Run("cancel_running_scanner", func(t *testing.T) {
		input := h.upload(t, "slow-wordlist", "text/vnd.contractor.wordlist", []byte("probe\n"), nil)
		runID := h.createRun(t, "ffuf-wordlist@1", map[string]string{"target": h.targetURL + "/slow/FUZZ"}, map[string]artifactRef{"wordlist": input})
		select {
		case <-h.fixture.slowStarted:
		case <-h.ctx.Done():
			t.Fatal("scanner never reached controlled cancellation fixture")
		}
		cancelWorkflowRun(t, h.client, h.baseURL, runID)
		status := h.terminal(t, runID, "cancelled")
		if len(status.Outputs) != 0 {
			t.Fatal("cancelled scan published a success output")
		}
		if len(status.Attempts) != 1 {
			t.Fatal("cancelled scan has no execution")
		}
		var termination runstore.StageTermination
		if err := json.Unmarshal(status.Attempts[0].Termination, &termination); err != nil || termination.Code != runstore.CancellationUserRequested {
			t.Fatalf("cancelled scan termination = %s", status.Attempts[0].Termination)
		}
		allocations, err := runstore.NewPostgresStore(h.pool).ListStageAllocations(h.ctx, status.Attempts[0].StageExecutionID)
		if err != nil || len(allocations) != 1 {
			t.Fatalf("cancelled scan allocations = %+v %v", allocations, err)
		}
		waitForObservedRuntimeAgents(t, h.ctx, h.server, []*childProcess{h.runtime}, h.client, h.baseURL, func(agents []observedRuntimeAgent) bool {
			for _, agent := range agents {
				if agent.InstanceID == allocations[0].RuntimeAgentInstanceID && agent.SlotState == "idle" && agent.CurrentAllocationID == nil && agent.AuthoritativeAllocationID == nil {
					return true
				}
			}
			return false
		})
	})
	t.Run("report_publication_failure", func(t *testing.T) {
		// A scoped storage fault exercises the real private Artifact API after the
		// real scanner exits, while allowing its receipt and diagnostics to persist.
		_, err := h.pool.Exec(h.ctx, `CREATE FUNCTION fail_scan_report() RETURNS trigger LANGUAGE plpgsql AS $$ BEGIN
		IF NEW.scope_kind = 'run' AND NEW.namespace = 'scanner' AND NEW.name = 'report-failure' THEN
		RAISE EXCEPTION 'controlled scan report storage fault'; END IF; RETURN NEW; END $$;
		CREATE TRIGGER fail_scan_report BEFORE INSERT ON artifact_bindings FOR EACH ROW EXECUTE FUNCTION fail_scan_report()`)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() {
			_, _ = h.pool.Exec(context.Background(), `DROP TRIGGER fail_scan_report ON artifact_bindings; DROP FUNCTION fail_scan_report()`)
		})
		input := h.upload(t, "report-failure-wordlist", "text/vnd.contractor.wordlist", []byte("hit\n"), nil)
		runID := h.createRun(t, "ffuf-report-failure@1", map[string]string{"target": h.targetURL + "/FUZZ"}, map[string]artifactRef{"wordlist": input})
		status := h.terminal(t, runID, "failed")
		h.assertExecution(t, status, "scan_ffuf", "tool_report_failed")
		if len(status.Outputs) != 0 {
			t.Fatal("report error published a success output")
		}
	})
	if os.Getenv("CONTRACTOR_SCAN_BROWSER") == "1" {
		t.Run("browser_upload_run_report", func(t *testing.T) { runScanBrowser(t, h) })
	}
}

func startScanStack(t *testing.T) *scanProcessHarness {
	t.Helper()
	return startScanStackForTools(t, "nuclei-target@1", "nuclei", "naabu", "sqlmap", "ffuf")
}

func startScanStackForTools(t *testing.T, capacityWorkflow string, binaries ...string) *scanProcessHarness {
	t.Helper()
	return startConfiguredScanStack(t, capacityWorkflow, nil, binaries...)
}

func startConfiguredScanStack(t *testing.T, capacityWorkflow string, configure func(*scanProcessHarness, string), binaries ...string) *scanProcessHarness {
	t.Helper()
	for _, binary := range binaries {
		if _, err := exec.LookPath(binary); err != nil {
			t.Fatalf("real scanner %s is required on PATH for the scan release gate", binary)
		}
	}
	databaseURL := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if databaseURL == "" {
		t.Fatal("CONTRACTOR_TEST_DATABASE_URL is required")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	t.Cleanup(cancel)
	h := &scanProcessHarness{ctx: ctx, repositoryRoot: repoRoot(t), temporaryRoot: t.TempDir(), client: &http.Client{Timeout: 10 * time.Second}, userID: "scan-e2e-" + randomHex(t, 8)}
	t.Cleanup(func() {
		if t.Failed() {
			if h.server != nil {
				t.Logf("scan Server diagnostics:\n%s", h.server.logs.redacted(publicToken))
			}
			if h.runtime != nil {
				t.Logf("scan Runtime diagnostics:\n%s", h.runtime.logs.redacted(publicToken))
			}
		}
	})
	h.databaseURL = isolatedDatabase(t, ctx, databaseURL)
	h.fixture = &scanHTTPFixture{slowStarted: make(chan struct{})}
	fixtureServer := httptest.NewServer(h.fixture)
	t.Cleanup(fixtureServer.Close)
	h.targetURL = fixtureServer.URL
	configRoot := filepath.Join(h.temporaryRoot, "configs")
	if err := os.CopyFS(configRoot, os.DirFS(filepath.Join(h.repositoryRoot, "configs", "scan"))); err != nil {
		t.Fatal(err)
	}
	naabuTemplate := filepath.Join(configRoot, "agent-templates", "naabu.yaml")
	target, _ := url.Parse(h.targetURL)
	replaceScanConfig(t, naabuTemplate, `value: "80,443"`, `value: "`+target.Port()+`"`)
	ffufWorkflow, err := os.ReadFile(filepath.Join(configRoot, "workflows", "ffuf.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	reportFailure := strings.ReplaceAll(string(ffufWorkflow), "name: ffuf-wordlist", "name: ffuf-report-failure")
	reportFailure = strings.ReplaceAll(reportFailure, "namespace: scanner, name: report", "namespace: scanner, name: report-failure")
	if err := os.WriteFile(filepath.Join(configRoot, "workflows", "ffuf_report_failure.yaml"), []byte(reportFailure), 0o600); err != nil {
		t.Fatal(err)
	}
	templates := filepath.Join(h.temporaryRoot, "nuclei-templates")
	if err := os.Mkdir(templates, 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(templates, "fixture.yaml"), []byte(`id: contractor-local-fixture
info:
  name: Controlled local release fixture
  author: contractor
  severity: info
http:
  - method: GET
    path: ["{{BaseURL}}/nuclei"]
    matchers:
      - type: word
        words: ["contractor-controlled-scan-fixture"]
`), 0o600); err != nil {
		t.Fatal(err)
	}
	if configure != nil {
		configure(h, configRoot)
	}
	serverBinary := filepath.Join(h.temporaryRoot, "contractor-server")
	runChecked(t, h.repositoryRoot, nil, "go", "build", "-o", serverBinary, "./cmd/contractor-server")
	runChecked(t, h.repositoryRoot, map[string]string{"CONTRACTOR_DATABASE_URL": h.databaseURL}, serverBinary, "migrate")
	h.pool, err = pgxpool.New(ctx, h.databaseURL)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(h.pool.Close)
	pkiRoot := filepath.Join(h.temporaryRoot, "pki")
	generator := localpki.Generator{}
	caPaths, err := generator.InitCA(pkiRoot, false)
	if err != nil {
		t.Fatal(err)
	}
	leaf := localpki.LeafOptions{IPAddresses: []net.IP{net.ParseIP("127.0.0.1")}}
	controlPaths, err := generator.IssueControlPlane(pkiRoot, localpki.ControlPlaneOptions{LeafOptions: leaf, URI: "urn:contractor:control-plane:scan-e2e"})
	if err != nil {
		t.Fatal(err)
	}
	publicAddress, privateAddress := freeAddress(t), freeAddress(t)
	h.baseURL, h.browserBaseURL = "http://"+publicAddress, "https://ui.contractor.invalid"
	if os.Getenv("CONTRACTOR_SCAN_BROWSER") == "1" {
		configureScanBrowser(t, h)
	}
	privateURL := "https://" + privateAddress
	serverEnvironment := map[string]string{
		"CONTRACTOR_DATABASE_URL": h.databaseURL, "CONTRACTOR_OPERATOR_CONFIG_ROOT": configRoot,
		"CONTRACTOR_PUBLIC_LISTEN": publicAddress, "CONTRACTOR_PRIVATE_LISTEN": privateAddress, "CONTRACTOR_PRIVATE_URL": privateURL,
		"CONTRACTOR_CA_FILE": caPaths.Certificate, "CONTRACTOR_CONTROL_PLANE_CERT_FILE": controlPaths.Certificate, "CONTRACTOR_CONTROL_PLANE_KEY_FILE": controlPaths.PrivateKey,
		"CONTRACTOR_PUBLIC_BEARER_TOKEN": publicToken,
		"CONTRACTOR_LOCAL_AUTH_FILE":     writeE2ELocalAuth(t, h.temporaryRoot, h.userID), "CONTRACTOR_BROWSER_ORIGINS": h.browserBaseURL,
		"CONTRACTOR_LLM_GATEWAY_TOKEN": "",
	}
	h.restartServer = func() {
		if h.server != nil {
			h.server.stop(t)
		}
		h.server = startProcess(t, "Go Scan Server", h.repositoryRoot, serverEnvironment, serverBinary, "serve")
		waitForHTTP(t, ctx, h.server, h.client, h.baseURL+"/readyz", http.StatusOK)
	}
	h.restartServer()
	python := os.Getenv("CONTRACTOR_SCAN_PYTHON")
	if python == "" {
		python = filepath.Join(h.repositoryRoot, "runtime", ".venv", "bin", "python")
	}
	if info, err := os.Stat(python); err != nil || info.IsDir() {
		t.Fatal("Python runtime virtual environment is required; run uv sync --locked in runtime or set CONTRACTOR_SCAN_PYTHON")
	}
	controlClient := newMTLSClient(t, caPaths.Certificate, controlPaths)
	startRuntime := func(name, path string) *childProcess {
		identity, err := generator.IssueAgent(pkiRoot, name, leaf)
		if err != nil {
			t.Fatal(err)
		}
		address := freeAddress(t)
		baseURL := "https://" + address
		process := startProcess(t, name, filepath.Join(h.repositoryRoot, "runtime"), map[string]string{
			"PYTHONUNBUFFERED": "1", "PYTHONPATH": filepath.Join(h.repositoryRoot, "runtime", "src"), "PATH": path, "NUCLEI_TEMPLATES_DIR": templates,
		}, python, "-m", "contractor_runtime", "--control-plane-url", privateURL,
			"--advertised-control-url", baseURL, "--advertised-a2a-url", baseURL,
			"--ca-file", caPaths.Certificate, "--certificate-file", identity.Certificate, "--private-key-file", identity.PrivateKey,
			"--listen", address, "--work-root", filepath.Join(h.temporaryRoot, name), "--request-timeout-seconds", "10", "--shutdown-grace-seconds", "5")
		waitForHTTP(t, ctx, process, controlClient, baseURL+"/healthz", http.StatusOK)
		waitForProcessLog(t, ctx, process, "runtime agent registered")
		return process
	}
	if capacityWorkflow != "" {
		t.Run("missing_binaries_wait_for_capacity", func(t *testing.T) {
			emptyBin := filepath.Join(h.temporaryRoot, "empty-bin")
			if err := os.Mkdir(emptyBin, 0o700); err != nil {
				t.Fatal(err)
			}
			emptyRuntime := startRuntime("scan-no-binaries", emptyBin)
			agents, _ := waitForObservedRuntimeAgents(t, ctx, h.server, []*childProcess{emptyRuntime}, h.client, h.baseURL, func(agents []observedRuntimeAgent) bool {
				return len(agents) == 1 && agents[0].ConfirmedLeaseUntil != nil
			})
			if len(runtimeToolset(agents[0], "scan@1").Tools) != 0 {
				t.Fatal("runtime advertises missing scanner binaries")
			}
			runID := h.createRun(t, capacityWorkflow, map[string]string{"target": h.targetURL}, nil)
			store := runstore.NewPostgresStore(h.pool)
			waiting := waitForUnallocatedPreparingExecution(t, ctx, store, runID)
			assertSameUnallocatedPreparingExecution(t, ctx, store, waiting)
			cancelWorkflowRun(t, h.client, h.baseURL, runID)
			h.runtime = emptyRuntime
			h.terminal(t, runID, "cancelled")
			emptyRuntime.stop(t)
		})
	}
	h.runtime = startRuntime("scan-real-binaries", os.Getenv("PATH"))
	waitForObservedRuntimeAgents(t, ctx, h.server, []*childProcess{h.runtime}, h.client, h.baseURL, func(agents []observedRuntimeAgent) bool {
		for _, agent := range agents {
			tools := runtimeToolset(agent, "scan@1").Tools
			available := agent.ConfirmedLeaseUntil != nil
			for _, binary := range binaries {
				available = available && containsString(tools, "scan_"+binary)
			}
			if available {
				return true
			}
		}
		return false
	})
	return h
}

func replaceScanConfig(t *testing.T, path, old, replacement string) {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Count(string(data), old) != 1 {
		t.Fatalf("expected one scan fixture configuration replacement in %s", path)
	}
	if err := os.WriteFile(path, []byte(strings.Replace(string(data), old, replacement, 1)), 0o600); err != nil {
		t.Fatal(err)
	}
}

func (h *scanProcessHarness) upload(t *testing.T, name, media string, data []byte, expected *string) artifactRef {
	t.Helper()
	request, err := http.NewRequest(http.MethodPut, h.baseURL+"/v1/artifacts/scans/"+name, bytes.NewReader(data))
	if err != nil {
		t.Fatal(err)
	}
	request.Header.Set("Authorization", "Bearer "+publicToken)
	request.Header.Set("Content-Type", media)
	code := http.StatusCreated
	if expected == nil {
		request.Header.Set("If-None-Match", "*")
	} else {
		request.Header.Set("If-Match", strconv.Quote(*expected))
		code = http.StatusOK
	}
	response := do(t, h.client, request, code)
	defer response.Body.Close()
	var body struct {
		Artifact  artifactRef `json:"artifact"`
		MediaType string      `json:"mediaType"`
		Size      int64       `json:"size"`
	}
	decodeResponse(t, response, &body)
	if body.Artifact.Revision == nil || body.Artifact.Namespace != "scans" || body.Artifact.Name != name {
		t.Fatal("upload returned no exact artifact")
	}
	return body.Artifact
}

func (h *scanProcessHarness) createRun(t *testing.T, workflow string, parameters map[string]string, inputs map[string]artifactRef) string {
	t.Helper()
	if parameters == nil {
		parameters = map[string]string{}
	}
	if inputs == nil {
		inputs = map[string]artifactRef{}
	}
	body, err := json.Marshal(map[string]any{"workflow": workflow, "parameters": parameters, "artifacts": inputs})
	if err != nil {
		t.Fatal(err)
	}
	request, err := http.NewRequest(http.MethodPost, h.baseURL+"/v1/runs", bytes.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	request.Header.Set("Authorization", "Bearer "+publicToken)
	request.Header.Set("Content-Type", "application/json")
	request.Header.Set("Idempotency-Key", "scan-"+randomHex(t, 8))
	response := do(t, h.client, request, http.StatusAccepted)
	defer response.Body.Close()
	var result runCreateResponse
	decodeResponse(t, response, &result)
	if result.RunID == "" {
		t.Fatal("missing created run ID")
	}
	return result.RunID
}

func (h *scanProcessHarness) terminal(t *testing.T, runID, state string) runStatus {
	t.Helper()
	ctx, cancel := context.WithTimeout(h.ctx, 90*time.Second)
	defer cancel()
	status := waitForRunTerminalAcross(t, ctx, h.server, []*childProcess{h.runtime}, h.client, h.baseURL, runID)
	if status.State != state {
		t.Fatalf("scan run state = %s, want %s; attempts=%+v\nruntime: %s\nserver: %s", status.State, state, status.Attempts, h.runtime.logs.redacted(publicToken), h.server.logs.redacted(publicToken))
	}
	return status
}

type scanProcessReport struct {
	SchemaVersion  int                    `json:"schemaVersion"`
	Tool           string                 `json:"tool"`
	InputDigest    string                 `json:"inputDigest"`
	InputArtifacts map[string]artifactRef `json:"inputArtifacts"`
	Observation    map[string]any         `json:"observation"`
}

func (h *scanProcessHarness) completedReport(t *testing.T, runID, tool string) (runStatus, scanProcessReport) {
	t.Helper()
	status := h.terminal(t, runID, "succeeded")
	output, ok := status.Outputs["report"]
	if !ok || output.Revision == nil {
		t.Fatal("successful scan has no exact report output")
	}
	data, media := download(t, h.client, h.baseURL+"/v1/runs/"+url.PathEscape(runID)+"/outputs/report")
	var report scanProcessReport
	if err := json.Unmarshal(data, &report); err != nil {
		t.Fatal(err)
	}
	if media != "application/json" || report.SchemaVersion != 1 || report.Tool != tool || !strings.HasPrefix(report.InputDigest, "sha256:") || report.Observation["status"] != "completed" || report.Observation["exitCode"] != float64(0) {
		t.Fatalf("invalid technical scan report: %+v", report)
	}
	return status, report
}

func (h *scanProcessHarness) assertScanInput(t *testing.T, status runStatus, report scanProcessReport, name string, original artifactRef) {
	t.Helper()
	if !reflect.DeepEqual(status.Inputs[name], report.InputArtifacts[name]) {
		t.Fatalf("report changed pinned run input: %+v != %+v", status.Inputs[name], report.InputArtifacts[name])
	}
	ref, ok := report.InputArtifacts[name]
	if !ok || ref.Namespace != "inputs" || ref.Name != name || ref.Revision == nil {
		t.Fatalf("report input is not exact run-scoped ref: %+v", report.InputArtifacts)
	}
	source, sourceMedia := download(t, h.client, h.baseURL+"/v1/artifacts/"+original.Namespace+"/"+original.Name+"?revision="+url.QueryEscape(*original.Revision))
	pinned, pinnedMedia := download(t, h.client, h.baseURL+"/v1/runs/"+status.RunID+"/artifacts/inputs/"+name+"?revision="+url.QueryEscape(*ref.Revision))
	if !bytes.Equal(source, pinned) || sourceMedia != pinnedMedia {
		t.Fatal("run input differs from selected source revision")
	}
}

func (h *scanProcessHarness) assertExecution(t *testing.T, status runStatus, tool, failure string) {
	t.Helper()
	if len(status.Attempts) != 1 || status.Attempts[0].Metrics == nil {
		t.Fatalf("scan attempt has no trusted metrics: %+v", status.Attempts)
	}
	attempt := status.Attempts[0]
	// The trusted total includes one Planner a2a.invoke and one scanner call.
	if attempt.Metrics.ModelCalls != 0 || attempt.Metrics.ToolCalls != 2 || !attempt.Metrics.ReportsComplete {
		t.Fatalf("scan is not one model-free worker invocation: %+v", attempt.Metrics)
	}
	if failure != "" {
		var result contracts.StageContentResult
		if err := json.Unmarshal(attempt.Result, &result); err != nil || result.Outcome != contracts.StageFailed || result.Error == nil || result.Error.Code != failure {
			t.Fatalf("scan result = %s, want failure %s", attempt.Result, failure)
		}
		if attempt.Metrics.ErrorCount == 0 {
			t.Fatal("scan failure missing from trusted diagnostics")
		}
		var diagnostics struct {
			Items []struct {
				Code string `json:"code"`
			} `json:"items"`
		}
		if err := json.Unmarshal(attempt.Diagnostics, &diagnostics); err != nil {
			t.Fatalf("scan diagnostics: %v", err)
		}
		found := failure != "tool_report_failed" && len(diagnostics.Items) > 0
		for _, item := range diagnostics.Items {
			found = found || item.Code == failure
		}
		if !found {
			t.Fatalf("scan diagnostics omit %s: %s", failure, attempt.Diagnostics)
		}
	}
	store := runstore.NewPostgresStore(h.pool)
	allocations, err := store.ListStageAllocations(h.ctx, attempt.StageExecutionID)
	if err != nil || len(allocations) != 1 || allocations[0].RuntimeAgentInstanceID == "" {
		t.Fatalf("scan bypassed real allocation: %+v %v", allocations, err)
	}
	reports, err := store.ListStageExecutionReports(h.ctx, attempt.StageExecutionID)
	if err != nil || len(reports) != 1 {
		t.Fatalf("trusted scan execution reports = %+v %v", reports, err)
	}
	metrics, ok := reports[0].Report.Worker.Metrics.Tools[tool]
	if !ok || metrics.Calls == nil || *metrics.Calls != 1 {
		t.Fatalf("scanner call missing from trusted report: %+v", reports[0])
	}
}
