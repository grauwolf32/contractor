//go:build e2e

package e2e

import (
	"archive/zip"
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"maps"
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

const (
	taintInitialApplication = `from framework import app

SOURCE_MARKER = "TAINT_SOURCE_CONTENT_CANARY"

@app.get("/items/{item_id}")
def handler(req):
    checked = validate_input(req)
    return query_db(checked)

def validate_input(value):
    return value if value.isdigit() else ""

def query_db(value):
    return database.execute("SELECT value FROM items WHERE id = ?", [value])
`
	taintAnnotatedApplication = `from framework import app

SOURCE_MARKER = "TAINT_SOURCE_CONTENT_CANARY"

# @trace target=TAINT_TARGET_VALUE_CANARY args=req:tainted calls=validate_input,query_db
@app.get("/items/{item_id}")
def handler(req):
    checked = validate_input(req)
    return query_db(checked)

# @validate arg=value kind=schema
def validate_input(value):
    return value if value.isdigit() else ""

# @sink kind=db.query arg=value
def query_db(value):
    return database.execute("SELECT value FROM items WHERE id = ?", [value])
`
	taintInitialDuplicates = `def duplicate(value):
    return value

def duplicate(value):
    return value + "!"
`
	taintAnnotatedDuplicates = `def duplicate(value):
    return value

# @trace target=TAINT_TARGET_VALUE_CANARY
def duplicate(value):
    return value + "!"
`
	taintCleanSource = `def clean_handler(value):
    return value
`
	taintExpectedDiff = `--- a/private-taint-path-canary/app.py
+++ b/private-taint-path-canary/app.py
@@ -2,13 +2,16 @@
 
 SOURCE_MARKER = "TAINT_SOURCE_CONTENT_CANARY"
 
+# @trace target=TAINT_TARGET_VALUE_CANARY args=req:tainted calls=validate_input,query_db
 @app.get("/items/{item_id}")
 def handler(req):
     checked = validate_input(req)
     return query_db(checked)
 
+# @validate arg=value kind=schema
 def validate_input(value):
     return value if value.isdigit() else ""
 
+# @sink kind=db.query arg=value
 def query_db(value):
     return database.execute("SELECT value FROM items WHERE id = ?", [value])
--- a/private-taint-path-canary/duplicates.py
+++ b/private-taint-path-canary/duplicates.py
@@ -1,5 +1,6 @@
 def duplicate(value):
     return value
 
+# @trace target=TAINT_TARGET_VALUE_CANARY
 def duplicate(value):
     return value + "!"
`
)

type taintWorkspaceState struct {
	APIVersion            string                    `json:"apiVersion"`
	Kind                  string                    `json:"kind"`
	BaseWorkspaceDigest   string                    `json:"baseWorkspaceDigest"`
	ResultWorkspaceDigest string                    `json:"resultWorkspaceDigest"`
	Operations            []taintWorkspaceOperation `json:"operations"`
}

type taintWorkspaceOperation struct {
	Op   string `json:"op"`
	Path string `json:"path"`
	Text string `json:"text,omitempty"`
}

func TestTaintAnnotationsAcrossRealRuntimeProcess(t *testing.T) {
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
		t.Fatalf("initialize taint-annotations test CA: %v", err)
	}
	leaf := localpki.LeafOptions{IPAddresses: []net.IP{net.ParseIP("127.0.0.1")}}
	controlPlanePaths, err := generator.IssueControlPlane(pkiRoot, localpki.ControlPlaneOptions{
		LeafOptions: leaf, URI: "urn:contractor:control-plane:taint-annotations-e2e",
	})
	if err != nil {
		t.Fatalf("issue taint-annotations Control Plane certificate: %v", err)
	}
	runtimeIdentity, err := generator.IssueAgent(pkiRoot, "taint-annotations-local", leaf)
	if err != nil {
		t.Fatalf("issue taint-annotations Runtime certificate: %v", err)
	}

	gateway := newTaintAnnotationGateway(llmGatewayToken)
	t.Cleanup(gateway.close)
	collector := newFakeOTLPCollector(taintOTLPCredential)
	t.Cleanup(collector.close)
	configRoot := stageE2EConfiguration(
		t, filepath.Join(repositoryRoot, "configs"),
		filepath.Join(temporaryRoot, "configs"), gateway.URL(),
	)

	publicAddress, privateAddress := freeAddress(t), freeAddress(t)
	publicBaseURL := "http://" + publicAddress
	privateBaseURL := "https://" + privateAddress
	userID := "taint-annotations-e2e-user-" + randomHex(t, 8)
	masterKeyFile := filepath.Join(temporaryRoot, "credential-master-key")
	masterKey := base64.StdEncoding.EncodeToString(bytes.Repeat([]byte{0x74}, 32))
	if err := os.WriteFile(masterKeyFile, []byte(masterKey), 0o600); err != nil {
		t.Fatal(err)
	}
	server := startProcess(t, "Go Server taint annotations", repositoryRoot, map[string]string{
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
	}, serverBinary, "serve",
		"--credential-master-key-file", masterKeyFile,
		"--runtime-request-timeout", "90s",
	)
	publicClient := &http.Client{Timeout: 8 * time.Second}
	waitForHTTP(t, ctx, server, publicClient, publicBaseURL+"/readyz", http.StatusOK)

	operations := &runtimeOperations{t: t, client: publicClient, baseURL: publicBaseURL}
	operations.createRuntimeCredential("taint-otel", "otlp-headers@1", map[string]any{
		"headers": map[string]string{"x-contractor-token": taintOTLPCredential},
	})
	debugConfig := operations.publishRuntimeConfig("taint-debug", "1", map[string]any{
		"worker": map[string]any{"telemetry": map[string]any{
			"adapter": "otlp-http@1", "endpoint": collector.URL(),
			"credential": "taint-otel", "captureContent": false, "flushTimeoutSeconds": 2,
		}},
	})
	operations.createRuntimeLabel("taint-debug", debugConfig)

	python := filepath.Join(repositoryRoot, "runtime", ".venv", "bin", "python")
	if info, statErr := os.Stat(python); statErr != nil || info.IsDir() {
		t.Fatalf("Python Runtime environment is missing at %s; run 'cd runtime && uv sync --locked'", python)
	}
	directAddress, advertisedAddress := freeAddress(t), freeAddress(t)
	directBaseURL := "https://" + directAddress
	advertisedBaseURL := "https://" + advertisedAddress
	workRoot := filepath.Join(temporaryRoot, "runtime-work")
	workspaceRoot := filepath.Join(temporaryRoot, "runtime-workspaces")
	releaseProxy := newRuntimeReleaseLossProxy(
		t, advertisedAddress, directBaseURL, caPaths.Certificate,
		runtimeIdentity, controlPlanePaths,
	)
	runtimeProcess := startTaintAnnotationsRuntime(
		t, repositoryRoot, python, privateBaseURL, advertisedBaseURL, directAddress,
		workRoot, workspaceRoot, caPaths.Certificate, runtimeIdentity,
	)
	runtimes := []*childProcess{runtimeProcess}
	t.Cleanup(func() {
		if !t.Failed() {
			return
		}
		for _, process := range []*childProcess{server, runtimeProcess} {
			t.Logf("%s logs:\n%s", process.name, process.logs.redacted(
				publicToken, llmGatewayToken, taintOTLPCredential,
				taintSourceCanary, taintTargetCanary,
			))
		}
		t.Logf("taint Gateway: calls=%d stages=%d failures=%v",
			gateway.Calls(), gateway.CompletedStages(), gateway.Failures())
	})
	controlClient := newMTLSClient(t, caPaths.Certificate, controlPlanePaths)
	waitForHTTP(t, ctx, runtimeProcess, controlClient, advertisedBaseURL+"/healthz", http.StatusOK)
	waitForProcessLog(t, ctx, runtimeProcess, "runtime agent registered")
	agents, initialSnapshot := waitForObservedRuntimeAgents(
		t, ctx, server, runtimes, publicClient, publicBaseURL,
		func(items []observedRuntimeAgent) bool {
			return len(items) == 1 && items[0].SlotState == "idle" &&
				items[0].ConfirmedLeaseUntil != nil
		},
	)
	runtimeAgent := agents[0]
	if got := runtimeToolset(runtimeAgent, "taint-annotations@1").Tools; !slices.Equal(got, []string{"annotate_sink", "annotate_trace", "annotate_validate"}) {
		t.Fatalf("Runtime taint-annotations capability = %v", got)
	}
	if got := runtimeToolset(runtimeAgent, "code-analysis@1").Tools; !slices.Equal(got, completeCodeAnalysisTools) {
		t.Fatalf("Runtime code-analysis capability = %v, want %v", got, completeCodeAnalysisTools)
	}
	if !slices.Contains(runtimeAgent.SupportedRuntimeAdapters, "otlp-http@1") {
		t.Fatalf("Runtime did not advertise otlp-http@1: %+v", runtimeAgent)
	}

	pool, err := pgxpool.New(ctx, isolateURL)
	if err != nil {
		t.Fatalf("open taint-annotations assertion database: %v", err)
	}
	t.Cleanup(pool.Close)
	store := runstore.NewPostgresStore(pool)

	initialFiles := map[string]string{
		taintSourcePath:    taintInitialApplication,
		taintDuplicatePath: taintInitialDuplicates,
	}
	sourceBytes := taintSourceArchive(t, initialFiles)
	source := uploadProjectArtifact(
		t, publicClient, publicBaseURL, "taint-annotations-source", "application/zip", sourceBytes,
	)
	initialRunID := createTaintAnnotationsRun(
		t, publicClient, publicBaseURL, "taint-annotations-initial", taintTargetCanary, source,
	)
	initialRun := waitForRunAcross(
		t, ctx, server, runtimes, gateway, publicClient, publicBaseURL, initialRunID,
	)
	initialAllocation := assertTaintRun(
		t, ctx, store, publicClient, publicBaseURL, initialRun,
		runtimeAgent.InstanceID, taintTraceReport, initialFiles,
		map[string]string{
			taintSourcePath:    taintAnnotatedApplication,
			taintDuplicatePath: taintAnnotatedDuplicates,
		},
		taintExpectedDiff,
	)
	select {
	case <-releaseProxy.dropped:
	case <-ctx.Done():
		t.Fatalf("wait for lost taint release acknowledgement: %v", ctx.Err())
	}

	// Cleanup already ran on the Runtime, but the Control Plane must retain the
	// authoritative fence until it obtains an acknowledgement from a retry.
	_, fencedSnapshot := waitForObservedRuntimeAgents(
		t, ctx, server, runtimes, publicClient, publicBaseURL,
		func(items []observedRuntimeAgent) bool {
			return len(items) == 1 && items[0].InstanceID == runtimeAgent.InstanceID &&
				items[0].SlotState == "fenced" && items[0].ObservedState == "idle" &&
				items[0].CurrentAllocationID == nil &&
				items[0].AuthoritativeAllocationID != nil &&
				*items[0].AuthoritativeAllocationID == initialAllocation.AllocationID
		},
	)
	releaseProxy.allowReleaseAcknowledgement()
	_, recoveredSnapshot := waitForObservedRuntimeAgents(
		t, ctx, server, runtimes, publicClient, publicBaseURL,
		func(items []observedRuntimeAgent) bool {
			return len(items) == 1 && items[0].InstanceID == runtimeAgent.InstanceID &&
				items[0].SlotState == "idle" && items[0].ObservedState == "idle" &&
				items[0].CurrentAllocationID == nil && items[0].AuthoritativeAllocationID == nil
		},
	)
	waitForRuntimeReleased(
		t, ctx, runtimeProcess, controlClient, advertisedBaseURL,
		initialAllocation.AllocationID, workRoot,
	)

	cleanFiles := map[string]string{taintCleanPath: taintCleanSource}
	cleanSourceBytes := taintSourceArchive(t, cleanFiles)
	cleanSource := uploadProjectArtifact(
		t, publicClient, publicBaseURL, "taint-annotations-clean-source",
		"application/zip", cleanSourceBytes,
	)
	cleanRunID := createTaintAnnotationsRun(
		t, publicClient, publicBaseURL, "taint-annotations-clean-reuse", "CLEAN_TARGET", cleanSource,
	)
	cleanRun := waitForRunAcross(
		t, ctx, server, runtimes, gateway, publicClient, publicBaseURL, cleanRunID,
	)
	cleanAllocation := assertTaintRun(
		t, ctx, store, publicClient, publicBaseURL, cleanRun,
		runtimeAgent.InstanceID, taintReuseReport, cleanFiles, cleanFiles, "",
	)
	waitForRuntimeReleased(
		t, ctx, runtimeProcess, controlClient, advertisedBaseURL,
		cleanAllocation.AllocationID, workRoot,
	)
	finalAgents, finalSnapshot := waitForObservedRuntimeAgents(
		t, ctx, server, runtimes, publicClient, publicBaseURL,
		func(items []observedRuntimeAgent) bool {
			return len(items) == 1 && items[0].InstanceID == runtimeAgent.InstanceID &&
				items[0].SlotState == "idle" && items[0].ObservedState == "idle" &&
				items[0].CurrentAllocationID == nil && items[0].AuthoritativeAllocationID == nil
		},
	)
	if len(finalAgents) != 1 || !workRootEmpty(workRoot) || !workRootEmpty(workspaceRoot) {
		t.Fatalf("taint Runtime retained allocation state: agents=%+v work=%t workspace=%t",
			finalAgents, workRootEmpty(workRoot), workRootEmpty(workspaceRoot))
	}
	if drops, failures := releaseProxy.snapshot(); drops == 0 || len(failures) != 0 {
		t.Fatalf("taint release-loss proxy = drops:%d failures:%v", drops, failures)
	}
	if gateway.CompletedStages() != 2 || len(gateway.Failures()) != 0 {
		t.Fatalf("taint Gateway = stages:%d calls:%d failures:%v",
			gateway.CompletedStages(), gateway.Calls(), gateway.Failures())
	}
	if collector.requests() == 0 || len(collector.failuresSnapshot()) != 0 {
		t.Fatalf("taint OTLP collector = requests:%d failures:%v",
			collector.requests(), collector.failuresSnapshot())
	}
	assertOTLPPayloadSafe(
		t, collector.payloads(), taintSourceCanary, taintTargetCanary,
		taintSourcePath, taintDuplicatePath, taintOTLPCredential,
		llmGatewayToken, publicToken,
	)

	for _, runID := range []string{initialRunID, cleanRunID} {
		assertTaintRetainedExecutionSafe(t, ctx, pool, runID)
	}
	assertTaintOperationalSurfacesSafe(
		t,
		[]string{initialSnapshot, fencedSnapshot, recoveredSnapshot, finalSnapshot},
		operations.evidence,
		[]*childProcess{server, runtimeProcess},
		temporaryRoot, workRoot, workspaceRoot,
	)
	assertUserArtifactUnchanged(
		t, publicClient, publicBaseURL, source, sourceBytes, "application/zip",
	)
	assertUserArtifactUnchanged(
		t, publicClient, publicBaseURL, cleanSource, cleanSourceBytes, "application/zip",
	)
}

func startTaintAnnotationsRuntime(
	t *testing.T,
	repositoryRoot, python, controlPlaneURL, advertisedURL, listenAddress,
	workRoot, workspaceRoot, caFile string,
	identity localpki.Paths,
) *childProcess {
	t.Helper()
	return startProcess(
		t, "Python local taint-annotations Runtime", filepath.Join(repositoryRoot, "runtime"),
		map[string]string{"PYTHONUNBUFFERED": "1"}, python,
		"-m", "contractor_runtime",
		"--control-plane-url", controlPlaneURL,
		"--advertised-control-url", advertisedURL,
		"--advertised-a2a-url", advertisedURL,
		"--ca-file", caFile,
		"--certificate-file", identity.Certificate,
		"--private-key-file", identity.PrivateKey,
		"--listen", listenAddress,
		"--work-root", workRoot,
		"--workspace-storage", "local",
		"--workspace-work-root", workspaceRoot,
		"--runtime-adapter", "otlp-http@1",
		"--request-timeout-seconds", "30",
		"--shutdown-grace-seconds", "8",
		"--heartbeat-interval-seconds", "1",
		"--confirmed-lease-seconds", "12",
	)
}

func createTaintAnnotationsRun(
	t *testing.T,
	client *http.Client,
	baseURL, idempotencyKey, target string,
	source artifactRef,
) string {
	t.Helper()
	body, err := json.Marshal(map[string]any{
		"workflow": "taint-trace-from-workspace@1",
		"parameters": map[string]string{
			"target": target, "objective": "Trace the selected request flow",
			"context": "Use only evidence in the supplied workspace",
		},
		"artifacts":     map[string]artifactRef{"source": source},
		"runtimeLabels": []string{"taint-debug"},
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
	if payload.RunID == "" || payload.State != "initializing" && payload.State != "running" {
		t.Fatalf("create taint Run response = %+v", payload)
	}
	return payload.RunID
}

func taintSourceArchive(t *testing.T, files map[string]string) []byte {
	t.Helper()
	var output bytes.Buffer
	writer := zip.NewWriter(&output)
	paths := make([]string, 0, len(files))
	for path := range files {
		paths = append(paths, path)
	}
	slices.Sort(paths)
	for _, path := range paths {
		header := &zip.FileHeader{Name: path, Method: zip.Store}
		header.SetMode(0o600)
		header.SetModTime(time.Date(2026, 9, 2, 0, 0, 0, 0, time.UTC))
		entry, err := writer.CreateHeader(header)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := entry.Write([]byte(files[path])); err != nil {
			t.Fatal(err)
		}
	}
	if err := writer.Close(); err != nil {
		t.Fatal(err)
	}
	return output.Bytes()
}

func assertTaintRun(
	t *testing.T,
	ctx context.Context,
	store runstore.Repository,
	client *http.Client,
	baseURL string,
	status runStatus,
	wantRuntimeInstanceID, wantReport string,
	sourceFiles, resultFiles map[string]string,
	wantDiff string,
) runstore.StageAllocation {
	t.Helper()
	if status.State != "succeeded" || len(status.Attempts) != 1 ||
		status.Attempts[0].State != "succeeded" || status.Attempts[0].Attempt != 1 {
		t.Fatalf("taint terminal Run = %+v", status)
	}
	for _, output := range []string{"report", "workspace_state", "workspace_diff"} {
		ref, ok := status.Outputs[output]
		if !ok || ref.Revision == nil {
			t.Fatalf("taint Run omitted exact output %q: %+v", output, status.Outputs)
		}
	}
	report, reportMediaType := download(
		t, client, baseURL+"/v1/runs/"+url.PathEscape(status.RunID)+"/outputs/report",
	)
	if string(report) != wantReport || reportMediaType != "text/markdown" {
		t.Fatalf("taint report = (%q, %q)", report, reportMediaType)
	}
	stateBytes, stateMediaType := download(
		t, client, baseURL+"/v1/runs/"+url.PathEscape(status.RunID)+"/outputs/workspace_state",
	)
	if stateMediaType != "application/vnd.contractor.workspace-overlay+json" {
		t.Fatalf("taint workspace state media type = %q", stateMediaType)
	}
	diff, diffMediaType := download(
		t, client, baseURL+"/v1/runs/"+url.PathEscape(status.RunID)+"/outputs/workspace_diff",
	)
	if diffMediaType != "text/x-diff" || string(diff) != wantDiff {
		t.Fatalf("taint workspace diff = (%q, %q), want exact %q", diff, diffMediaType, wantDiff)
	}
	assertTaintWorkspaceState(t, stateBytes, sourceFiles, resultFiles)
	allocation := onlyRunAllocation(t, ctx, store, status.RunID)
	if allocation.RuntimeAgentInstanceID != wantRuntimeInstanceID {
		t.Fatalf("taint Run %s used Runtime %s, want %s",
			status.RunID, allocation.RuntimeAgentInstanceID, wantRuntimeInstanceID)
	}
	return allocation
}

func assertTaintWorkspaceState(
	t *testing.T,
	payload []byte,
	sourceFiles, resultFiles map[string]string,
) {
	t.Helper()
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(payload, &raw); err != nil {
		t.Fatalf("decode taint workspace state: %v", err)
	}
	wantKeys := []string{
		"apiVersion", "baseWorkspaceDigest", "kind", "operations", "resultWorkspaceDigest",
	}
	gotKeys := make([]string, 0, len(raw))
	for key := range raw {
		gotKeys = append(gotKeys, key)
	}
	slices.Sort(gotKeys)
	if !slices.Equal(gotKeys, wantKeys) {
		t.Fatalf("taint workspace state keys = %v", gotKeys)
	}
	var state taintWorkspaceState
	if err := json.Unmarshal(payload, &state); err != nil {
		t.Fatalf("decode typed taint workspace state: %v", err)
	}
	if state.APIVersion != "contractor.workspace/v1" || state.Kind != "WorkspaceOverlay" ||
		!strings.HasPrefix(state.BaseWorkspaceDigest, "sha256:") ||
		!strings.HasPrefix(state.ResultWorkspaceDigest, "sha256:") {
		t.Fatalf("invalid taint workspace state envelope: %+v", state)
	}

	applied := maps.Clone(sourceFiles)
	wantChangedPaths := make([]string, 0)
	for path, result := range resultFiles {
		if sourceFiles[path] != result {
			wantChangedPaths = append(wantChangedPaths, path)
		}
	}
	slices.Sort(wantChangedPaths)
	if (len(wantChangedPaths) == 0) != (state.BaseWorkspaceDigest == state.ResultWorkspaceDigest) {
		t.Fatalf("taint workspace digest transition disagrees with operations: %+v", state)
	}
	if len(state.Operations) != len(wantChangedPaths) {
		t.Fatalf("taint workspace operations = %+v, want changes in %v", state.Operations, wantChangedPaths)
	}
	for index, operation := range state.Operations {
		if operation.Op != "write_file" || operation.Path != wantChangedPaths[index] ||
			operation.Text != resultFiles[operation.Path] {
			t.Fatalf("taint workspace operation %d = %+v", index, operation)
		}
		applied[operation.Path] = operation.Text
	}
	if !maps.Equal(applied, resultFiles) {
		t.Fatalf("taint workspace state did not reproduce exact result: got=%v want=%v",
			applied, resultFiles)
	}
}

func assertTaintRetainedExecutionSafe(
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
			t.Fatalf("read retained taint execution surface: %v", err)
		}
		for _, canary := range taintOperationalCanaries() {
			if strings.Contains(retained, canary) {
				t.Fatalf("retained execution surface for %s exposed %q", runID, canary)
			}
		}
	}
}

func assertTaintOperationalSurfacesSafe(
	t *testing.T,
	snapshots []string,
	operationEvidence [][]byte,
	processes []*childProcess,
	privatePaths ...string,
) {
	t.Helper()
	canaries := append(taintOperationalCanaries(), privatePaths...)
	for _, canary := range canaries {
		for _, snapshot := range snapshots {
			if strings.Contains(snapshot, canary) {
				t.Fatalf("Operations snapshot exposed %q", canary)
			}
		}
		for _, evidence := range operationEvidence {
			if bytes.Contains(evidence, []byte(canary)) {
				t.Fatalf("Operations mutation response exposed %q", canary)
			}
		}
		for _, process := range processes {
			if strings.Contains(process.logs.redacted(), canary) {
				t.Fatalf("%s logs exposed %q", process.name, canary)
			}
		}
	}
}

func taintOperationalCanaries() []string {
	return []string{
		taintSourceCanary, taintTargetCanary, taintSourcePath, taintDuplicatePath,
		taintOTLPCredential, publicToken, llmGatewayToken,
	}
}
