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
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/localpki"
	contractormemory "github.com/grauwolf32/contractor/internal/memory"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
	collectortracev1 "go.opentelemetry.io/proto/otlp/collector/trace/v1"
	"google.golang.org/protobuf/encoding/protojson"
	"google.golang.org/protobuf/proto"
)

const memoryTelemetryLabel = "memory-release-audit"

func TestSharedMemoryMVPProcesses(t *testing.T) {
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
		LeafOptions: leaf, URI: "urn:contractor:control-plane:shared-memory-e2e",
	})
	if err != nil {
		t.Fatalf("issue Control Plane certificate: %v", err)
	}
	agentPaths := make([]localpki.Paths, 2)
	for index := range agentPaths {
		agentPaths[index], err = generator.IssueAgent(
			pkiRoot, fmt.Sprintf("shared-memory-e2e-agent-%d", index+1), leaf,
		)
		if err != nil {
			t.Fatalf("issue Runtime Agent %d certificate: %v", index+1, err)
		}
	}

	gateway := newSharedMemoryGateway(llmGatewayToken)
	t.Cleanup(gateway.close)
	configRoot := stageE2EConfiguration(
		t, filepath.Join(repositoryRoot, "configs"),
		filepath.Join(temporaryRoot, "configs"), gateway.URL(),
	)
	publicAddress := freeAddress(t)
	privateAddress := freeAddress(t)
	publicBaseURL := "http://" + publicAddress
	privateBaseURL := "https://" + privateAddress
	userID := "shared-memory-e2e-user-" + randomHex(t, 8)
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
	publicClient := &http.Client{Timeout: 5 * time.Second}
	waitForHTTP(t, ctx, server, publicClient, publicBaseURL+"/readyz", http.StatusOK)
	collector := newFakeOTLPCollector("")
	t.Cleanup(collector.close)
	operations := &runtimeOperations{t: t, client: publicClient, baseURL: publicBaseURL}
	telemetryConfig := operations.publishRuntimeConfig("memory-release-audit", "1", map[string]any{
		"planner": map[string]any{"telemetry": map[string]any{
			"adapter": "otlp-http@1", "endpoint": collector.URL(),
			"captureContent": false, "flushTimeoutSeconds": 2,
		}},
		"worker": map[string]any{"telemetry": map[string]any{
			"adapter": "otlp-http@1", "endpoint": collector.URL(),
			"captureContent": false, "flushTimeoutSeconds": 2,
		}},
	})
	operations.createRuntimeLabel(memoryTelemetryLabel, telemetryConfig)

	python := filepath.Join(repositoryRoot, "runtime", ".venv", "bin", "python")
	if info, statErr := os.Stat(python); statErr != nil || info.IsDir() {
		t.Fatalf("Python Runtime environment is missing at %s; run 'cd runtime && uv sync --locked'", python)
	}
	controlClient := newMTLSClient(t, caPaths.Certificate, controlPlanePaths)
	runtimes := make([]*childProcess, 0, 2)
	runtimeURLs := make([]string, 0, 2)
	workRoots := make([]string, 0, 2)
	for index := range agentPaths {
		runtimeAddress := freeAddress(t)
		runtimeBaseURL := "https://" + runtimeAddress
		workRoot := filepath.Join(temporaryRoot, fmt.Sprintf("runtime-work-%d", index+1))
		process := startProcess(
			t, fmt.Sprintf("Python Runtime Agent %d", index+1), filepath.Join(repositoryRoot, "runtime"),
			map[string]string{"PYTHONUNBUFFERED": "1"},
			python, "-m", "contractor_runtime",
			"--control-plane-url", privateBaseURL,
			"--advertised-control-url", runtimeBaseURL,
			"--advertised-a2a-url", runtimeBaseURL,
			"--ca-file", caPaths.Certificate,
			"--certificate-file", agentPaths[index].Certificate,
			"--private-key-file", agentPaths[index].PrivateKey,
			"--listen", runtimeAddress,
			"--work-root", workRoot,
			"--request-timeout-seconds", "10",
			"--shutdown-grace-seconds", "5",
		)
		waitForHTTP(t, ctx, process, controlClient, runtimeBaseURL+"/healthz", http.StatusOK)
		waitForProcessLog(t, ctx, process, "runtime agent registered")
		runtimes = append(runtimes, process)
		runtimeURLs = append(runtimeURLs, runtimeBaseURL)
		workRoots = append(workRoots, workRoot)
	}

	pool, err := pgxpool.New(ctx, isolateURL)
	if err != nil {
		t.Fatalf("open assertion database: %v", err)
	}
	t.Cleanup(pool.Close)
	store := runstore.NewPostgresStore(pool)

	firstRunID := createEmptyWorkflowRun(
		t, publicClient, publicBaseURL, "shared-memory-streamline@1", "shared-memory-streamline-first",
		memoryTelemetryLabel,
	)
	firstStatus := waitForSharedMemoryRun(
		t, ctx, server, runtimes, gateway, publicClient, publicBaseURL, firstRunID,
	)
	firstAllocations := assertStreamlineMemoryRun(t, ctx, pool, store, firstRunID, firstStatus)
	assertOwnerMemoryArtifact(
		t, publicClient, publicBaseURL, firstRunID, "shared", streamlineNoteName,
		streamlineSeed+"\n"+streamlineFirstAppend+"\n"+streamlineRetryAppend+"\n"+streamlineConfirmAppend,
	)

	routerRunID := createEmptyWorkflowRun(
		t, publicClient, publicBaseURL, "shared-memory-router@1", "shared-memory-router",
		memoryTelemetryLabel,
	)
	routerStatus := waitForSharedMemoryRun(
		t, ctx, server, runtimes, gateway, publicClient, publicBaseURL, routerRunID,
	)
	routerAllocations := assertRouterMemoryRun(t, ctx, pool, store, routerRunID, routerStatus)

	secondRunID := createEmptyWorkflowRun(
		t, publicClient, publicBaseURL, "shared-memory-streamline@1", "shared-memory-streamline-second",
		memoryTelemetryLabel,
	)
	secondStatus := waitForSharedMemoryRun(
		t, ctx, server, runtimes, gateway, publicClient, publicBaseURL, secondRunID,
	)
	secondAllocations := assertStreamlineMemoryRun(t, ctx, pool, store, secondRunID, secondStatus)

	emptyStarts, existingStarts := gateway.StartCounts()
	if emptyStarts != 2 || existingStarts != 2 {
		t.Fatalf("coordinate namespace starts empty/existing = %d/%d, want 2/2", emptyStarts, existingStarts)
	}
	var firstRevision, secondRevision string
	if err := pool.QueryRow(ctx, `
SELECT current_revision FROM artifact_bindings
WHERE scope_kind = 'run' AND scope_id = $1 AND namespace = 'shared' AND name = $2`,
		firstRunID, "memory."+streamlineNoteName,
	).Scan(&firstRevision); err != nil {
		t.Fatal(err)
	}
	if err := pool.QueryRow(ctx, `
SELECT current_revision FROM artifact_bindings
WHERE scope_kind = 'run' AND scope_id = $1 AND namespace = 'shared' AND name = $2`,
		secondRunID, "memory."+streamlineNoteName,
	).Scan(&secondRevision); err != nil {
		t.Fatal(err)
	}
	if firstRevision == secondRevision {
		t.Fatalf("separate Runs unexpectedly share current revision %q", firstRevision)
	}

	allAllocations := append(firstAllocations, routerAllocations...)
	allAllocations = append(allAllocations, secondAllocations...)
	assertNoPlacementInMemoryGateway(t, gateway.Transcripts(), allAllocations, runtimeURLs)
	if failures := gateway.Failures(); len(failures) != 0 {
		t.Fatalf("shared-memory Gateway validation failures: %v", failures)
	}
	for index := range runtimes {
		waitForRuntimeIdle(t, ctx, runtimes[index], controlClient, runtimeURLs[index], workRoots[index])
	}
	canaries := sharedMemoryCanaries()
	assertMemoryHTTPFailureRedacted(t, publicClient, publicBaseURL, canaries)
	assertNoMemoryOnlyDatabaseInventory(t, ctx, pool)
	assertAllNonPayloadDatabaseColumnsRedacted(t, ctx, pool, canaries)
	assertDecodedMemoryOTLPRedacted(t, collector.payloads(), canaries)
	if failures := collector.failuresSnapshot(); len(failures) != 0 {
		t.Fatalf("shared-memory OTLP collector failures: %v", failures)
	}
	for _, secret := range []string{publicToken, llmGatewayToken} {
		if strings.Contains(server.logs.redacted(), secret) {
			t.Fatal("Server logs contain a configured secret")
		}
		for _, process := range runtimes {
			if strings.Contains(process.logs.redacted(), secret) {
				t.Fatalf("%s logs contain a configured secret", process.name)
			}
		}
	}
	assertNoMemoryCanaryForms(t, "Server logs", []byte(server.logs.redacted()), canaries)
	for _, process := range runtimes {
		assertNoMemoryCanaryForms(t, process.name+" logs", []byte(process.logs.redacted()), canaries)
	}
}

func createEmptyWorkflowRun(
	t *testing.T, client *http.Client, baseURL, workflow, idempotencyKey string, labels ...string,
) string {
	t.Helper()
	requestBody := map[string]any{
		"workflow": workflow, "parameters": map[string]string{}, "artifacts": map[string]artifactRef{},
	}
	if len(labels) != 0 {
		requestBody["runtimeLabels"] = labels
	}
	body, err := json.Marshal(requestBody)
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
	var payload runCreateResponse
	decodeResponse(t, response, &payload)
	if payload.RunID == "" || payload.State != "running" {
		t.Fatalf("create empty Run response = %+v", payload)
	}
	return payload.RunID
}

func waitForSharedMemoryRun(
	t *testing.T,
	ctx context.Context,
	server *childProcess,
	runtimes []*childProcess,
	gateway *sharedMemoryGateway,
	client *http.Client,
	baseURL, runID string,
) runStatus {
	t.Helper()
	ticker := time.NewTicker(200 * time.Millisecond)
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
			if status.State == "succeeded" {
				return status
			}
			if status.State == "failed" || status.State == "cancelled" {
				t.Fatalf("Run %s reached %s: %+v\n%s", runID, status.State, status,
					sharedMemoryDiagnostics(server, runtimes, gateway))
			}
		}
		for _, process := range append([]*childProcess{server}, runtimes...) {
			if exited, processErr := process.exited(); exited {
				t.Fatalf("%s exited while Run was active: %v\n%s", process.name, processErr,
					sharedMemoryDiagnostics(server, runtimes, gateway))
			}
		}
		select {
		case <-ctx.Done():
			t.Fatalf("wait for Run %s: %v\n%s", runID, ctx.Err(),
				sharedMemoryDiagnostics(server, runtimes, gateway))
		case <-ticker.C:
		}
	}
}

func sharedMemoryDiagnostics(
	server *childProcess, runtimes []*childProcess, gateway *sharedMemoryGateway,
) string {
	var result strings.Builder
	result.WriteString("server:\n")
	result.WriteString(server.logs.redacted(publicToken, llmGatewayToken))
	for _, process := range runtimes {
		result.WriteString("\n" + process.name + ":\n")
		result.WriteString(process.logs.redacted(publicToken, llmGatewayToken))
	}
	result.WriteString(fmt.Sprintf("\ngateway observations=%+v failures=%v",
		gateway.Observations(), gateway.Failures()))
	return result.String()
}

func assertStreamlineMemoryRun(
	t *testing.T,
	ctx context.Context,
	pool *pgxpool.Pool,
	store *runstore.PostgresStore,
	runID string,
	status runStatus,
) []runstore.StageAllocation {
	t.Helper()
	// The release scenario enables Planner OTLP, whose bounded supplementary
	// telemetry.export record adds one safe tool-call metric to each attempt.
	assertPublicAttemptMetrics(t, status, [][2]int64{{9, 9}, {9, 9}, {9, 9}})
	if len(status.Outputs) != 0 {
		t.Fatalf("shared-memory Run %s exposed outputs: %+v", runID, status.Outputs)
	}
	executions := requireExecutions(t, ctx, store, runID, 3)
	first, retry, confirm := executions[0], executions[1], executions[2]
	if first.StageName != "coordinate" || first.Attempt != 1 || first.State != runstore.StageFailed ||
		first.AcceptedResult == nil || first.AcceptedResult.Outcome != contracts.StageFailed ||
		first.AcceptedResult.Error == nil || first.AcceptedResult.Error.Code != "memory_retry_probe" ||
		!first.AcceptedResult.Error.Retryable || first.FinalizationID == nil || first.TerminalAt == nil {
		t.Fatalf("first shared-memory attempt = %+v", first)
	}
	assertSucceededExecution(t, retry, 2)
	if retry.StageName != "coordinate" || retry.PreviousExecutionID == nil ||
		*retry.PreviousExecutionID != first.StageExecutionID {
		t.Fatalf("shared-memory retry linkage = %+v", retry)
	}
	assertSucceededExecution(t, confirm, 1)
	if confirm.StageName != "confirm" || confirm.PreviousExecutionID != nil {
		t.Fatalf("later shared-memory Stage = %+v", confirm)
	}
	decisions, err := store.ListStageTransitionDecisions(ctx, runID)
	if err != nil || len(decisions) != 3 ||
		decisions[0].Action != runstore.StageTransitionRetry ||
		decisions[1].Action != runstore.StageTransitionNext ||
		decisions[2].Action != runstore.StageTransitionSucceed {
		t.Fatalf("shared-memory transition decisions = (%+v, %v)", decisions, err)
	}
	allocations := make([]runstore.StageAllocation, 0, 3)
	for _, execution := range executions {
		allocations = append(allocations, assertCompleteAllocations(t, ctx, store, execution, 1, "builder")...)
	}
	expected := []string{
		streamlineSeed,
		streamlineSeed + "\n" + streamlineFirstAppend,
		streamlineSeed + "\n" + streamlineFirstAppend + "\n" + streamlineRetryAppend,
		streamlineSeed + "\n" + streamlineFirstAppend + "\n" + streamlineRetryAppend + "\n" + streamlineConfirmAppend,
	}
	assertPersistedMemory(
		t, ctx, pool, runID, "shared", streamlineNoteName, streamlineDescription,
		[]string{streamlineTag}, expected,
	)
	assertMemoryDiagnosticsRedacted(t, ctx, pool, runID, streamlineNoteName, []string{
		streamlineSeed, streamlineFirstAppend, streamlineRetryAppend,
		streamlineConfirmAppend, streamlineDescription, streamlineTag,
	})
	return allocations
}

func assertRouterMemoryRun(
	t *testing.T,
	ctx context.Context,
	pool *pgxpool.Pool,
	store *runstore.PostgresStore,
	runID string,
	status runStatus,
) []runstore.StageAllocation {
	t.Helper()
	assertPublicAttemptMetrics(t, status, [][2]int64{{17, 16}})
	executions := requireExecutions(t, ctx, store, runID, 1)
	assertSucceededExecution(t, executions[0], 1)
	allocations := assertCompleteAllocations(t, ctx, store, executions[0], 2, "builder", "reviewer")
	assertPersistedMemory(
		t, ctx, pool, runID, "builder_notes", routerNoteName, routerBuilderDescription,
		[]string{routerBuilderTag}, []string{
			routerBuilderSeed,
			routerBuilderSeed + "\n" + routerBuilderPlanner,
			routerBuilderSeed + "\n" + routerBuilderPlanner + "\n" + routerBuilderWorker,
		},
	)
	assertPersistedMemory(
		t, ctx, pool, runID, "reviewer_notes", routerNoteName, routerReviewerDescription,
		[]string{routerReviewerTag}, []string{routerReviewerSeed},
	)
	assertMemoryDiagnosticsRedacted(t, ctx, pool, runID, routerNoteName, []string{
		routerBuilderSeed, routerBuilderPlanner, routerBuilderWorker,
		routerBuilderDescription, routerBuilderTag,
		routerReviewerSeed, routerReviewerDescription, routerReviewerTag,
	})
	return allocations
}

func assertPersistedMemory(
	t *testing.T,
	ctx context.Context,
	pool *pgxpool.Pool,
	runID, namespace, name, description string,
	tags, expectedContents []string,
) {
	t.Helper()
	rows, err := pool.Query(ctx, `
SELECT binding.current_revision, binding.frozen, binding.created_at,
       revision.revision, revision.created_at, version.media_type, blob.payload
FROM artifact_bindings AS binding
JOIN artifact_binding_revisions AS revision
  ON revision.scope_kind = binding.scope_kind AND revision.scope_id = binding.scope_id
 AND revision.namespace = binding.namespace AND revision.name = binding.name
JOIN artifact_versions AS version ON version.version_id = revision.version_id
JOIN artifact_blobs AS blob ON blob.sha256 = version.blob_sha256
WHERE binding.scope_kind = 'run' AND binding.scope_id = $1
  AND binding.namespace = $2 AND binding.name = $3
ORDER BY revision.created_at, revision.revision`, runID, namespace, "memory."+name)
	if err != nil {
		t.Fatal(err)
	}
	defer rows.Close()
	var currentRevision string
	var bindingCreatedAt time.Time
	revisions := make([]string, 0, len(expectedContents))
	contents := make([]string, 0, len(expectedContents))
	var previousRevisionAt time.Time
	for rows.Next() {
		var current string
		var frozen bool
		var bindingCreated time.Time
		var revision string
		var revisionCreated time.Time
		var mediaType string
		var payload []byte
		if err := rows.Scan(
			&current, &frozen, &bindingCreated, &revision, &revisionCreated, &mediaType, &payload,
		); err != nil {
			t.Fatal(err)
		}
		if frozen || mediaType != contractormemory.MediaType || bindingCreated.IsZero() || revisionCreated.IsZero() ||
			revisionCreated.Before(bindingCreated) ||
			(!previousRevisionAt.IsZero() && !revisionCreated.After(previousRevisionAt)) {
			t.Fatalf("invalid Memory metadata for %s/%s: frozen=%t media=%q binding=%s revision=%s",
				namespace, name, frozen, mediaType, bindingCreated, revisionCreated)
		}
		if currentRevision != "" && (currentRevision != current || !bindingCreatedAt.Equal(bindingCreated)) {
			t.Fatalf("Memory binding metadata changed across immutable revisions for %s/%s", namespace, name)
		}
		currentRevision, bindingCreatedAt, previousRevisionAt = current, bindingCreated, revisionCreated
		note, decodeErr := contractormemory.Decode("memory."+name, payload)
		if decodeErr != nil || note.Name != name || note.Description != description ||
			note.Ordinal != 0 || !equalStringSlices(note.Tags, tags) {
			t.Fatalf("decode persisted Memory %s/%s = (%+v, %v)", namespace, name, note, decodeErr)
		}
		revisions = append(revisions, revision)
		contents = append(contents, note.Content)
	}
	if err := rows.Err(); err != nil {
		t.Fatal(err)
	}
	if !equalStringSlices(contents, expectedContents) || len(revisions) != len(expectedContents) ||
		len(revisions) == 0 || currentRevision != revisions[len(revisions)-1] {
		t.Fatalf("Memory %s/%s revisions/content = %v/%q current=%q, want %q",
			namespace, name, revisions, contents, currentRevision, expectedContents)
	}
	seen := make(map[string]bool, len(revisions))
	for _, revision := range revisions {
		if revision == "" || seen[revision] {
			t.Fatalf("Memory %s/%s has invalid immutable revisions %v", namespace, name, revisions)
		}
		seen[revision] = true
	}
}

func assertMemoryDiagnosticsRedacted(
	t *testing.T,
	ctx context.Context,
	pool *pgxpool.Pool,
	runID, noteName string,
	canaries []string,
) {
	t.Helper()
	queries := []string{
		`SELECT COALESCE(string_agg(event::text, E'\n'), '')
FROM planner_events WHERE run_id = $1`,
		`SELECT COALESCE(string_agg(data::text, E'\n'), '')
FROM workflow_run_events WHERE run_id = $1`,
		`SELECT COALESCE(string_agg(session.state::text, E'\n'), '')
FROM planner_sessions AS session
JOIN stage_executions AS execution ON execution.stage_execution_id = session.stage_execution_id
WHERE execution.run_id = $1`,
		`SELECT COALESCE(string_agg(report.report::text, E'\n'), '')
FROM planner_execution_reports AS report
JOIN stage_executions AS execution ON execution.stage_execution_id = report.stage_execution_id
WHERE execution.run_id = $1`,
		`SELECT COALESCE(string_agg(report.report::text, E'\n'), '')
FROM allocation_execution_reports AS report
JOIN stage_executions AS execution ON execution.stage_execution_id = report.stage_execution_id
WHERE execution.run_id = $1`,
		`SELECT COALESCE(string_agg(metrics.metrics::text || metrics.summary::text, E'\n'), '')
FROM stage_metrics AS metrics
JOIN stage_executions AS execution ON execution.stage_execution_id = metrics.stage_execution_id
WHERE execution.run_id = $1`,
	}
	combined := ""
	for _, query := range queries {
		var value string
		if err := pool.QueryRow(ctx, query, runID).Scan(&value); err != nil {
			t.Fatalf("read retained Memory diagnostics: %v", err)
		}
		assertNoMemoryCanaryForms(t, "Run "+runID+" diagnostics", []byte(value), canaries)
		combined += value
	}
	if !strings.Contains(combined, "read_memory") || !strings.Contains(combined, noteName) {
		t.Fatalf("Run %s retained diagnostics lost safe Memory operation/name dimensions", runID)
	}
}

func sharedMemoryCanaries() []string {
	return []string{
		streamlineSeed, streamlineFirstAppend, streamlineRetryAppend,
		streamlineConfirmAppend, streamlineDescription, streamlineTag,
		routerBuilderSeed, routerBuilderPlanner, routerBuilderWorker,
		routerBuilderDescription, routerBuilderTag,
		routerReviewerSeed, routerReviewerDescription, routerReviewerTag,
	}
}

func memoryCanaryForms(canary string) []string {
	encoded, _ := json.Marshal(canary)
	candidates := []string{
		canary,
		string(encoded),
		strings.TrimSuffix(strings.TrimPrefix(string(encoded), `"`), `"`),
		url.QueryEscape(canary),
		url.PathEscape(canary),
	}
	result := make([]string, 0, len(candidates))
	seen := map[string]bool{}
	for _, candidate := range candidates {
		if candidate != "" && !seen[candidate] {
			seen[candidate] = true
			result = append(result, candidate)
		}
	}
	return result
}

func assertNoMemoryCanaryForms(t *testing.T, surface string, data []byte, canaries []string) {
	t.Helper()
	for _, canary := range canaries {
		for _, form := range memoryCanaryForms(canary) {
			if bytes.Contains(data, []byte(form)) {
				t.Fatalf("%s retained Memory canary form %q", surface, form)
			}
		}
	}
}

func assertMemoryHTTPFailureRedacted(
	t *testing.T,
	client *http.Client,
	baseURL string,
	canaries []string,
) {
	t.Helper()
	body, err := json.Marshal(map[string]any{
		"workflow":                "shared-memory-streamline@1",
		"parameters":              map[string]string{},
		"artifacts":               map[string]artifactRef{},
		"unexpectedMemoryPayload": strings.Join(canaries, " | "),
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
	request.Header.Set("Idempotency-Key", "shared-memory-redacted-failure")
	response := do(t, client, request, http.StatusBadRequest)
	defer response.Body.Close()
	failure, err := io.ReadAll(io.LimitReader(response.Body, 64<<10))
	if err != nil {
		t.Fatal(err)
	}
	assertNoMemoryCanaryForms(t, "public HTTP failure", failure, canaries)
}

func assertNoMemoryOnlyDatabaseInventory(
	t *testing.T,
	ctx context.Context,
	pool *pgxpool.Pool,
) {
	t.Helper()
	rows, err := pool.Query(ctx, `
SELECT table_name, column_name
FROM information_schema.columns
WHERE table_schema = current_schema()
  AND (lower(table_name) LIKE '%memory%' OR lower(column_name) LIKE '%memory%')
ORDER BY table_name, ordinal_position`)
	if err != nil {
		t.Fatal(err)
	}
	defer rows.Close()
	var inventory []string
	for rows.Next() {
		var table, column string
		if err := rows.Scan(&table, &column); err != nil {
			t.Fatal(err)
		}
		inventory = append(inventory, table+"."+column)
	}
	if err := rows.Err(); err != nil {
		t.Fatal(err)
	}
	if len(inventory) != 0 {
		t.Fatalf("Memory-specific database inventory exists: %v", inventory)
	}
}

func assertAllNonPayloadDatabaseColumnsRedacted(
	t *testing.T,
	ctx context.Context,
	pool *pgxpool.Pool,
	canaries []string,
) {
	t.Helper()
	rows, err := pool.Query(ctx, `
SELECT table_name, column_name
FROM information_schema.columns
WHERE table_schema = current_schema()
ORDER BY table_name, ordinal_position`)
	if err != nil {
		t.Fatal(err)
	}
	type columnRef struct{ table, column string }
	var columns []columnRef
	for rows.Next() {
		var column columnRef
		if err := rows.Scan(&column.table, &column.column); err != nil {
			rows.Close()
			t.Fatal(err)
		}
		if column.table == "artifact_blobs" && column.column == "payload" {
			continue
		}
		columns = append(columns, column)
	}
	if err := rows.Err(); err != nil {
		rows.Close()
		t.Fatal(err)
	}
	rows.Close()
	for _, column := range columns {
		query := fmt.Sprintf(
			"SELECT COALESCE(string_agg(%s::text, E'\\n'), '') FROM %s",
			pgx.Identifier{column.column}.Sanitize(), pgx.Identifier{column.table}.Sanitize(),
		)
		var retained string
		if err := pool.QueryRow(ctx, query).Scan(&retained); err != nil {
			t.Fatalf("scan retained column %s.%s: %v", column.table, column.column, err)
		}
		assertNoMemoryCanaryForms(
			t, "PostgreSQL "+column.table+"."+column.column, []byte(retained), canaries,
		)
	}
}

func assertDecodedMemoryOTLPRedacted(
	t *testing.T,
	payloads [][]byte,
	canaries []string,
) {
	t.Helper()
	if len(payloads) == 0 {
		t.Fatal("shared-memory OTLP collector received no payloads")
	}
	var decoded bytes.Buffer
	for index, payload := range payloads {
		var request collectortracev1.ExportTraceServiceRequest
		if err := proto.Unmarshal(payload, &request); err != nil {
			t.Fatalf("decode OTLP payload %d: %v", index, err)
		}
		encoded, err := (protojson.MarshalOptions{UseProtoNames: true}).Marshal(&request)
		if err != nil {
			t.Fatalf("project decoded OTLP payload %d: %v", index, err)
		}
		decoded.Write(encoded)
		decoded.WriteByte('\n')
	}
	assertNoMemoryCanaryForms(t, "decoded OTLP", decoded.Bytes(), canaries)
	projection := decoded.String()
	for _, scope := range []string{"contractor.planner.", "contractor.runtime."} {
		if !strings.Contains(projection, scope) {
			t.Fatalf("decoded OTLP has no safe %q scope", scope)
		}
	}
}

func assertOwnerMemoryArtifact(
	t *testing.T,
	client *http.Client,
	baseURL, runID, namespace, name, expectedContent string,
) {
	t.Helper()
	request, _ := http.NewRequest(
		http.MethodGet,
		baseURL+"/v1/runs/"+url.PathEscape(runID)+"/artifacts?namespace="+url.QueryEscape(namespace),
		nil,
	)
	request.Header.Set("Authorization", "Bearer "+publicToken)
	response := do(t, client, request, http.StatusOK)
	defer response.Body.Close()
	var page struct {
		Items []struct {
			Artifact  artifactRef `json:"artifact"`
			MediaType string      `json:"mediaType"`
			Size      int64       `json:"size"`
			Current   bool        `json:"current"`
			Frozen    bool        `json:"frozen"`
			CreatedAt time.Time   `json:"createdAt"`
		} `json:"items"`
		Page json.RawMessage `json:"page"`
	}
	decodeResponse(t, response, &page)
	if len(page.Items) != 1 || page.Items[0].Artifact.Namespace != namespace ||
		page.Items[0].Artifact.Name != "memory."+name || page.Items[0].Artifact.Revision == nil ||
		page.Items[0].MediaType != contractormemory.MediaType || !page.Items[0].Current || page.Items[0].Frozen {
		t.Fatalf("owner Run Artifact page = %+v", page)
	}
	data, mediaType := download(
		t, client,
		baseURL+"/v1/runs/"+url.PathEscape(runID)+"/artifacts/"+
			url.PathEscape(namespace)+"/"+url.PathEscape("memory."+name),
	)
	note, err := contractormemory.Decode("memory."+name, data)
	if err != nil || note.Content != expectedContent || mediaType != contractormemory.MediaType {
		t.Fatalf("owner Memory Artifact = (%+v, %q, %v)", note, mediaType, err)
	}
}

func assertNoPlacementInMemoryGateway(
	t *testing.T,
	transcripts []string,
	allocations []runstore.StageAllocation,
	runtimeURLs []string,
) {
	t.Helper()
	combined := strings.Join(transcripts, "\n")
	if combined == "" || strings.Contains(combined, "/private/v1/allocations/") {
		t.Fatal("Memory Gateway transcript is absent or contains a private allocation route")
	}
	for _, allocation := range allocations {
		for _, physical := range []string{allocation.AllocationID, allocation.RuntimeAgentInstanceID} {
			if physical != "" && strings.Contains(combined, physical) {
				t.Fatalf("Memory Gateway transcript leaked physical identity %q", physical)
			}
		}
	}
	for _, runtimeURL := range runtimeURLs {
		if strings.Contains(combined, runtimeURL) {
			t.Fatalf("Memory Gateway transcript leaked Runtime URL %q", runtimeURL)
		}
	}
}
