//go:build e2e

package e2e

import (
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
	"reflect"
	"sort"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/localpki"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/jackc/pgx/v5/pgxpool"
	collectortracev1 "go.opentelemetry.io/proto/otlp/collector/trace/v1"
	commonv1 "go.opentelemetry.io/proto/otlp/common/v1"
	"google.golang.org/protobuf/proto"
)

const metadataLabelOTLPToken = "RUN_METADATA_LABEL_OTLP_HEADER_SECRET"

type metadataRunCreateResponse struct {
	RunID                string            `json:"runId"`
	State                string            `json:"state"`
	RuntimeLabels        []string          `json:"runtimeLabels"`
	Labels               map[string]string `json:"labels"`
	RuntimeConfiguration json.RawMessage   `json:"runtimeConfiguration"`
}

type metadataRunPage struct {
	Items []struct {
		RunID  string            `json:"runId"`
		State  string            `json:"state"`
		Labels map[string]string `json:"labels"`
	} `json:"items"`
	Page struct {
		HasMore    bool   `json:"hasMore"`
		NextCursor string `json:"nextCursor,omitempty"`
	} `json:"page"`
}

func TestRunMetadataLabelsAcrossProcesses(t *testing.T) {
	if testing.Short() {
		t.Skip("end-to-end process test")
	}
	databaseURL := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if databaseURL == "" {
		t.Fatal("CONTRACTOR_TEST_DATABASE_URL is required")
	}
	repositoryRoot := repoRoot(t)
	temporaryRoot := t.TempDir()
	ctx, cancel := context.WithTimeout(context.Background(), 210*time.Second)
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
		LeafOptions: leaf, URI: "urn:contractor:control-plane:run-metadata-label-e2e",
	})
	if err != nil {
		t.Fatalf("issue Control Plane certificate: %v", err)
	}
	runtimeIdentity, err := generator.IssueAgent(pkiRoot, "run-metadata-label-runtime", leaf)
	if err != nil {
		t.Fatalf("issue Runtime Agent certificate: %v", err)
	}

	gateway := newFakeGateway(llmGatewayToken)
	t.Cleanup(gateway.close)
	collector := newFakeOTLPCollector(metadataLabelOTLPToken)
	t.Cleanup(collector.close)
	configRoot := stageE2EConfiguration(
		t, filepath.Join(repositoryRoot, "configs", "e2e"),
		filepath.Join(temporaryRoot, "configs"), gateway.URL(),
	)
	publicAddress, privateAddress, runtimeAddress := freeAddress(t), freeAddress(t), freeAddress(t)
	publicBaseURL := "http://" + publicAddress
	privateBaseURL := "https://" + privateAddress
	userID := "metadata-label-e2e-user-" + randomHex(t, 8)
	masterKeyFile := filepath.Join(temporaryRoot, "credential-master-key")
	masterKey := base64.StdEncoding.EncodeToString(bytes.Repeat([]byte{0x6b}, 32))
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
		"CONTRACTOR_LOCAL_AUTH_FILE":         writeE2ELocalAuth(t, temporaryRoot, userID),
		"CONTRACTOR_BROWSER_ORIGINS":         "https://ui.contractor.invalid",
	}
	server := startProcess(
		t, "Run metadata label Go Server", repositoryRoot, serverEnvironment,
		serverBinary, "serve", "--credential-master-key-file", masterKeyFile,
	)
	publicClient := &http.Client{Timeout: 5 * time.Second}
	waitForHTTP(t, ctx, server, publicClient, publicBaseURL+"/readyz", http.StatusOK)

	python := filepath.Join(repositoryRoot, "runtime", ".venv", "bin", "python")
	if info, statErr := os.Stat(python); statErr != nil || info.IsDir() {
		t.Fatalf("Python Runtime environment is missing at %s; run 'cd runtime && uv sync --locked'", python)
	}
	workRoot := filepath.Join(temporaryRoot, "runtime-work")
	runtimeProcess := startRuntimeWithAdapters(
		t, "Run metadata label Python Runtime", repositoryRoot, python,
		privateBaseURL, runtimeAddress, workRoot, caPaths.Certificate,
		runtimeIdentity, []string{"otlp-http@1"},
	)
	controlClient := newMTLSClient(t, caPaths.Certificate, controlPlanePaths)
	waitForHTTP(
		t, ctx, runtimeProcess, controlClient,
		"https://"+runtimeAddress+"/healthz", http.StatusOK,
	)
	waitForProcessLog(t, ctx, runtimeProcess, "runtime agent registered")

	pool, err := pgxpool.New(ctx, isolateURL)
	if err != nil {
		t.Fatalf("open assertion database: %v", err)
	}
	t.Cleanup(pool.Close)
	store := runstore.NewPostgresStore(pool)
	operations := &runtimeOperations{t: t, client: publicClient, baseURL: publicBaseURL}
	uploaded := uploadInput(t, publicClient, publicBaseURL)

	assertInvalidMetadataRunRequestsAreAtomic(t, ctx, pool, operations, uploaded, userID)
	operations.createRuntimeCredential("metadata-label-otel", "otlp-headers@1", map[string]any{
		"headers": map[string]string{"x-contractor-token": metadataLabelOTLPToken},
	})
	debugConfig := operations.publishRuntimeConfig("metadata-label-debug", "1", map[string]any{
		"planner": map[string]any{"telemetry": map[string]any{
			"adapter": "otlp-http@1", "endpoint": collector.URL(),
			"credential": "metadata-label-otel", "captureContent": false,
			"flushTimeoutSeconds": 2,
		}},
		"worker": map[string]any{"telemetry": map[string]any{
			"adapter": "otlp-http@1", "endpoint": collector.URL(),
			"credential": "metadata-label-otel", "captureContent": false,
			"flushTimeoutSeconds": 2,
		}},
	})
	operations.createRuntimeLabel("debug", debugConfig)

	suffix := randomHex(t, 6)
	sharedLabels := map[string]string{
		"purpose":      "eval",
		"eval.name":    "metadata-process-suite-" + suffix,
		"eval.id":      "metadata-process-group=" + suffix,
		"eval.fixture": "metadata-fixture-" + suffix,
		"eval.case":    "metadata-case-" + suffix,
	}
	labelsA := cloneStrings(sharedLabels)
	labelsA["eval.leg"] = "a"
	labelsA["eval.sample"] = "1"
	labelsB := cloneStrings(sharedLabels)
	labelsB["eval.leg"] = "b"
	labelsB["eval.sample"] = "2"

	baseline := createMetadataRun(
		t, operations, "metadata-baseline-"+suffix, uploaded, []string{"debug"}, nil,
	)
	baselineStatus := waitForRunAcross(
		t, ctx, server, []*childProcess{runtimeProcess}, gateway,
		publicClient, publicBaseURL, baseline.RunID,
	)
	assertMetadataRunResult(t, publicClient, publicBaseURL, baselineStatus, nil)
	baselineAllocation := onlyRunAllocation(t, ctx, store, baseline.RunID)
	gateway.ResetScenario()

	runA := createMetadataRun(
		t, operations, "metadata-a-"+suffix, uploaded, []string{"debug"}, labelsA,
	)
	assertChangedMetadataReplayConflicts(t, operations, uploaded, "metadata-a-"+suffix, labelsA)
	statusA := waitForRunAcross(
		t, ctx, server, []*childProcess{runtimeProcess}, gateway,
		publicClient, publicBaseURL, runA.RunID,
	)
	assertMetadataRunResult(t, publicClient, publicBaseURL, statusA, labelsA)
	allocationA := onlyRunAllocation(t, ctx, store, runA.RunID)
	gateway.ResetScenario()

	collector.setReject(true)
	runB := createMetadataRun(
		t, operations, "metadata-b-"+suffix, uploaded, []string{"debug"}, labelsB,
	)
	statusB := waitForRunAcross(
		t, ctx, server, []*childProcess{runtimeProcess}, gateway,
		publicClient, publicBaseURL, runB.RunID,
	)
	assertMetadataRunResult(t, publicClient, publicBaseURL, statusB, labelsB)
	allocationB := onlyRunAllocation(t, ctx, store, runB.RunID)
	failedExportMetrics := onlyRuntimeAdapterMetrics(t, ctx, store, runB.RunID, "otlp-http@1")
	if failedExportMetrics.FailedOperations == 0 || failedExportMetrics.FlushSucceeded == nil ||
		*failedExportMetrics.FlushSucceeded {
		t.Fatalf("metadata Run exporter failure was not retained as supplementary metrics: %+v", failedExportMetrics)
	}
	collector.setReject(false)
	gateway.ResetScenario()

	reuse := createMetadataRun(
		t, operations, "metadata-reuse-"+suffix, uploaded, []string{"debug"}, nil,
	)
	reuseStatus := waitForRunAcross(
		t, ctx, server, []*childProcess{runtimeProcess}, gateway,
		publicClient, publicBaseURL, reuse.RunID,
	)
	assertMetadataRunResult(t, publicClient, publicBaseURL, reuseStatus, nil)
	reuseAllocation := onlyRunAllocation(t, ctx, store, reuse.RunID)
	waitForRuntimeReleased(
		t, ctx, runtimeProcess, controlClient, "https://"+runtimeAddress,
		reuseAllocation.AllocationID, workRoot,
	)

	assertMetadataDoesNotChangePlacement(
		t, baselineAllocation, allocationA, allocationB, reuseAllocation,
	)
	assertExactInputLineage(t, ctx, pool, uploaded, []string{
		baseline.RunID, runA.RunID, runB.RunID, reuse.RunID,
	})
	assertMetadataLabelQueries(
		t, operations, sharedLabels["eval.id"], runA, runB, labelsA, labelsB,
	)
	assertMetadataOTLPRoots(
		t, collector.payloads(), map[string]map[string]string{
			baseline.RunID: {}, runA.RunID: labelsA, runB.RunID: labelsB, reuse.RunID: {},
		},
	)
	assertMetadataAbsentFromGateway(t, gateway.Payloads(), labelsA, labelsB)
	assertMetadataAbsentFromExecutionRetention(
		t, ctx, pool, []string{runA.RunID, runB.RunID}, sharedLabels,
	)
	assertForeignOwnerCannotDiscoverMetadataRuns(
		t, ctx, repositoryRoot, temporaryRoot, serverBinary, masterKeyFile,
		serverEnvironment, publicClient, sharedLabels["eval.id"], runA.RunID,
	)

	for _, process := range []*childProcess{server, runtimeProcess} {
		logs := process.logs.redacted()
		for _, value := range []string{
			sharedLabels["eval.name"], sharedLabels["eval.id"],
			sharedLabels["eval.fixture"], sharedLabels["eval.case"],
		} {
			if strings.Contains(logs, value) {
				t.Fatalf("%s logs exposed model-inert Run metadata value %q", process.name, value)
			}
		}
		if strings.Contains(logs, metadataLabelOTLPToken) {
			t.Fatalf("%s logs exposed OTLP credential", process.name)
		}
	}
	for _, payload := range collector.payloads() {
		if bytes.Contains(payload, []byte(metadataLabelOTLPToken)) {
			t.Fatal("OTLP payload exposed its header credential")
		}
	}
}

func cloneStrings(source map[string]string) map[string]string {
	result := make(map[string]string, len(source))
	for key, value := range source {
		result[key] = value
	}
	return result
}

func metadataRunBody(
	input artifactRef,
	runtimeLabels []string,
	labels map[string]string,
) map[string]any {
	body := map[string]any{
		"workflow": "artifact-copy@1", "parameters": map[string]string{},
		"artifacts":     map[string]artifactRef{"source": input},
		"runtimeLabels": append([]string(nil), runtimeLabels...),
	}
	if labels != nil {
		body["labels"] = cloneStrings(labels)
	}
	return body
}

func createMetadataRun(
	t *testing.T,
	operations *runtimeOperations,
	idempotencyKey string,
	input artifactRef,
	runtimeLabels []string,
	labels map[string]string,
) metadataRunCreateResponse {
	t.Helper()
	body := metadataRunBody(input, runtimeLabels, labels)
	headers := map[string]string{"Idempotency-Key": idempotencyKey}
	created := operations.requestResponse(
		http.MethodPost, "/v1/runs", body, http.StatusAccepted, headers,
	)
	var result metadataRunCreateResponse
	if err := json.Unmarshal(created.body, &result); err != nil || result.RunID == "" || result.State != "running" ||
		result.Labels == nil || !reflect.DeepEqual(result.RuntimeLabels, runtimeLabels) ||
		!maps.Equal(result.Labels, labels) {
		t.Fatalf("create metadata Run = (%+v, %v): %s", result, err, created.body)
	}
	replayed := operations.requestResponse(
		http.MethodPost, "/v1/runs", body, http.StatusAccepted, headers,
	)
	operations.requireReplay(replayed, "metadata Run create")
	var replay metadataRunCreateResponse
	if err := json.Unmarshal(replayed.body, &replay); err != nil || replay.RunID != result.RunID ||
		replay.Labels == nil || !maps.Equal(replay.Labels, result.Labels) ||
		!bytes.Equal(replay.RuntimeConfiguration, result.RuntimeConfiguration) {
		t.Fatalf("metadata Run replay changed immutable identity: first=%+v replay=%+v error=%v", result, replay, err)
	}
	return result
}

func assertChangedMetadataReplayConflicts(
	t *testing.T,
	operations *runtimeOperations,
	input artifactRef,
	idempotencyKey string,
	original map[string]string,
) {
	t.Helper()
	changed := cloneStrings(original)
	changed["eval.leg"] = "changed-after-response-loss"
	response := operations.requestResponse(
		http.MethodPost, "/v1/runs", metadataRunBody(input, []string{"debug"}, changed),
		http.StatusConflict, map[string]string{"Idempotency-Key": idempotencyKey},
	)
	var problem struct {
		Code string `json:"code"`
	}
	if err := json.Unmarshal(response.body, &problem); err != nil || problem.Code != "conflict" {
		t.Fatalf("changed metadata replay response = (%+v, %v): %s", problem, err, response.body)
	}
}

func assertInvalidMetadataRunRequestsAreAtomic(
	t *testing.T,
	ctx context.Context,
	pool *pgxpool.Pool,
	operations *runtimeOperations,
	input artifactRef,
	ownerID string,
) {
	t.Helper()
	var before int
	if err := pool.QueryRow(ctx, `SELECT count(*) FROM workflow_runs WHERE owner_id = $1`, ownerID).Scan(&before); err != nil {
		t.Fatal(err)
	}
	tooMany := make(map[string]string, contracts.MaxRunMetadataLabels+1)
	for index := 0; index <= contracts.MaxRunMetadataLabels; index++ {
		tooMany["extra."+strings.Repeat("x", index/10)+string(rune('a'+index%10))] = "value"
	}
	cases := []any{
		nil,
		[]string{"legacy-runtime-label-array"},
		map[string]string{"contractor.secret": "rejected"},
		tooMany,
	}
	for index, labels := range cases {
		body := metadataRunBody(input, []string{}, map[string]string{})
		body["labels"] = labels
		operations.request(
			http.MethodPost, "/v1/runs", body, http.StatusBadRequest,
			map[string]string{"Idempotency-Key": "metadata-invalid-" + string(rune('a'+index))},
		)
	}
	var after, labelRows, runScopes int
	if err := pool.QueryRow(ctx, `SELECT count(*) FROM workflow_runs WHERE owner_id = $1`, ownerID).Scan(&after); err != nil {
		t.Fatal(err)
	}
	if err := pool.QueryRow(ctx, `SELECT count(*) FROM workflow_run_metadata_labels`).Scan(&labelRows); err != nil {
		t.Fatal(err)
	}
	if err := pool.QueryRow(ctx, `SELECT count(*) FROM artifact_scopes WHERE scope_kind = 'run'`).Scan(&runScopes); err != nil {
		t.Fatal(err)
	}
	if before != after || labelRows != 0 || runScopes != 0 {
		t.Fatalf("invalid metadata requests mutated storage: runs %d->%d labels=%d scopes=%d", before, after, labelRows, runScopes)
	}
}

func assertMetadataRunResult(
	t *testing.T,
	client *http.Client,
	baseURL string,
	status runStatus,
	labels map[string]string,
) {
	t.Helper()
	if status.State != "succeeded" || status.Labels == nil || !maps.Equal(status.Labels, labels) ||
		!reflect.DeepEqual(status.RuntimeLabels, []string{"debug"}) || len(status.Attempts) != 1 ||
		status.Attempts[0].State != "succeeded" || status.Attempts[0].Stage != "copy" {
		t.Fatalf("metadata Run semantic result changed: %+v", status)
	}
	var result contracts.StageContentResult
	if err := json.Unmarshal(status.Attempts[0].Result, &result); err != nil ||
		result.Outcome != contracts.StageSucceeded || result.Summary != "Source artifact copied byte-for-byte" {
		t.Fatalf("metadata Run Worker result = (%+v, %v)", result, err)
	}
	output, ok := status.Outputs["result"]
	if !ok || output.Revision == nil {
		t.Fatalf("metadata Run has no exact output: %+v", status.Outputs)
	}
	payload, mediaType := download(
		t, client, baseURL+"/v1/runs/"+url.PathEscape(status.RunID)+"/outputs/result",
	)
	if string(payload) != e2eInput || mediaType != e2eMediaType {
		t.Fatalf("metadata Run output = (%q, %q)", payload, mediaType)
	}
}

func assertMetadataDoesNotChangePlacement(t *testing.T, allocations ...runstore.StageAllocation) {
	t.Helper()
	if len(allocations) < 2 {
		t.Fatal("placement comparison requires at least two allocations")
	}
	fingerprint := func(allocation runstore.StageAllocation) any {
		return struct {
			LogicalAgent                string
			Namespace                   string
			AgentTemplate               contracts.AgentTemplateRef
			WorkerRuntime               contracts.WorkerRuntimeRef
			RuntimeAgentID              string
			RuntimeAgentInstanceID      string
			RuntimeAgentLabelRevision   uint64
			RuntimeConfigurationVersion string
			RuntimeConfiguration        *runstore.AllocationRuntimeConfiguration
		}{
			allocation.LogicalAgentName, allocation.Namespace,
			allocation.AgentTemplateRef, allocation.WorkerRuntimeRef,
			allocation.RuntimeAgentID, allocation.RuntimeAgentInstanceID,
			allocation.RuntimeAgentLabelRevision,
			allocation.RuntimeConfigurationSchemaVersion,
			allocation.RuntimeConfiguration,
		}
	}
	want := fingerprint(allocations[0])
	for _, allocation := range allocations[1:] {
		if !reflect.DeepEqual(fingerprint(allocation), want) {
			t.Fatalf("metadata labels changed placement/configuration: want=%+v got=%+v", want, fingerprint(allocation))
		}
	}
}

func assertExactInputLineage(
	t *testing.T,
	ctx context.Context,
	pool *pgxpool.Pool,
	input artifactRef,
	runIDs []string,
) {
	t.Helper()
	if input.Revision == nil {
		t.Fatal("input fixture is not exact")
	}
	for _, runID := range runIDs {
		var matches int
		if err := pool.QueryRow(ctx, `
SELECT count(*)
FROM artifact_lineage
WHERE target_scope_kind = 'run' AND target_scope_id = $1
  AND target_namespace = 'inputs' AND target_name = 'source'
  AND source_scope_kind = 'user' AND source_namespace = $2 AND source_name = $3
  AND source_revision = $4 AND lineage_kind = 'input_fork'`,
			runID, input.Namespace, input.Name, *input.Revision,
		).Scan(&matches); err != nil || matches != 1 {
			t.Fatalf("Run %s exact input lineage count = %d, error=%v", runID, matches, err)
		}
	}
}

func assertMetadataLabelQueries(
	t *testing.T,
	operations *runtimeOperations,
	evalID string,
	runA, runB metadataRunCreateResponse,
	labelsA, labelsB map[string]string,
) {
	t.Helper()
	first := listMetadataRuns(operations, []string{"purpose=eval", "eval.id=" + evalID}, "succeeded", 1, "")
	if len(first.Items) != 1 || !first.Page.HasMore || first.Page.NextCursor == "" {
		t.Fatalf("first metadata group page = %+v", first)
	}
	second := listMetadataRuns(
		operations, []string{"purpose=eval", "eval.id=" + evalID},
		"succeeded", 1, first.Page.NextCursor,
	)
	if len(second.Items) != 1 || second.Page.HasMore {
		t.Fatalf("second metadata group page = %+v", second)
	}
	gotIDs := []string{first.Items[0].RunID, second.Items[0].RunID}
	sort.Strings(gotIDs)
	wantIDs := []string{runA.RunID, runB.RunID}
	sort.Strings(wantIDs)
	if !reflect.DeepEqual(gotIDs, wantIDs) {
		t.Fatalf("metadata group IDs = %v, want %v", gotIDs, wantIDs)
	}
	for _, item := range append(first.Items, second.Items...) {
		want := labelsA
		if item.RunID == runB.RunID {
			want = labelsB
		}
		if !maps.Equal(item.Labels, want) {
			t.Fatalf("metadata list projection for %s = %v, want %v", item.RunID, item.Labels, want)
		}
	}
	for leg, runID := range map[string]string{"a": runA.RunID, "b": runB.RunID} {
		page := listMetadataRuns(
			operations, []string{"eval.id=" + evalID, "eval.leg=" + leg}, "succeeded", 10, "",
		)
		if len(page.Items) != 1 || page.Items[0].RunID != runID {
			t.Fatalf("metadata leg %s page = %+v, want %s", leg, page, runID)
		}
	}
	if page := listMetadataRuns(
		operations, []string{"eval.id=" + evalID, "eval.leg=missing"}, "succeeded", 10, "",
	); len(page.Items) != 0 || page.Page.HasMore {
		t.Fatalf("unmatched metadata selector page = %+v", page)
	}
}

func listMetadataRuns(
	operations *runtimeOperations,
	selectors []string,
	state string,
	limit int,
	cursor string,
) metadataRunPage {
	operations.t.Helper()
	if limit < 1 || limit > 100 {
		operations.t.Fatalf("metadata process page limit %d is outside API bounds", limit)
	}
	query := url.Values{"limit": {strconv.Itoa(limit)}}
	if state != "" {
		query.Set("state", state)
	}
	if cursor != "" {
		query.Set("cursor", cursor)
	}
	for _, selector := range selectors {
		query.Add("label", selector)
	}
	data := operations.request(http.MethodGet, "/v1/runs?"+query.Encode(), nil, http.StatusOK, nil)
	var page metadataRunPage
	if err := json.Unmarshal(data, &page); err != nil {
		operations.t.Fatalf("decode metadata Run page: %v: %s", err, data)
	}
	return page
}

func assertMetadataOTLPRoots(
	t *testing.T,
	payloads [][]byte,
	expectedByRun map[string]map[string]string,
) {
	t.Helper()
	rootNames := map[string]bool{
		"contractor.planner.invocation": true,
		"contractor.worker.a2a_task":    true,
	}
	seen := make(map[string]map[string]bool)
	for index, payload := range payloads {
		var request collectortracev1.ExportTraceServiceRequest
		if err := proto.Unmarshal(payload, &request); err != nil {
			t.Fatalf("decode metadata OTLP payload %d: %v", index, err)
		}
		for _, resourceSpans := range request.ResourceSpans {
			if resourceSpans.Resource == nil {
				t.Fatalf("metadata OTLP payload %d has no Resource", index)
			}
			resource := otlpStringAttributes(resourceSpans.Resource.Attributes)
			runID := resource["contractor.run.id"]
			expected, relevant := expectedByRun[runID]
			if !relevant {
				continue
			}
			for key := range resource {
				if strings.HasPrefix(key, "contractor.run.label.") {
					t.Fatalf("Run %s exported metadata label %q as a process Resource attribute", runID, key)
				}
			}
			if labels := otlpStringSliceAttribute(resourceSpans.Resource.Attributes, "contractor.run.labels"); !reflect.DeepEqual(labels, []string{"debug"}) {
				t.Fatalf("Run %s Runtime label Resource attribute = %v, want independent debug", runID, labels)
			}
			for _, scope := range resourceSpans.ScopeSpans {
				for _, span := range scope.Spans {
					attributes := otlpStringAttributes(span.Attributes)
					dynamic := make(map[string]string)
					for key, value := range attributes {
						if strings.HasPrefix(key, "contractor.run.label.") {
							dynamic[strings.TrimPrefix(key, "contractor.run.label.")] = value
						}
					}
					if !rootNames[span.Name] {
						if len(dynamic) != 0 {
							t.Fatalf("Run %s child span %s exposed metadata labels %v", runID, span.Name, dynamic)
						}
						continue
					}
					if !maps.Equal(dynamic, expected) {
						t.Fatalf("Run %s root span %s labels = %v, want %v", runID, span.Name, dynamic, expected)
					}
					if seen[runID] == nil {
						seen[runID] = make(map[string]bool)
					}
					seen[runID][span.Name] = true
				}
			}
		}
	}
	for runID := range expectedByRun {
		for root := range rootNames {
			if !seen[runID][root] {
				t.Errorf("Run %s has no captured %s root span", runID, root)
			}
		}
	}
}

func otlpStringAttributes(attributes []*commonv1.KeyValue) map[string]string {
	result := make(map[string]string)
	for _, attribute := range attributes {
		if attribute != nil && attribute.Value != nil {
			if value, ok := attribute.Value.Value.(*commonv1.AnyValue_StringValue); ok {
				result[attribute.Key] = value.StringValue
			}
		}
	}
	return result
}

func otlpStringSliceAttribute(attributes []*commonv1.KeyValue, key string) []string {
	for _, attribute := range attributes {
		if attribute == nil || attribute.Key != key || attribute.Value == nil {
			continue
		}
		array, ok := attribute.Value.Value.(*commonv1.AnyValue_ArrayValue)
		if !ok || array.ArrayValue == nil {
			return nil
		}
		result := make([]string, 0, len(array.ArrayValue.Values))
		for _, item := range array.ArrayValue.Values {
			if item != nil {
				if value, ok := item.Value.(*commonv1.AnyValue_StringValue); ok {
					result = append(result, value.StringValue)
				}
			}
		}
		return result
	}
	return nil
}

func assertMetadataAbsentFromGateway(
	t *testing.T,
	payloads [][]byte,
	labelSets ...map[string]string,
) {
	t.Helper()
	if len(payloads) == 0 {
		t.Fatal("recording Gateway received no model requests")
	}
	for index, payload := range payloads {
		for _, key := range []string{"eval.id", "eval.name", "eval.fixture", "eval.case", "eval.sample", "eval.leg"} {
			if bytes.Contains(payload, []byte(key)) {
				t.Fatalf("Gateway payload %d exposed Run metadata key %q", index, key)
			}
		}
		for _, labels := range labelSets {
			for key, value := range labels {
				if key == "purpose" || value == "a" || value == "b" || value == "1" || value == "2" {
					continue
				}
				if bytes.Contains(payload, []byte(value)) {
					t.Fatalf("Gateway payload %d exposed Run metadata value for %s", index, key)
				}
			}
		}
	}
}

func assertMetadataAbsentFromExecutionRetention(
	t *testing.T,
	ctx context.Context,
	pool *pgxpool.Pool,
	runIDs []string,
	labels map[string]string,
) {
	t.Helper()
	for key, value := range labels {
		if key == "purpose" {
			continue
		}
		var leaked bool
		err := pool.QueryRow(ctx, `
SELECT
  EXISTS (
    SELECT 1 FROM stage_executions
    WHERE run_id = ANY($2::text[]) AND (
      position($1 in stage_spec_snapshot::text) > 0 OR
      position($1 in stage_context_snapshot::text) > 0 OR
      position($1 in coalesce(candidate_stage_result::text, '')) > 0 OR
      position($1 in coalesce(accepted_stage_result::text, '')) > 0 OR
      position($1 in coalesce(stage_termination::text, '')) > 0
    )
  ) OR EXISTS (
    SELECT 1 FROM planner_sessions AS session
    JOIN stage_executions AS execution USING (stage_execution_id)
    WHERE execution.run_id = ANY($2::text[]) AND position($1 in session.state::text) > 0
  ) OR EXISTS (
    SELECT 1 FROM planner_events AS event
    JOIN planner_sessions AS session USING (session_id)
    JOIN stage_executions AS execution USING (stage_execution_id)
    WHERE execution.run_id = ANY($2::text[]) AND position($1 in event.event::text) > 0
  ) OR EXISTS (
    SELECT 1 FROM workflow_run_events
    WHERE run_id = ANY($2::text[]) AND position($1 in data::text) > 0
  ) OR EXISTS (
    SELECT 1 FROM stage_execution_reports AS report
    JOIN stage_executions AS execution USING (stage_execution_id)
    WHERE execution.run_id = ANY($2::text[]) AND position($1 in report.report::text) > 0
  ) OR EXISTS (
    SELECT 1 FROM allocation_execution_reports AS report
    JOIN stage_executions AS execution USING (stage_execution_id)
    WHERE execution.run_id = ANY($2::text[]) AND position($1 in report.report::text) > 0
  ) OR EXISTS (
    SELECT 1 FROM planner_execution_reports AS report
    JOIN stage_executions AS execution USING (stage_execution_id)
    WHERE execution.run_id = ANY($2::text[]) AND position($1 in report.report::text) > 0
  ) OR EXISTS (
    SELECT 1 FROM stage_metrics AS metrics
    JOIN stage_executions AS execution USING (stage_execution_id)
    WHERE execution.run_id = ANY($2::text[]) AND (
      position($1 in metrics.metrics::text) > 0 OR position($1 in metrics.summary::text) > 0
    )
  ) OR EXISTS (
    SELECT 1 FROM artifact_binding_revisions AS revision
    JOIN artifact_versions AS version USING (version_id)
    JOIN artifact_blobs AS blob ON blob.sha256 = version.blob_sha256
    WHERE revision.scope_kind = 'run' AND revision.scope_id = ANY($2::text[])
      AND position(convert_to($1, 'UTF8') in blob.payload) > 0
  )`, value, runIDs).Scan(&leaked)
		if err != nil {
			t.Fatalf("scan retained execution surfaces for %s: %v", key, err)
		}
		if leaked {
			t.Fatalf("Run metadata value for %s entered retained execution content", key)
		}
	}
}

func assertForeignOwnerCannotDiscoverMetadataRuns(
	t *testing.T,
	ctx context.Context,
	repositoryRoot, temporaryRoot, serverBinary, masterKeyFile string,
	baseEnvironment map[string]string,
	primaryClient *http.Client,
	evalID, runID string,
) {
	t.Helper()
	foreignPublicAddress, foreignPrivateAddress := freeAddress(t), freeAddress(t)
	foreignUserID := "metadata-label-foreign-user-" + randomHex(t, 8)
	foreignToken := "metadata-label-foreign-token-" + randomHex(t, 8)
	environment := cloneStrings(baseEnvironment)
	environment["CONTRACTOR_PUBLIC_LISTEN"] = foreignPublicAddress
	environment["CONTRACTOR_PRIVATE_LISTEN"] = foreignPrivateAddress
	environment["CONTRACTOR_PRIVATE_URL"] = "https://" + foreignPrivateAddress
	environment["CONTRACTOR_PUBLIC_USER_ID"] = foreignUserID
	environment["CONTRACTOR_PUBLIC_BEARER_TOKEN"] = foreignToken
	environment["CONTRACTOR_LOCAL_AUTH_FILE"] = writeE2ELocalAuth(t, temporaryRoot, foreignUserID)
	foreignServer := startProcess(
		t, "foreign-owner Go Server", repositoryRoot, environment,
		serverBinary, "serve", "--credential-master-key-file", masterKeyFile,
	)
	foreignBaseURL := "http://" + foreignPublicAddress
	waitForHTTP(t, ctx, foreignServer, primaryClient, foreignBaseURL+"/readyz", http.StatusOK)

	query := url.Values{"limit": {"10"}, "label": {"eval.id=" + evalID}}
	request, _ := http.NewRequestWithContext(ctx, http.MethodGet, foreignBaseURL+"/v1/runs?"+query.Encode(), nil)
	request.Header.Set("Authorization", "Bearer "+foreignToken)
	response := do(t, primaryClient, request, http.StatusOK)
	var page metadataRunPage
	decodeResponse(t, response, &page)
	response.Body.Close()
	if len(page.Items) != 0 || page.Page.HasMore {
		t.Fatalf("foreign owner inferred metadata group: %+v", page)
	}

	detailRequest, _ := http.NewRequestWithContext(
		ctx, http.MethodGet, foreignBaseURL+"/v1/runs/"+url.PathEscape(runID), nil,
	)
	detailRequest.Header.Set("Authorization", "Bearer "+foreignToken)
	detail := do(t, primaryClient, detailRequest, http.StatusNotFound)
	defer detail.Body.Close()
	var problem map[string]any
	decodeResponse(t, detail, &problem)
	encoded, _ := json.Marshal(problem)
	if bytes.Contains(encoded, []byte(evalID)) || bytes.Contains(encoded, []byte(runID)) {
		t.Fatalf("foreign detail error exposed Run identity or metadata: %s", encoded)
	}
	if exited, processErr := foreignServer.exited(); exited {
		t.Fatalf("foreign owner Server exited: %v\n%s", processErr, foreignServer.logs.redacted(foreignToken))
	}
}
