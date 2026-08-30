package public

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/planner"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/telemetry"
)

const testBearerToken = "test-bearer-token"

type handlerFixture struct {
	handler    http.Handler
	configs    *config.Manager
	repository *fakeArtifactRepository
	artifacts  *artifacts.Service
	runs       *fakeRunStore
	unit       *fakeUnitOfWork
	notifier   *recordingRunNotifier
	metrics    *fakeMetricsReader
	plans      *fakePlannerPlanReader
}

func newHandlerFixture(t *testing.T) handlerFixture {
	return newHandlerFixtureWithConfig(t, "../../config/testdata/valid")
}

func newHandlerFixtureWithConfig(t *testing.T, configRoot string) handlerFixture {
	t.Helper()
	manager, err := config.NewManager(config.ManagerOptions{
		OperatorRoot: configRoot,
		ManagedRoot:  filepath.Join(t.TempDir(), "managed-configs"),
		Descriptors:  config.MVPDescriptors(),
	})
	if err != nil {
		t.Fatalf("load test config: %v", err)
	}
	repository := newFakeArtifactRepository()
	service := artifacts.NewService(repository)
	runs := newFakeRunStore()
	unit := &fakeUnitOfWork{runs: runs, artifacts: service}
	notifier := &recordingRunNotifier{}
	metrics := &fakeMetricsReader{records: map[string]telemetry.StageMetricsRecord{}}
	plans := &fakePlannerPlanReader{plans: map[string]planner.PlannerPlanProjection{}}
	handler, err := NewHandler(Dependencies{
		Config: manager, ConfigurationPublisher: manager,
		Runs: runs, Artifacts: service, Transactions: unit,
		Metrics:      metrics,
		PlannerPlans: plans,
		BearerToken:  contracts.NewSecretString(testBearerToken), UserID: "user-1",
		NewID:        func(prefix string) (string, error) { return prefix + "fixed", nil },
		NewRequestID: func() (string, error) { return "request-fixed", nil },
		RunNotifier:  notifier,
		Now: func() time.Time {
			return time.Date(2026, 8, 29, 12, 0, 0, 0, time.UTC)
		},
	})
	if err != nil {
		t.Fatalf("NewHandler: %v", err)
	}
	return handlerFixture{
		handler: handler, configs: manager, repository: repository, artifacts: service,
		runs: runs, unit: unit, notifier: notifier, metrics: metrics, plans: plans,
	}
}

func authenticatedRequest(method, target string, body *bytes.Reader) *http.Request {
	request := httptest.NewRequest(method, target, body)
	request.Header.Set("Authorization", "Bearer "+testBearerToken)
	if method == http.MethodPost && target == "/v1/runs" {
		request.Header.Set(idempotencyKeyHeader, "test-create-run")
	}
	return request
}

func TestAuthenticationAndRequestID(t *testing.T) {
	fixture := newHandlerFixture(t)
	for _, authorization := range []string{"", "Bearer wrong"} {
		request := httptest.NewRequest(http.MethodGet, "/v1/runs/missing", nil)
		if authorization != "" {
			request.Header.Set("Authorization", authorization)
		}
		response := httptest.NewRecorder()
		fixture.handler.ServeHTTP(response, request)
		if response.Code != http.StatusUnauthorized || response.Header().Get("X-Request-ID") != "request-fixed" {
			t.Fatalf("auth response = status %d, request ID %q", response.Code, response.Header().Get("X-Request-ID"))
		}
		assertErrorCode(t, response, "unauthorized")
	}
}

func TestArtifactCreateUpdateAndExactRead(t *testing.T) {
	fixture := newHandlerFixture(t)
	create := authenticatedRequest(http.MethodPut, "/v1/artifacts/projects/source", bytes.NewReader([]byte("first")))
	create.Header.Set("Content-Type", "text/plain")
	create.Header.Set("If-None-Match", "*")
	created := httptest.NewRecorder()
	fixture.handler.ServeHTTP(created, create)
	if created.Code != http.StatusCreated || created.Header().Get("ETag") == "" {
		t.Fatalf("create = status %d, ETag %q, body %s", created.Code, created.Header().Get("ETag"), created.Body.String())
	}

	update := authenticatedRequest(http.MethodPut, "/v1/artifacts/projects/source", bytes.NewReader([]byte("second")))
	update.Header.Set("Content-Type", "text/plain")
	update.Header.Set("If-Match", created.Header().Get("ETag"))
	updated := httptest.NewRecorder()
	fixture.handler.ServeHTTP(updated, update)
	if updated.Code != http.StatusOK || updated.Header().Get("ETag") == created.Header().Get("ETag") {
		t.Fatalf("update = status %d, ETag %q, body %s", updated.Code, updated.Header().Get("ETag"), updated.Body.String())
	}

	exact := authenticatedRequest(http.MethodGet, "/v1/artifacts/projects/source?revision=revision-1", bytes.NewReader(nil))
	read := httptest.NewRecorder()
	fixture.handler.ServeHTTP(read, exact)
	if read.Code != http.StatusOK || read.Body.String() != "first" || read.Header().Get("ETag") != `"revision-1"` {
		t.Fatalf("exact read = status %d, ETag %q, body %q", read.Code, read.Header().Get("ETag"), read.Body.String())
	}
}

func TestArtifactResponseLossRetryFailsCASWithoutAnotherRevision(t *testing.T) {
	fixture := newHandlerFixture(t)
	create := func() *httptest.ResponseRecorder {
		request := authenticatedRequest(
			http.MethodPut,
			"/v1/artifacts/projects/source",
			bytes.NewReader([]byte("first")),
		)
		request.Header.Set("Content-Type", "text/plain")
		request.Header.Set("If-None-Match", "*")
		response := httptest.NewRecorder()
		fixture.handler.ServeHTTP(response, request)
		return response
	}
	first := create()
	if first.Code != http.StatusCreated {
		t.Fatalf("first create = %d %s", first.Code, first.Body.String())
	}
	retry := create()
	if retry.Code != http.StatusConflict || fixture.repository.next != 1 {
		t.Fatalf("lost-response create retry = status %d revisions %d body %s", retry.Code, fixture.repository.next, retry.Body.String())
	}

	oldETag := first.Header().Get("ETag")
	update := func() *httptest.ResponseRecorder {
		request := authenticatedRequest(
			http.MethodPut,
			"/v1/artifacts/projects/source",
			bytes.NewReader([]byte("second")),
		)
		request.Header.Set("Content-Type", "text/plain")
		request.Header.Set("If-Match", oldETag)
		response := httptest.NewRecorder()
		fixture.handler.ServeHTTP(response, request)
		return response
	}
	accepted := update()
	if accepted.Code != http.StatusOK {
		t.Fatalf("first update = %d %s", accepted.Code, accepted.Body.String())
	}
	lostResponseRetry := update()
	if lostResponseRetry.Code != http.StatusConflict || fixture.repository.next != 2 {
		t.Fatalf("lost-response update retry = status %d revisions %d body %s", lostResponseRetry.Code, fixture.repository.next, lostResponseRetry.Body.String())
	}
	current := authenticatedRequest(
		http.MethodGet,
		"/v1/artifacts/projects/source",
		bytes.NewReader(nil),
	)
	read := httptest.NewRecorder()
	fixture.handler.ServeHTTP(read, current)
	if read.Code != http.StatusOK || read.Body.String() != "second" ||
		read.Header().Get("ETag") != accepted.Header().Get("ETag") {
		t.Fatalf("current artifact after retry = %d etag=%q body=%q", read.Code, read.Header().Get("ETag"), read.Body.String())
	}
}

func TestArtifactUploadIsBoundedBeforeRepository(t *testing.T) {
	fixture := newHandlerFixture(t)
	request := authenticatedRequest(
		http.MethodPut, "/v1/artifacts/projects/large",
		bytes.NewReader(bytes.Repeat([]byte("x"), artifacts.MaxPayloadSize+1)),
	)
	request.Header.Set("Content-Type", "application/octet-stream")
	response := httptest.NewRecorder()
	fixture.handler.ServeHTTP(response, request)
	if response.Code != http.StatusRequestEntityTooLarge || fixture.repository.writes != 0 {
		t.Fatalf("oversized upload = status %d, writes %d", response.Code, fixture.repository.writes)
	}
	assertErrorCode(t, response, "artifact_too_large")
}

func TestCreateRunStrictValidationOccursBeforeTransaction(t *testing.T) {
	fixture := newHandlerFixture(t)
	requests := []string{
		`{"workflow":"artifact-copy@1","parameters":{"objective":42},"artifacts":{"source":{"namespace":"projects","name":"source"}}}`,
		`{"workflow":"artifact-copy@1","parameters":{},"artifacts":{},"extra":true}`,
		`{"workflow":"artifact-copy@1","parameters":{},"artifacts":{}}`,
		`{"workflow":"artifact-copy@1","parameters":{},"artifacts":{"source":{"namespace":"projects","name":"source"}},"executionConfig":null}`,
		`{"workflow":"artifact-copy@1","parameters":{},"artifacts":{"source":{"namespace":"projects","name":"source"}},"executionConfig":{"workers":{"modelPolicy":null}}}`,
		`{"workflow":"artifact-copy@1","parameters":{},"artifacts":{"source":{"namespace":"projects","name":"source"}},"executionConfig":{"workers":{}}}`,
	}
	for _, body := range requests {
		request := authenticatedRequest(http.MethodPost, "/v1/runs", bytes.NewReader([]byte(body)))
		response := httptest.NewRecorder()
		fixture.handler.ServeHTTP(response, request)
		if response.Code != http.StatusBadRequest {
			t.Fatalf("invalid Run request status = %d, body %s", response.Code, response.Body.String())
		}
		assertErrorCode(t, response, "invalid_request")
	}
	if fixture.unit.calls != 0 || len(fixture.runs.runs) != 0 || fixture.notifier.calls != 0 {
		t.Fatalf("invalid requests used transaction %d times or created %d Runs", fixture.unit.calls, len(fixture.runs.runs))
	}
}

func TestCreateRunForksInputAndReturnsRunning(t *testing.T) {
	fixture := newHandlerFixture(t)
	user, _ := fixture.artifacts.User("user-1")
	written, err := user.Write(
		t.Context(), contracts.ArtifactRef{Namespace: "projects", Name: "source"},
		artifacts.Payload{MediaType: "text/plain", Data: []byte("source")}, nil,
	)
	if err != nil {
		t.Fatal(err)
	}
	body := `{"workflow":"artifact-copy@1","parameters":{"objective":"copy"},"artifacts":{"source":{"namespace":"projects","name":"source"}}}`
	request := authenticatedRequest(http.MethodPost, "/v1/runs", bytes.NewReader([]byte(body)))
	response := httptest.NewRecorder()
	fixture.handler.ServeHTTP(response, request)
	if response.Code != http.StatusAccepted {
		t.Fatalf("create Run = status %d, body %s", response.Code, response.Body.String())
	}
	if fixture.notifier.calls != 1 {
		t.Fatalf("Scheduler wake calls = %d", fixture.notifier.calls)
	}
	run := fixture.runs.runs["run_fixed"]
	if run.State != runstore.RunRunning || run.OwnerID != "user-1" || len(run.WorkflowSnapshot) == 0 {
		t.Fatalf("stored Run = %+v", run)
	}
	runArtifacts, _ := fixture.artifacts.Run("run_fixed")
	forked, err := runArtifacts.Read(t.Context(), contracts.ArtifactRef{Namespace: "inputs", Name: "source"})
	if err != nil || string(forked.Payload.Data) != "source" || written.Ref.Revision == nil {
		t.Fatalf("forked input = (%+v, %v)", forked, err)
	}
}

func TestCreateRunResponseLossRetryReturnsExistingRun(t *testing.T) {
	fixture := newHandlerFixture(t)
	user, _ := fixture.artifacts.User("user-1")
	if _, err := user.Write(
		t.Context(), contracts.ArtifactRef{Namespace: "projects", Name: "source"},
		artifacts.Payload{MediaType: "text/plain", Data: []byte("source")}, nil,
	); err != nil {
		t.Fatal(err)
	}
	body := []byte(`{"workflow":"artifact-copy@1","parameters":{"objective":"copy"},"artifacts":{"source":{"namespace":"projects","name":"source"}}}`)

	first := authenticatedRequest(http.MethodPost, "/v1/runs", bytes.NewReader(body))
	firstResponse := httptest.NewRecorder()
	fixture.handler.ServeHTTP(firstResponse, first)
	if firstResponse.Code != http.StatusAccepted {
		t.Fatalf("first create = %d %s", firstResponse.Code, firstResponse.Body.String())
	}

	retry := authenticatedRequest(http.MethodPost, "/v1/runs", bytes.NewReader(body))
	retryResponse := httptest.NewRecorder()
	fixture.handler.ServeHTTP(retryResponse, retry)
	if retryResponse.Code != http.StatusAccepted ||
		retryResponse.Header().Get("Idempotency-Replayed") != "true" ||
		retryResponse.Body.String() != firstResponse.Body.String() {
		t.Fatalf("retry create = %d headers=%v body=%s", retryResponse.Code, retryResponse.Header(), retryResponse.Body.String())
	}
	if len(fixture.runs.runs) != 1 || fixture.notifier.calls != 1 || fixture.repository.writes != 1 {
		t.Fatalf("retry side effects = runs:%d wakes:%d artifact writes:%d",
			len(fixture.runs.runs), fixture.notifier.calls, fixture.repository.writes)
	}

	different := []byte(`{"workflow":"artifact-copy@1","parameters":{"objective":"different"},"artifacts":{"source":{"namespace":"projects","name":"source"}}}`)
	conflict := authenticatedRequest(http.MethodPost, "/v1/runs", bytes.NewReader(different))
	conflictResponse := httptest.NewRecorder()
	fixture.handler.ServeHTTP(conflictResponse, conflict)
	if conflictResponse.Code != http.StatusConflict {
		t.Fatalf("idempotency key reuse = %d %s", conflictResponse.Code, conflictResponse.Body.String())
	}
}

func TestCreateRunPinsExecutionConfigAndDetectsSelectorIdempotencyConflict(t *testing.T) {
	fixture := newHandlerFixtureWithConfig(t, "../../../configs")
	user, _ := fixture.artifacts.User("user-1")
	if _, err := user.Write(
		t.Context(), contracts.ArtifactRef{Namespace: "projects", Name: "source"},
		artifacts.Payload{MediaType: "text/plain", Data: []byte("source")}, nil,
	); err != nil {
		t.Fatal(err)
	}
	requestBody := func(policy string) []byte {
		return []byte(`{"workflow":"artifact-copy@1","parameters":{},"artifacts":{"source":{"namespace":"projects","name":"source"}},"executionConfig":{"workers":{"modelPolicy":"` + policy + `"}}}`)
	}

	first := authenticatedRequest(http.MethodPost, "/v1/runs", bytes.NewReader(requestBody("worker@1")))
	firstResponse := httptest.NewRecorder()
	fixture.handler.ServeHTTP(firstResponse, first)
	if firstResponse.Code != http.StatusAccepted {
		t.Fatalf("first create = %d %s", firstResponse.Code, firstResponse.Body.String())
	}
	stored := fixture.runs.runs["run_fixed"]
	var workflow config.ResolvedWorkflow
	if err := json.Unmarshal(stored.WorkflowSnapshot, &workflow); err != nil {
		t.Fatal(err)
	}
	selection := workflow.Stages["copy"].ExecutionConfig.Agents["builder"]
	if selection.ModelPolicy.Ref.PolicyID != "worker" ||
		selection.Origins.ModelPolicy != "run.executionConfig.workers" {
		t.Fatalf("stored resolved executionConfig = %+v", selection)
	}

	conflict := authenticatedRequest(
		http.MethodPost, "/v1/runs", bytes.NewReader(requestBody("domain_worker@1")),
	)
	conflictResponse := httptest.NewRecorder()
	fixture.handler.ServeHTTP(conflictResponse, conflict)
	if conflictResponse.Code != http.StatusConflict || len(fixture.runs.runs) != 1 || fixture.repository.writes != 1 {
		t.Fatalf(
			"selector conflict = status:%d runs:%d writes:%d body:%s",
			conflictResponse.Code, len(fixture.runs.runs), fixture.repository.writes,
			conflictResponse.Body.String(),
		)
	}
}

func TestCreateRunPinsNamedEscalationVariant(t *testing.T) {
	fixture := newHandlerFixtureWithConfig(t, "../../../configs")
	user, _ := fixture.artifacts.User("user-1")
	if _, err := user.Write(
		t.Context(), contracts.ArtifactRef{Namespace: "projects", Name: "source-archive"},
		artifacts.Payload{MediaType: "application/zip", Data: []byte("source")}, nil,
	); err != nil {
		t.Fatal(err)
	}
	body := []byte(`{"workflow":"openapi-from-source@1","parameters":{},"artifacts":{"source":{"namespace":"projects","name":"source-archive"}}}`)
	request := authenticatedRequest(http.MethodPost, "/v1/runs", bytes.NewReader(body))
	response := httptest.NewRecorder()
	fixture.handler.ServeHTTP(response, request)
	if response.Code != http.StatusAccepted {
		t.Fatalf("create = %d %s", response.Code, response.Body.String())
	}
	stored := fixture.runs.runs["run_fixed"]
	var workflow config.ResolvedWorkflow
	if err := json.Unmarshal(stored.WorkflowSnapshot, &workflow); err != nil {
		t.Fatal(err)
	}
	variant := workflow.Stages["openapi_validate"].On.Failed.Escalate.ExecutionConfig
	if variant.Ref == nil || variant.Ref.ConfigID != "strong-oas-review" ||
		variant.Effective.Agents["validator"].ModelPolicy.Ref.PolicyID != "strong_domain_worker" {
		t.Fatalf("stored escalation variant = %+v", variant)
	}
}

func TestCreateRunRequiresValidIdempotencyKey(t *testing.T) {
	fixture := newHandlerFixture(t)
	body := []byte(`{"workflow":"artifact-copy@1","parameters":{},"artifacts":{"source":{"namespace":"projects","name":"source"}}}`)
	for _, key := range []string{"", "two words", strings.Repeat("x", 129)} {
		request := authenticatedRequest(http.MethodPost, "/v1/runs", bytes.NewReader(body))
		request.Header.Del(idempotencyKeyHeader)
		if key != "" {
			request.Header.Set(idempotencyKeyHeader, key)
		}
		response := httptest.NewRecorder()
		fixture.handler.ServeHTTP(response, request)
		if response.Code != http.StatusBadRequest {
			t.Fatalf("key %q status = %d %s", key, response.Code, response.Body.String())
		}
	}
}

func TestCreateRunDigestNormalizesEquivalentEmptyMappings(t *testing.T) {
	omitted, err := createRunRequestDigest(createRunRequest{Workflow: "empty@1"})
	if err != nil {
		t.Fatal(err)
	}
	explicit, err := createRunRequestDigest(createRunRequest{
		Workflow: "empty@1", Parameters: map[string]string{},
		Artifacts: map[string]contracts.ArtifactRef{},
	})
	if err != nil || explicit != omitted {
		t.Fatalf("semantic request digests = omitted:%q explicit:%q error:%v", omitted, explicit, err)
	}
	var emptyExecutionConfig config.ExecutionConfigPatch
	if err := json.Unmarshal([]byte(`{"stages":{}}`), &emptyExecutionConfig); err != nil {
		t.Fatal(err)
	}
	withEmptyExecutionConfig, err := createRunRequestDigest(createRunRequest{
		Workflow: "empty@1", ExecutionConfig: emptyExecutionConfig,
	})
	if err != nil || withEmptyExecutionConfig != omitted {
		t.Fatalf(
			"empty executionConfig digest = %q, omitted = %q, error = %v",
			withEmptyExecutionConfig, omitted, err,
		)
	}
}

type recordingRunNotifier struct {
	calls         int
	cancellations []string
}

func (n *recordingRunNotifier) Wake() { n.calls++ }
func (n *recordingRunNotifier) Cancel(runID string) {
	n.cancellations = append(n.cancellations, runID)
}

func TestCancelRunIsOwnedStrictDurableAndIdempotent(t *testing.T) {
	fixture := newHandlerFixture(t)
	fixture.runs.runs["run-cancel"] = runstore.WorkflowRun{
		RunID: "run-cancel", OwnerID: "user-1", WorkflowName: "artifact-copy", WorkflowVersion: "1",
		State: runstore.RunRunning,
	}

	request := authenticatedRequest(
		http.MethodPost, "/v1/runs/run-cancel/cancel",
		bytes.NewReader([]byte(`{"reason":"  user changed direction  "}`)),
	)
	response := httptest.NewRecorder()
	fixture.handler.ServeHTTP(response, request)
	if response.Code != http.StatusAccepted {
		t.Fatalf("cancel Run = status %d, body %s", response.Code, response.Body.String())
	}
	var first cancelRunResponse
	if err := json.Unmarshal(response.Body.Bytes(), &first); err != nil {
		t.Fatal(err)
	}
	if first.State != runstore.RunCancelling || first.Cancellation == nil ||
		first.Cancellation.Reason == nil || *first.Cancellation.Reason != "user changed direction" ||
		first.Cancellation.RequestedBy == nil || *first.Cancellation.RequestedBy != "user-1" {
		t.Fatalf("cancel response = %+v", first)
	}

	repeat := authenticatedRequest(
		http.MethodPost, "/v1/runs/run-cancel/cancel",
		bytes.NewReader([]byte(`{"reason":"must not replace winner"}`)),
	)
	repeated := httptest.NewRecorder()
	fixture.handler.ServeHTTP(repeated, repeat)
	stored := fixture.runs.runs["run-cancel"]
	if repeated.Code != http.StatusAccepted || stored.Cancellation == nil ||
		stored.Cancellation.Reason == nil || *stored.Cancellation.Reason != "user changed direction" ||
		len(fixture.notifier.cancellations) != 2 {
		t.Fatalf("repeat = status %d, cancellation %+v, notifications %v", repeated.Code, stored.Cancellation, fixture.notifier.cancellations)
	}

	terminalRun := fixture.runs.runs["run-cancel"]
	terminalRun.State = runstore.RunCancelled
	fixture.runs.runs["run-cancel"] = terminalRun
	terminal := httptest.NewRecorder()
	fixture.handler.ServeHTTP(terminal, authenticatedRequest(
		http.MethodPost, "/v1/runs/run-cancel/cancel", bytes.NewReader([]byte(`{}`)),
	))
	if terminal.Code != http.StatusOK || len(fixture.notifier.cancellations) != 2 {
		t.Fatalf("terminal repeat = status %d, notifications %v", terminal.Code, fixture.notifier.cancellations)
	}
}

func TestCancelRunRejectsInvalidOrForeignRequests(t *testing.T) {
	fixture := newHandlerFixture(t)
	fixture.runs.runs["foreign"] = runstore.WorkflowRun{RunID: "foreign", OwnerID: "user-2", State: runstore.RunRunning}
	fixture.runs.runs["owned"] = runstore.WorkflowRun{RunID: "owned", OwnerID: "user-1", State: runstore.RunSucceeded}

	tests := []struct {
		method string
		target string
		body   string
		status int
	}{
		{http.MethodPost, "/v1/runs/foreign/cancel", `{}`, http.StatusNotFound},
		{http.MethodPost, "/v1/runs/owned/cancel?force=true", `{}`, http.StatusBadRequest},
		{http.MethodPost, "/v1/runs/owned/cancel", `{"reason":" "}`, http.StatusBadRequest},
		{http.MethodPost, "/v1/runs/owned/cancel", `{"unexpected":true}`, http.StatusBadRequest},
		{http.MethodPost, "/v1/runs/owned/cancel", `null`, http.StatusBadRequest},
		{http.MethodGet, "/v1/runs/owned/cancel", `{}`, http.StatusMethodNotAllowed},
	}
	for _, test := range tests {
		response := httptest.NewRecorder()
		fixture.handler.ServeHTTP(response, authenticatedRequest(test.method, test.target, bytes.NewReader([]byte(test.body))))
		if response.Code != test.status {
			t.Errorf("%s %s = %d, want %d: %s", test.method, test.target, response.Code, test.status, response.Body.String())
		}
	}
	if fixture.runs.runs["owned"].Cancellation != nil || len(fixture.notifier.cancellations) != 0 {
		t.Fatal("invalid cancellation changed state or notified Scheduler")
	}
}

func TestRunStatusExposesOnlySafeMetricsSummary(t *testing.T) {
	fixture := newHandlerFixture(t)
	snapshot, err := config.Load("../../config/testdata/valid", config.MVPDescriptors())
	if err != nil {
		t.Fatal(err)
	}
	workflow, err := snapshot.Workflow("artifact-copy@1")
	if err != nil {
		t.Fatal(err)
	}
	stageSnapshot, err := json.Marshal(workflow.Stages[workflow.EntryStage])
	if err != nil {
		t.Fatal(err)
	}
	fixture.runs.runs["run-metrics"] = runstore.WorkflowRun{
		RunID: "run-metrics", OwnerID: "user-1", WorkflowName: "artifact-copy",
		WorkflowVersion: "1", State: runstore.RunRunning,
	}
	fixture.runs.executions["run-metrics"] = []runstore.StageExecution{{
		StageExecutionID: "stage-metrics", RunID: "run-metrics", StageName: "copy",
		Attempt: 1, ExecutionConfigVariant: runstore.StageExecutionConfigBase,
		StageSpecSnapshot: stageSnapshot, State: runstore.StageRunning,
	}}
	fixture.metrics.records["stage-metrics"] = telemetry.StageMetricsRecord{
		StageExecutionID: "stage-metrics",
		Metrics: contracts.StageMetrics{
			Workers: map[string]contracts.ExecutionReport{
				"builder": {
					ReportID: "worker-secret", Complete: true,
					Metrics: contracts.ExecutionMetrics{Tools: map[string]contracts.ToolMetrics{}},
					ToolCalls: []contracts.ToolCallRecord{{
						CallID: "call-secret", Tool: "probe",
						Arguments: map[string]any{"token": "must-not-be-public"},
						Outcome:   contracts.ToolCallSucceeded,
					}},
					Errors: []contracts.ExecutionError{},
				},
			},
			Runtime: map[string]contracts.RuntimeReport{"builder": {Complete: true}},
		},
		Summary: telemetry.Summary{
			ReportsComplete: true, ModelCalls: 2, ToolCalls: 1,
		},
	}

	response := httptest.NewRecorder()
	fixture.handler.ServeHTTP(response, authenticatedRequest(
		http.MethodGet, "/v1/runs/run-metrics", bytes.NewReader(nil),
	))
	if response.Code != http.StatusOK {
		t.Fatalf("status response = %d: %s", response.Code, response.Body.String())
	}
	if strings.Contains(response.Body.String(), "must-not-be-public") ||
		strings.Contains(response.Body.String(), "call-secret") {
		t.Fatalf("public status leaked detailed telemetry: %s", response.Body.String())
	}
	var result runStatusResponse
	if err := json.Unmarshal(response.Body.Bytes(), &result); err != nil {
		t.Fatal(err)
	}
	if len(result.Attempts) != 1 || result.Attempts[0].Metrics == nil ||
		result.Attempts[0].Metrics.ModelCalls != 2 || result.Attempts[0].Metrics.ToolCalls != 1 {
		t.Fatalf("public metrics summary = %+v", result.Attempts)
	}
	if result.Attempts[0].ExecutionConfig.Variant != runstore.StageExecutionConfigBase ||
		result.Attempts[0].ExecutionConfig.Agents["builder"].ModelPolicy.PolicyID != "worker" {
		t.Fatalf("public effective executionConfig refs = %+v", result.Attempts[0].ExecutionConfig)
	}
}

func TestRunStatusExposesEscalationLineageAndSafeEffectiveRefs(t *testing.T) {
	fixture := newHandlerFixture(t)
	snapshot, err := config.Load("../../../configs", config.MVPDescriptors())
	if err != nil {
		t.Fatal(err)
	}
	workflow, err := snapshot.Workflow("openapi-from-source@1")
	if err != nil {
		t.Fatal(err)
	}
	stage := workflow.Stages["openapi_validate"]
	action := stage.On.Failed
	if action.Escalate == nil || action.Escalate.ExecutionConfig.Ref == nil {
		t.Fatal("repository Workflow has no named failed escalation")
	}
	baseSnapshot, err := json.Marshal(stage)
	if err != nil {
		t.Fatal(err)
	}
	stage.ExecutionConfig = action.Escalate.ExecutionConfig.Effective
	escalatedSnapshot, err := json.Marshal(stage)
	if err != nil {
		t.Fatal(err)
	}
	previous, ordinal := "stage-base", 1
	fixture.runs.runs["run-escalation"] = runstore.WorkflowRun{
		RunID: "run-escalation", OwnerID: "user-1", WorkflowName: "openapi-from-source",
		WorkflowVersion: "1", State: runstore.RunFailed,
	}
	fixture.runs.executions["run-escalation"] = []runstore.StageExecution{
		{
			StageExecutionID: "stage-base", RunID: "run-escalation", StageName: "openapi_validate",
			Attempt: 1, ExecutionConfigVariant: runstore.StageExecutionConfigBase,
			StageSpecSnapshot: baseSnapshot, State: runstore.StageFailed,
		},
		{
			StageExecutionID: "stage-escalated", RunID: "run-escalation", StageName: "openapi_validate",
			Attempt: 2, PreviousExecutionID: &previous,
			ExecutionConfigVariant: runstore.StageExecutionConfigFailedEscalation,
			EscalationOrdinal:      &ordinal,
			StageSpecSnapshot:      escalatedSnapshot, State: runstore.StageFailed,
		},
	}
	fixture.runs.decisions["run-escalation"] = []runstore.StageTransitionDecision{
		{
			SourceExecutionID: "stage-base", RunID: "run-escalation",
			Action: runstore.StageTransitionEscalate, TargetStageName: stringTestPointer("openapi_validate"),
			TargetExecutionID: stringTestPointer("stage-escalated"), EscalationOrdinal: &ordinal,
		},
		{
			SourceExecutionID: "stage-escalated", RunID: "run-escalation",
			Action: runstore.StageTransitionFail, EscalationOrdinal: &ordinal,
			EscalationExhausted: true,
		},
	}

	response := httptest.NewRecorder()
	fixture.handler.ServeHTTP(response, authenticatedRequest(
		http.MethodGet, "/v1/runs/run-escalation", bytes.NewReader(nil),
	))
	if response.Code != http.StatusOK {
		t.Fatalf("status response = %d: %s", response.Code, response.Body.String())
	}
	var result runStatusResponse
	if err := json.Unmarshal(response.Body.Bytes(), &result); err != nil {
		t.Fatal(err)
	}
	if len(result.Attempts) != 2 || len(result.Transitions) != 2 ||
		result.Attempts[1].PreviousExecutionID == nil || *result.Attempts[1].PreviousExecutionID != "stage-base" ||
		result.Attempts[1].ExecutionConfig.Ref == nil ||
		result.Attempts[1].ExecutionConfig.Ref.ConfigID != "strong-oas-review" ||
		result.Attempts[1].ExecutionConfig.Agents["validator"].ModelPolicy.PolicyID != "strong_domain_worker" ||
		result.Transitions[0].Action != runstore.StageTransitionEscalate ||
		!result.Transitions[1].EscalationExhausted {
		t.Fatalf("public escalation read model = %+v", result)
	}
	if strings.Contains(response.Body.String(), "http://litellm") ||
		strings.Contains(response.Body.String(), "credential") {
		t.Fatalf("public escalation read model leaked connection details: %s", response.Body.String())
	}
}

func stringTestPointer(value string) *string { return &value }

func TestRunOutputDownloadRequiresOwnerAndReturnsExactMetadata(t *testing.T) {
	fixture := newHandlerFixture(t)
	fixture.runs.runs["run-owned"] = runstore.WorkflowRun{
		RunID: "run-owned", OwnerID: "user-1", WorkflowName: "artifact-copy", WorkflowVersion: "1", State: runstore.RunSucceeded,
	}
	runArtifacts, _ := fixture.artifacts.Run("run-owned")
	source, err := runArtifacts.Write(
		t.Context(), contracts.ArtifactRef{Namespace: "builder", Name: "result"},
		artifacts.Payload{MediaType: "text/plain", Data: []byte("finished")}, nil,
	)
	if err != nil {
		t.Fatal(err)
	}
	bound, err := fixture.artifacts.BindOutputExact(t.Context(), "run-owned", "result", source.Ref, nil)
	if err != nil {
		t.Fatal(err)
	}

	request := authenticatedRequest(http.MethodGet, "/v1/runs/run-owned/outputs/result", bytes.NewReader(nil))
	response := httptest.NewRecorder()
	fixture.handler.ServeHTTP(response, request)
	if response.Code != http.StatusOK || response.Body.String() != "finished" || response.Header().Get("ETag") != quotedETag(bound.TargetRef.Revision) {
		t.Fatalf("output = status %d, ETag %q, body %q", response.Code, response.Header().Get("ETag"), response.Body.String())
	}

	fixture.runs.runs["run-owned"] = runstore.WorkflowRun{RunID: "run-owned", OwnerID: "another-user"}
	readsBefore := fixture.repository.reads
	denied := httptest.NewRecorder()
	fixture.handler.ServeHTTP(denied, request)
	if denied.Code != http.StatusNotFound || fixture.repository.reads != readsBefore {
		t.Fatalf("foreign output = status %d, reads before/after %d/%d", denied.Code, readsBefore, fixture.repository.reads)
	}
}

func assertErrorCode(t *testing.T, response *httptest.ResponseRecorder, expected string) {
	t.Helper()
	var body errorResponse
	if err := json.Unmarshal(response.Body.Bytes(), &body); err != nil {
		t.Fatalf("decode error response: %v; body %q", err, response.Body.String())
	}
	if body.Code != expected || strings.TrimSpace(body.Message) == "" ||
		body.RequestID != "request-fixed" || response.Header().Get("X-Request-ID") != body.RequestID {
		t.Fatalf("error response = %+v, want code %q", body, expected)
	}
}

type fakeMetricsReader struct {
	records map[string]telemetry.StageMetricsRecord
}

func (f *fakeMetricsReader) GetStageMetrics(
	_ context.Context, stageExecutionID string,
) (telemetry.StageMetricsRecord, error) {
	record, ok := f.records[stageExecutionID]
	if !ok {
		return telemetry.StageMetricsRecord{}, telemetry.ErrNotFound
	}
	return record, nil
}
