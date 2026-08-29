package public

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/telemetry"
)

const testBearerToken = "test-bearer-token"

type handlerFixture struct {
	handler    http.Handler
	repository *fakeArtifactRepository
	artifacts  *artifacts.Service
	runs       *fakeRunStore
	unit       *fakeUnitOfWork
	notifier   *recordingRunNotifier
	metrics    *fakeMetricsReader
}

func newHandlerFixture(t *testing.T) handlerFixture {
	t.Helper()
	snapshot, err := config.Load("../../config/testdata/valid", config.MVPDescriptors())
	if err != nil {
		t.Fatalf("load test config: %v", err)
	}
	repository := newFakeArtifactRepository()
	service := artifacts.NewService(repository)
	runs := newFakeRunStore()
	unit := &fakeUnitOfWork{runs: runs, artifacts: service}
	notifier := &recordingRunNotifier{}
	metrics := &fakeMetricsReader{records: map[string]telemetry.StageMetricsRecord{}}
	handler, err := NewHandler(Dependencies{
		Config: snapshot, Runs: runs, Artifacts: service, Transactions: unit,
		Metrics:     metrics,
		BearerToken: contracts.NewSecretString(testBearerToken), UserID: "user-1",
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
		handler: handler, repository: repository, artifacts: service,
		runs: runs, unit: unit, notifier: notifier, metrics: metrics,
	}
}

func authenticatedRequest(method, target string, body *bytes.Reader) *http.Request {
	request := httptest.NewRequest(method, target, body)
	request.Header.Set("Authorization", "Bearer "+testBearerToken)
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
	fixture.runs.runs["run-metrics"] = runstore.WorkflowRun{
		RunID: "run-metrics", OwnerID: "user-1", WorkflowName: "artifact-copy",
		WorkflowVersion: "1", State: runstore.RunRunning,
	}
	fixture.runs.executions["run-metrics"] = []runstore.StageExecution{{
		StageExecutionID: "stage-metrics", RunID: "run-metrics", StageName: "copy",
		Attempt: 1, State: runstore.StageRunning,
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
}

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
	if body.Code != expected || strings.TrimSpace(body.Message) == "" {
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
