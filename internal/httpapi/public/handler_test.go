package public

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/runstore"
)

const testBearerToken = "test-bearer-token"

type handlerFixture struct {
	handler    http.Handler
	repository *fakeArtifactRepository
	artifacts  *artifacts.Service
	runs       *fakeRunStore
	unit       *fakeUnitOfWork
	notifier   *recordingRunNotifier
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
	handler, err := NewHandler(Dependencies{
		Config: snapshot, Runs: runs, Artifacts: service, Transactions: unit,
		BearerToken: contracts.NewSecretString(testBearerToken), UserID: "user-1",
		NewID:        func(prefix string) (string, error) { return prefix + "fixed", nil },
		NewRequestID: func() (string, error) { return "request-fixed", nil },
		RunNotifier:  notifier,
	})
	if err != nil {
		t.Fatalf("NewHandler: %v", err)
	}
	return handlerFixture{
		handler: handler, repository: repository, artifacts: service,
		runs: runs, unit: unit, notifier: notifier,
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

type recordingRunNotifier struct{ calls int }

func (n *recordingRunNotifier) Wake() { n.calls++ }

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
