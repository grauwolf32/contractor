package public

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/grauwolf32/contractor/internal/runstore"
)

func TestResumeRunOwnedStrictAndIdempotent(t *testing.T) {
	fixture := newHandlerFixture(t)
	fixture.runs.runs["run-resume"] = runstore.WorkflowRun{RunID: "run-resume", OwnerID: "user-1", State: runstore.RunFailed}
	fixture.runs.executions["run-resume"] = []runstore.StageExecution{{RunID: "run-resume", StageExecutionID: "failed-stage", StageName: "build", State: runstore.StageFailed, Attempt: 1}}
	post := func(body string) *httptest.ResponseRecorder {
		request := authenticatedRequest(http.MethodPost, "/v1/runs/run-resume/resume", bytes.NewReader([]byte(body)))
		response := httptest.NewRecorder()
		fixture.handler.ServeHTTP(response, request)
		return response
	}
	for _, body := range []string{`null`, `{}`, `{"stageExecutionId":"failed-stage","extra":true}`} {
		if response := post(body); response.Code != 400 {
			t.Fatalf("invalid request %s: %d", body, response.Code)
		}
	}
	if response := post(`{"stageExecutionId":"stale-stage"}`); response.Code != 409 {
		t.Fatalf("stale: %d", response.Code)
	}
	response := post(`{"stageExecutionId":"failed-stage"}`)
	if response.Code != 202 {
		t.Fatalf("resume: %d %s", response.Code, response.Body.String())
	}
	var result runstore.ResumeRunResult
	if err := json.Unmarshal(response.Body.Bytes(), &result); err != nil {
		t.Fatal(err)
	}
	replay := post(`{"stageExecutionId":"failed-stage"}`)
	if replay.Code != 202 || replay.Body.String() != response.Body.String() || len(fixture.runs.executions["run-resume"]) != 2 {
		t.Fatal("idempotency failed")
	}
	run := fixture.runs.runs["run-resume"]
	run.OwnerID = "other-owner"
	fixture.runs.runs["run-resume"] = run
	if response := post(`{"stageExecutionId":"failed-stage"}`); response.Code != 404 {
		t.Fatalf("owner: %d", response.Code)
	}
}
