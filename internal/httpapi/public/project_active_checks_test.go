package public

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts"
)

func TestProjectActiveChecksStrictRequestsAndCAS(t *testing.T) {
	fixture := newHandlerFixture(t)
	call := func(method, body, revision string) *httptest.ResponseRecorder {
		path := "/v1/projects"
		if method == http.MethodPatch {
			path += "/project_fixed"
		}
		request := authenticatedRequest(method, path, bytes.NewReader([]byte(body)))
		request.Header.Set("Content-Type", "application/json")
		request.Header.Set("Idempotency-Key", "project-policy-create")
		if revision != "" {
			request.Header.Set("If-Match", revision)
		}
		response := httptest.NewRecorder()
		fixture.handler.ServeHTTP(response, request)
		return response
	}
	for _, value := range []string{`null`, `""`, `"enabled"`, `true`, `"Automatic"`} {
		if r := call(http.MethodPost, `{"kind":"project","name":"Policy","activeChecks":`+value+`}`, ""); r.Code != 400 {
			t.Fatalf("create %s: %d %s", value, r.Code, r.Body.String())
		}
	}
	created := call(http.MethodPost, `{"kind":"project","name":"Policy"}`, "")
	var project projectResponse
	if err := json.Unmarshal(created.Body.Bytes(), &project); err != nil || created.Code != 201 || project.ActiveChecks != contracts.ActiveChecksAutomatic {
		t.Fatalf("default policy: %d %s %v", created.Code, created.Body.String(), err)
	}
	for _, value := range []string{`null`, `""`, `"enabled"`, `true`} {
		if r := call(http.MethodPatch, `{"activeChecks":`+value+`}`, `"1"`); r.Code != 400 {
			t.Fatalf("update %s: %d %s", value, r.Code, r.Body.String())
		}
	}
	updated := call(http.MethodPatch, `{"activeChecks":"approval-required"}`, `"1"`)
	if err := json.Unmarshal(updated.Body.Bytes(), &project); err != nil || updated.Code != 200 || project.ActiveChecks != contracts.ActiveChecksApprovalRequired {
		t.Fatalf("update policy: %d %s %v", updated.Code, updated.Body.String(), err)
	}
	if r := call(http.MethodPatch, `{"activeChecks":"automatic"}`, `"1"`); r.Code != 412 {
		t.Fatalf("stale policy: %d %s", r.Code, r.Body.String())
	}
	preserved := call(http.MethodPatch, `{"description":"Keep policy"}`, `"2"`)
	if err := json.Unmarshal(preserved.Body.Bytes(), &project); err != nil || preserved.Code != 200 || project.ActiveChecks != contracts.ActiveChecksApprovalRequired {
		t.Fatalf("omitted policy lost: %s %v", preserved.Body.String(), err)
	}
}
