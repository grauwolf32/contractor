package public

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/runstore"
)

func TestReviewAmbiguousCredentialsCannotCancelRun(t *testing.T) {
	for _, test := range []struct {
		name   string
		mutate func(*http.Request)
		status int
	}{
		{"duplicate authorization", func(r *http.Request) {
			r.Header.Add("Authorization", "Bearer "+testBearerToken)
			r.Header.Add("Authorization", "Bearer "+testBearerToken)
		}, http.StatusUnauthorized},
		{"wrong bearer with valid cookie", func(r *http.Request) {
			r.Header.Set("Authorization", "Bearer invalid")
		}, http.StatusUnauthorized},
		{"empty authorization with valid cookie", func(r *http.Request) {
			r.Header.Set("Authorization", "")
		}, http.StatusUnauthorized},
		{"wrong scheme with valid cookie", func(r *http.Request) {
			r.Header.Set("Authorization", "Basic invalid")
		}, http.StatusUnauthorized},
		{"duplicate session cookie", func(r *http.Request) {
			r.Header.Add("Cookie", r.Header.Get("Cookie"))
		}, http.StatusUnauthorized},
		{"duplicate origin", func(r *http.Request) {
			r.Header.Add("Origin", r.Header.Get("Origin"))
		}, http.StatusForbidden},
		{"duplicate CSRF", func(r *http.Request) {
			r.Header.Add("X-CSRF-Token", r.Header.Get("X-CSRF-Token"))
		}, http.StatusForbidden},
		{"comma joined origins", func(r *http.Request) {
			r.Header.Set("Origin", testBrowserOrigin+", https://attacker.invalid")
		}, http.StatusForbidden},
		{"foreign CSRF", func(r *http.Request) {
			r.Header.Set("X-CSRF-Token", strings.Repeat("a", 43))
		}, http.StatusForbidden},
		{"valid session control", func(*http.Request) {}, http.StatusAccepted},
	} {
		t.Run(test.name, func(t *testing.T) {
			fixture := newHandlerFixture(t)
			login := loginBrowser(t, fixture.handler, "admin", testAuthPassword, testBrowserOrigin)
			if login.response.Code != http.StatusOK {
				t.Fatalf("login status=%d", login.response.Code)
			}
			const runID = "run-auth-review"
			fixture.runs.runs[runID] = runstore.WorkflowRun{
				RunID: runID, OwnerID: "user-1", WorkflowName: "artifact-copy",
				WorkflowVersion: "1", WorkflowSchemaVersion: contracts.APIVersion, State: runstore.RunRunning,
			}
			request := browserRequest(http.MethodPost, "/v1/runs/"+runID+"/cancel", []byte(`{}`), login.cookie, testBrowserOrigin, login.session.CSRFToken)
			request.Header.Set("Content-Type", "application/json")
			test.mutate(request)
			response := httptest.NewRecorder()
			fixture.handler.ServeHTTP(response, request)
			if response.Code != test.status {
				t.Fatalf("status=%d, want=%d", response.Code, test.status)
			}
			cancelled := fixture.runs.runs[runID].Cancellation != nil
			if cancelled != (test.status == http.StatusAccepted) {
				t.Fatalf("domain cancellation=%v for status=%d", cancelled, response.Code)
			}
			for _, secret := range []string{testBearerToken, testAuthPassword, login.cookie.Value, login.session.CSRFToken} {
				if strings.Contains(response.Body.String(), secret) {
					t.Fatal("authentication material appeared in the response")
				}
			}
		})
	}
}
