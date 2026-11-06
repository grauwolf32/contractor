package public

import (
	"bytes"
	"context"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/grauwolf32/contractor/internal/auth"
	"github.com/grauwolf32/contractor/internal/runstore"
)

type recordingGatewayRetry struct{ calls int }

func (g *recordingGatewayRetry) Retry(context.Context, string, string) error { g.calls++; return nil }

func TestGatewayRetryRejectsInactiveRunBeforeChangingMemory(t *testing.T) {
	for _, state := range []runstore.WorkflowRunState{runstore.RunInitializing, runstore.RunPending, runstore.RunRunning, runstore.RunWaiting, runstore.RunCancelling, runstore.RunCancelled, runstore.RunFailed, runstore.RunSucceeded} {
		t.Run(string(state), func(t *testing.T) {
			f := newHandlerFixture(t)
			f.runs.runs["run"] = runstore.WorkflowRun{RunID: "run", OwnerID: "user-1", State: state}
			gateway := &recordingGatewayRetry{}
			h := runHandlers{RunDependencies: RunDependencies{Runs: f.runs, GatewayRecovery: gateway, RunNotifier: f.notifier}}
			request := authenticatedRequest(http.MethodPost, "/v1/runs/run/retry-gateway", bytes.NewReader([]byte(`{}`)))
			request = request.WithContext(auth.WithPrincipal(request.Context(), auth.Principal{UserID: "user-1"}))
			request.SetPathValue("runId", "run")
			request.Header.Set("Content-Type", "application/json")
			response := httptest.NewRecorder()
			serve(h.retryRunGateway)(response, request)
			allowed := state == runstore.RunPending || state == runstore.RunRunning || state == runstore.RunWaiting
			want, calls := http.StatusConflict, 0
			if allowed {
				want, calls = http.StatusAccepted, 1
			}
			if response.Code != want || gateway.calls != calls {
				t.Fatalf("status=%d calls=%d want=%d/%d: %s", response.Code, gateway.calls, want, calls, response.Body.String())
			}
		})
	}
}
