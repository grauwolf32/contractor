package public

import (
	"bytes"
	"net/http"
	"net/http/httptest"
	"testing"
)

// RunNotifier is optional; a committed Queue control change must still be
// answered when no Scheduler wake-up is wired.
func TestOwnerQueueControlUpdateWithoutRunNotifier(t *testing.T) {
	fixture := newHandlerFixtureWithAuth(
		t, "../../config/testdata/valid", newTestAuthentication(t), mustTestOrigins(t), false, nil,
		func(dependencies *Dependencies) { dependencies.RunNotifier = nil },
	)
	request := authenticatedRequest(http.MethodPut, "/v1/queue/control", bytes.NewReader([]byte(`{"paused":true}`)))
	request.Header.Set("Content-Type", "application/json")
	request.Header.Set("If-Match", `"0"`)
	response := httptest.NewRecorder()
	fixture.handler.ServeHTTP(response, request)
	if response.Code != http.StatusOK || response.Header().Get("ETag") != `"1"` ||
		!fixture.runs.queueControls["user-1"].Paused {
		t.Fatalf("Queue control update = %d headers=%v body=%s", response.Code, response.Header(), response.Body.String())
	}
}
