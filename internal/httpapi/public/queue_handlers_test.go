package public

import (
	"bytes"
	"net/http"
	"net/http/httptest"
	"testing"
)

// A committed Queue control change wakes the Scheduler, so a resumed Queue
// admits work without waiting for the next poll.
func TestOwnerQueueControlUpdateWakesTheScheduler(t *testing.T) {
	fixture := newHandlerFixture(t)
	request := authenticatedRequest(http.MethodPut, "/v1/queue/control", bytes.NewReader([]byte(`{"paused":true}`)))
	request.Header.Set("Content-Type", "application/json")
	request.Header.Set("If-Match", `"0"`)
	response := httptest.NewRecorder()
	fixture.handler.ServeHTTP(response, request)
	if response.Code != http.StatusOK || response.Header().Get("ETag") != `"1"` ||
		!fixture.runs.queueControls["user-1"].Paused || fixture.notifier.calls != 1 {
		t.Fatalf("Queue control update = %d headers=%v body=%s wakes=%d",
			response.Code, response.Header(), response.Body.String(), fixture.notifier.calls)
	}
}
