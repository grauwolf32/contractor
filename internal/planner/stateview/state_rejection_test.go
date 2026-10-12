package stateview

import (
	"errors"
	"net/http"
	"testing"

	"github.com/grauwolf32/contractor/internal/planner"
)

// TestServiceKeepsRejectedStateReadNonRetryable pins the Planner side of
// AWC-R115: a State snapshot that the Server rejects, for example one whose
// completion diagnostics carry an unknown kind or phase, is a non-retryable
// unavailable projection rather than a retry loop or a partial State.
func TestServiceKeepsRejectedStateReadNonRetryable(t *testing.T) {
	reader := &recordingReader{err: &planner.WorkerStateReadError{
		StatusCode: http.StatusOK, Code: "worker_state_response_invalid", Retryable: false,
	}}
	service := testService(t, reader, Binding{
		LogicalName: "builder", Handle: stateHandle("allocation-builder"), WorkspaceEligible: false,
	})
	service.RecordCompletion("builder", "1.1", stateCompletion("worker-completed", "1.1", 7))
	_, err := service.GetWorkerToolUsage(t.Context(), "builder")
	var typed *Error
	if !errors.As(err, &typed) || typed.Code != CodeUnavailable || typed.Retryable {
		t.Fatalf("rejected State projection error = %#v", err)
	}
}
