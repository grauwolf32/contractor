package privateartifacts

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
)

// interruptedBody yields its data and then fails like a connection that
// closes, or a socket read that misses its deadline, before the declared
// length has arrived.
type interruptedBody struct{ data []byte }

func (b *interruptedBody) Read(target []byte) (int, error) {
	if len(b.data) == 0 {
		return 0, io.ErrUnexpectedEOF
	}
	n := copy(target, b.data)
	b.data = b.data[n:]
	return n, nil
}

// TestPrivateArtifactPayloadInterruptionIsRetryable pins the private outcome
// of an Artifact payload that stops before its declared length (BLB-R28,
// FAIL-BLB-1): 500 internal_error with retryable true, which the Runtime's
// Artifact client reports to its callers as retryable, and no write; the
// complete payload is then accepted with the same precondition.
func TestPrivateArtifactPayloadInterruptionIsRetryable(t *testing.T) {
	repository := newMemoryRepository()
	handler := newTestHandler(t, &fakeRegistry{grant: testGrant("run-a")}, repository)
	request := trustedRequest(http.MethodPut, "/private/v1/allocations/allocation-1/artifacts/worker/interrupted",
		&interruptedBody{data: []byte("part")})
	request.ContentLength = 64
	request.Header.Set("Content-Type", "text/plain")
	request.Header.Set("If-None-Match", "*")
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, request)
	var failure errorResponse
	if err := json.Unmarshal(response.Body.Bytes(), &failure); err != nil || response.Code != http.StatusInternalServerError ||
		failure.Code != "internal_error" || !failure.Retryable || repository.writeCalls != 0 {
		t.Fatalf("interrupted payload = %d %s with %d writes, want retryable 500 internal_error and no write",
			response.Code, response.Body.String(), repository.writeCalls)
	}
	if response := putArtifact(t, handler, "worker", "interrupted", "*", []byte("complete")); response.Code != http.StatusCreated {
		t.Fatalf("complete payload = %d %s, want 201", response.Code, response.Body.String())
	}
}
