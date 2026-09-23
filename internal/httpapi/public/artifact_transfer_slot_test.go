package public

import (
	"bytes"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/artifacts"
)

// Artifact byte reads share a few global transfer slots. Requests that are
// rejected for their method or ownership must be answered without one, so
// they neither wait for nor exhaust transfer capacity.
func TestArtifactReadsRejectBeforeTakingATransferSlot(t *testing.T) {
	fixture := newHandlerFixture(t)
	runtime := artifacts.NewBlobRuntime(nil, nil)
	ctx := artifacts.WithBlobRuntime(t.Context(), runtime)
	for {
		_, release, err := artifacts.AcquireTransfer(ctx)
		if err != nil {
			break
		}
		defer release()
	}

	for _, test := range []struct {
		method, target string
		status         int
	}{
		{http.MethodHead, "/v1/artifacts/projects/source", http.StatusMethodNotAllowed},
		{http.MethodHead, "/v1/runs/foreign/artifacts/inputs/source", http.StatusMethodNotAllowed},
		{http.MethodGet, "/v1/runs/foreign/artifacts/inputs/source", http.StatusNotFound},
		{http.MethodHead, "/v1/projects/foreign/artifacts/sources/service", http.StatusMethodNotAllowed},
		{http.MethodGet, "/v1/projects/foreign/artifacts/sources/service", http.StatusNotFound},
		{http.MethodHead, "/v1/runs/foreign/outputs/result", http.StatusMethodNotAllowed},
		{http.MethodGet, "/v1/runs/foreign/outputs/result", http.StatusNotFound},
		// An accepted read still needs a slot.
		{http.MethodGet, "/v1/artifacts/projects/source", http.StatusServiceUnavailable},
	} {
		request := authenticatedRequest(test.method, test.target, bytes.NewReader(nil)).WithContext(ctx)
		response := httptest.NewRecorder()
		fixture.handler.ServeHTTP(response, request)
		if response.Code != test.status {
			t.Errorf("%s %s = %d, want %d: %s", test.method, test.target, response.Code, test.status, response.Body.String())
		}
		if test.status != http.StatusServiceUnavailable &&
			strings.Contains(response.Body.String(), "artifact_transfer_capacity") {
			t.Errorf("%s %s waited for transfer capacity", test.method, test.target)
		}
	}
}
