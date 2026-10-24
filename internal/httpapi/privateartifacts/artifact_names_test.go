package privateartifacts

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"
)

// TestPrivateArtifactNamesRejectBeforeStore rejects a malformed namespace or
// name as request syntax: 400 invalid_request before the allocation is
// looked up and before a byte of the payload is read or stored, so the answer
// is the same for an unknown allocation (ART-R45). The router answers first:
// HEAD on the same path is 405 whatever the names (ART-R120).
func TestPrivateArtifactNamesRejectBeforeStore(t *testing.T) {
	repository := newMemoryRepository()
	handler := newTestHandler(t, &fakeRegistry{grant: testGrant("run-a")}, repository)
	malformed := []string{"review notes", "отчет", "_report", strings.Repeat("a", 129)}
	for _, allocationID := range []string{"allocation-1", "allocation-missing"} {
		for _, name := range malformed {
			for _, path := range []string{"worker/" + url.PathEscape(name), url.PathEscape(name) + "/report"} {
				body := &countingReader{data: []byte("report")}
				request := trustedRequest(http.MethodPut,
					fmt.Sprintf("/private/v1/allocations/%s/artifacts/%s", allocationID, path), body)
				request.Header.Set("Content-Type", "text/plain")
				request.Header.Set("If-None-Match", "*")
				response := httptest.NewRecorder()
				handler.ServeHTTP(response, request)
				var failure errorResponse
				_ = json.Unmarshal(response.Body.Bytes(), &failure)
				if response.Code != http.StatusBadRequest || failure.Code != "invalid_request" {
					t.Errorf("%s %q: %d %s", allocationID, path, response.Code, response.Body.String())
				}
				if body.reads != 0 || repository.writeCalls != 0 {
					t.Errorf("%s %q read the body %d times and wrote %d times", allocationID, path, body.reads, repository.writeCalls)
				}
				target := fmt.Sprintf("/private/v1/allocations/%s/artifacts/%s", allocationID, path)
				head := httptest.NewRecorder()
				handler.ServeHTTP(head, trustedRequest(http.MethodHead, target, nil))
				checkPrivateMethodNotAllowed(t, head, target)
			}
		}
	}
	if repository.readCalls != 0 || repository.metadataCalls != 0 {
		t.Fatalf("malformed names reached the store: %d payload and %d metadata reads", repository.readCalls, repository.metadataCalls)
	}
	response := putArtifact(t, handler, "worker", "review_notes", "*", []byte("report"))
	if response.Code != http.StatusCreated {
		t.Fatalf("portable name: %d %s", response.Code, response.Body.String())
	}
}
