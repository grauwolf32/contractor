package privateartifacts

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
)

// Go owns ordinary Artifact transport, not a second parser for the Runtime's
// cookie document. The shared valid payload must survive that boundary exactly.
func TestSharedHTTPCookieOpaqueArtifact(t *testing.T) {
	fixture := contractstest.ReadFile(t, "api", "testdata", "http-cookie-session.json")
	var value struct {
		Document json.RawMessage `json:"document"`
	}
	if err := json.Unmarshal(fixture, &value); err != nil {
		t.Fatal(err)
	}
	repository := newMemoryRepository()
	handler := newTestHandler(t, &fakeRegistry{grant: testGrant("run-a")}, repository)
	const path = "/private/v1/allocations/allocation-1/artifacts/builder/http-cookie-session"
	request := trustedRequest(http.MethodPut, path, bytes.NewReader(value.Document))
	request.Header.Set("Content-Type", "application/vnd.contractor.http-cookie-session+json")
	request.Header.Set("If-None-Match", "*")
	written := httptest.NewRecorder()
	handler.ServeHTTP(written, request)
	if written.Code != http.StatusCreated {
		t.Fatalf("cookie Artifact write = %d", written.Code)
	}
	read := httptest.NewRecorder()
	handler.ServeHTTP(read, trustedRequest(http.MethodGet, path, nil))
	if read.Code != http.StatusOK || !bytes.Equal(read.Body.Bytes(), value.Document) || read.Header().Get("Content-Type") != request.Header.Get("Content-Type") {
		t.Fatalf("ordinary cookie Artifact bytes/metadata changed (status %d)", read.Code)
	}
}
