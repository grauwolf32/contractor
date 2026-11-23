package privateartifacts

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// The private update precondition is one strong entity tag read verbatim
// (ART-R45, API-R70). An escaped, single- or back-quoted spelling of the
// current revision, an empty or weak tag and a list are bad requests that
// never reach the store, while the plain quoted revision still updates the
// binding.
func TestPrivateArtifactIfMatchAcceptsOnlyOneStrongEntityTag(t *testing.T) {
	repository := newMemoryRepository()
	repository.seed("run-a", "notes", "draft", "7", []byte("before"))
	handler := newTestHandler(t, &fakeRegistry{grant: testGrant("run-a")}, repository)
	update := func(ifMatch string) *httptest.ResponseRecorder {
		t.Helper()
		request := trustedRequest(
			http.MethodPut,
			"/private/v1/allocations/allocation-1/artifacts/notes/draft",
			strings.NewReader("after"),
		)
		request.Header.Set("Content-Type", "text/plain")
		request.Header.Set("If-Match", ifMatch)
		response := httptest.NewRecorder()
		handler.ServeHTTP(response, request)
		return response
	}
	for _, ifMatch := range []string{
		// Escaped spellings of the current revision 7.
		`"\x37"`, `"\067"`, `"\u0037"`, `"\U00000037"`,
		// Other quoting of the current revision.
		`'7'`, "`7`", `7`, ` "7"`, `"7" `, `"7"x`, `"7\"`,
		// Empty, weak and list values.
		`""`, `"`, `W/"7"`, `"7", "7"`, `"7","8"`, `*`,
	} {
		t.Run(ifMatch, func(t *testing.T) {
			assertPrivateErrorEnvelope(t, update(ifMatch), http.StatusBadRequest, "invalid_request", false)
		})
	}
	if repository.writeCalls != 0 {
		t.Fatalf("a malformed If-Match reached the store %d times", repository.writeCalls)
	}
	if current, _ := repository.current("run-a", "notes", "draft"); current.revision != "7" ||
		string(current.data) != "before" {
		t.Fatalf("binding after rejected preconditions = %q %q", current.revision, current.data)
	}
	response := update(`"7"`)
	if response.Code != http.StatusOK || response.Header().Get("ETag") == `"7"` {
		t.Fatalf("plain quoted If-Match = %d headers=%v %s", response.Code, response.Header(), response.Body.String())
	}
	if current, _ := repository.current("run-a", "notes", "draft"); string(current.data) != "after" {
		t.Fatalf("binding after the plain update = %q", current.data)
	}
}
