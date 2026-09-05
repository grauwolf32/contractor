package privateartifacts

import (
	"net/http"
	"net/url"
	"strings"
	"testing"
)

func TestPrivateArtifactNamesRejectBeforeStore(t *testing.T) {
	repository := newMemoryRepository()
	handler := newTestHandler(t, &fakeRegistry{grant: testGrant("run-a")}, repository)
	for _, name := range []string{"review notes", "отчет", "_report", strings.Repeat("a", 129)} {
		response := putArtifact(t, handler, "worker", url.PathEscape(name), "*", []byte("report"))
		if response.Code != http.StatusBadRequest {
			t.Errorf("%q: %d %s", name, response.Code, response.Body.String())
		}
	}
	response := putArtifact(t, handler, "worker", "review_notes", "*", []byte("report"))
	if response.Code != http.StatusCreated {
		t.Fatalf("portable name: %d %s", response.Code, response.Body.String())
	}
}
