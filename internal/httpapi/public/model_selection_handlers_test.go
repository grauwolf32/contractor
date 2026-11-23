package public

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/grauwolf32/contractor/internal/config"
)

func TestModelSelectionReplaceAndConflictPreserveSavedModels(t *testing.T) {
	fixture := newHandlerFixture(t)
	get := func() (*httptest.ResponseRecorder, config.ModelSelectionInfo) {
		response := httptest.NewRecorder()
		fixture.handler.ServeHTTP(response, authenticatedRequest(http.MethodGet, "/v1/operations/settings/models", bytes.NewReader(nil)))
		var value config.ModelSelectionInfo
		if err := json.Unmarshal(response.Body.Bytes(), &value); err != nil || response.Code != 200 {
			t.Fatalf("GET models: %d %s %v", response.Code, response.Body.String(), err)
		}
		return response, value
	}
	original, initial := get()
	if initial.Selection.DefaultModel != "fixture-a" || original.Header().Get("ETag") == "" {
		t.Fatal("missing initial model selection")
	}
	put := func(body, etag string) *httptest.ResponseRecorder {
		request := authenticatedRequest(http.MethodPut, "/v1/operations/settings/models", bytes.NewReader([]byte(body)))
		request.Header.Set("Content-Type", "application/json")
		request.Header.Set("If-Match", etag)
		response := httptest.NewRecorder()
		fixture.handler.ServeHTTP(response, request)
		return response
	}
	body := `{"defaultModel":"fixture-b","roles":{"summarizer":"fixture-a"}}`
	replaced := put(body, original.Header().Get("ETag"))
	if replaced.Code != 200 {
		t.Fatalf("PUT models: %d %s", replaced.Code, replaced.Body.String())
	}
	saved, current := get()
	if current.Selection.DefaultModel != "fixture-b" || current.Selection.Roles["summarizer"] != "fixture-a" || current.Revision != "2" {
		t.Fatalf("saved models: %+v", current)
	}
	noop := put(body, saved.Header().Get("ETag"))
	if noop.Code != 200 || noop.Body.String() != saved.Body.String() {
		t.Fatalf("no-op models changed: %d %s", noop.Code, noop.Body.String())
	}
	for _, test := range []struct {
		body, etag string
		status     int
	}{
		{`{"defaultModel":"fixture-a","roles":{}}`, original.Header().Get("ETag"), 412},
		{`{"defaultModel":"missing","roles":{}}`, saved.Header().Get("ETag"), 400},
		{`{"defaultModel":"fixture-a","roles":{"worker":null}}`, saved.Header().Get("ETag"), 400},
		{`{"defaultModel":"fixture-a","roles":{"other":"fixture-b"}}`, saved.Header().Get("ETag"), 400},
		{`{"defaultModel":"fixture-a","roles":{},"model":"other"}`, saved.Header().Get("ETag"), 400},
		{`{"defaultModel":"fixture-a","defaultModel":"fixture-b","roles":{}}`, saved.Header().Get("ETag"), 400},
		{body, `W/` + saved.Header().Get("ETag"), 400},
		{body, `"2"`, 400},
	} {
		response := put(test.body, test.etag)
		if response.Code != test.status {
			t.Fatalf("rejected models: %d want %d: %s", response.Code, test.status, response.Body.String())
		}
		latest, _ := get()
		if latest.Body.String() != saved.Body.String() {
			t.Fatal("rejected model replacement changed saved settings")
		}
	}
}
