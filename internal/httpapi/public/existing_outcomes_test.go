package public

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

// TestOperationsSettingsKeepUnsupportedMediaType keeps the documented 415
// unsupported_media_type of both Operations settings updates (OPS-R60,
// OPS-R225): a body without exactly application/json changes nothing, and
// both operations declare the response.
func TestOperationsSettingsKeepUnsupportedMediaType(t *testing.T) {
	document := loadPublicOpenAPI(t)
	fixture := newHandlerFixture(t)
	models := httptest.NewRecorder()
	fixture.handler.ServeHTTP(models, authenticatedRequest(http.MethodGet, "/v1/operations/settings/models", bytes.NewReader(nil)))
	if models.Code != http.StatusOK {
		t.Fatalf("GET model selection = %d %s", models.Code, models.Body.String())
	}
	for _, update := range []struct {
		operation, path, body, ifMatch string
	}{
		{"putModelSelection", "/v1/operations/settings/models", `{"defaultModel":"fixture-b","roles":{}}`, models.Header().Get("ETag")},
		{"putSchedulerSettings", "/v1/operations/settings/scheduler", `{"maxConcurrentRuns":2}`, `"1"`},
	} {
		operation := document.Paths.Find(update.path).Put
		if operation == nil || operation.OperationID != update.operation || operation.Responses.Status(http.StatusUnsupportedMediaType) == nil {
			t.Fatalf("%s does not declare its 415 response", update.operation)
		}
		for _, contentType := range []string{"", "text/plain", "application/json; charset=utf-8"} {
			request := authenticatedRequest(http.MethodPut, update.path, bytes.NewReader([]byte(update.body)))
			if contentType == "" {
				request.Header.Del("Content-Type")
			} else {
				request.Header.Set("Content-Type", contentType)
			}
			request.Header.Set("If-Match", update.ifMatch)
			response := httptest.NewRecorder()
			fixture.handler.ServeHTTP(response, request)
			var failure errorResponse
			if err := json.Unmarshal(response.Body.Bytes(), &failure); err != nil ||
				response.Code != http.StatusUnsupportedMediaType || failure.Code != "unsupported_media_type" {
				t.Errorf("%s with Content-Type %q = %d %s, want 415 unsupported_media_type",
					update.operation, contentType, response.Code, response.Body.String())
			}
		}
	}
	after := httptest.NewRecorder()
	fixture.handler.ServeHTTP(after, authenticatedRequest(http.MethodGet, "/v1/operations/settings/models", bytes.NewReader(nil)))
	if after.Body.String() != models.Body.String() {
		t.Fatal("a rejected media type changed the model selection")
	}
}

// TestAllocationHistoryKeepsItsPageMaximum keeps the allocation history page
// at its own maximum of 100 (OPS-R168, LIM-OPS-28), distinct from the shared
// page bound, in the contract and in the handler.
func TestAllocationHistoryKeepsItsPageMaximum(t *testing.T) {
	document := loadPublicOpenAPI(t)
	operation := document.Paths.Find("/v1/operations/allocation-history").Get
	limit := operation.Parameters.GetByInAndName("query", "limit")
	if limit == nil || limit.Schema == nil || limit.Schema.Value.Max == nil || *limit.Schema.Value.Max != 100 {
		t.Fatal("listAllocationResourceHistory does not declare its limit maximum of 100")
	}
	fixture := newHandlerFixture(t)
	for query, status := range map[string]int{"limit=100": http.StatusOK, "limit=101": http.StatusBadRequest} {
		response := httptest.NewRecorder()
		fixture.handler.ServeHTTP(response, authenticatedRequest(http.MethodGet, "/v1/operations/allocation-history?"+query, bytes.NewReader(nil)))
		if response.Code != status {
			t.Errorf("allocation history %s = %d %s, want %d", query, response.Code, response.Body.String(), status)
		}
	}
}
