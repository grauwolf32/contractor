package public

import (
	"bytes"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestAuditApprovalRoutesAreRemoved(t *testing.T) {
	fixture := newHandlerFixture(t)
	for _, test := range []struct{ method, path string }{
		{http.MethodGet, "/v1/reviews"},
		{http.MethodGet, "/v1/audits/audit-one/reviews"},
		{http.MethodGet, "/v1/audits/audit-one/reviews/request-one"},
		{http.MethodPost, "/v1/audits/audit-one/reviews/request-one/decisions"},
	} {
		t.Run(test.method+test.path, func(t *testing.T) {
			request := authenticatedRequest(test.method, test.path, bytes.NewReader([]byte(`{"action":"approve","rationale":"Historical consent grants no authority"}`)))
			request.Header.Set("Content-Type", "application/json")
			request.Header.Set("Idempotency-Key", "obsolete-approval")
			request.Header.Set("If-Match", `"1"`)
			response := httptest.NewRecorder()
			fixture.handler.ServeHTTP(response, request)
			if response.Code != http.StatusNotFound {
				t.Fatalf("obsolete approval route returned %d: %s", response.Code, response.Body.String())
			}
		})
	}
}
