package public

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestJSONRequestNULEscapeRuleAndOpaqueExceptions(t *testing.T) {
	for _, tc := range []struct {
		name, body string
		invalid    bool
	}{
		{"value", `{"note":"a\u0000b"}`, true},
		{"member", `{"a\u0000b":"value"}`, true},
		{"nested", `{"note":[{"value":"\u0000"}]}`, true},
		{"literal escape", `{"note":"a\\u0000b"}`, false},
		{"other control", `{"note":"a\u0001b"}`, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			request := httptest.NewRequest(http.MethodPost, "/", strings.NewReader(tc.body))
			var result any
			err := decodeJSON(httptest.NewRecorder(), request, &result)
			if (err != nil) != tc.invalid {
				t.Fatalf("decodeJSON error = %v, invalid=%t", err, tc.invalid)
			}
		})
	}
	// These values are hashed or encrypted as bytes and never cast to jsonb.
	for _, body := range []string{`{"password":"a\u0000b"}`, `{"material":{"token":"a\u0000b"}}`} {
		request := httptest.NewRequest(http.MethodPost, "/", strings.NewReader(body))
		var result any
		if err := decodeOpaqueJSONBounded(httptest.NewRecorder(), request, &result, maxJSONRequestSize); err != nil {
			t.Fatalf("opaque input = %v", err)
		}
	}
}

func TestAuditJSONNULEscapeIsRejectedBeforeIdentityAndService(t *testing.T) {
	// Even a missing profile or unavailable store cannot precede this check.
	// Nil service and ID generator deliberately fail if either is reached.
	h := &auditHandlers{}
	request := auditAuthenticatedRequest(http.MethodPost, "/v1/projects/ordinary/audits", []byte(`{"profile":{"name":"profile","version":"1"},"inputs":{},"runtimeLabels":[],"scope":{"objective":"a\u0000b"}}`))
	request.SetPathValue("projectId", "ordinary")
	request.Header.Set("Content-Type", "application/json")
	request.Header.Set("Idempotency-Key", "nul-boundary")
	response := httptest.NewRecorder()
	serve(h.createAudit)(response, request)
	if response.Code != http.StatusBadRequest {
		t.Fatalf("Audit boundary = %d %s", response.Code, response.Body.String())
	}
}

func TestRuntimeConfigJSONNULEscapeIsRejectedBeforePublication(t *testing.T) {
	h := &runtimeConfigurationHandlers{}
	request := httptest.NewRequest(http.MethodPost, "/v1/runtime-configs", strings.NewReader(`{"description":"a\u0000b"}`))
	request.Header.Set("Content-Type", "application/json")
	request.Header.Set("Idempotency-Key", "nul-boundary")
	response := httptest.NewRecorder()
	serve(h.publishRuntimeConfig)(response, request)
	if response.Code != http.StatusBadRequest {
		t.Fatalf("RuntimeConfig boundary = %d %s", response.Code, response.Body.String())
	}
}
