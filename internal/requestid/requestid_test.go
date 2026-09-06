package requestid

import (
	"bytes"
	"errors"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestMiddlewareGeneratesPropagatesAndLogsCorrelationID(t *testing.T) {
	const injectedSecret = "secret-do-not-log"
	var logs bytes.Buffer
	logger := slog.New(slog.NewJSONHandler(&logs, nil))
	handler := Middleware(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if From(r.Context()) != "request-fixed" {
			t.Fatalf("context request ID = %q", From(r.Context()))
		}
		http.Error(w, "safe failure", http.StatusInternalServerError)
	}), Options{
		Generator: func() (string, error) { return "request-fixed", nil },
		Logger:    logger, Boundary: "test",
	})
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, httptest.NewRequest(http.MethodGet, "/failure/"+injectedSecret, nil))
	if response.Header().Get(Header) != "request-fixed" ||
		!strings.Contains(logs.String(), `"request_id":"request-fixed"`) {
		t.Fatalf("response headers=%v logs=%s", response.Header(), logs.String())
	}
	if strings.Contains(logs.String(), injectedSecret) {
		t.Fatalf("request path leaked injected secret: %s", logs.String())
	}
}

func TestMiddlewareTrustsOnlyValidPrivateIncomingIDAndHasFallback(t *testing.T) {
	handler := Middleware(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusNoContent)
	}), Options{
		TrustIncoming: true,
		Generator:     func() (string, error) { return "", errors.New("entropy unavailable") },
	})
	trusted := httptest.NewRequest(http.MethodGet, "/", nil)
	trusted.Header.Set(Header, "upstream:request-1")
	trustedResponse := httptest.NewRecorder()
	handler.ServeHTTP(trustedResponse, trusted)
	if trustedResponse.Header().Get(Header) != "upstream:request-1" {
		t.Fatalf("trusted correlation ID = %q", trustedResponse.Header().Get(Header))
	}

	invalid := httptest.NewRequest(http.MethodGet, "/", nil)
	invalid.Header.Set(Header, "contains a space")
	invalidResponse := httptest.NewRecorder()
	handler.ServeHTTP(invalidResponse, invalid)
	if value := invalidResponse.Header().Get(Header); !Valid(value) || !strings.HasPrefix(value, "request_fallback_") {
		t.Fatalf("fallback correlation ID = %q", value)
	}
}

func TestEnsurePreservesContextAndCreatesBoundedOutgoingID(t *testing.T) {
	if got := Ensure(With(t.Context(), "upstream-request")); got != "upstream-request" {
		t.Fatalf("preserved request ID = %q", got)
	}
	if got := Ensure(t.Context()); !Valid(got) || !strings.HasPrefix(got, "request_") {
		t.Fatalf("generated outgoing request ID = %q", got)
	}
}

func TestFailureAnnotationUnwrapsAndKeepsTheFirstCause(t *testing.T) {
	var logs bytes.Buffer
	handler := Middleware(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		wrapped := &annotationWriter{ResponseWriter: w}
		AnnotateFailure(wrapped, "", "storage_failed")
		AnnotateFailure(wrapped, "GET /v1/resources/{id}", "unknown")
		wrapped.WriteHeader(http.StatusInternalServerError)
	}), Options{Logger: slog.New(slog.NewJSONHandler(&logs, nil))})
	handler.ServeHTTP(httptest.NewRecorder(), httptest.NewRequest(http.MethodGet, "/v1/resources/private-value", nil))
	if !strings.Contains(logs.String(), `"cause":"storage_failed"`) ||
		!strings.Contains(logs.String(), `"operation":"GET /v1/resources/{id}"`) ||
		strings.Contains(logs.String(), "private-value") || strings.Count(logs.String(), "\n") != 1 {
		t.Fatalf("diagnostic = %s", &logs)
	}
	// A direct handler call outside the middleware remains supported.
	AnnotateFailure(httptest.NewRecorder(), "operation", "unknown")
}

type annotationWriter struct{ http.ResponseWriter }

func (w *annotationWriter) Unwrap() http.ResponseWriter { return w.ResponseWriter }
