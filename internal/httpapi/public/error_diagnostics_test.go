package public

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"log/slog"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgconn"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/credentials"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/grauwolf32/contractor/internal/requestid"
)

func TestPublicErrorDiagnosticsAreCorrelatedRedactedAndDoNotChangeEnvelopes(t *testing.T) {
	const secret = "recognizable-password-session-sql-artifact-secret"
	for _, test := range []struct {
		name, cause, code string
		err               error
		status            int
	}{
		{"storage", "storage_constraint_failed", "internal_error", &pgconn.PgError{
			Code: "23503", Message: secret, Detail: secret, Where: secret, InternalQuery: secret,
		}, http.StatusInternalServerError},
		{"storage-unknown-code", "storage_failed", "internal_error", &pgconn.PgError{
			Code: secret, Message: secret,
		}, http.StatusInternalServerError},
		{"downstream", "downstream_unavailable", "gateway_unavailable", credentials.ErrGatewayUnavailable, http.StatusBadGateway},
		{"cancelled", "cancelled", "internal_error", context.Canceled, http.StatusInternalServerError},
		{"deadline", "deadline_exceeded", "internal_error", context.DeadlineExceeded, http.StatusInternalServerError},
		{"unknown", "unknown", "internal_error", errors.New(secret), http.StatusInternalServerError},
	} {
		t.Run(test.name, func(t *testing.T) {
			var logs bytes.Buffer
			logger := slog.New(slog.NewJSONHandler(&logs, nil))
			h := &handler{}
			mux := http.NewServeMux()
			mux.HandleFunc("POST /v1/diagnostics/{resource}", func(w http.ResponseWriter, _ *http.Request) {
				h.handleError(w, fmt.Errorf("%s: %w", secret, test.err))
			})
			boundary := withAPIVersion(requestid.Middleware(withErrorDiagnostics(mux), requestid.Options{
				Generator: func() (string, error) { return "request-diagnostic", nil },
				Logger:    logger, Boundary: "public-api",
			}))
			request := httptest.NewRequest(http.MethodPost, "/v1/diagnostics/"+secret+"?token="+secret, strings.NewReader(secret))
			request.Header.Set("Authorization", "Bearer "+secret)
			request.Header.Set("Cookie", "session="+secret)
			request.Header.Set(requestid.Header, secret)
			response := httptest.NewRecorder()
			boundary.ServeHTTP(response, request)
			if response.Code != test.status || response.Header().Get(requestid.Header) != "request-diagnostic" {
				t.Fatalf("response = %d %v", response.Code, response.Header())
			}
			var envelope map[string]any
			if err := json.Unmarshal(response.Body.Bytes(), &envelope); err != nil {
				t.Fatal(err)
			}
			if len(envelope) != 4 || envelope["code"] != test.code || envelope["retryable"] != true || envelope["requestId"] != "request-diagnostic" {
				t.Fatalf("public envelope changed: %v", envelope)
			}
			if strings.Contains(logs.String()+response.Body.String(), secret) {
				t.Fatal("diagnostic or response leaked private error/request material")
			}
			decoder := json.NewDecoder(&logs)
			var event map[string]any
			if err := decoder.Decode(&event); err != nil {
				t.Fatal(err)
			}
			if event["msg"] != "HTTP request failed" || event["cause"] != test.cause ||
				event["operation"] != "POST /v1/diagnostics/{resource}" || event["request_id"] != "request-diagnostic" ||
				event["status"] != float64(test.status) || event["boundary"] != "public-api" {
				t.Fatalf("diagnostic = %v", event)
			}
			if err := decoder.Decode(&event); err != io.EOF {
				t.Fatalf("expected exactly one request diagnostic, got another event: %v (%v)", event, err)
			}
		})
	}
}

func TestPublicExpectedErrorsKeepContractsWithoutFailureLogs(t *testing.T) {
	for _, test := range []struct {
		err       error
		status    int
		code      string
		retryable bool
	}{
		{contracts.ErrValidation, http.StatusBadRequest, "invalid_request", false},
		{projectstore.ErrNotFound, http.StatusNotFound, "not_found", false},
		{projectstore.ErrPrecondition, http.StatusPreconditionFailed, "precondition_failed", false},
		{artifacts.ErrArtifactConflict, http.StatusConflict, "conflict", true},
	} {
		t.Run(test.code, func(t *testing.T) {
			var logs bytes.Buffer
			h := &handler{}
			mux := http.NewServeMux()
			mux.HandleFunc("GET /v1/diagnostics", func(w http.ResponseWriter, _ *http.Request) {
				h.handleError(w, fmt.Errorf("private request payload: %w", test.err))
			})
			boundary := requestid.Middleware(withErrorDiagnostics(mux), requestid.Options{
				Logger:    slog.New(slog.NewJSONHandler(&logs, nil)),
				Generator: func() (string, error) { return "request-expected", nil },
			})
			response := httptest.NewRecorder()
			boundary.ServeHTTP(response, httptest.NewRequest(http.MethodGet, "/v1/diagnostics", nil))
			var envelope errorResponse
			if err := json.Unmarshal(response.Body.Bytes(), &envelope); err != nil {
				t.Fatal(err)
			}
			if response.Code != test.status || envelope.Code != test.code || envelope.Retryable != test.retryable || envelope.RequestID != "request-expected" {
				t.Fatalf("response changed = %d %+v", response.Code, envelope)
			}
			if logs.Len() != 0 {
				t.Fatalf("expected rejection emitted failure logs: %s", &logs)
			}
		})
	}
}

func TestPublicAuthenticationRejectionDoesNotEmitDiagnostic(t *testing.T) {
	var logs bytes.Buffer
	fixture := newHandlerFixtureWithAuth(t, "../../config/testdata/valid", newTestAuthentication(t), mustTestOrigins(t), false,
		slog.New(slog.NewJSONHandler(&logs, nil)))
	response := httptest.NewRecorder()
	request := httptest.NewRequest(http.MethodGet, "/v1/projects", nil)
	request.Header.Set("Authorization", "Bearer private-invalid-token")
	fixture.handler.ServeHTTP(response, request)
	if response.Code != http.StatusUnauthorized || response.Header().Get(requestid.Header) != "request-fixed" || logs.Len() != 0 {
		t.Fatalf("authentication response=%d logs=%s", response.Code, &logs)
	}
	forbidden := httptest.NewRecorder()
	request = authenticatedRequest(http.MethodGet, "/v1/projects", bytes.NewReader(nil))
	request.Header.Set("Origin", "https://forbidden.example.test")
	fixture.handler.ServeHTTP(forbidden, request)
	if forbidden.Code != http.StatusForbidden || forbidden.Header().Get(requestid.Header) != "request-fixed" || logs.Len() != 0 {
		t.Fatalf("authorization response=%d logs=%s", forbidden.Code, &logs)
	}
}

func TestDiagnosticCauseUsesOnlyClosedCategories(t *testing.T) {
	for _, test := range []struct {
		err   error
		cause string
	}{
		{&pgconn.PgError{Code: "40001"}, "storage_serialization_conflict"},
		{&pgconn.PgError{Code: "40P01"}, "storage_deadlock"},
		{&pgconn.PgError{Code: "55P03"}, "storage_lock_unavailable"},
		{&pgconn.PgError{Code: "57014"}, "storage_query_cancelled"},
		{&pgconn.PgError{Code: "53300"}, "storage_unavailable"},
		{pgx.ErrTxClosed, "storage_transaction_failed"},
		{credentials.ErrCrypto, "credential_crypto_failed"},
		{credentials.ErrKeyUnavailable, "credential_key_unavailable"},
		{&net.DNSError{Err: "private network payload", Name: "private-host", IsTimeout: true}, "network_timeout"},
		{&net.DNSError{Err: "private network payload", Name: "private-host"}, "network_failed"},
		{&fs.PathError{Op: "read", Path: "private-path", Err: fs.ErrPermission}, "storage_io_failed"},
		{diagnosticOpaqueError{}, "unknown"},
	} {
		t.Run(test.cause, func(t *testing.T) {
			if got := diagnosticCause(test.err); got != test.cause {
				t.Fatalf("cause = %q, want %q", got, test.cause)
			}
		})
	}
}

type diagnosticOpaqueError struct{}

func (diagnosticOpaqueError) Error() string { panic("diagnostics must not stringify errors") }
