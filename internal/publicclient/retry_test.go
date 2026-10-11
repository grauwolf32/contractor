package publicclient

import (
	"bytes"
	"context"
	"crypto/x509"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"sync/atomic"
	"testing"
	"time"
)

func TestIdempotentCreateReplaysExactRequestAfterTransientResponses(t *testing.T) {
	for _, first := range []struct {
		name, version, body string
		status              int
	}{
		{name: "headerless proxy", status: http.StatusBadGateway},
		{name: "versioned retryable envelope", version: APIVersion, status: http.StatusInternalServerError,
			body: `{"code":"busy","message":"try again","retryable":true}`},
	} {
		t.Run(first.name, func(t *testing.T) {
			var calls atomic.Int32
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				body, err := io.ReadAll(r.Body)
				if err != nil {
					t.Error(err)
				}
				if r.Method != http.MethodPost || string(body) != `{"name":"demo"}` ||
					r.Header.Get("Idempotency-Key") != "same-key" || r.Header.Get("Authorization") != "Bearer secret" {
					t.Errorf("replayed request = %s %q %#v", r.Method, body, r.Header)
				}
				if calls.Add(1) == 1 {
					if first.version != "" {
						w.Header().Set(APIVersionHeader, first.version)
					}
					w.WriteHeader(first.status)
					_, _ = io.WriteString(w, first.body)
					return
				}
				w.Header().Set(APIVersionHeader, APIVersion)
				w.WriteHeader(http.StatusCreated)
			}))
			defer server.Close()

			transport, err := transport("")
			if err != nil {
				t.Fatal(err)
			}
			client := &http.Client{Transport: &checkedTransport{
				base: transport, origin: server.URL, token: "secret", timeout: time.Second,
			}}
			request, err := http.NewRequestWithContext(t.Context(), http.MethodPost,
				server.URL+"/v1/projects", bytes.NewBufferString(`{"name":"demo"}`))
			if err != nil {
				t.Fatal(err)
			}
			request.Header.Set("Idempotency-Key", "same-key")
			response, err := client.Do(request)
			if err != nil {
				t.Fatal(err)
			}
			_ = response.Body.Close()
			if response.StatusCode != http.StatusCreated || calls.Load() != 2 {
				t.Fatalf("status = %d; calls = %d", response.StatusCode, calls.Load())
			}
		})
	}
}

func TestIdempotentCreateRetriesPerAttemptTimeout(t *testing.T) {
	var calls atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, err := io.ReadAll(r.Body)
		if err != nil || string(body) != `{"name":"demo"}` || r.Header.Get("Idempotency-Key") != "same-key" {
			t.Errorf("timed-out retry body/key = %q/%q, error=%v", body, r.Header.Get("Idempotency-Key"), err)
		}
		if calls.Add(1) == 1 {
			time.Sleep(90 * time.Millisecond)
		}
		w.Header().Set(APIVersionHeader, APIVersion)
		w.WriteHeader(http.StatusCreated)
	}))
	defer server.Close()
	base, err := transport("")
	if err != nil {
		t.Fatal(err)
	}
	client := &http.Client{Transport: &checkedTransport{
		base: base, origin: server.URL, token: "secret", timeout: 30 * time.Millisecond,
	}}
	request, err := http.NewRequestWithContext(t.Context(), http.MethodPost,
		server.URL+"/v1/projects", bytes.NewBufferString(`{"name":"demo"}`))
	if err != nil {
		t.Fatal(err)
	}
	request.Header.Set("Idempotency-Key", "same-key")
	response, err := client.Do(request)
	if err != nil {
		t.Fatal(err)
	}
	_ = response.Body.Close()
	if response.StatusCode != http.StatusCreated || calls.Load() != 2 {
		t.Fatalf("status = %d; calls = %d", response.StatusCode, calls.Load())
	}
}

func TestUnsafeWriteAndVersionMismatchAreNotRetried(t *testing.T) {
	for _, test := range []struct {
		name, method, version string
		status                int
	}{
		{name: "artifact PUT", method: http.MethodPut, version: APIVersion, status: http.StatusBadGateway},
		{name: "mismatched success version", method: http.MethodGet, version: "contractor.public.v2", status: http.StatusOK},
	} {
		t.Run(test.name, func(t *testing.T) {
			var calls atomic.Int32
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				calls.Add(1)
				w.Header().Set(APIVersionHeader, test.version)
				w.WriteHeader(test.status)
			}))
			defer server.Close()
			base, err := transport("")
			if err != nil {
				t.Fatal(err)
			}
			client := &http.Client{Transport: &checkedTransport{
				base: base, origin: server.URL, token: "secret", timeout: time.Second,
			}}
			request, err := http.NewRequestWithContext(t.Context(), test.method,
				server.URL+"/v1/artifacts/example", nil)
			if err != nil {
				t.Fatal(err)
			}
			response, err := client.Do(request)
			if test.method == http.MethodGet {
				var compatibility *CompatibilityError
				if !errors.As(err, &compatibility) {
					t.Fatalf("error = %v", err)
				}
			} else {
				if err != nil || response.StatusCode != test.status {
					t.Fatalf("response = %v; error = %v", response, err)
				}
				_ = response.Body.Close()
			}
			if calls.Load() != 1 {
				t.Fatalf("calls = %d", calls.Load())
			}
		})
	}
}

func TestRetryStopsAtCallerDeadline(t *testing.T) {
	var calls atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		calls.Add(1)
		w.WriteHeader(http.StatusServiceUnavailable)
	}))
	defer server.Close()
	base, err := transport("")
	if err != nil {
		t.Fatal(err)
	}
	client := &http.Client{Transport: &checkedTransport{
		base: base, origin: server.URL, token: "secret", timeout: time.Second,
	}}
	ctx, cancel := context.WithTimeout(t.Context(), 30*time.Millisecond)
	defer cancel()
	request, err := http.NewRequestWithContext(ctx, http.MethodGet, server.URL+"/v1/workflows", nil)
	if err != nil {
		t.Fatal(err)
	}
	_, err = client.Do(request)
	if !errors.Is(err, context.DeadlineExceeded) || calls.Load() != 1 {
		t.Fatalf("error = %v; calls = %d", err, calls.Load())
	}
}

func TestTransientClassificationDoesNotHideTLSConfigurationErrors(t *testing.T) {
	if IsTransient(&url.Error{Op: "Get", Err: x509.UnknownAuthorityError{}}) {
		t.Fatal("unknown CA was classified as a recoverable watch failure")
	}
	if !IsTransient(&url.Error{Op: "Get", Err: context.DeadlineExceeded}) {
		t.Fatal("request timeout was classified as permanent")
	}
}

// The Server marks every error retryable or not, and the client follows that
// flag on every status: retryable false is never retried, even for 502, 503
// or 504, and an idempotency conflict is never replayed.
func TestServerRetryableFlagDecidesRetriesOnEveryStatus(t *testing.T) {
	for _, test := range []struct {
		name      string
		status    int
		code      string
		retryable bool
		calls     int32
	}{
		{name: "503 not retryable", status: http.StatusServiceUnavailable, code: "artifact_content_corrupt", calls: 1},
		{name: "503 retryable", status: http.StatusServiceUnavailable, code: "storage_transaction_conflict", retryable: true, calls: 2},
		{name: "502 not retryable", status: http.StatusBadGateway, code: "gateway_rejected", calls: 1},
		{name: "504 not retryable", status: http.StatusGatewayTimeout, code: "git_import_timeout", calls: 1},
		{name: "idempotency conflict", status: http.StatusConflict, code: "idempotency_conflict", calls: 1},
		{name: "409 retryable", status: http.StatusConflict, code: "conflict", retryable: true, calls: 2},
	} {
		t.Run(test.name, func(t *testing.T) {
			var calls atomic.Int32
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set(APIVersionHeader, APIVersion)
				if calls.Add(1) == 1 {
					w.WriteHeader(test.status)
					_, _ = fmt.Fprintf(w, `{"code":%q,"message":"failed","retryable":%t,"requestId":"request-1"}`,
						test.code, test.retryable)
					return
				}
				w.WriteHeader(http.StatusCreated)
			}))
			defer server.Close()
			base, err := transport("")
			if err != nil {
				t.Fatal(err)
			}
			client := &http.Client{Transport: &checkedTransport{
				base: base, origin: server.URL, token: "secret", timeout: time.Second,
			}}
			request, err := http.NewRequestWithContext(t.Context(), http.MethodPost,
				server.URL+"/v1/projects", bytes.NewBufferString(`{"name":"demo"}`))
			if err != nil {
				t.Fatal(err)
			}
			request.Header.Set("Idempotency-Key", "same-key")
			response, err := client.Do(request)
			if err != nil {
				t.Fatal(err)
			}
			body, _ := io.ReadAll(response.Body)
			_ = response.Body.Close()
			want := http.StatusCreated
			if test.calls == 1 {
				want = test.status
			}
			if response.StatusCode != want || calls.Load() != test.calls {
				t.Fatalf("status = %d body=%s; calls = %d, want %d after %d calls",
					response.StatusCode, body, calls.Load(), want, test.calls)
			}
			apiErr := &APIError{Status: test.status, Code: test.code, Retryable: test.retryable}
			if IsTransient(apiErr) != test.retryable {
				t.Fatalf("IsTransient(%+v) = %t", apiErr, !test.retryable)
			}
		})
	}
	// Without the API version header the response is an intermediary's, and
	// only its status can be judged.
	if !IsTransient(&IntermediaryError{Status: http.StatusServiceUnavailable}) ||
		IsTransient(&IntermediaryError{Status: http.StatusConflict}) {
		t.Fatal("intermediary errors are not classified by status")
	}
}
