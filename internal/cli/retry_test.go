package cli

import (
	"bytes"
	"context"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
)

func TestCreateFailureShowsGeneratedKeyForManualRetry(t *testing.T) {
	for _, test := range []struct {
		name, path string
		args       []string
	}{
		{name: "project", path: "/v1/projects", args: []string{"project", "create", "demo"}},
		{name: "run", path: "/v1/runs", args: []string{"run", "create", "flow@1"}},
	} {
		t.Run(test.name, func(t *testing.T) {
			var calls atomic.Int32
			var mu sync.Mutex
			var firstKey, firstBody string
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				body, err := io.ReadAll(r.Body)
				if err != nil {
					t.Error(err)
				}
				key := r.Header.Get("Idempotency-Key")
				if r.URL.Path != test.path || key == "" || r.Method != http.MethodPost {
					t.Errorf("create request = %s %s, key=%q", r.Method, r.URL.Path, key)
				}
				mu.Lock()
				if calls.Add(1) == 1 {
					firstKey, firstBody = key, string(body)
				} else if key != firstKey || string(body) != firstBody {
					t.Errorf("create retry changed key/body: %q %q, want %q %q", key, body, firstKey, firstBody)
				}
				mu.Unlock()
				w.WriteHeader(http.StatusServiceUnavailable)
			}))
			defer server.Close()
			var stdout, stderr bytes.Buffer
			command := New(strings.NewReader(""), &stdout, &stderr, func(name string) string {
				if name == "CONTRACTOR_API_TOKEN" {
					return "secret"
				}
				return ""
			})
			err := command.Run(t.Context(), append([]string{"--server", server.URL}, test.args...))
			mu.Lock()
			key := firstKey
			mu.Unlock()
			if err == nil || calls.Load() != 3 || key == "" || stdout.Len() != 0 ||
				!strings.Contains(stderr.String(), "--idempotency-key "+key) {
				t.Fatalf("error=%v calls=%d key=%q stdout=%q stderr=%q",
					err, calls.Load(), key, stdout.String(), stderr.String())
			}
		})
	}
}

func TestExplicitCreateKeyNeedsNoGeneratedKeyHint(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Idempotency-Key") != "operator-key" {
			t.Errorf("key = %q", r.Header.Get("Idempotency-Key"))
		}
		w.Header().Set("X-Contractor-API-Version", "contractor.public.v1")
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusConflict)
		_, _ = io.WriteString(w, `{"code":"conflict","message":"exists","retryable":false}`)
	}))
	defer server.Close()
	var stdout, stderr bytes.Buffer
	command := New(strings.NewReader(""), &stdout, &stderr, func(name string) string {
		if name == "CONTRACTOR_API_TOKEN" {
			return "secret"
		}
		return ""
	})
	err := command.Run(t.Context(), []string{
		"--server", server.URL, "project", "create", "demo", "--idempotency-key", "operator-key",
	})
	if err == nil || strings.Contains(stderr.String(), "--idempotency-key") {
		t.Fatalf("error=%v stderr=%q", err, stderr.String())
	}
}

func TestWatchContinuesAfterExhaustedTransientPoll(t *testing.T) {
	var calls atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/runs/run-1" {
			t.Errorf("path = %q", r.URL.Path)
		}
		if calls.Add(1) <= 3 {
			w.WriteHeader(http.StatusBadGateway)
			return
		}
		w.Header().Set("X-Contractor-API-Version", "contractor.public.v1")
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"runId":"run-1","state":"succeeded","workflow":"flow@1"}`)
	}))
	defer server.Close()
	var stdout, stderr bytes.Buffer
	command := New(strings.NewReader(""), &stdout, &stderr, func(name string) string {
		if name == "CONTRACTOR_API_TOKEN" {
			return "secret"
		}
		return ""
	})
	err := command.Run(t.Context(), []string{
		"--server", server.URL, "--output", "name", "run", "watch", "run-1", "--interval", "100ms", "--wait-timeout", "2s",
	})
	if err != nil || calls.Load() != 4 || stdout.String() != "run-1\n" {
		t.Fatalf("error=%v calls=%d stdout=%q stderr=%q", err, calls.Load(), stdout.String(), stderr.String())
	}
}

func TestWatchStopsOnPermanentErrorAndOverallTimeout(t *testing.T) {
	for _, test := range []struct {
		name   string
		status int
		limit  string
	}{
		{name: "unauthorized", status: http.StatusUnauthorized, limit: "2s"},
		{name: "missing", status: http.StatusNotFound, limit: "2s"},
		{name: "transient until deadline", status: http.StatusBadGateway, limit: "500ms"},
	} {
		t.Run(test.name, func(t *testing.T) {
			var calls atomic.Int32
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				calls.Add(1)
				w.Header().Set("X-Contractor-API-Version", "contractor.public.v1")
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(test.status)
				_, _ = io.WriteString(w, `{"code":"unavailable","message":"unavailable","retryable":true}`)
			}))
			defer server.Close()
			var stdout, stderr bytes.Buffer
			command := New(strings.NewReader(""), &stdout, &stderr, func(name string) string {
				if name == "CONTRACTOR_API_TOKEN" {
					return "secret"
				}
				return ""
			})
			err := command.Run(t.Context(), []string{
				"--server", server.URL, "run", "watch", "run-1", "--interval", "100ms", "--wait-timeout", test.limit,
			})
			if test.status == http.StatusBadGateway {
				if !errors.Is(err, context.DeadlineExceeded) || calls.Load() < 3 {
					t.Fatalf("error=%v calls=%d", err, calls.Load())
				}
			} else if err == nil || calls.Load() != 1 {
				t.Fatalf("error=%v calls=%d", err, calls.Load())
			}
		})
	}
}
