package cli

import (
	"bytes"
	"context"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// Artifact transfers bound inactivity, not total duration: a download that
// keeps making progress outlasts --timeout, while ordinary commands do not.
func TestTransferCommandsOutlastTheTimeoutWhileMakingProgress(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("X-Contractor-API-Version", "contractor.public.v1")
		if r.URL.Path == "/v1/workflows" {
			w.Header().Set("Content-Type", "application/json")
		} else {
			w.Header().Set("Content-Type", "application/octet-stream")
		}
		flusher := w.(http.Flusher)
		for range 10 {
			_, _ = w.Write([]byte(" "))
			flusher.Flush()
			select {
			case <-time.After(40 * time.Millisecond):
			case <-r.Context().Done():
				return
			}
		}
		if r.URL.Path == "/v1/workflows" {
			_, _ = w.Write([]byte(`{"items":[],"page":{"hasMore":false}}`))
		}
	}))
	defer server.Close()
	var stdout bytes.Buffer
	command := New(nil, &stdout, io.Discard, func(name string) string {
		switch name {
		case "CONTRACTOR_CLI_CONFIG":
			return filepath.Join(t.TempDir(), "contexts.yaml")
		case "CONTRACTOR_API_TOKEN":
			return "test-token"
		}
		return ""
	})
	global := []string{"--server", server.URL, "--timeout", "200ms"}
	if err := command.Run(context.Background(), append(global, "workflow", "list")); !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("ordinary command = %v, want the whole-request deadline", err)
	}
	if err := command.Run(context.Background(), append(global, "artifact", "get", "documents/brief")); err != nil {
		t.Fatalf("steady slow download: %v", err)
	}
	if stdout.String() != strings.Repeat(" ", 10) {
		t.Fatalf("downloaded %q", stdout.String())
	}
}

func TestContextCheckUsesGlobalTimeout(t *testing.T) {
	for _, source := range []string{"flag", "environment"} {
		t.Run(source, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				select {
				case <-r.Context().Done():
				case <-time.After(time.Second):
					w.WriteHeader(http.StatusOK)
				}
			}))
			defer server.Close()
			path := filepath.Join(t.TempDir(), "contexts.yaml")
			if _, err := NewContextStore(path).Put("probe", ServerContext{Server: server.URL}, true); err != nil {
				t.Fatal(err)
			}
			command := New(nil, io.Discard, io.Discard, func(name string) string {
				switch name {
				case "CONTRACTOR_CLI_CONFIG":
					return path
				case "CONTRACTOR_API_TOKEN":
					return "test-token"
				case "CONTRACTOR_TIMEOUT":
					if source == "environment" {
						return "50ms"
					}
				}
				return ""
			})
			for _, args := range [][]string{{"check"}, {"context", "check", "probe"}} {
				if source == "flag" {
					args = append([]string{"--timeout", "50ms"}, args...)
				}
				if err := command.Run(context.Background(), args); !errors.Is(err, context.DeadlineExceeded) {
					t.Fatalf("%v: expected request deadline, got %v", args, err)
				}
			}
		})
	}
}
