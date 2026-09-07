package cli

import (
	"context"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"testing"
	"time"
)

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
