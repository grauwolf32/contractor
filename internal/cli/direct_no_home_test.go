package cli

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/grauwolf32/contractor/internal/publicclient"
)

func TestDirectCommandsDoNotNeedUserConfigDirectory(t *testing.T) {
	t.Setenv("HOME", "")
	t.Setenv("XDG_CONFIG_HOME", "")
	t.Setenv("CONTRACTOR_CLI_CONFIG", "")
	if _, err := os.UserConfigDir(); err == nil {
		t.Fatal("test environment unexpectedly has a user config directory")
	}

	for _, test := range []struct {
		name      string
		args      func(string, string) []string
		envServer bool
		envToken  bool
		wantToken string
	}{
		{
			name: "server flag and environment token", envToken: true, wantToken: "env-token",
			args: func(server, _ string) []string { return []string{"--server", server, "workflow", "list"} },
		},
		{
			name: "server environment and environment token", envServer: true, envToken: true, wantToken: "env-token",
			args: func(_, _ string) []string { return []string{"workflow", "list"} },
		},
		{
			name: "server flag and token file", wantToken: "file-token",
			args: func(server, tokenFile string) []string {
				return []string{"--server", server, "--token-file", tokenFile, "workflow", "list"}
			},
		},
		{
			name: "direct check", envToken: true, wantToken: "env-token",
			args: func(server, _ string) []string { return []string{"--server", server, "check"} },
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			var requests atomic.Int32
			server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
				requests.Add(1)
				if request.URL.Path != "/v1/workflows" || request.Header.Get("Authorization") != "Bearer "+test.wantToken {
					t.Errorf("unexpected direct request: %s, authorization=%q", request.URL.Path, request.Header.Get("Authorization"))
				}
				writer.Header().Set(publicclient.APIVersionHeader, publicclient.APIVersion)
				writer.Header().Set("Content-Type", "application/json")
				_, _ = io.WriteString(writer, `{"items":[],"page":{"hasMore":false}}`)
			}))
			defer server.Close()
			tokenFile := filepath.Join(t.TempDir(), "api.token")
			if err := os.WriteFile(tokenFile, []byte("file-token\n"), 0o600); err != nil {
				t.Fatal(err)
			}
			command := New(nil, io.Discard, io.Discard, func(name string) string {
				switch name {
				case "CONTRACTOR_SERVER":
					if test.envServer {
						return server.URL
					}
				case "CONTRACTOR_API_TOKEN":
					if test.envToken {
						return "env-token"
					}
				}
				return ""
			})
			if err := command.Run(context.Background(), test.args(server.URL, tokenFile)); err != nil {
				t.Fatal(err)
			}
			if requests.Load() != 1 {
				t.Fatalf("Server received %d requests, want one", requests.Load())
			}
		})
	}
}

func TestContextCommandsStillNeedUserConfigDirectory(t *testing.T) {
	t.Setenv("HOME", "")
	t.Setenv("XDG_CONFIG_HOME", "")
	t.Setenv("CONTRACTOR_CLI_CONFIG", "")
	for _, test := range []struct {
		name string
		args []string
		env  map[string]string
	}{
		{name: "no server", args: []string{"workflow", "list"}},
		{name: "named context", args: []string{"--context", "named", "--server", "http://127.0.0.1:9", "workflow", "list"}},
		{name: "context environment", args: []string{"workflow", "list"}, env: map[string]string{
			"CONTRACTOR_CONTEXT": "named", "CONTRACTOR_SERVER": "http://127.0.0.1:9",
		}},
		{name: "context command", args: []string{"context", "list"}},
	} {
		t.Run(test.name, func(t *testing.T) {
			command := New(nil, io.Discard, io.Discard, func(name string) string { return test.env[name] })
			err := command.Run(context.Background(), test.args)
			if err == nil || !strings.Contains(err.Error(), "resolve user configuration directory") {
				t.Fatalf("context resolution = %v, want a clear config-directory error", err)
			}
		})
	}
}
