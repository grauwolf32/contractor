package cli

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"

	"github.com/grauwolf32/contractor/internal/publicclient"
)

func TestTokenPrecedenceIsFlagThenContextThenEnvironment(t *testing.T) {
	var authorization string
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		if authorization == "" {
			authorization = request.Header.Get("Authorization")
		}
		writer.Header().Set(publicclient.APIVersionHeader, publicclient.APIVersion)
		writer.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(writer, `{"items":[],"page":{"hasMore":false}}`)
	}))
	defer server.Close()

	directory := t.TempDir()
	writeToken := func(name, token string) string {
		path := filepath.Join(directory, name)
		if err := os.WriteFile(path, []byte(token+"\n"), 0o600); err != nil {
			t.Fatal(err)
		}
		return path
	}
	flagToken := writeToken("flag.token", "flag-token")
	contextToken := writeToken("context.token", "context-token")

	for _, test := range []struct {
		name         string
		contextToken string
		args         []string
		want         string
	}{
		{"flag overrides context and environment", contextToken, []string{"--token-file", flagToken, "workflow", "list"}, "flag-token"},
		{"context overrides environment", contextToken, []string{"workflow", "list"}, "context-token"},
		{"environment is the fallback", "", []string{"workflow", "list"}, "env-token"},
		{"context check uses context token file", contextToken, []string{"context", "check", "probe"}, "context-token"},
		{"context check falls back to environment", "", []string{"context", "check", "probe"}, "env-token"},
	} {
		t.Run(test.name, func(t *testing.T) {
			authorization = ""
			path := filepath.Join(t.TempDir(), "contexts.json")
			if _, err := NewContextStore(path).Put("probe", ServerContext{Server: server.URL, TokenFile: test.contextToken}, true); err != nil {
				t.Fatal(err)
			}
			command := New(nil, io.Discard, io.Discard, func(name string) string {
				switch name {
				case "CONTRACTOR_CLI_CONFIG":
					return path
				case "CONTRACTOR_API_TOKEN":
					return "env-token"
				}
				return ""
			})
			_ = command.Run(context.Background(), test.args)
			if authorization != "Bearer "+test.want {
				t.Fatalf("Authorization = %q, want Bearer %s", authorization, test.want)
			}
		})
	}
}
