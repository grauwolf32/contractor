package cli

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/publicclient"
)

func TestProjectSitemapCLIUsesGeneratedOwnerClientAndPinnedIntent(t *testing.T) {
	var calls []string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls = append(calls, r.Method+" "+r.URL.Path)
		w.Header().Set(publicclient.APIVersionHeader, publicclient.APIVersion)
		w.Header().Set("Content-Type", "application/json")
		if r.Method == "GET" {
			if r.URL.Query().Get("revision") != "rev-selected" {
				t.Error("exact selection changed")
			}
			_, _ = io.WriteString(w, `{"revision":"rev-selected","document":{"version":"project-sitemap@1","projection":"sitemap-safe-projection@1","groups":[],"sources":[],"rules":[],"markers":[],"gaps":[]}}`)
			return
		}
		if r.Header.Get("Idempotency-Key") != "original-key" {
			t.Error("mutation key changed")
		}
		var body map[string]any
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Error(err)
		}
		if body["expectedRevision"] != "rev-original" {
			t.Error("CLI refreshed stale edit intent")
		}
		_, _ = io.WriteString(w, `{"artifact":{"namespace":"sitemap","name":"map","revision":"rev-result"},"changed":true,"replayed":false}`)
	}))
	defer server.Close()
	file := filepath.Join(t.TempDir(), "mutation.json")
	if err := os.WriteFile(file, []byte(`{"operation":"remove","expectedRevision":"rev-original","markerId":"hidden","memberIds":["selected"]}`), 0600); err != nil {
		t.Fatal(err)
	}
	var output, stderr bytes.Buffer
	command := New(strings.NewReader(""), &output, &stderr, func(name string) string {
		if name == "CONTRACTOR_API_TOKEN" {
			return "test-token"
		}
		return ""
	})
	for _, args := range [][]string{
		{"get", "project-one", "--revision", "rev-selected"},
		{"mutate", "project-one", "--file", file, "--idempotency-key", "original-key"},
	} {
		prefix := []string{"--server", server.URL, "--output", "json", "project", "sitemap"}
		if err := command.Run(context.Background(), append(prefix, args...)); err != nil {
			t.Fatalf("command: %v; %s", err, stderr.String())
		}
	}
	if len(calls) != 2 || calls[0] != "GET /v1/projects/project-one/sitemap" || calls[1] != "POST /v1/projects/project-one/sitemap/mutations" {
		t.Fatalf("unexpected extra reads or calls: %v", calls)
	}
}
