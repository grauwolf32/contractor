package e2e

import (
	"bytes"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
)

func TestProjectFindingBacktraceScriptRestartsRevisionBoundTraversal(t *testing.T) {
	if _, err := exec.LookPath("bash"); err != nil {
		t.Skip("bash is unavailable")
	}
	if _, err := exec.LookPath("jq"); err != nil {
		t.Skip("jq is unavailable")
	}

	var generation atomic.Int32
	generation.Store(1)
	var provenanceCalls atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		if request.Header.Get("Authorization") != "Bearer test-token" {
			http.Error(response, `{"code":"unauthorized"}`, http.StatusUnauthorized)
			return
		}
		response.Header().Set("Content-Type", "application/json")
		revision := generation.Load()
		switch request.URL.Path {
		case "/v1/projects/project-script/findings/finding-script":
			fmt.Fprintf(response, `{"findingId":"finding-script","state":"proposed","revision":%d}`, revision)
		case "/v1/projects/project-script/finding-sources":
			provenanceCalls.Add(1)

			if revision == 1 && request.URL.Query().Get("cursor") == "" {
				fmt.Fprint(response, `{"projectRevision":1,"items":[],"page":{"hasMore":true,"nextCursor":"stale-cursor"}}`)
				return
			}
			if revision == 1 {
				generation.Store(2)
				http.Error(response, `{"code":"conflict"}`, http.StatusConflict)
				return
			}
			fmt.Fprint(response, `{"projectRevision":2,"items":[{"receiptId":"stable-page"}],"page":{"hasMore":false}}`)
		case "/v1/projects/project-script/finding-sources/stable-page":
			fmt.Fprint(response, `{"receiptId":"stable-page"}`)
		case "/v1/projects/project-script/finding-sources/stable-page/history":
			fmt.Fprint(response, `{"projectRevision":2,"items":[],"page":{"hasMore":false}}`)
		default:
			http.NotFound(response, request)
		}
	}))
	defer server.Close()

	script := filepath.Join("..", "..", "docs", "examples", "project-finding-backtrace.sh")
	command := exec.Command("bash", script)
	command.Env = append(os.Environ(),
		"CONTRACTOR_API_URL="+server.URL,
		"CONTRACTOR_API_TOKEN=test-token",
		"PROJECT_ID=project-script",
		"FINDING_ID=finding-script",
	)
	var output, diagnostics bytes.Buffer
	command.Stdout, command.Stderr = &output, &diagnostics
	if err := command.Run(); err != nil {
		t.Fatalf("backtrace script: %v\nstdout:\n%s\nstderr:\n%s", err, output.String(), diagnostics.String())
	}
	if provenanceCalls.Load() != 4 || strings.Contains(output.String(), "stale-page") ||
		!strings.Contains(output.String(), `"receiptId":"stable-page"`) ||
		!strings.Contains(output.String(), `"findingId":"finding-script"`) ||
		!strings.Contains(diagnostics.String(), "restarting snapshot (1/5)") {
		t.Fatalf("revision-bound traversal calls=%d\nstdout:\n%s\nstderr:\n%s",
			provenanceCalls.Load(), output.String(), diagnostics.String())
	}
}
