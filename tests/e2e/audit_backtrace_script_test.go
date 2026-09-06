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

func TestAuditFindingBacktraceScriptRestartsRevisionBoundTraversal(t *testing.T) {
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
		case "/v1/audits/audit-script":
			fmt.Fprintf(response, `{"auditId":"audit-script","revision":%d}`, revision)
		case "/v1/audits/audit-script/findings/finding-script":
			fmt.Fprintf(response, `{"findingId":"finding-script","state":"proposed","revision":%d}`, revision)
		case "/v1/audits/audit-script/findings/finding-script/provenance":
			provenanceCalls.Add(1)
			if request.URL.Query().Get("auditRevision") != fmt.Sprint(revision) ||
				request.URL.Query().Get("findingRevision") != fmt.Sprint(revision) {
				http.Error(response, `{"code":"conflict"}`, http.StatusConflict)
				return
			}
			if revision == 1 && request.URL.Query().Get("cursor") == "" {
				fmt.Fprint(response, `{"items":[{"recordId":"stale-page"}],"page":{"hasMore":true,"nextCursor":"stale-cursor"}}`)
				return
			}
			if revision == 1 {
				generation.Store(2)
				http.Error(response, `{"code":"conflict"}`, http.StatusConflict)
				return
			}
			fmt.Fprint(response, `{"items":[{"recordId":"stable-page"}],"page":{"hasMore":false}}`)
		default:
			http.NotFound(response, request)
		}
	}))
	defer server.Close()

	script := filepath.Join("..", "..", "docs", "examples", "audit-finding-backtrace.sh")
	command := exec.Command("bash", script)
	command.Env = append(os.Environ(),
		"CONTRACTOR_API_URL="+server.URL,
		"CONTRACTOR_API_TOKEN=test-token",
		"AUDIT_ID=audit-script",
		"FINDING_ID=finding-script",
	)
	var output, diagnostics bytes.Buffer
	command.Stdout, command.Stderr = &output, &diagnostics
	if err := command.Run(); err != nil {
		t.Fatalf("backtrace script: %v\nstdout:\n%s\nstderr:\n%s", err, output.String(), diagnostics.String())
	}
	if provenanceCalls.Load() != 3 || strings.Contains(output.String(), "stale-page") ||
		!strings.Contains(output.String(), `"recordId":"stable-page"`) ||
		!strings.Contains(output.String(), `"findingId": "finding-script"`) ||
		!strings.Contains(diagnostics.String(), "restarting snapshot (1/5)") {
		t.Fatalf("revision-bound traversal calls=%d\nstdout:\n%s\nstderr:\n%s",
			provenanceCalls.Load(), output.String(), diagnostics.String())
	}
}
