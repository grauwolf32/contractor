//go:build e2e

package e2e

import (
	"net"
	"net/http"
	"net/http/httptest"
	"net/http/httputil"
	"net/url"
	"os"
	"path/filepath"
	"testing"
)

type researchBrowserTransport struct{ internalURL, baseURL, apiURL, root string }

// Real HTTPS transports preserve the Server's browser session/CSRF boundary.
// The proxies forward bytes unchanged; no API responses are mocked.
func configureResearchBrowser(t *testing.T, api, root string) *researchBrowserTransport {
	t.Helper()
	proxy := func(target string) string {
		parsed, err := url.Parse(target)
		if err != nil {
			t.Fatal(err)
		}
		server := httptest.NewTLSServer(httputil.NewSingleHostReverseProxy(parsed))
		t.Cleanup(server.Close)
		return server.URL
	}
	internal := "http://" + freeAddress(t)
	return &researchBrowserTransport{internalURL: internal, baseURL: proxy(internal), apiURL: proxy(api), root: root}
}

func runResearchBrowser(t *testing.T, h *findingsHarness, projectID, auditID, scenario string) {
	t.Helper()
	transport := h.browser
	if transport == nil {
		t.Fatal("research browser transport was not configured before Server startup")
	}
	uiRoot := filepath.Join(repoRoot(t), "ui")
	dist := filepath.Join(transport.root, "ui-dist")
	runChecked(t, uiRoot, nil, "node", "node_modules/vite/bin/vite.js", "build", "--outDir", dist)
	parsed, err := url.Parse(transport.internalURL)
	if err != nil {
		t.Fatal(err)
	}
	_, port, err := net.SplitHostPort(parsed.Host)
	if err != nil {
		t.Fatal(err)
	}
	ui := startProcess(t, "research browser UI", uiRoot, map[string]string{
		"NODE_ENV": "production", "CONTRACTOR_UI_API_BASE_URL": transport.apiURL,
		"CONTRACTOR_UI_HOST": "127.0.0.1", "CONTRACTOR_UI_PORT": port, "CONTRACTOR_UI_DIST_DIR": dist,
	}, "node", "server/index.mjs")
	waitForHTTP(t, h.ctx, ui, http.DefaultClient, transport.internalURL+"/runtime-config.json", http.StatusOK)
	evidence := os.Getenv("CONTRACTOR_AUDIT_RESEARCH_EVIDENCE_DIR")
	if evidence == "" {
		evidence = filepath.Join(transport.root, "research-browser")
	}
	evidence, err = filepath.Abs(filepath.Join(evidence, scenario))
	if err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(evidence, 0700); err != nil {
		t.Fatal(err)
	}
	report := filepath.Join(evidence, "browser-report.json")
	runChecked(t, uiRoot, map[string]string{
		"CONTRACTOR_UI_E2E_BASE_URL": transport.baseURL, "CONTRACTOR_UI_E2E_API_URL": transport.apiURL,
		"CONTRACTOR_UI_E2E_USERNAME": "admin", "CONTRACTOR_UI_E2E_PASSWORD": "contractor e2e local password",
		"CONTRACTOR_RESEARCH_PROJECT": projectID, "CONTRACTOR_RESEARCH_AUDIT": auditID, "CONTRACTOR_RESEARCH_SCENARIO": scenario,
		"CONTRACTOR_AUDIT_RESEARCH_EVIDENCE_DIR": evidence, "CONTRACTOR_UI_E2E_OUTPUT_DIR": filepath.Join(evidence, "playwright"),
		"PLAYWRIGHT_JSON_OUTPUT_FILE": report,
	}, "node", "node_modules/@playwright/test/cli.js", "test", "--config=playwright.research.config.ts", "--reporter=line,json")
	runChecked(t, uiRoot, nil, "node", "--input-type=module", "-e", `import {readFile} from 'node:fs/promises'; import {validateBrowserReport} from './server/browser-gate.mjs'; validateBrowserReport(JSON.parse(await readFile(process.argv[1],'utf8')),['audit-research.production.ts']);`, report)
	t.Logf("real Audit research browser evidence: %s", evidence)
}
