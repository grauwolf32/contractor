//go:build e2e

package e2e

import (
	"net"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"testing"
)

func runCompositionBrowser(t *testing.T, h *scanProcessHarness, controlURL, token string) {
	t.Helper()
	uiRoot := filepath.Join(h.repositoryRoot, "ui")
	runChecked(t, uiRoot, nil, "node", "node_modules/vite/bin/vite.js", "build")
	internal, err := url.Parse(h.browserInternalURL)
	if err != nil {
		t.Fatal(err)
	}
	_, port, err := net.SplitHostPort(internal.Host)
	if err != nil {
		t.Fatal(err)
	}
	ui := startProcess(t, "composition browser UI", uiRoot, map[string]string{
		"NODE_ENV": "production", "CONTRACTOR_UI_API_BASE_URL": h.browserAPIURL,
		"CONTRACTOR_UI_HOST": "127.0.0.1", "CONTRACTOR_UI_PORT": port, "CONTRACTOR_UI_DIST_DIR": filepath.Join(uiRoot, "dist"),
	}, "node", "server/index.mjs")
	waitForHTTP(t, h.ctx, ui, http.DefaultClient, h.browserInternalURL+"/runtime-config.json", http.StatusOK)
	evidence := os.Getenv("CONTRACTOR_AUDIT_COMPOSITION_EVIDENCE_DIR")
	if evidence == "" {
		evidence = filepath.Join(h.temporaryRoot, "composition-browser")
	}
	evidence, err = filepath.Abs(evidence)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(evidence, 0700); err != nil {
		t.Fatal(err)
	}
	source := filepath.Join(h.temporaryRoot, "composition-source.zip")
	if err := os.WriteFile(source, compositionSourceArchive(t), 0600); err != nil {
		t.Fatal(err)
	}
	report := filepath.Join(evidence, "browser-report.json")
	runChecked(t, uiRoot, map[string]string{
		"CONTRACTOR_UI_E2E_BASE_URL": h.browserBaseURL, "CONTRACTOR_UI_E2E_API_URL": h.browserAPIURL,
		"CONTRACTOR_UI_E2E_USERNAME": "admin", "CONTRACTOR_UI_E2E_PASSWORD": "contractor e2e local password",
		"CONTRACTOR_COMPOSITION_SOURCE": source, "CONTRACTOR_COMPOSITION_TARGET": h.targetURL,
		"CONTRACTOR_COMPOSITION_CONTROL": controlURL, "CONTRACTOR_COMPOSITION_CONTROL_TOKEN": token,
		"CONTRACTOR_AUDIT_COMPOSITION_EVIDENCE_DIR": evidence, "CONTRACTOR_UI_E2E_OUTPUT_DIR": filepath.Join(evidence, "playwright"),
		"PLAYWRIGHT_JSON_OUTPUT_FILE": report,
	}, "node", "node_modules/@playwright/test/cli.js", "test", "--config=playwright.composition.config.ts", "--reporter=line,json")
	// The same report guard used by the existing browser stack rejects skips,
	// missing selections, failed attempts and an empty successful invocation.
	runChecked(t, uiRoot, nil, "node", "--input-type=module", "-e", `import {readFile} from 'node:fs/promises'; import {validateBrowserReport} from './server/browser-gate.mjs'; validateBrowserReport(JSON.parse(await readFile(process.argv[1],'utf8')),['audit-composition.production.ts']);`, report)
	t.Logf("real source-composition browser evidence: %s", evidence)
}
