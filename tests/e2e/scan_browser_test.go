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

// The browser uses real HTTPS origins and the ordinary session/CSRF boundary.
// Only the loopback HTTP transport is proxied; no API response is fabricated.
func configureScanBrowser(t *testing.T, h *scanProcessHarness) {
	t.Helper()
	proxy := func(target string) string {
		u, err := url.Parse(target)
		if err != nil {
			t.Fatal(err)
		}
		server := httptest.NewTLSServer(httputil.NewSingleHostReverseProxy(u))
		t.Cleanup(server.Close)
		return server.URL
	}
	h.browserInternalURL = "http://" + freeAddress(t)
	h.browserBaseURL = proxy(h.browserInternalURL)
	h.browserAPIURL = proxy(h.baseURL)
}

func runScanBrowser(t *testing.T, h *scanProcessHarness) {
	t.Helper()
	uiRoot := filepath.Join(h.repositoryRoot, "ui")
	if _, err := os.Stat(filepath.Join(uiRoot, "node_modules", ".bin", "playwright")); err != nil {
		t.Fatal("browser scan gate requires installed UI dependencies and Chromium")
	}
	if os.Getenv("CONTRACTOR_SCAN_UI_PREBUILT") != "1" {
		runChecked(t, uiRoot, nil, "node", "node_modules/vite/bin/vite.js", "build")
	}
	internal, err := url.Parse(h.browserInternalURL)
	if err != nil {
		t.Fatal(err)
	}
	_, port, err := net.SplitHostPort(internal.Host)
	if err != nil {
		t.Fatal(err)
	}
	ui := startProcess(t, "scan browser UI", uiRoot, map[string]string{
		"NODE_ENV":                   "production",
		"CONTRACTOR_UI_API_BASE_URL": h.browserAPIURL,
		"CONTRACTOR_UI_HOST":         "127.0.0.1",
		"CONTRACTOR_UI_PORT":         port,
		"CONTRACTOR_UI_DIST_DIR":     filepath.Join(uiRoot, "dist"),
	}, "node", "server/index.mjs")
	waitForHTTP(t, h.ctx, ui, http.DefaultClient, h.browserInternalURL+"/runtime-config.json", http.StatusOK)
	evidence := os.Getenv("CONTRACTOR_SCAN_EVIDENCE_DIR")
	if evidence == "" {
		evidence = filepath.Join(h.temporaryRoot, "scan-browser-evidence")
	}
	evidence, err = filepath.Abs(evidence)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(evidence, 0o700); err != nil {
		t.Fatal(err)
	}
	runChecked(t, uiRoot, map[string]string{
		"CONTRACTOR_SCAN_BROWSER":      "1",
		"CONTRACTOR_UI_E2E_BASE_URL":   h.browserBaseURL,
		"CONTRACTOR_UI_E2E_API_URL":    h.browserAPIURL,
		"CONTRACTOR_UI_E2E_USERNAME":   "admin",
		"CONTRACTOR_UI_E2E_PASSWORD":   "contractor e2e local password",
		"CONTRACTOR_SCAN_TARGET_URL":   h.targetURL,
		"CONTRACTOR_SCAN_EVIDENCE_DIR": evidence,
		"CONTRACTOR_UI_E2E_OUTPUT_DIR": filepath.Join(evidence, "playwright"),
	}, "node", "node_modules/@playwright/test/cli.js", "test", "e2e/scan-tools.spec.ts")
	t.Logf("real scanner browser evidence: %s", evidence)
}
