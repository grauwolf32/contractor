//go:build e2e

package e2e

import (
	"context"
	"encoding/json"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func runToolsetBrowser(t *testing.T, root, temp, uiAddr, baseURL, endpoint, sourceRun string) {
	t.Helper()
	_, port, _ := net.SplitHostPort(uiAddr)
	ui := startProcess(t, "toolset Node UI", filepath.Join(root, "ui"), map[string]string{"NODE_ENV": "production", "CONTRACTOR_UI_API_BASE_URL": baseURL, "CONTRACTOR_UI_HOST": "127.0.0.1", "CONTRACTOR_UI_PORT": port, "CONTRACTOR_UI_DIST_DIR": filepath.Join(root, "ui/dist"), "CONTRACTOR_TEST_DATABASE_URL": ""}, "node", "server/index.mjs")
	ctx, cancel := context.WithTimeout(t.Context(), 10*time.Second)
	defer cancel()
	waitForHTTP(t, ctx, ui, &http.Client{Timeout: time.Second}, "http://"+uiAddr+"/runtime-config.json", 200)
	output := os.Getenv("CONTRACTOR_TOOLSET_BROWSER_REPORT")
	if output == "" {
		output = filepath.Join(temp, "toolset-browser.json")
	}
	runChecked(t, filepath.Join(root, "ui"), map[string]string{"CONTRACTOR_UI_E2E_BASE_URL": baseURL, "CONTRACTOR_UI_E2E_API_URL": baseURL, "CONTRACTOR_UI_E2E_OUTPUT_DIR": filepath.Join(temp, "browser-output"), "CONTRACTOR_TOOLSET_MCP_URL": endpoint, "CONTRACTOR_TOOLSET_SOURCE_RUN": sourceRun, "PLAYWRIGHT_JSON_OUTPUT_FILE": output, "CONTRACTOR_TEST_DATABASE_URL": ""}, "corepack", "pnpm", "exec", "playwright", "test", "e2e/toolset-connections.spec.ts", "--reporter=json")
	raw, err := os.ReadFile(output)
	if err != nil {
		t.Fatal(err)
	}
	var report struct {
		Stats struct{ Expected, Skipped, Unexpected, Flaky int }
	}
	if json.Unmarshal(raw, &report) != nil || report.Stats.Expected != 4 || report.Stats.Skipped != 0 || report.Stats.Unexpected != 0 || report.Stats.Flaky != 0 {
		t.Fatalf("mandatory browser cases did not all pass: %+v", report.Stats)
	}
	t.Log("4 production browser cases passed, zero skips or retries")
}
