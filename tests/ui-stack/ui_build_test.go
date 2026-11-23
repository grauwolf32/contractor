//go:build e2e

package uistack

import (
	"bytes"
	"context"
	"io"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"testing"
	"time"
)

func buildBrowserUI(t *testing.T, uiRoot, temporaryRoot string) string {
	t.Helper()
	environment := cleanEnvironment(nil, "PATH", "HOME", "XDG_CACHE_HOME", "COREPACK_HOME", "TMPDIR", "LANG", "LC_ALL", "TZ")
	dist := filepath.Join(temporaryRoot, "ui-dist")
	runChecked(t, uiRoot, environment, "node", filepath.Join(uiRoot, "node_modules", "vite", "bin", "vite.js"), "build", "--outDir", dist)
	return dist
}

// Exercise Vite and the production static server, with every mutable build
// inside the staged fixture. A sibling build must not invalidate a live index.
func TestBrowserUIBuildSurvivesSiblingRebuild(t *testing.T) {
	repository := repoRoot(t)
	temporary := t.TempDir()
	stagedUI := filepath.Join(temporary, "ui")
	if err := os.MkdirAll(stagedUI, 0700); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(filepath.Join(repository, "ui", "node_modules"), filepath.Join(stagedUI, "node_modules")); err != nil {
		t.Fatal(err)
	}
	write := func(name, content string) {
		t.Helper()
		if err := os.WriteFile(filepath.Join(stagedUI, name), []byte(content), 0600); err != nil {
			t.Fatal(err)
		}
	}
	write("package.json", `{"private":true,"type":"module"}`)
	write("index.html", `<html><body><script type="module" src="/entry.js"></script></body></html>`)
	write("entry.js", `document.body.dataset.build = "original";`)
	dist := buildBrowserUI(t, stagedUI, temporary)

	address := freeAddress(t, "127.0.0.1")
	_, port, err := net.SplitHostPort(address)
	if err != nil {
		t.Fatal(err)
	}
	server := startProcess(t, "isolated UI build regression", filepath.Join(repository, "ui"),
		cleanEnvironment(map[string]string{
			"NODE_ENV":                   "production",
			"CONTRACTOR_UI_API_BASE_URL": "http://127.0.0.1:1",
			"CONTRACTOR_UI_HOST":         "127.0.0.1",
			"CONTRACTOR_UI_PORT":         port,
			"CONTRACTOR_UI_DIST_DIR":     dist,
		}, "PATH", "LANG", "LC_ALL", "TZ"), "node", "server/index.mjs")
	t.Cleanup(func() { server.stop(t) })
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	client := &http.Client{Timeout: 5 * time.Second}
	base := "http://" + address
	waitForHTTP(t, ctx, server, client, base+"/healthz", http.StatusOK, nil)
	get := func(path string) []byte {
		t.Helper()
		response, err := client.Get(base + path)
		if err != nil {
			t.Fatal(err)
		}
		defer response.Body.Close()
		data, err := io.ReadAll(io.LimitReader(response.Body, 128<<10))
		if err != nil {
			t.Fatal(err)
		}
		if response.StatusCode != http.StatusOK {
			t.Fatalf("live UI asset %s returned %d after a sibling build", path, response.StatusCode)
		}
		return data
	}
	index := get("/")
	match := regexp.MustCompile(`<script\b[^>]*\bsrc="([^"]+)"`).FindSubmatch(index)
	if len(match) != 2 {
		t.Fatalf("Vite index has no module script: %s", index)
	}
	script := string(match[1])
	original := get(script)
	if !bytes.Contains(original, []byte("original")) {
		t.Fatal("original build was not served")
	}
	write("entry.js", `document.body.dataset.build = "replacement";`)
	runChecked(t, stagedUI, cleanEnvironment(nil, "PATH", "HOME", "TMPDIR"),
		"node", filepath.Join(stagedUI, "node_modules", "vite", "bin", "vite.js"), "build")
	if !bytes.Equal(index, get("/")) || !bytes.Equal(original, get(script)) {
		t.Fatal("a sibling build replaced a live UI snapshot")
	}
}
