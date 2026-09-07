// Package configtest builds isolated catalogs for tests of policy overrides and budgets.
package configtest

import (
	"embed"
	"io/fs"
	"os"
	"path/filepath"
	"testing"
)

//go:embed testdata/model-policies/*.yaml testdata/agent-templates/*.yaml testdata/execution-configs/*.yaml testdata/escalation/workflows/*.yaml
var fixtures embed.FS

// CopyWithPolicies copies a catalog and overlays static testdata resources.
// The test artifact builder and escalation profile select stable test-* policies;
// the remaining manifests retain their production selections.
func CopyWithPolicies(t testing.TB, source string) string {
	t.Helper()
	root := filepath.Join(t.TempDir(), "configs")
	if err := os.CopyFS(root, os.DirFS(source)); err != nil {
		t.Fatal(err)
	}
	data, err := fs.Sub(fixtures, "testdata")
	if err != nil {
		t.Fatal(err)
	}
	err = fs.WalkDir(data, ".", func(path string, entry fs.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		target := filepath.Join(root, filepath.FromSlash(path))
		if entry.IsDir() {
			if path == "escalation" {
				return fs.SkipDir
			}
			return os.MkdirAll(target, 0o755)
		}
		content, err := fs.ReadFile(data, path)
		if err != nil {
			return err
		}
		return os.WriteFile(target, content, 0o644)
	})
	if err != nil {
		t.Fatal(err)
	}
	return root
}

// CopyWithEscalation adds a minimal validation Workflow to the isolated catalog.
// The production OpenAPI Workflows keep their ordinary retry transitions.
func CopyWithEscalation(t testing.TB, source string) string {
	t.Helper()
	root := CopyWithPolicies(t, source)
	const relative = "workflows/openapi_from_workspace_v5.yaml"
	content, err := fixtures.ReadFile("testdata/escalation/" + relative)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, relative), content, 0o644); err != nil {
		t.Fatal(err)
	}
	return root
}
