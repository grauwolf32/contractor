// Package configtest builds isolated catalogs for tests of policy overrides and budgets.
package configtest

import (
	"embed"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

//go:embed testdata/model-policies/*.yaml testdata/agent-templates/*.yaml testdata/workflows/*.yaml testdata/instructions/*.md testdata/escalation/workflows/*.yaml
var fixtures embed.FS

// CopyWithPolicies copies a catalog and overlays static testdata resources.
// The test artifact builder and inline escalation select stable test-* policies;
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

// CopyWithModelVariants adds two authored Workflow variants with equal budgets and contracts.
func CopyWithModelVariants(t testing.TB, source string) string {
	t.Helper()
	root := CopyWithPolicies(t, source)
	workflow, err := os.ReadFile(filepath.Join(root, "workflows/artifact_copy.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	policy, err := os.ReadFile(filepath.Join(root, "model-policies/test-worker.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	for _, variant := range []string{"a", "b"} {
		id := "test-variant-worker-" + variant
		body := strings.Replace(string(policy), "name: test-worker", "name: "+id, 1)
		body = strings.Replace(body, "model: worker-model", "model: variant-model-"+variant, 1)
		if err := os.WriteFile(filepath.Join(root, "model-policies/"+id+".yaml"), []byte(body), 0o600); err != nil {
			t.Fatal(err)
		}
		body = strings.Replace(string(workflow), "name: artifact-copy", "name: artifact-copy-"+variant, 1)
		body = strings.Replace(body, "modelPolicy: test-worker@1", "modelPolicy: "+id+"@1", 1)
		if err := os.WriteFile(filepath.Join(root, "workflows/artifact_copy_"+variant+".yaml"), []byte(body), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	return root
}
