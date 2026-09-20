package agentinstructions

import (
	"io/fs"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"go.yaml.in/yaml/v4"
)

// Archived profiles are checked as frozen experiment data. Loading them through
// today's strict profile parser would require keeping obsolete production schemas.
func copyArchivedExecutionConfigs(t *testing.T, destination, source string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Join(destination, "audit-profiles"), 0700); err != nil {
		t.Fatal(err)
	}
	err := fs.WalkDir(os.DirFS(source), ".", func(path string, entry fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if entry.IsDir() {
			if path == "audit-profiles" {
				return fs.SkipDir
			}
			return os.MkdirAll(filepath.Join(destination, path), 0700)
		}
		data, err := os.ReadFile(filepath.Join(source, path))
		if err != nil {
			return err
		}
		return os.WriteFile(filepath.Join(destination, path), data, 0600)
	})
	if err != nil {
		t.Fatal(err)
	}
}

func assertArchivedProfileWrapper(t *testing.T, baseline, path, selector, workflow string) {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var original, wrapper map[string]any
	if err := yaml.Unmarshal([]byte(baseline), &original); err != nil {
		t.Fatal(err)
	}
	if err := yaml.Unmarshal(data, &wrapper); err != nil {
		t.Fatal(err)
	}
	metadata := wrapper["metadata"].(map[string]any)
	if metadata["name"].(string)+"@"+metadata["version"].(string) != selector {
		t.Fatal("profile wrapper identity differs from the release manifest")
	}
	checkBinding := func(document map[string]any) map[string]any {
		return document["spec"].(map[string]any)["workflows"].(map[string]any)["check"].(map[string]any)
	}
	binding := checkBinding(wrapper)
	if binding["ref"] != workflow {
		t.Fatal("profile must pin its arm's exact child workflow")
	}
	wrapper["metadata"] = original["metadata"]
	binding["ref"] = checkBinding(original)["ref"]
	if !reflect.DeepEqual(original, wrapper) {
		t.Fatal("archived Audit wrapper changes policy, inventory or evidence mapping")
	}
}
