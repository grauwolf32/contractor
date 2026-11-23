package agentinstructions

import (
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"go.yaml.in/yaml/v4"
)

// Archive documents remain data: no alias rewriting or current catalog loading.
func archiveDocument(t *testing.T, data []byte) map[string]any {
	t.Helper()
	var document map[string]any
	if err := yaml.Unmarshal(data, &document); err != nil {
		t.Fatal(err)
	}
	return document
}

// Keep raw bytes so each comparison gets its own detached document.
func archivedCatalog(t *testing.T, frozen map[string]string, overlays ...string) map[string][]byte {
	t.Helper()
	catalog := map[string][]byte{}
	add := func(data []byte) {
		document := archiveDocument(t, data)
		metadata := document["metadata"].(map[string]any)
		key := fmt.Sprintf("%s/%s@%s", document["kind"], metadata["name"], metadata["version"])
		if _, exists := catalog[key]; exists {
			t.Fatalf("duplicate archived identity %s", key)
		}
		catalog[key] = data
	}
	for path, raw := range frozen {
		if filepath.Ext(path) == ".yaml" {
			add([]byte(raw))
		}
	}
	for _, directory := range overlays {
		paths, err := filepath.Glob(filepath.Join(directory, "*", "*.yaml"))
		if err != nil {
			t.Fatal(err)
		}
		for _, path := range paths {
			data, err := os.ReadFile(path)
			if err != nil {
				t.Fatal(err)
			}
			add(data)
		}
	}
	return catalog
}

func archivedSelection(t *testing.T, catalog map[string][]byte, kind, selector string) map[string]any {
	t.Helper()
	raw, ok := catalog[kind+"/"+selector]
	if !ok {
		t.Fatalf("missing archived %s %s", kind, selector)
	}
	return archiveDocument(t, raw)
}

func assertArchivedTemplateContract(t *testing.T, before, after map[string]any) {
	t.Helper()
	// Different versions and instruction references are the only template changes.
	after["metadata"].(map[string]any)["version"] = before["metadata"].(map[string]any)["version"]
	after["spec"].(map[string]any)["instructions"] = before["spec"].(map[string]any)["instructions"]
	if !reflect.DeepEqual(before, after) {
		t.Fatalf("%v changes archived execution capabilities or policy beyond instructions", after["metadata"])
	}
}

func assertArchivedWorkflowContract(t *testing.T, catalog map[string][]byte, before, after map[string]any) {
	t.Helper()
	after["metadata"] = before["metadata"]
	beforeStages := before["spec"].(map[string]any)["stages"].(map[string]any)
	afterStages := after["spec"].(map[string]any)["stages"].(map[string]any)
	for name, raw := range afterStages {
		stage := raw.(map[string]any)
		priorRaw, ok := beforeStages[name]
		if !ok {
			t.Fatalf("unexpected archived Stage %s", name)
		}
		prior := priorRaw.(map[string]any)
		stage["instructions"] = prior["instructions"]
		for role, bindingRaw := range stage["agents"].(map[string]any) {
			binding := bindingRaw.(map[string]any)
			oldRaw, ok := prior["agents"].(map[string]any)[role]
			if !ok {
				t.Fatalf("unexpected archived Agent role %s", role)
			}
			old := oldRaw.(map[string]any)
			assertArchivedTemplateContract(t,
				archivedSelection(t, catalog, "AgentTemplate", old["template"].(string)),
				archivedSelection(t, catalog, "AgentTemplate", binding["template"].(string)))
			binding["template"] = old["template"]
		}
	}
	if !reflect.DeepEqual(before, after) {
		t.Fatal("archived A/B differs beyond instructions and dependent refs")
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
