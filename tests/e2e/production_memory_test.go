//go:build e2e

package e2e

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/config"
	"go.yaml.in/yaml/v4"
)

func TestProductionMemoryTemplatesAcrossProcesses(t *testing.T) {
	runSharedMemoryProcesses(t, true)
}

// Stage exact working template and Workflow bytes, including their selected
// policy and instructions. Only the test Gateway URL is substituted by the harness.
func stageProductionMemoryConfiguration(t *testing.T, repositoryRoot, target string) string {
	t.Helper()
	// Keep this process catalog scoped to the Memory scenarios. Unrelated e2e
	// templates can require different bodies of the same test-only policy selector.
	for _, group := range []string{"agent-templates", "workflows", "audit-profiles"} {
		directory := filepath.Join(target, group)
		entries, err := os.ReadDir(directory)
		if os.IsNotExist(err) {
			continue
		}
		if err != nil {
			t.Fatal(err)
		}
		for _, entry := range entries {
			if group == "workflows" && (entry.Name() == "shared_memory_streamline.yaml" || entry.Name() == "shared_memory_router.yaml") {
				continue
			}
			if err := os.RemoveAll(filepath.Join(directory, entry.Name())); err != nil {
				t.Fatal(err)
			}
		}
	}
	var catalog struct {
		Templates []struct {
			Legacy, Active string
			ActiveFile     string `json:"active_file"`
		} `json:"templates"`
		Workflows []struct {
			Legacy, Active string
			ActiveFile     string `json:"active_file"`
		} `json:"workflows"`
	}
	data, err := os.ReadFile(filepath.Join(repositoryRoot, "configs/memory-catalog.json"))
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(data, &catalog); err != nil {
		t.Fatal(err)
	}
	template, workflow := "", ""
	copyFile := func(relative string) []byte {
		data, err := os.ReadFile(filepath.Join(repositoryRoot, relative))
		if err != nil {
			t.Fatal(err)
		}
		dest := filepath.Join(target, strings.TrimPrefix(relative, "configs/"))
		if err := os.MkdirAll(filepath.Dir(dest), 0o700); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(dest, data, 0o600); err != nil {
			t.Fatal(err)
		}
		return data
	}
	var selected struct {
		Spec struct {
			Instructions struct{ Ref string }
			ModelPolicy  string `yaml:"modelPolicy"`
			Summarizer   *struct {
				ModelPolicy string `yaml:"modelPolicy"`
			}
		}
	}
	for _, entry := range catalog.Templates {
		if entry.Legacy == "artifact_builder@1" {
			template = entry.Active
			if err := yaml.Unmarshal(copyFile(entry.ActiveFile), &selected); err != nil {
				t.Fatal(err)
			}
		}
	}
	for _, entry := range catalog.Workflows {
		if entry.Legacy == "artifact-copy@1" {
			workflow = entry.Active
			copyFile(entry.ActiveFile)
		}
	}
	if template == "" || workflow == "" {
		t.Fatal("production Memory successors missing")
	}
	copyFile("configs/" + selected.Spec.Instructions.Ref)
	copyFile("configs/instructions/copy-planner.md")
	policies := map[string]bool{selected.Spec.ModelPolicy: true}
	if selected.Spec.Summarizer != nil {
		policies[selected.Spec.Summarizer.ModelPolicy] = true
	}
	files, err := filepath.Glob(filepath.Join(repositoryRoot, "configs/model-policies/*.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	for _, file := range files {
		data, err := os.ReadFile(file)
		if err != nil {
			t.Fatal(err)
		}
		var doc struct {
			Metadata struct{ Name, Version string }
		}
		if err := yaml.Unmarshal(data, &doc); err != nil {
			t.Fatal(err)
		}
		selector := doc.Metadata.Name + "@" + doc.Metadata.Version
		if policies[selector] {
			destination := filepath.Join(target, "model-policies", filepath.Base(file))
			if existing, err := os.ReadFile(destination); err == nil {
				var prior struct {
					Metadata struct{ Name, Version string }
				}
				if err := yaml.Unmarshal(existing, &prior); err != nil {
					t.Fatal(err)
				}
				if prior.Metadata.Name+"@"+prior.Metadata.Version != selector {
					destination = filepath.Join(target, "model-policies", "production_"+filepath.Base(file))
				}
			}
			if err := os.WriteFile(destination, data, 0o600); err != nil {
				t.Fatal(err)
			}
			delete(policies, selector)
		}
	}
	if len(policies) != 0 {
		t.Fatalf("production policies missing: %v", policies)
	}
	for _, file := range []string{"shared_memory_streamline.yaml", "shared_memory_router.yaml"} {
		path := filepath.Join(target, "workflows", file)
		data, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		text := strings.ReplaceAll(string(data), "shared_memory_worker@1", template)
		text = strings.ReplaceAll(text, "shared_memory_reviewer@1", template)
		if err := os.WriteFile(path, []byte(text), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	if _, err := config.Load(target, config.MVPDescriptors()); err != nil {
		t.Fatal(err)
	}
	return workflow
}

func productionMemoryCopyResponse(step int, request map[string]any) (map[string]any, string, string, error) {
	switch step {
	case 1:
		return gatewayToolCall("production-read", "read_artifact", map[string]any{"namespace": "inputs", "name": "source", "revision": nil})
	case 2:
		if data, ok := lastStringValue(request, "dataBase64"); !ok || data == "" {
			return nil, "", "", fmt.Errorf("production input not observed")
		}
		return gatewayToolCall("production-note", "write_memory", map[string]any{"name": "copy_progress", "content": "Input inspected; produce the declared copied artifact"})
	case 3:
		if err := requireMemoryNote(request, "production-note", "copy_progress", "Input inspected; produce the declared copied artifact", "", []string{}, 0); err != nil {
			return nil, "", "", err
		}
		data, ok := lastStringValue(request, "dataBase64")
		if !ok {
			return nil, "", "", fmt.Errorf("production input lost after Memory write")
		}
		return gatewayToolCall("production-write", "write_artifact", map[string]any{"namespace": "builder", "name": "copied", "media_type": "text/plain", "data_base64": data, "expected_revision": nil})
	case 4:
		if _, ok := lastExactArtifact(request, "builder", "copied"); !ok {
			return nil, "", "", fmt.Errorf("production output exact revision not observed")
		}
		return map[string]any{"role": "assistant", "content": "production-copy-worker updated isolated note"}, "stop", "<final>", nil
	default:
		return nil, "", "", fmt.Errorf("production copy exceeded script: %d", step)
	}
}
