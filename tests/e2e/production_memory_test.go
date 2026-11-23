//go:build e2e

package e2e

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/configload"
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
			Summarizer   *struct {
				Instructions struct{ Ref string }
			}
		}
	}
	template = "artifact_builder@1"
	if err := yaml.Unmarshal(copyFile("configs/agent-templates/artifact_builder.yaml"), &selected); err != nil {
		t.Fatal(err)
	}
	workflow = "artifact-report-memory-fixture@1"
	fixture, err := os.ReadFile(filepath.Join(repoRoot(t), "tests/e2e/testdata/production-memory/workflow.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(target, "workflows", "artifact_report_memory.yaml"), fixture, 0600); err != nil {
		t.Fatal(err)
	}
	if template == "" || workflow == "" {
		t.Fatal("production Memory successors missing")
	}
	copyFile("configs/" + selected.Spec.Instructions.Ref)
	copyFile("configs/instructions/report-planner.md")
	var fixturePins struct {
		Spec struct {
			ExecutionConfig struct {
				Workers struct {
					ModelPolicy string `yaml:"modelPolicy"`
				}
				Stages map[string]struct {
					Agents map[string]struct {
						SummarizerModelPolicy string `yaml:"summarizerModelPolicy"`
					}
				}
			} `yaml:"executionConfig"`
		}
	}
	if err := yaml.Unmarshal(fixture, &fixturePins); err != nil {
		t.Fatal(err)
	}
	mainPolicy := fixturePins.Spec.ExecutionConfig.Workers.ModelPolicy
	policies := map[string]bool{mainPolicy: true}
	summaryPolicy := ""
	if selected.Spec.Summarizer != nil {
		summaryPolicy = fixturePins.Spec.ExecutionConfig.Stages["copy"].Agents["builder"].SummarizerModelPolicy
		if summaryPolicy == "" {
			t.Fatal("report fixture has no summary policy")
		}
		policies[summaryPolicy] = true
		if ref := selected.Spec.Summarizer.Instructions.Ref; ref != "" {
			copyFile("configs/" + ref)
		}
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
		var authored map[string]any
		if err := yaml.Unmarshal([]byte(text), &authored); err != nil {
			t.Fatal(err)
		}
		workers := authored["spec"].(map[string]any)["executionConfig"].(map[string]any)["workers"].(map[string]any)
		workers["modelPolicy"] = mainPolicy
		if summaryPolicy != "" {
			workers["summarizerModelPolicy"] = summaryPolicy
		}
		encoded, err := yaml.Marshal(authored)
		if err != nil {
			t.Fatal(err)
		}
		text = string(encoded)
		if err := os.WriteFile(path, []byte(text), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	if _, _, err := configload.LoadUnionReadOnly(target, filepath.Join(t.TempDir(), "managed")); err != nil {
		t.Fatal(err)
	}
	return workflow
}

func productionMemoryCopyResponse(step int, request map[string]any) (map[string]any, string, string, error) {
	switch step {
	case 1:
		return gatewayToolCall("production-read", "read_artifact", map[string]any{"namespace": "inputs", "name": "source"})
	case 2:
		if data, ok := lastStringValue(request, "text"); !ok || data == "" {
			return nil, "", "", fmt.Errorf("production input not observed")
		}
		return gatewayToolCall("production-note", "write_memory", map[string]any{"name": "copy_progress", "content": "Input inspected; produce the declared copied artifact"})
	case 3:
		if err := requireMemoryNote(request, "production-note", "copy_progress", "Input inspected; produce the declared copied artifact", "", []string{}, 0); err != nil {
			return nil, "", "", err
		}
		_, ok := lastStringValue(request, "text")
		if !ok {
			return nil, "", "", fmt.Errorf("production input lost after Memory write")
		}
		return gatewayToolCall("production-write", "write_text_artifact", map[string]any{"name": "copied", "media_type": "text/plain", "text": e2eReport})
	case 4:
		if _, ok := lastExactArtifact(request, "builder", "copied"); !ok {
			return nil, "", "", fmt.Errorf("production output exact revision not observed")
		}
		return map[string]any{"role": "assistant", "content": "production-copy-worker updated isolated note"}, "stop", "<final>", nil
	default:
		return nil, "", "", fmt.Errorf("production copy exceeded script: %d", step)
	}
}
