//go:build e2e

package e2e

import (
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/grauwolf32/contractor/internal/configload"
	"go.yaml.in/yaml/v4"
)

func TestProductionMemoryConfigurationStaging(t *testing.T) {
	for _, configured := range []bool{true, false} {
		name := "configured instructions"
		if !configured {
			name = "legacy summarizer instructions omitted"
		}
		t.Run(name, func(t *testing.T) {
			source := t.TempDir()
			configRoot := filepath.Join(source, "configs")
			if err := os.CopyFS(configRoot, os.DirFS(filepath.Join(repoRoot(t), "configs"))); err != nil {
				t.Fatal(err)
			}
			path := filepath.Join(configRoot, "agent-templates/artifact_builder_v2_memory.yaml")
			data, err := os.ReadFile(path)
			if err != nil {
				t.Fatal(err)
			}
			var document map[string]any
			if err := yaml.Unmarshal(data, &document); err != nil {
				t.Fatal(err)
			}
			spec := document["spec"].(map[string]any)
			// Different refs catch accidental hard-coded dependency paths; the
			// production process test separately uses the unmodified catalog.
			workerRef := "instructions/staging-worker.md"
			spec["instructions"] = map[string]any{"ref": workerRef}
			if err := os.WriteFile(filepath.Join(configRoot, workerRef), []byte("Worker: literal {state}, Unicode Ж\n"), 0o600); err != nil {
				t.Fatal(err)
			}
			summarizer := spec["summarizer"].(map[string]any)
			if configured {
				ref := "instructions/staging-summarizer.md"
				summarizer["instructions"] = map[string]any{"ref": ref}
				if err := os.WriteFile(filepath.Join(configRoot, ref), []byte("Summarizer: preserve {state}, Unicode Ж\n"), 0o600); err != nil {
					t.Fatal(err)
				}
			} else {
				delete(summarizer, "instructions")
			}
			data, err = yaml.Marshal(document)
			if err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(path, data, 0o600); err != nil {
				t.Fatal(err)
			}
			expectedCatalog, err := configload.Load(configRoot)
			if err != nil {
				t.Fatal(err)
			}
			expected, err := expectedCatalog.AgentTemplate("artifact_builder@2")
			if err != nil {
				t.Fatal(err)
			}
			if (expected.Summarizer.Instructions != nil) != configured {
				t.Fatal("source catalog did not preserve configured versus omitted instructions")
			}
			target := stageE2EConfiguration(t, filepath.Join(configRoot, "e2e"), filepath.Join(t.TempDir(), "configs"), "http://127.0.0.1:1/v1")
			stageProductionMemoryConfiguration(t, source, target)
			actualCatalog, err := configload.Load(target)
			if err != nil {
				t.Fatal(err)
			}
			actual, err := actualCatalog.AgentTemplate("artifact_builder@2")
			if err != nil {
				t.Fatal(err)
			}
			gateway := newSharedMemoryGateway("test-only")
			t.Cleanup(gateway.close)
			gateway.workerModel = actual.ModelPolicy.Model
			for _, marker := range []string{
				"MEMORY_STREAMLINE_FIRST_WORKER", "MEMORY_STREAMLINE_RETRY_WORKER",
				"MEMORY_STREAMLINE_CONTINUE_WORKER", "MEMORY_ROUTER_BUILDER_WORKER",
				"MEMORY_ROUTER_REVIEWER_WORKER", "Copy the source artifact into the declared result artifact",
			} {
				_, worker, err := gateway.sharedMemoryScenario(expected.ModelPolicy.Model, marker)
				if err != nil || !worker {
					t.Fatalf("selected Worker model/context rejected: %v", err)
				}
				if _, _, err := gateway.sharedMemoryScenario(expected.ModelPolicy.Model+"-wrong", marker); err == nil {
					t.Fatal("accepted an unselected Worker model")
				}
			}
			finalizerInput, err := json.Marshal(workerResultFinalizerInput{
				SubtaskID: "0", ResultText: "streamline-first-worker updated isolated note",
			})
			if err != nil {
				t.Fatal(err)
			}
			request := map[string]any{
				"model": expected.ModelPolicy.Model,
				"messages": []any{map[string]any{
					"role": "user", "content": workerResultFinalizerMarker + string(finalizerInput),
				}},
			}
			_, reason, model, _, _, _, err := gateway.next(request)
			if err != nil || reason != "stop" || model != expected.ModelPolicy.Model {
				t.Fatalf("selected result-finalizer model rejected: model=%q reason=%q error=%v", model, reason, err)
			}
			request["model"] = expected.ModelPolicy.Model + "-wrong"
			if _, _, _, _, _, _, err := gateway.next(request); err == nil {
				t.Fatal("result finalizer accepted an unselected Worker model")
			}
			request["model"] = expected.ModelPolicy.Model
			request["tools"] = []any{map[string]any{"function": map[string]any{"name": "read_memory"}}}
			if _, _, _, _, _, _, err := gateway.next(request); err == nil {
				t.Fatal("result finalizer accepted model-visible tools")
			}
			if !reflect.DeepEqual(actual, expected) {
				t.Fatal("staging changed the resolved production template, instructions or digests")
			}
		})
	}
}
