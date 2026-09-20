//go:build e2e

package e2e

import (
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/grauwolf32/contractor/internal/config"
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
			expectedCatalog, err := config.Load(configRoot, config.MVPDescriptors())
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
			actualCatalog, err := config.Load(target, config.MVPDescriptors())
			if err != nil {
				t.Fatal(err)
			}
			actual, err := actualCatalog.AgentTemplate("artifact_builder@2")
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(actual, expected) {
				t.Fatal("staging changed the resolved production template, instructions or digests")
			}
		})
	}
}
