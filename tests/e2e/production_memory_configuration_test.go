//go:build e2e

package e2e

import (
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/configload"
	"github.com/grauwolf32/contractor/internal/configload/configloadtest"
	"go.yaml.in/yaml/v4"
)

func TestProductionMemoryConfigurationStaging(t *testing.T) {
	assertProductionPlannerCompactionCatalog(t)
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
			// The shipped source was validated above. This private fixture
			// deliberately tests different instruction references.
			if err := os.Remove(filepath.Join(configRoot, "catalog.lock.json")); err != nil {
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
			expectedCatalog, _, err := configload.LoadUnionReadOnly(configRoot, filepath.Join(t.TempDir(), "managed"))
			if err != nil {
				t.Fatal(err)
			}
			expected, err := configloadtest.AgentTemplate(expectedCatalog, "artifact_builder@2")
			if err != nil {
				t.Fatal(err)
			}
			if (expected.Summarizer.Instructions != nil) != configured {
				t.Fatal("source catalog did not preserve configured versus omitted instructions")
			}
			target := stageE2EConfiguration(t, filepath.Join(configRoot, "e2e"), filepath.Join(t.TempDir(), "configs"), "http://127.0.0.1:1/v1")
			stageProductionMemoryConfiguration(t, source, target)
			actualCatalog, _, err := configload.LoadUnionReadOnly(target, filepath.Join(t.TempDir(), "managed"))
			if err != nil {
				t.Fatal(err)
			}
			actual, err := configloadtest.AgentTemplate(actualCatalog, "artifact_builder@2")
			if err != nil {
				t.Fatal(err)
			}
			gateway := newSharedMemoryGateway("test-only")
			t.Cleanup(gateway.close)
			expectedWorkflow, err := expectedCatalog.Workflow("artifact-copy@2")
			if err != nil {
				t.Fatal(err)
			}
			actualWorkflow, err := actualCatalog.Workflow("artifact-copy@2")
			if err != nil {
				t.Fatal(err)
			}
			expectedPolicy := expectedWorkflow.Stages["copy"].ExecutionConfig.Agents["builder"].ModelPolicy
			actualPolicy := actualWorkflow.Stages["copy"].ExecutionConfig.Agents["builder"].ModelPolicy
			if !reflect.DeepEqual(actualPolicy, expectedPolicy) {
				t.Fatal("staging changed Workflow policy")
			}
			gateway.workerModel = actualPolicy.Model
			for _, marker := range []string{
				"MEMORY_STREAMLINE_FIRST_WORKER", "MEMORY_STREAMLINE_RETRY_WORKER",
				"MEMORY_STREAMLINE_CONTINUE_WORKER", "MEMORY_ROUTER_BUILDER_WORKER",
				"MEMORY_ROUTER_REVIEWER_WORKER", "Copy the source artifact into the declared result artifact",
			} {
				_, worker, err := gateway.sharedMemoryScenario(expectedPolicy.Model, marker)
				if err != nil || !worker {
					t.Fatalf("selected Worker model/context rejected: %v", err)
				}
				if _, _, err := gateway.sharedMemoryScenario(expectedPolicy.Model+"-wrong", marker); err == nil {
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
				"model": expectedPolicy.Model,
				"messages": []any{map[string]any{
					"role": "user", "content": workerResultFinalizerMarker + string(finalizerInput),
				}},
			}
			_, reason, model, _, _, _, err := gateway.next(request)
			if err != nil || reason != "stop" || model != expectedPolicy.Model {
				t.Fatalf("selected result-finalizer model rejected: model=%q reason=%q error=%v", model, reason, err)
			}
			request["model"] = expectedPolicy.Model + "-wrong"
			if _, _, _, _, _, _, err := gateway.next(request); err == nil {
				t.Fatal("result finalizer accepted an unselected Worker model")
			}
			request["model"] = expectedPolicy.Model
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

// Read the shipped catalog only in this e2e fixture gate (AGENTS.md 2).
func assertProductionPlannerCompactionCatalog(t *testing.T) {
	t.Helper()
	root := filepath.Join(repoRoot(t), "configs")
	catalog, _, err := configload.LoadUnionReadOnly(root, filepath.Join(t.TempDir(), "managed"))
	if err != nil {
		t.Fatal(err)
	}
	direct, continuations := 0, 0
	for _, workflow := range catalog.Workflows() {
		for name, stage := range workflow.Stages {
			check := func(context *resolved.PlannerContextManagement, selection *resolved.ResolvedConsumerExecutionConfig) {
				t.Helper()
				if context == nil || context.Strategy != resolved.PlannerContextCompact ||
					context.Instructions.Ref != "instructions/planner-compaction.md" ||
					selection == nil || selection.CompactorModelPolicy == nil ||
					selection.CompactorModelPolicy.Ref.PolicyID != "compactor" || selection.CompactorModelPolicy.Ref.Version != "1" {
					t.Fatalf("%s@%s Stage %s has no shared Planner compaction policy", workflow.Ref.Name, workflow.Ref.Version, name)
				}
			}
			if stage.Planner.Selector() == resolved.StreamlinePlanner || stage.Planner.Selector() == resolved.RouterPlanner {
				check(stage.PlannerContextManagement, stage.ExecutionConfig.Planner)
				direct++
			}
			if stage.On.Summarized != nil && stage.On.Summarized.Escalate != nil {
				target := stage.On.Summarized.Escalate
				check(target.PlannerContextManagement, target.ExecutionConfig.Effective.Planner)
				continuations++
			}
		}
	}
	// project-findings-review@1's analyze Stage adds one summarized
	// continuation to the twenty retained by the common Audit cutover.
	if direct != 8 || continuations != 21 {
		t.Fatalf("modelled Planner coverage = %d direct/%d continuations, want 8/21", direct, continuations)
	}
	loaded := filepath.Join(t.TempDir(), "catalog")
	if err := os.CopyFS(loaded, os.DirFS(root)); err != nil {
		t.Fatal(err)
	}
	// Expanding native YAML references must preserve every normalized value,
	// instruction digest and model pin across the entire production catalog.
	entries, err := os.ReadDir(filepath.Join(loaded, "workflows"))
	if err != nil {
		t.Fatal(err)
	}
	for _, entry := range entries {
		if entry.IsDir() || filepath.Ext(entry.Name()) != ".yaml" {
			continue
		}
		path := filepath.Join(loaded, "workflows", entry.Name())
		data, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		var value map[string]any
		if err := yaml.Unmarshal(data, &value); err != nil {
			t.Fatal(err)
		}
		expanded, err := yaml.Marshal(value)
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, expanded, 0o600); err != nil {
			t.Fatal(err)
		}
	}
	inline, _, err := configload.LoadUnionReadOnly(loaded, filepath.Join(t.TempDir(), "managed"))
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(catalog.Workflows(), inline.Workflows()) {
		t.Fatal("YAML references changed normalized Workflow behavior")
	}
	for _, name := range []string{"streamline_review_workflow.yaml", "router_openapi_workflow.yaml"} {
		data, err := os.ReadFile(filepath.Join(root, "examples", name))
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(loaded, "workflows", name), data, 0o600); err != nil {
			t.Fatal(err)
		}
	}
	if _, err := configload.UpdateCatalogLock(loaded, filepath.Join(t.TempDir(), "managed")); err != nil {
		t.Fatalf("Planner examples cannot use the shipped compactor policy: %v", err)
	}
}
