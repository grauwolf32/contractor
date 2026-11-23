package config

import (
	"encoding/json"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
	"go.yaml.in/yaml/v4"
)

const compactWorkflowPath = "workflows/compact_copy.yaml"

// TestSharedWorkflowContextManagementCases holds Workflow authoring to the
// cases the Studio editor also reads (WFL-R91..WFL-R93).
func TestSharedWorkflowContextManagementCases(t *testing.T) {
	var fixture struct {
		Description string
		Cases       []struct {
			Name            string
			Path            []string
			Value           any
			Valid           bool
			CatalogOnly     bool
			ExecutionConfig map[string]any
		}
	}
	if err := json.Unmarshal(contractstest.ReadFile(t, "api", "testdata", "workflow-context-management-cases.json"), &fixture); err != nil {
		t.Fatal(err)
	}
	if len(fixture.Cases) == 0 {
		t.Fatal("no context management cases")
	}
	for _, tc := range fixture.Cases {
		t.Run(tc.Name, func(t *testing.T) {
			var source map[string]any
			if err := yaml.Unmarshal(readFile(t, filepath.Join("testdata", "context-compaction", compactWorkflowPath)), &source); err != nil {
				t.Fatal(err)
			}
			node := source["spec"].(map[string]any)
			for _, part := range tc.Path[:len(tc.Path)-1] {
				next, ok := node[part].(map[string]any)
				if !ok {
					next = map[string]any{}
					node[part] = next
				}
				node = next
			}
			node[tc.Path[len(tc.Path)-1]] = tc.Value
			if tc.ExecutionConfig != nil {
				source["spec"].(map[string]any)["executionConfig"] = tc.ExecutionConfig
			}
			data, err := yaml.Marshal(source)
			if err != nil {
				t.Fatal(err)
			}
			root := copyCoreFixture(t, "context-compaction")
			writeFile(t, filepath.Join(root, compactWorkflowPath), data)
			_, err = LoadUnionReadOnly(root, filepath.Join(t.TempDir(), "managed"))
			if (err == nil) != tc.Valid {
				t.Fatalf("valid=%t, want %t: %v", err == nil, tc.Valid, err)
			}
		})
	}
}

func TestCompactBindingResolvesExplicitPinsWithoutChangingTemplate(t *testing.T) {
	t.Parallel()
	snapshot := mustLoad(t, copyCoreFixture(t, "context-compaction"))
	workflow, err := snapshot.Workflow("compact-copy@1")
	if err != nil {
		t.Fatal(err)
	}
	binding := workflow.Stages["copy"].Agents["builder"]
	management := binding.ContextManagement
	if management.Strategy != contracts.ContextStrategyCompact || management.WorkerCompaction == nil {
		t.Fatalf("contextManagement = %+v", management)
	}
	compaction := *management.WorkerCompaction
	if compaction.ContextWindowRatio != contracts.DefaultWorkerCompactionContextRatio ||
		compaction.KeepRecentGroups != contracts.DefaultWorkerCompactionRecentGroups ||
		compaction.MaxCompactions != contracts.DefaultWorkerCompactions || compaction.IncludeReasoning {
		t.Fatalf("defaults were not normalized: %+v", compaction)
	}
	if workflow.Stages["copy"].ExecutionConfig.Agents["builder"].CompactorModelPolicy.Ref.PolicyID != "test-compactor" || workflow.Stages["copy"].ExecutionConfig.Agents["builder"].CompactorModelPolicy.MaxModelCalls != 1 {
		t.Fatalf("compactor policy = %+v", workflow.Stages["copy"].ExecutionConfig.Agents["builder"].CompactorModelPolicy)
	}
	assertDigest(t, workflow.Stages["copy"].ExecutionConfig.Agents["builder"].CompactorModelPolicy.Ref.Digest)
	if compaction.Instructions.Ref != "instructions/worker-compaction.md" ||
		!strings.Contains(compaction.Instructions.Text, "same subtask") {
		t.Fatalf("instructions = %+v", compaction.Instructions)
	}
	assertDigest(t, compaction.Instructions.Digest)

	// The binding keeps the selected template and its digest unchanged; the
	// compact branch replaces the summarizer only in the effective branch.
	template, err := snapshot.agentTemplateForTest("compact_builder@1")
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(binding.Template.Ref, template.Ref) || binding.Template.Summarizer == nil {
		t.Fatalf("template changed: %+v", binding.Template.Ref)
	}
	effective := contracts.EffectiveContextManagement(management, binding.Template)
	if !reflect.DeepEqual(effective, management) {
		t.Fatalf("effective branch = %+v", effective)
	}
	selection := workflow.Stages["copy"].ExecutionConfig.Agents["builder"]
	if selection.SummarizerModelPolicy != nil {
		t.Fatalf("compact binding received a summarizer selection: %+v", selection.SummarizerModelPolicy)
	}

	// Resolved Workflow and Stage snapshots round-trip the explicit branch.
	encoded, err := json.Marshal(workflow)
	if err != nil {
		t.Fatal(err)
	}
	decoded, err := resolved.DecodeResolvedWorkflowSnapshot(encoded)
	if err != nil || !reflect.DeepEqual(decoded, workflow) {
		t.Fatalf("Workflow snapshot round trip: %v", err)
	}
	if !strings.Contains(string(encoded), `"contextManagement":{"strategy":"compact@1"`) {
		t.Fatalf("snapshot lacks the explicit branch: %s", encoded)
	}
}

func TestOmittedContextManagementResolvesToExplicitTemplate(t *testing.T) {
	t.Parallel()
	snapshot := mustLoad(t, copyCoreFixture(t))
	workflow, err := snapshot.Workflow("artifact-copy@1")
	if err != nil {
		t.Fatal(err)
	}
	stage := workflow.Stages["copy"]
	if got := stage.Agents["builder"].ContextManagement; got != (contracts.WorkerContextManagement{Strategy: contracts.ContextStrategyTemplate}) {
		t.Fatalf("omitted contextManagement = %+v", got)
	}
	encoded, err := json.Marshal(stage)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(encoded), `"contextManagement":{"strategy":"template@1"}`) {
		t.Fatalf("stage snapshot omitted the explicit strategy: %s", encoded)
	}
	// A persisted snapshot without the member or with a foreign shape is
	// rejected, never inferred.
	for _, replacement := range []string{
		``,
		`,"contextManagement":null`,
		`,"contextManagement":{}`,
		`,"contextManagement":{"strategy":"none"}`,
		`,"contextManagement":{"strategy":"terminal@1"}`,
		`,"contextManagement":{"strategy":"template@1","maxCompactions":2}`,
		`,"contextManagement":{"strategy":"compact@1"}`,
	} {
		invalid := strings.Replace(string(encoded), `,"contextManagement":{"strategy":"template@1"}`, replacement, 1)
		if invalid == string(encoded) {
			t.Fatal("snapshot member was not replaced")
		}
		if _, err := resolved.DecodeResolvedStageSnapshot([]byte(invalid)); err == nil {
			t.Fatalf("persisted stage with %q accepted", replacement)
		}
	}
}
