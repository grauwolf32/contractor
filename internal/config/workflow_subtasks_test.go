package config

import (
	"encoding/json"
	"fmt"
	"path/filepath"
	"testing"

	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
)

func TestSharedWorkflowSubtaskLimits(t *testing.T) {
	var fixture struct {
		Limits []struct {
			Name      string
			Present   bool
			Value     json.RawMessage
			Valid     bool
			Effective int
		}
	}
	if err := json.Unmarshal(contractstest.ReadFile(t, "api", "testdata", "planner-subtask-cases.json"), &fixture); err != nil {
		t.Fatal(err)
	}
	baseline, err := mustLoad(t, copyCoreFixture(t)).Workflow("artifact-copy@1")
	if err != nil {
		t.Fatal(err)
	}
	baselineSnapshot, err := resolved.EncodeResolvedWorkflowSnapshot(baseline)
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range fixture.Limits {
		t.Run(tc.Name, func(t *testing.T) {
			var document map[string]json.RawMessage
			if err := json.Unmarshal(baselineSnapshot, &document); err != nil {
				t.Fatal(err)
			}
			if tc.Present {
				document["maxSubtasks"] = tc.Value
				var stages map[string]map[string]json.RawMessage
				if err := json.Unmarshal(document["stages"], &stages); err != nil {
					t.Fatal(err)
				}
				for _, stage := range stages {
					stage["maxSubtasks"] = tc.Value
				}
				stagesJSON, err := json.Marshal(stages)
				if err != nil {
					t.Fatal(err)
				}
				document["stages"] = stagesJSON
			}
			workflowSnapshot, err := json.Marshal(document)
			if err != nil {
				t.Fatal(err)
			}
			if _, err := resolved.DecodeResolvedWorkflowSnapshot(workflowSnapshot); (err == nil) != tc.Valid {
				t.Fatalf("Workflow snapshot valid=%v: %v", tc.Valid, err)
			}
			root := copyCoreFixture(t)
			if tc.Present {
				replaceFile(t, filepath.Join(root, "workflows/artifact_copy.yaml"), "spec:\n", fmt.Sprintf("spec:\n  maxSubtasks: %s\n", tc.Value))
			}
			snapshot, err := Load(root)
			if (err == nil) != tc.Valid {
				t.Fatalf("valid=%v, load error=%v", tc.Valid, err)
			}
			stageSnapshot := []byte(`{"session":"isolated"}`)
			if tc.Present {
				stageSnapshot = []byte(fmt.Sprintf(`{"session":"isolated","maxSubtasks":%s}`, tc.Value))
			}
			if _, err := resolved.DecodeResolvedStageSnapshot(stageSnapshot); (err == nil) != tc.Valid {
				t.Fatalf("Stage snapshot valid=%v: %v", tc.Valid, err)
			}
			if err != nil {
				return
			}
			workflow, err := snapshot.Workflow("artifact-copy@1")
			if err != nil {
				t.Fatal(err)
			}
			if workflow.MaxSubtasks.Effective() != tc.Effective || workflow.Stages["copy"].MaxSubtasks != workflow.MaxSubtasks {
				t.Fatalf("Workflow/Stage limit selection differs: %+v", workflow.MaxSubtasks)
			}
			encoded, err := resolved.EncodeResolvedWorkflowSnapshot(workflow)
			if err != nil {
				t.Fatal(err)
			}
			restored, err := resolved.DecodeResolvedWorkflowSnapshot(encoded)
			if err != nil || restored.MaxSubtasks != workflow.MaxSubtasks {
				t.Fatalf("snapshot selection = (%+v, %v)", restored.MaxSubtasks, err)
			}
			cloned := workflow.Clone()
			stage := cloned.Stages["copy"]
			stage.MaxSubtasks = 1
			cloned.Stages["copy"] = stage
			if workflow.Stages["copy"].MaxSubtasks != workflow.MaxSubtasks {
				t.Fatal("clone mutated source budget")
			}
			if tc.Effective != 1 {
				bad, err := resolved.EncodeResolvedWorkflowSnapshot(cloned)
				if err != nil {
					t.Fatal(err)
				}
				if _, err := resolved.DecodeResolvedWorkflowSnapshot(bad); err == nil {
					t.Fatal("Stage budget drift was accepted")
				}
			}
		})
	}
}

func TestSummaryContinuationRetainsWorkflowSubtaskLimit(t *testing.T) {
	root, path := summaryWorkflowFixture(t)
	replaceFile(t, path, "spec:\n", "spec:\n  maxSubtasks: 64\n")
	snapshot := mustLoad(t, root)
	workflow, err := snapshot.ResolveRunWorkflow(t.Context(), "artifact-copy@1", resolved.ExecutionConfigPatch{}, developmentCredentialLookup(t, snapshot))
	if err != nil {
		t.Fatal(err)
	}
	stage := workflow.Stages["copy"]
	if stage.MaxSubtasks.Effective() != 64 || stage.On.Summarized == nil {
		t.Fatal("summary policy lost the Workflow limit")
	}
	encoded, err := resolved.EncodeResolvedWorkflowSnapshot(workflow)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := resolved.DecodeResolvedWorkflowSnapshot(encoded); err != nil {
		t.Fatal(err)
	}
}
