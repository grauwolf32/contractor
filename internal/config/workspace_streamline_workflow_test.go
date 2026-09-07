package config

import (
	"reflect"
	"testing"
)

func TestRepositoryWorkspaceStreamlinePreservesCurrentContract(t *testing.T) {
	t.Parallel()

	for _, test := range []struct{ baseline, streamline string }{
		{"likec4-from-workspace@5", "likec4-from-workspace-streamline@2"},
		{"openapi-from-workspace@5", "openapi-from-workspace-streamline@1"},
	} {
		t.Run(test.streamline, func(t *testing.T) {
			snapshot := mustLoad(t, repositoryConfigRoot, MVPDescriptors())
			baseline, err := snapshot.Workflow(test.baseline)
			if err != nil {
				t.Fatal(err)
			}
			streamline, err := snapshot.Workflow(test.streamline)
			if err != nil {
				t.Fatal(err)
			}
			if streamline.EntryStage != baseline.EntryStage ||
				!reflect.DeepEqual(streamline.Parameters, baseline.Parameters) ||
				!reflect.DeepEqual(streamline.Inputs, baseline.Inputs) ||
				!reflect.DeepEqual(streamline.Outputs, baseline.Outputs) ||
				len(streamline.Stages) != len(baseline.Stages) {
				t.Fatal("workspace Streamline Workflow drifted from the current passthrough contract")
			}

			plannerPolicy, err := snapshot.ModelPolicy("project_planner@1")
			if err != nil {
				t.Fatal(err)
			}
			workerPolicy, err := snapshot.ModelPolicy("project_worker@1")
			if err != nil {
				t.Fatal(err)
			}
			for name, baselineStage := range baseline.Stages {
				stage := streamline.Stages[name]
				if stage.Planner != (PlannerRef{PlannerID: "streamline", Version: "1"}) {
					t.Fatalf("Stage %q Planner = %+v", name, stage.Planner)
				}
				if stage.ExecutionConfig.Planner == nil ||
					stage.ExecutionConfig.Planner.ModelPolicy.Ref != plannerPolicy.Ref ||
					stage.ExecutionConfig.Planner.Credential == nil ||
					stage.ExecutionConfig.Planner.Credential.CredentialID != "development-planner" {
					t.Fatalf("Stage %q Planner execution config = %+v", name, stage.ExecutionConfig.Planner)
				}
				for logicalName, selection := range stage.ExecutionConfig.Agents {
					if selection.ModelPolicy.Ref != workerPolicy.Ref || selection.Credential == nil ||
						selection.Credential.CredentialID != "development-worker" {
						t.Fatalf("Stage %q Agent %q execution config = %+v", name, logicalName, selection)
					}
				}

				actual := cloneStage(stage)
				actual.Planner = baselineStage.Planner
				actual.ExecutionConfig = baselineStage.ExecutionConfig
				if baselineStage.On.Failed.Escalate != nil && actual.On.Failed.Escalate != nil {
					// OpenAPI escalation still selects the strong validator while
					// retaining the Streamline planner for that invocation.
					effective := &actual.On.Failed.Escalate.ExecutionConfig.Effective
					if !reflect.DeepEqual(effective.Planner, stage.ExecutionConfig.Planner) {
						t.Fatalf("Stage %q escalation lost the Streamline planner", name)
					}
					effective.Planner = baselineStage.On.Failed.Escalate.ExecutionConfig.Effective.Planner
				}
				if !reflect.DeepEqual(actual, baselineStage) {
					t.Fatalf("Stage %q changed outside Planner and execution policy", name)
				}
			}
		})
	}
}
