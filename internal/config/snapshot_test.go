package config

import "testing"

func TestSnapshotAccessorsReturnDeepCopies(t *testing.T) {
	t.Parallel()

	snapshot := mustLoad(t, repositoryConfigRoot, MVPDescriptors())

	policy, _ := snapshot.ModelPolicy("worker@1")
	*policy.Temperature = 99
	policyAgain, _ := snapshot.ModelPolicy("worker@1")
	if policyAgain.Temperature == nil || *policyAgain.Temperature != 0.1 {
		t.Fatalf("ModelPolicy mutation leaked into Snapshot: %+v", policyAgain)
	}

	template, _ := snapshot.AgentTemplate("artifact_builder@1")
	template.Toolsets[0].Tools[0] = "corrupted"
	*template.ModelPolicy.Temperature = 99
	templateAgain, _ := snapshot.AgentTemplate("artifact_builder@1")
	if templateAgain.Toolsets[0].Tools[0] != "list_artifacts" || *templateAgain.ModelPolicy.Temperature != 0.1 {
		t.Fatalf("AgentTemplate mutation leaked into Snapshot: %+v", templateAgain)
	}

	workflow, _ := snapshot.Workflow("artifact-copy@1")
	workflow.Inputs["source"] = ArtifactSlot{Required: false, MediaTypes: []string{"corrupted/type"}}
	stage := workflow.Stages["copy"]
	stage.Result.Artifacts["copied"] = ArtifactSlot{MediaTypes: []string{"corrupted/type"}}
	binding := stage.Agents["builder"]
	binding.Template.Toolsets[0].Tools[0] = "corrupted"
	stage.Agents["builder"] = binding
	stage.WorkflowOutputs["result"] = "corrupted"
	workflow.Stages["copy"] = stage

	workflowAgain, _ := snapshot.Workflow("artifact-copy@1")
	stageAgain := workflowAgain.Stages["copy"]
	if !workflowAgain.Inputs["source"].Required || workflowAgain.Inputs["source"].MediaTypes[0] != "text/plain" {
		t.Fatalf("Workflow input mutation leaked into Snapshot: %+v", workflowAgain.Inputs)
	}
	if stageAgain.Result.Artifacts["copied"].MediaTypes[0] != "text/plain" ||
		stageAgain.Agents["builder"].Template.Toolsets[0].Tools[0] != "list_artifacts" ||
		stageAgain.WorkflowOutputs["result"] != "copied" {
		t.Fatalf("nested Workflow mutation leaked into Snapshot: %+v", stageAgain)
	}
}
