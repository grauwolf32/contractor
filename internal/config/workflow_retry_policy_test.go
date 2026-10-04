package config

import "testing"

// retryRunPolicyFixtureRoot is a frozen catalog whose single Stage retries a
// failed attempt and fails an interrupted one.
const retryRunPolicyFixtureRoot = "testdata/retry-run-policy"

func TestResolveRunWorkflowKeepsRunWorkerPolicyOnRetryingStage(t *testing.T) {
	t.Parallel()
	snapshot := mustLoad(t, retryRunPolicyFixtureRoot, MVPDescriptors())
	credentials := developmentCredentialLookup(t, snapshot)

	base, err := snapshot.ResolveRunWorkflow(t.Context(), "artifact-copy-retry@1", ExecutionConfigPatch{}, credentials)
	if err != nil {
		t.Fatal(err)
	}
	baseStage := base.Stages["copy"]
	assertBoundedRetry(t, baseStage.On.Failed, 2)
	if baseStage.On.Interrupted.Kind != TransitionFail || baseStage.On.Interrupted.Retry != nil {
		t.Fatalf("interrupted transition = %+v, want fail", baseStage.On.Interrupted)
	}
	if selection := baseStage.ExecutionConfig.Agents["builder"]; selection.ModelPolicy.Ref.PolicyID != "test-worker" ||
		selection.Origins.ModelPolicy != "agentTemplate.modelPolicy" {
		t.Fatalf("default selection = %+v", selection)
	}

	patch := decodeExecutionConfigPatch(t, `{"workers":{"modelPolicy":"test-retry-worker@1"}}`)
	workflow, err := snapshot.ResolveRunWorkflow(t.Context(), "artifact-copy-retry@1", patch, credentials)
	if err != nil {
		t.Fatal(err)
	}
	stage := workflow.Stages["copy"]
	assertBoundedRetry(t, stage.On.Failed, 2)
	selection := stage.ExecutionConfig.Agents["builder"]
	if selection.ModelPolicy.Ref.PolicyID != "test-retry-worker" || selection.ModelPolicy.Model != "retry-worker-model" ||
		selection.Origins.ModelPolicy != "run.executionConfig.workers" {
		t.Fatalf("retrying Stage lost the Run policy: %+v", selection)
	}
}
