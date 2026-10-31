package config

import (
	"os"
	"testing"

	"github.com/grauwolf32/contractor/internal/config/resolved"
)

// retryRunPolicyFixtureRoot is a frozen catalog whose single Stage retries a
// failed attempt and fails an interrupted one.
const retryRunPolicyFixtureRoot = "testdata/retry-run-policy"

func TestResolveRunWorkflowKeepsWorkflowWorkerPolicyOnRetryingStage(t *testing.T) {
	t.Parallel()
	root := t.TempDir()
	if err := os.CopyFS(root, os.DirFS(retryRunPolicyFixtureRoot)); err != nil {
		t.Fatal(err)
	}
	snapshot := mustLoad(t, root)
	credentials := developmentCredentialLookup(t, snapshot)

	base, err := snapshot.ResolveRunWorkflow(t.Context(), "artifact-copy-retry@1", credentials)
	if err != nil {
		t.Fatal(err)
	}
	baseStage := base.Stages["copy"]
	assertBoundedRetry(t, baseStage.On.Failed, 2)
	if baseStage.On.Interrupted.Kind != resolved.TransitionFail || baseStage.On.Interrupted.Retry != nil {
		t.Fatalf("interrupted transition = %+v, want fail", baseStage.On.Interrupted)
	}
	if selection := baseStage.ExecutionConfig.Agents["builder"]; selection.ModelPolicy.Ref.PolicyID != "test-worker" ||
		selection.Origins.ModelPolicy != "agentTemplate.modelPolicy" {
		t.Fatalf("default selection = %+v", selection)
	}

	writeWorkflowExecutionConfig(t, root, "artifact_copy_retry.yaml", `{"workers":{"modelPolicy":"test-retry-worker@1"}}`)
	snapshot = mustLoad(t, root)
	workflow, err := snapshot.ResolveRunWorkflow(t.Context(), "artifact-copy-retry@1", credentials)
	if err != nil {
		t.Fatal(err)
	}
	stage := workflow.Stages["copy"]
	assertBoundedRetry(t, stage.On.Failed, 2)
	selection := stage.ExecutionConfig.Agents["builder"]
	if selection.ModelPolicy.Ref.PolicyID != "test-retry-worker" || selection.ModelPolicy.Model != "retry-worker-model" ||
		selection.Origins.ModelPolicy != "workflow.executionConfig.workers" {
		t.Fatalf("retrying Stage lost the Workflow policy: %+v", selection)
	}
}
