package config

import (
	"bytes"
	"path/filepath"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/config/resolved"
)

const summaryTransitionYAML = `        summarized:
          escalate:
            maxAttempts: 1
            planner: streamline@1
            deadlineSeconds: 600
            executionConfig:
              planner:
                modelPolicy: test-planner@1
                llmGateway: local-litellm@1
`

func summaryWorkflowFixture(t *testing.T) (string, string) {
	t.Helper()
	root := copyCoreFixture(t)
	installWorkerSummarizer(t, root)
	path := filepath.Join(root, "workflows/artifact_copy.yaml")
	appendFile(t, path, summaryTransitionYAML)
	return root, path
}

func TestSummaryEscalationPinsItsIndependentPlannerAndClonesPolicy(t *testing.T) {
	root, _ := summaryWorkflowFixture(t)
	snapshot := mustLoad(t, root)
	workflow, err := snapshot.ResolveRunWorkflow(t.Context(), "artifact-copy@1", resolved.ExecutionConfigPatch{}, developmentCredentialLookup(t, snapshot))
	if err != nil {
		t.Fatal(err)
	}
	stage := workflow.Stages["copy"]
	if stage.ExecutionConfig.Planner != nil || stage.On.Summarized == nil || stage.On.Summarized.Escalate == nil {
		t.Fatalf("base passthrough or continuation config is invalid: %+v", stage)
	}
	policy := stage.On.Summarized.Escalate
	if policy.Planner == nil || *policy.Planner != (resolved.PlannerRef{PlannerID: "streamline", Version: "1"}) ||
		policy.MaxAttempts != 1 || policy.DeadlineSeconds != 600 ||
		policy.ExecutionConfig.Effective.Planner == nil ||
		policy.ExecutionConfig.Effective.Planner.ModelPolicy.Ref.PolicyID != "test-planner" ||
		policy.ExecutionConfig.Effective.Agents["builder"].ModelPolicy.Ref != stage.ExecutionConfig.Agents["builder"].ModelPolicy.Ref {
		t.Fatalf("summary escalation changed the Worker or failed to pin its Planner: %+v", policy)
	}
	encoded, err := resolved.EncodeResolvedWorkflowSnapshot(workflow)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := resolved.DecodeResolvedWorkflowSnapshot(encoded); err != nil {
		t.Fatalf("persisted summary policy failed strict validation: %v", err)
	}
	for _, injected := range []string{`"then":null,`, `"then":{"kind":"fail"},`} {
		corrupt := bytes.Replace(encoded, []byte(`"deadlineSeconds":600,`), []byte(`"deadlineSeconds":600,`+injected), 1)
		if bytes.Equal(corrupt, encoded) {
			t.Fatal("snapshot mutation did not select the summary policy")
		}
		if _, err := resolved.DecodeResolvedWorkflowSnapshot(corrupt); err == nil {
			t.Fatal("persisted summary policy accepted an authored then")
		}
	}
	cloned := workflow.Clone()
	cloned.Stages["copy"].On.Summarized.Escalate.Planner.PlannerID = "changed"
	if policy.Planner.PlannerID != "streamline" {
		t.Fatal("Workflow clone shared its summary Planner ref")
	}
}

func TestSummaryStrategyCannotMixWithModelEscalation(t *testing.T) {
	root, _ := summaryWorkflowFixture(t)
	workflow, err := mustLoad(t, root).Workflow("artifact-copy@1")
	if err != nil {
		t.Fatal(err)
	}
	stage := workflow.Stages["copy"]
	policy := stage.ExecutionConfig.Agents["builder"].ModelPolicy
	stage.On.Failed = resolved.TransitionAction{Kind: resolved.TransitionEscalate, Escalate: &resolved.EscalateTransition{
		MaxAttempts: 1, Then: resolved.TransitionAction{Kind: resolved.TransitionFail},
		ExecutionConfig: resolved.ResolvedEscalationExecutionConfig{Override: resolved.ResolvedStageExecutionConfigOverride{
			Agents: map[string]resolved.ResolvedExecutionSelectionOverride{"builder": {ModelPolicy: &policy}},
		}},
	}}
	workflow.Stages["copy"] = stage
	if err := resolved.ResolveEscalationVariants(&workflow); err == nil || !strings.Contains(err.Error(), "cannot share a Stage") {
		t.Fatalf("mixed strategy/model policies were not rejected at their boundary: %v", err)
	}
}

func TestSummaryEscalationRejectsInvalidOrUnboundedPolicies(t *testing.T) {
	for _, test := range []struct{ name, old, replacement string }{
		{"missing Planner", "            planner: streamline@1\n", ""},
		{"null Planner", "planner: streamline@1", "planner: null"},
		{"wrong Planner", "planner: streamline@1", "planner: router@1"},
		{"missing deadline", "            deadlineSeconds: 600\n", ""},
		{"zero deadline", "deadlineSeconds: 600", "deadlineSeconds: 0"},
		{"oversized deadline", "deadlineSeconds: 600", "deadlineSeconds: 86401"},
		{"multiple strategy attempts", "maxAttempts: 1", "maxAttempts: 2"},
		{"unreachable exhaustion edge", "deadlineSeconds: 600", "deadlineSeconds: 600\n            then: {fail: {}}"},
		{"null exhaustion edge", "deadlineSeconds: 600", "deadlineSeconds: 600\n            then: null"},
		{"missing model", "                modelPolicy: test-planner@1\n", ""},
		{"missing Gateway", "                llmGateway: local-litellm@1\n", ""},
		{"unknown field", "deadlineSeconds: 600", "deadlineSeconds: 600\n            unknownField: true"},
		{"duplicate field", "planner: streamline@1", "planner: streamline@1\n            planner: streamline@1"},
		{"null summary policy", summaryTransitionYAML, "        summarized: null\n"},
		{"failure instead of escalation", summaryTransitionYAML, "        summarized: {fail: {}}\n"},
	} {
		t.Run(test.name, func(t *testing.T) {
			root, path := summaryWorkflowFixture(t)
			replaceFile(t, path, test.old, test.replacement)
			if _, err := Load(root); err == nil {
				t.Fatal("invalid summary escalation was accepted")
			}
		})
	}
	t.Run("no summarizer", func(t *testing.T) {
		root, _ := summaryWorkflowFixture(t)
		replaceFile(t, templatePath(root), workerSummarizerBlock, "")
		if _, err := Load(root); err == nil {
			t.Fatal("summary Worker without a summarizer was accepted")
		}
	})
	t.Run("ordinary namespace required", func(t *testing.T) {
		root, _ := summaryWorkflowFixture(t)
		workflow, err := mustLoad(t, root).Workflow("artifact-copy@1")
		if err != nil {
			t.Fatal(err)
		}
		stage := workflow.Stages["copy"]
		binding := stage.Agents["builder"]
		binding.Namespace = "inputs"
		stage.Agents["builder"] = binding
		workflow.Stages["copy"] = stage
		if err := resolved.ResolveEscalationVariants(&workflow); err == nil || !strings.Contains(err.Error(), "writable ordinary") {
			t.Fatalf("reserved summary namespace was not rejected at its boundary: %v", err)
		}
	})
}

func TestSummaryStrategyFieldsCannotChangeOrdinaryEscalation(t *testing.T) {
	root, path := summaryWorkflowFixture(t)
	data := string(readFile(t, path))
	data = strings.Replace(data, "        failed: {fail: {}}\n", "", 1)
	data = strings.Replace(data, "        summarized:\n", "        failed:\n", 1)
	writeFile(t, path, []byte(data))
	if _, err := Load(root); err == nil {
		t.Fatal("on.failed accepted a strategy override intended for summarized work")
	}
}
