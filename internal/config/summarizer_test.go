package config

import (
	"encoding/json"
	"path/filepath"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts"
)

func TestWorkerSummarizerGoldenDigestMatchesGoCanonicalization(t *testing.T) {
	t.Parallel()

	var allocation contracts.AllocationSpec
	encoded := readFile(
		t, filepath.Join("..", "..", "api", "testdata", "v1alpha1", "valid", "allocation-spec-summarizer.json"),
	)
	if err := json.Unmarshal(encoded, &allocation); err != nil {
		t.Fatal(err)
	}
	if err := allocation.Validate(); err != nil {
		t.Fatal(err)
	}
	template := allocation.AgentTemplate
	if template.Summarizer == nil {
		t.Fatal("golden allocation omitted summarizer")
	}
	if got := digestBytes([]byte(template.Instructions.Text)); got != template.Instructions.Digest {
		t.Fatalf("instruction digest = %s, want %s", got, template.Instructions.Digest)
	}
	for _, policy := range []contracts.ResolvedModelPolicy{
		template.ModelPolicy, template.Summarizer.ModelPolicy,
	} {
		got, err := modelPolicyDigest(
			Selector{ID: policy.Ref.PolicyID, Version: policy.Ref.Version}, policy,
		)
		if err != nil {
			t.Fatal(err)
		}
		if got != policy.Ref.Digest {
			t.Fatalf("ModelPolicy %s digest = %s, want %s", policy.Ref.PolicyID, got, policy.Ref.Digest)
		}
	}
	got, err := agentTemplateDigest(
		Selector{ID: template.Ref.TemplateID, Version: template.Ref.Version}, template,
	)
	if err != nil {
		t.Fatal(err)
	}
	if got != template.Ref.Digest {
		t.Fatalf("AgentTemplate digest = %s, want %s", got, template.Ref.Digest)
	}
}

func TestWorkerSummarizerResolvesPinsAndPublishesSafeConfiguration(t *testing.T) {
	t.Parallel()

	root := copyConfigTree(t)
	installWorkerSummarizer(t, root)
	snapshot := mustLoad(t, root, MVPDescriptors())

	template, err := snapshot.AgentTemplate("artifact_builder@1")
	if err != nil {
		t.Fatal(err)
	}
	policy, err := snapshot.ModelPolicy("terminal_summarizer@1")
	if err != nil {
		t.Fatal(err)
	}
	if template.Summarizer == nil || template.Summarizer.ModelPolicy.Ref != policy.Ref ||
		template.Summarizer.ModelPolicy.Model != "worker-summarizer-model" ||
		template.Summarizer.CumulativeBudget == nil || *template.Summarizer.CumulativeBudget != 20_000 ||
		template.Summarizer.ContextWindowRatio != 0.9 ||
		template.ModelPolicy.ContextWindowTokens != 131_072 ||
		template.Summarizer.ModelPolicy.ContextWindowTokens != 131_072 {
		t.Fatalf("resolved summarizer = %+v", template.Summarizer)
	}

	workflow, err := snapshot.ResolveRunWorkflow(
		t.Context(), "artifact-copy@1", ExecutionConfigPatch{}, developmentCredentialLookup(t, snapshot),
	)
	if err != nil {
		t.Fatal(err)
	}
	pinned := workflow.Stages["copy"].Agents["builder"].Template.Summarizer
	if pinned == nil || pinned.ModelPolicy.Ref != policy.Ref ||
		pinned.CumulativeBudget == nil || *pinned.CumulativeBudget != 20_000 {
		t.Fatalf("Run-resolved summarizer = %+v", pinned)
	}

	resource, err := snapshot.Configuration(ConfigurationAgentTemplates, "artifact_builder@1")
	if err != nil {
		t.Fatal(err)
	}
	body := resource.Body.(map[string]any)
	safe := body["summarizer"].(map[string]any)
	if safe["modelPolicy"] != policy.Ref || safe["cumulativeBudget"] != 20_000 ||
		safe["contextWindowRatio"] != 0.9 {
		t.Fatalf("safe AgentTemplate resource summarizer = %#v", safe)
	}
	encoded, err := json.Marshal(resource)
	if err != nil {
		t.Fatal(err)
	}
	for _, forbidden := range []string{"llmGatewayToken", "credential", "http://", "https://"} {
		if strings.Contains(string(encoded), forbidden) {
			t.Fatalf("safe resource contains %q: %s", forbidden, encoded)
		}
	}
}

func TestWorkerSummarizerDigestCoversExactPolicyAndThresholds(t *testing.T) {
	t.Parallel()

	baselineRoot := copyConfigTree(t)
	installWorkerSummarizer(t, baselineRoot)
	baseline := mustLoad(t, baselineRoot, MVPDescriptors())
	baselineTemplate, _ := baseline.AgentTemplate("artifact_builder@1")
	baselineWorker, _ := baseline.ModelPolicy("worker@1")
	baselineSummary, _ := baseline.ModelPolicy("terminal_summarizer@1")

	thresholdRoot := copyConfigTree(t)
	installWorkerSummarizer(t, thresholdRoot)
	replaceFile(
		t, filepath.Join(thresholdRoot, "agent-templates/artifact_builder.yaml"),
		"contextWindowRatio: 0.9", "contextWindowRatio: 0.85",
	)
	threshold := mustLoad(t, thresholdRoot, MVPDescriptors())
	thresholdTemplate, _ := threshold.AgentTemplate("artifact_builder@1")
	thresholdSummary, _ := threshold.ModelPolicy("terminal_summarizer@1")
	if thresholdTemplate.Ref.Digest == baselineTemplate.Ref.Digest {
		t.Fatal("summarizer threshold change did not alter AgentTemplate digest")
	}
	if thresholdSummary.Ref.Digest != baselineSummary.Ref.Digest {
		t.Fatal("summarizer threshold unexpectedly altered ModelPolicy digest")
	}

	policyRoot := copyConfigTree(t)
	installWorkerSummarizer(t, policyRoot)
	replaceFile(
		t, filepath.Join(policyRoot, "model-policies/terminal_summarizer.yaml"),
		"model: worker-summarizer-model", "model: worker-summarizer-model-v2",
	)
	policyVariant := mustLoad(t, policyRoot, MVPDescriptors())
	policyTemplate, _ := policyVariant.AgentTemplate("artifact_builder@1")
	policySummary, _ := policyVariant.ModelPolicy("terminal_summarizer@1")
	policyWorker, _ := policyVariant.ModelPolicy("worker@1")
	if policySummary.Ref.Digest == baselineSummary.Ref.Digest ||
		policyTemplate.Ref.Digest == baselineTemplate.Ref.Digest {
		t.Fatal("resolved summarizer policy change did not alter both policy and template digests")
	}
	if policyWorker.Ref.Digest != baselineWorker.Ref.Digest {
		t.Fatal("summarizer policy change unexpectedly altered normal Worker policy digest")
	}

	omittedRoot := copyConfigTree(t)
	writeWorkerSummarizerPolicy(t, omittedRoot)
	omitted := mustLoad(t, omittedRoot, MVPDescriptors())
	omittedTemplate, _ := omitted.AgentTemplate("artifact_builder@1")
	original := mustLoad(t, repositoryConfigRoot, MVPDescriptors())
	originalTemplate, _ := original.AgentTemplate("artifact_builder@1")
	if omittedTemplate.Summarizer != nil || omittedTemplate.Ref.Digest != originalTemplate.Ref.Digest {
		t.Fatalf("unreferenced summarizer changed omitted template: %+v", omittedTemplate.Summarizer)
	}
}

func TestWorkerSummarizerDefaultsContextWindowRatioBeforeDigesting(t *testing.T) {
	t.Parallel()

	explicitRoot := copyConfigTree(t)
	installWorkerSummarizer(t, explicitRoot)
	explicitSnapshot := mustLoad(t, explicitRoot, MVPDescriptors())
	explicitTemplate, _ := explicitSnapshot.AgentTemplate("artifact_builder@1")

	defaultedRoot := copyConfigTree(t)
	installWorkerSummarizer(t, defaultedRoot)
	replaceFile(t, templatePath(defaultedRoot), "    contextWindowRatio: 0.9\n", "")
	defaultedSnapshot := mustLoad(t, defaultedRoot, MVPDescriptors())
	defaultedTemplate, _ := defaultedSnapshot.AgentTemplate("artifact_builder@1")

	if defaultedTemplate.Summarizer == nil ||
		defaultedTemplate.Summarizer.ContextWindowRatio != 0.9 ||
		defaultedTemplate.Ref.Digest != explicitTemplate.Ref.Digest {
		t.Fatalf("defaulted summarizer = %+v", defaultedTemplate.Summarizer)
	}
}

func TestWorkerSummarizerSnapshotIsDeepAndRemainsPinnedAcrossReload(t *testing.T) {
	t.Parallel()

	root := copyConfigTree(t)
	installWorkerSummarizer(t, root)
	first := mustLoad(t, root, MVPDescriptors())
	template, _ := first.AgentTemplate("artifact_builder@1")
	oldDigest := template.Ref.Digest
	*template.Summarizer.CumulativeBudget = 1
	template.Summarizer.ContextWindowRatio = 0.5
	*template.Summarizer.ModelPolicy.Temperature = 99

	again, _ := first.AgentTemplate("artifact_builder@1")
	if again.Summarizer == nil || *again.Summarizer.CumulativeBudget != 20_000 ||
		again.Summarizer.ContextWindowRatio != 0.9 ||
		again.Summarizer.ModelPolicy.Temperature == nil || *again.Summarizer.ModelPolicy.Temperature != 0.1 {
		t.Fatalf("summarizer mutation leaked into Snapshot: %+v", again.Summarizer)
	}

	replaceFile(
		t, filepath.Join(root, "model-policies/terminal_summarizer.yaml"),
		"model: worker-summarizer-model", "model: worker-summarizer-model-v2",
	)
	second := mustLoad(t, root, MVPDescriptors())
	pinnedAgain, _ := first.AgentTemplate("artifact_builder@1")
	newTemplate, _ := second.AgentTemplate("artifact_builder@1")
	if pinnedAgain.Ref.Digest != oldDigest ||
		pinnedAgain.Summarizer.ModelPolicy.Model != "worker-summarizer-model" ||
		newTemplate.Ref.Digest == oldDigest ||
		newTemplate.Summarizer.ModelPolicy.Model != "worker-summarizer-model-v2" {
		t.Fatalf("old/new snapshot pinning failed: old=%+v new=%+v", pinnedAgain, newTemplate)
	}
}

func TestWorkerSummarizerRejectsInvalidAuthoring(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name   string
		mutate func(*testing.T, string)
		want   string
	}{
		{"empty block", func(t *testing.T, root string) {
			replaceFile(t, templatePath(root), workerSummarizerBlock, "  summarizer: {}\n")
		}, "spec.summarizer"},
		{"unknown field", func(t *testing.T, root string) {
			replaceFile(t, templatePath(root), "    contextWindowRatio: 0.9", "    contextWindowRatio: 0.9\n    unknown: true")
		}, "unknown"},
		{"duplicate field", func(t *testing.T, root string) {
			replaceFile(t, templatePath(root), "    contextWindowRatio: 0.9", "    contextWindowRatio: 0.9\n    contextWindowRatio: 0.8")
		}, "already defined"},
		{"zero total", func(t *testing.T, root string) {
			replaceFile(t, templatePath(root), "cumulativeBudget: 20000", "cumulativeBudget: 0")
		}, "cumulativeBudget"},
		{"zero context ratio", func(t *testing.T, root string) {
			replaceFile(t, templatePath(root), "contextWindowRatio: 0.9", "contextWindowRatio: 0")
		}, "contextWindowRatio"},
		{"unit context ratio", func(t *testing.T, root string) {
			replaceFile(t, templatePath(root), "contextWindowRatio: 0.9", "contextWindowRatio: 1")
		}, "contextWindowRatio"},
		{"non-finite context ratio", func(t *testing.T, root string) {
			replaceFile(t, templatePath(root), "contextWindowRatio: 0.9", "contextWindowRatio: .nan")
		}, "contextWindowRatio"},
		{"total equals Worker hard bound", func(t *testing.T, root string) {
			replaceFile(t, templatePath(root), "cumulativeBudget: 20000", "cumulativeBudget: 32768")
		}, "below Worker maxTotalTokens"},
		{"unknown policy", func(t *testing.T, root string) {
			replaceFile(t, templatePath(root), "terminal_summarizer@1", "missing@1")
		}, "unknown ModelPolicy"},
		{"missing output budget", func(t *testing.T, root string) {
			replaceFile(t, summaryPolicyPath(root), "  maxOutputTokens: 2048\n", "")
		}, "requires maxOutputTokens"},
		{"missing summary context window", func(t *testing.T, root string) {
			replaceFile(t, summaryPolicyPath(root), "  contextWindowTokens: 131072\n", "")
		}, "requires contextWindowTokens"},
		{"missing Worker context window", func(t *testing.T, root string) {
			replaceFile(t, workerPolicyPath(root), "  contextWindowTokens: 131072\n", "")
		}, "summarized Worker modelPolicy requires contextWindowTokens"},
		{"missing exact model call", func(t *testing.T, root string) {
			replaceFile(t, summaryPolicyPath(root), "  maxModelCalls: 1\n", "")
		}, "requires maxModelCalls=1"},
		{"more than one model call", func(t *testing.T, root string) {
			replaceFile(t, summaryPolicyPath(root), "maxModelCalls: 1", "maxModelCalls: 2")
		}, "requires maxModelCalls=1"},
		{"tool budget", func(t *testing.T, root string) {
			replaceFile(t, summaryPolicyPath(root), "  maxModelCalls: 1", "  maxModelCalls: 1\n  maxToolCalls: 1")
		}, "must omit maxToolCalls"},
		{"Worker-call budget", func(t *testing.T, root string) {
			replaceFile(t, summaryPolicyPath(root), "  maxModelCalls: 1", "  maxModelCalls: 1\n  maxWorkerCalls: 1")
		}, "must omit maxWorkerCalls"},
	}
	for _, test := range tests {
		test := test
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			root := copyConfigTree(t)
			installWorkerSummarizer(t, root)
			test.mutate(t, root)
			snapshot, err := Load(root, MVPDescriptors())
			if err == nil || snapshot != nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("Load() = (%v, %v), want error containing %q", snapshot, err, test.want)
			}
		})
	}
}

func TestWorkerSummarizerRejectsEffectiveRunPolicyBelowSoftThreshold(t *testing.T) {
	t.Parallel()

	root := copyConfigTree(t)
	installWorkerSummarizer(t, root)
	writeFile(t, filepath.Join(root, "model-policies/low_worker.yaml"), []byte(lowWorkerPolicyYAML))
	snapshot := mustLoad(t, root, MVPDescriptors())
	patch := decodeExecutionConfigPatch(t, `{"workers":{"modelPolicy":"low_worker@1"}}`)
	resolved, err := snapshot.ResolveRunWorkflow(
		t.Context(), "artifact-copy@1", patch, developmentCredentialLookup(t, snapshot),
	)
	if err == nil || !strings.Contains(err.Error(), "below Worker maxTotalTokens") {
		t.Fatalf("ResolveRunWorkflow() = (%+v, %v), want summarizer threshold error", resolved, err)
	}
}

func TestExecutionConfigCannotOverrideWorkerSummarizer(t *testing.T) {
	t.Parallel()

	for _, raw := range []string{
		`{"workers":{"summarizer":"terminal_summarizer@1"}}`,
		`{"stages":{"copy":{"agents":{"builder":{"summarizer":null}}}}}`,
	} {
		var patch ExecutionConfigPatch
		if err := json.Unmarshal([]byte(raw), &patch); err == nil {
			t.Fatalf("executionConfig accepted summarizer override: %s", raw)
		}
	}
}

func installWorkerSummarizer(t *testing.T, root string) {
	t.Helper()
	writeWorkerSummarizerPolicy(t, root)
	replaceFile(
		t, workerPolicyPath(root),
		"  model: worker-model\n",
		"  model: worker-model\n  contextWindowTokens: 131072\n",
	)
	replaceFile(
		t, templatePath(root),
		"  modelPolicy: worker@1\n  toolsets:",
		"  modelPolicy: worker@1\n"+workerSummarizerBlock+"  toolsets:",
	)
}

func writeWorkerSummarizerPolicy(t *testing.T, root string) {
	t.Helper()
	writeFile(t, summaryPolicyPath(root), []byte(workerSummarizerPolicyYAML))
}

func templatePath(root string) string {
	return filepath.Join(root, "agent-templates/artifact_builder.yaml")
}

func summaryPolicyPath(root string) string {
	return filepath.Join(root, "model-policies/terminal_summarizer.yaml")
}

func workerPolicyPath(root string) string {
	return filepath.Join(root, "model-policies/worker.yaml")
}

const workerSummarizerBlock = `  summarizer:
    modelPolicy: terminal_summarizer@1
    contextWindowRatio: 0.9
    cumulativeBudget: 20000
`

const workerSummarizerPolicyYAML = `apiVersion: contractor/v1alpha1
kind: ModelPolicy
metadata:
  name: terminal_summarizer
  version: "1"
spec:
  model: worker-summarizer-model
  contextWindowTokens: 131072
  maxOutputTokens: 2048
  maxModelCalls: 1
  temperature: 0.1
`

const lowWorkerPolicyYAML = `apiVersion: contractor/v1alpha1
kind: ModelPolicy
metadata:
  name: low_worker
  version: "1"
spec:
  model: worker-model
  contextWindowTokens: 131072
  maxOutputTokens: 4096
  maxModelCalls: 8
  maxToolCalls: 16
  maxTotalTokens: 16000
  temperature: 0.1
`
