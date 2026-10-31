package config

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contentdigest"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
	"github.com/grauwolf32/contractor/internal/contracts/control"
)

func TestSharedWorkerSummarizerGoldenDigestMatchesGoCanonicalization(t *testing.T) {
	t.Parallel()

	var allocation control.AllocationSpec
	encoded := contractstest.ReadFile(t, "api", "testdata", "v1alpha1", "valid", "allocation-spec-summarizer.json")
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
	if got := contentdigest.Bytes([]byte(template.Instructions.Text)); got != template.Instructions.Digest {
		t.Fatalf("instruction digest = %s, want %s", got, template.Instructions.Digest)
	}
	for _, policy := range []contracts.ResolvedModelPolicy{
		allocation.ModelPolicy, *allocation.SummarizerModelPolicy,
	} {
		got, err := modelPolicyDigest(
			resolved.Selector{ID: policy.Ref.PolicyID, Version: policy.Ref.Version}, policy,
		)
		if err != nil {
			t.Fatal(err)
		}
		if got != policy.Ref.Digest {
			t.Fatalf("ModelPolicy %s digest = %s, want %s", policy.Ref.PolicyID, got, policy.Ref.Digest)
		}
	}
	got, err := agentTemplateDigest(
		resolved.Selector{ID: template.Ref.TemplateID, Version: template.Ref.Version}, template,
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

	root := copyCoreFixture(t)
	installWorkerSummarizer(t, root)
	snapshot := mustLoad(t, root)

	template, err := snapshot.agentTemplateForTest("artifact_builder@1")
	if err != nil {
		t.Fatal(err)
	}
	policy, err := snapshot.ModelPolicy("terminal_summarizer@1")
	if err != nil {
		t.Fatal(err)
	}
	if template.Summarizer == nil ||
		template.Summarizer.CumulativeBudget == nil || *template.Summarizer.CumulativeBudget != 20_000 ||
		template.Summarizer.ContextWindowRatio != 0.9 {
		t.Fatalf("resolved summarizer = %+v", template.Summarizer)
	}

	workflow, err := snapshot.ResolveRunWorkflow(
		t.Context(), "artifact-copy@1", developmentCredentialLookup(t, snapshot),
	)
	if err != nil {
		t.Fatal(err)
	}
	pinned := workflow.Stages["copy"].Agents["builder"].Template.Summarizer
	if pinned == nil || workflow.Stages["copy"].ExecutionConfig.Agents["builder"].SummarizerModelPolicy.Ref != policy.Ref ||
		pinned.CumulativeBudget == nil || *pinned.CumulativeBudget != 20_000 {
		t.Fatalf("Run-resolved summarizer = %+v", pinned)
	}

	resource, err := snapshot.Configuration(ConfigurationAgentTemplates, "artifact_builder@1")
	if err != nil {
		t.Fatal(err)
	}
	body := resource.Body.(map[string]any)
	safe := body["summarizer"].(map[string]any)
	if safe["modelPolicy"] != nil || body["modelPolicy"] != nil || safe["cumulativeBudget"] != 20_000 ||
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

	baselineRoot := copyCoreFixture(t)
	installWorkerSummarizer(t, baselineRoot)
	baseline := mustLoad(t, baselineRoot)
	baselineTemplate, _ := baseline.agentTemplateForTest("artifact_builder@1")
	baselineWorker, _ := baseline.ModelPolicy("test-worker@1")
	baselineSummary, _ := baseline.ModelPolicy("terminal_summarizer@1")

	thresholdRoot := copyCoreFixture(t)
	installWorkerSummarizer(t, thresholdRoot)
	replaceFile(
		t, filepath.Join(thresholdRoot, "agent-templates/artifact_builder.yaml"),
		"contextWindowRatio: 0.9", "contextWindowRatio: 0.85",
	)
	threshold := mustLoad(t, thresholdRoot)
	thresholdTemplate, _ := threshold.agentTemplateForTest("artifact_builder@1")
	thresholdSummary, _ := threshold.ModelPolicy("terminal_summarizer@1")
	if thresholdTemplate.Ref.Digest == baselineTemplate.Ref.Digest {
		t.Fatal("summarizer threshold change did not alter AgentTemplate digest")
	}
	if thresholdSummary.Ref.Digest != baselineSummary.Ref.Digest {
		t.Fatal("summarizer threshold unexpectedly altered ModelPolicy digest")
	}

	policyRoot := copyCoreFixture(t)
	installWorkerSummarizer(t, policyRoot)
	replaceFile(
		t, filepath.Join(policyRoot, "model-policies/terminal_summarizer.yaml"),
		"model: worker-summarizer-model", "model: worker-summarizer-model-v2",
	)
	policyVariant := mustLoad(t, policyRoot)
	policyTemplate, _ := policyVariant.agentTemplateForTest("artifact_builder@1")
	policySummary, _ := policyVariant.ModelPolicy("terminal_summarizer@1")
	policyWorker, _ := policyVariant.ModelPolicy("test-worker@1")
	if policySummary.Ref.Digest == baselineSummary.Ref.Digest ||
		policyTemplate.Ref.Digest != baselineTemplate.Ref.Digest {
		t.Fatal("policy must change its own digest and leave the template digest unchanged")
	}
	if policyWorker.Ref.Digest != baselineWorker.Ref.Digest {
		t.Fatal("summarizer policy change unexpectedly altered normal Worker policy digest")
	}

	omittedRoot := copyCoreFixture(t)
	writeWorkerSummarizerPolicy(t, omittedRoot)
	omitted := mustLoad(t, omittedRoot)
	omittedTemplate, _ := omitted.agentTemplateForTest("artifact_builder@1")
	original := mustLoad(t, copyCoreFixture(t))
	originalTemplate, _ := original.agentTemplateForTest("artifact_builder@1")
	if omittedTemplate.Summarizer != nil || omittedTemplate.Ref.Digest != originalTemplate.Ref.Digest {
		t.Fatalf("unreferenced summarizer changed omitted template: %+v", omittedTemplate.Summarizer)
	}
}

func TestWorkerSummarizerDefaultsContextWindowRatioBeforeDigesting(t *testing.T) {
	t.Parallel()

	explicitRoot := copyCoreFixture(t)
	installWorkerSummarizer(t, explicitRoot)
	explicitSnapshot := mustLoad(t, explicitRoot)
	explicitTemplate, _ := explicitSnapshot.agentTemplateForTest("artifact_builder@1")

	defaultedRoot := copyCoreFixture(t)
	installWorkerSummarizer(t, defaultedRoot)
	replaceFile(t, templatePath(defaultedRoot), "    contextWindowRatio: 0.9\n", "")
	defaultedSnapshot := mustLoad(t, defaultedRoot)
	defaultedTemplate, _ := defaultedSnapshot.agentTemplateForTest("artifact_builder@1")

	if defaultedTemplate.Summarizer == nil ||
		defaultedTemplate.Summarizer.ContextWindowRatio != 0.9 ||
		defaultedTemplate.Ref.Digest != explicitTemplate.Ref.Digest {
		t.Fatalf("defaulted summarizer = %+v", defaultedTemplate.Summarizer)
	}
}

func TestWorkerSummarizerSnapshotIsDeepAndRemainsPinnedAcrossReload(t *testing.T) {
	t.Parallel()

	root := copyCoreFixture(t)
	installWorkerSummarizer(t, root)
	first := mustLoad(t, root)
	template, _ := first.agentTemplateForTest("artifact_builder@1")
	oldDigest := template.Ref.Digest
	*template.Summarizer.CumulativeBudget = 1
	template.Summarizer.ContextWindowRatio = 0.5

	again, _ := first.agentTemplateForTest("artifact_builder@1")
	if again.Summarizer == nil || *again.Summarizer.CumulativeBudget != 20_000 ||
		again.Summarizer.ContextWindowRatio != 0.9 {
		t.Fatalf("summarizer mutation leaked into Snapshot: %+v", again.Summarizer)
	}

	replaceFile(
		t, filepath.Join(root, "model-policies/terminal_summarizer.yaml"),
		"model: worker-summarizer-model", "model: worker-summarizer-model-v2",
	)
	second := mustLoad(t, root)
	oldWorkflow, _ := first.Workflow("artifact-copy@1")
	newWorkflow, _ := second.Workflow("artifact-copy@1")
	if oldWorkflow.Stages["copy"].ExecutionConfig.Agents["builder"].SummarizerModelPolicy.Model != "worker-summarizer-model" || newWorkflow.Stages["copy"].ExecutionConfig.Agents["builder"].SummarizerModelPolicy.Model != "worker-summarizer-model-v2" {
		t.Fatal("effective summary policies did not remain pinned")
	}
	pinnedAgain, _ := first.agentTemplateForTest("artifact_builder@1")
	newTemplate, _ := second.agentTemplateForTest("artifact_builder@1")
	if pinnedAgain.Ref.Digest != oldDigest ||
		newTemplate.Ref.Digest != oldDigest {
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
			replaceFile(t, filepath.Join(root, "workflows/artifact_copy.yaml"), "terminal_summarizer@1", "missing@1")
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
			root := copyCoreFixture(t)
			installWorkerSummarizer(t, root)
			test.mutate(t, root)
			snapshot, err := LoadUnionReadOnly(root, filepath.Join(t.TempDir(), "managed"))
			if err == nil || snapshot != nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("Load() = (%v, %v), want error containing %q", snapshot, err, test.want)
			}
		})
	}
}

func TestWorkerSummarizerRejectsEffectiveWorkflowPolicyBelowSoftThreshold(t *testing.T) {
	t.Parallel()

	root := copyCoreFixture(t)
	installWorkerSummarizer(t, root)
	writeFile(t, filepath.Join(root, "model-policies/test-low-worker.yaml"), []byte(lowWorkerPolicyYAML))
	writeWorkflowExecutionConfig(t, root, "artifact_copy.yaml", `{"workers":{"modelPolicy":"test-low-worker@1","summarizerModelPolicy":"terminal_summarizer@1"}}`)
	workflow, err := LoadUnionReadOnly(root, filepath.Join(t.TempDir(), "managed"))
	if err == nil || !strings.Contains(err.Error(), "below Worker maxTotalTokens") {
		t.Fatalf("Load() = (%+v, %v), want summarizer threshold error", workflow, err)
	}
}

func TestExecutionConfigCannotOverrideWorkerSummarizer(t *testing.T) {
	t.Parallel()

	for _, raw := range []string{
		`{"workers":{"summarizer":"terminal_summarizer@1"}}`,
		`{"stages":{"copy":{"agents":{"builder":{"summarizer":null}}}}}`,
	} {
		root := copyCoreFixture(t)
		writeWorkflowExecutionConfig(t, root, "artifact_copy.yaml", raw)
		if _, err := LoadUnionReadOnly(root, filepath.Join(t.TempDir(), "managed")); err == nil {
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
		"  toolsets:",
		workerSummarizerBlock+"  toolsets:",
	)
	err := filepath.WalkDir(filepath.Join(root, "workflows"), func(path string, entry os.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if !entry.IsDir() && strings.HasSuffix(path, ".yaml") {
			data := string(readFile(t, path))
			if strings.Contains(data, "template: artifact_builder@1") {
				data = strings.Replace(data, "    workers:\n", "    workers:\n      summarizerModelPolicy: terminal_summarizer@1\n", 1)
				writeFile(t, path, []byte(data))
			}
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
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
	return filepath.Join(root, "model-policies/test-worker.yaml")
}

const workerSummarizerBlock = `  summarizer:
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
  name: test-low-worker
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

func TestWorkerSummarizerIncludeReasoningResolvesPinsAndDigests(t *testing.T) {
	t.Parallel()

	load := func(t *testing.T, line string) *Snapshot {
		t.Helper()
		root := copyCoreFixture(t)
		installWorkerSummarizer(t, root)
		if line != "" {
			replaceFile(t, templatePath(root), "    cumulativeBudget: 20000\n", "    cumulativeBudget: 20000\n"+line+"\n")
		}
		return mustLoad(t, root)
	}
	template := func(t *testing.T, snapshot *Snapshot) contracts.ResolvedAgentTemplate {
		t.Helper()
		value, err := snapshot.agentTemplateForTest("artifact_builder@1")
		if err != nil {
			t.Fatal(err)
		}
		return value
	}
	omitted := template(t, load(t, ""))
	explicitFalse := template(t, load(t, "    includeReasoning: false"))
	enabledSnapshot := load(t, "    includeReasoning: true")
	enabled := template(t, enabledSnapshot)

	if omitted.Summarizer.IncludeReasoning || explicitFalse.Summarizer.IncludeReasoning ||
		!enabled.Summarizer.IncludeReasoning {
		t.Fatalf("resolved includeReasoning = %v/%v/%v, want false/false/true",
			omitted.Summarizer.IncludeReasoning, explicitFalse.Summarizer.IncludeReasoning,
			enabled.Summarizer.IncludeReasoning)
	}
	// Authoring omission and explicit false are the same body; enabling
	// reasoning is a different exact body.
	if omitted.Ref.Digest != explicitFalse.Ref.Digest || enabled.Ref.Digest == omitted.Ref.Digest {
		t.Fatalf("digests = %s/%s/%s", omitted.Ref.Digest, explicitFalse.Ref.Digest, enabled.Ref.Digest)
	}

	workflow, err := enabledSnapshot.ResolveRunWorkflow(
		t.Context(), "artifact-copy@1", developmentCredentialLookup(t, enabledSnapshot),
	)
	if err != nil {
		t.Fatal(err)
	}
	pinned := workflow.Stages["copy"].Agents["builder"].Template
	if pinned.Summarizer == nil || !pinned.Summarizer.IncludeReasoning || pinned.Ref.Digest != enabled.Ref.Digest {
		t.Fatalf("Run-resolved summarizer = %+v", pinned.Summarizer)
	}
	encoded, err := resolved.EncodeResolvedWorkflowSnapshot(workflow)
	if err != nil {
		t.Fatal(err)
	}
	reloaded, err := resolved.DecodeResolvedWorkflowSnapshot(encoded)
	if err != nil {
		t.Fatal(err)
	}
	if !reloaded.Stages["copy"].Agents["builder"].Template.Summarizer.IncludeReasoning {
		t.Fatal("persisted snapshot lost includeReasoning")
	}

	for _, test := range []struct {
		snapshot *Snapshot
		want     bool
	}{{load(t, ""), false}, {enabledSnapshot, true}} {
		resource, err := test.snapshot.Configuration(ConfigurationAgentTemplates, "artifact_builder@1")
		if err != nil {
			t.Fatal(err)
		}
		safe := resource.Body.(map[string]any)["summarizer"].(map[string]any)
		if safe["includeReasoning"] != test.want {
			t.Fatalf("public AgentTemplate summarizer = %#v, want includeReasoning %v", safe, test.want)
		}
	}
}

func TestWorkerSummarizerIncludeReasoningRejectsInvalidAuthoring(t *testing.T) {
	t.Parallel()

	for name, test := range map[string]struct{ line, want string }{
		"null":             {"    includeReasoning: null", "includeReasoning must be a boolean"},
		"tilde":            {"    includeReasoning: ~", "includeReasoning must be a boolean"},
		"empty":            {"    includeReasoning:", "includeReasoning must be a boolean"},
		"quoted string":    {`    includeReasoning: "true"`, "includeReasoning must be a boolean"},
		"YAML 1.1 yes":     {"    includeReasoning: yes", "includeReasoning must be a boolean"},
		"integer":          {"    includeReasoning: 1", "includeReasoning must be a boolean"},
		"tagged string":    {"    includeReasoning: !!bool yes", "includeReasoning must be a boolean"},
		"sequence":         {"    includeReasoning: [true]", "includeReasoning must be a boolean"},
		"duplicate":        {"    includeReasoning: true\n    includeReasoning: false", "already defined"},
		"case alias":       {"    IncludeReasoning: true", "not found"},
		"snake case alias": {"    include_reasoning: true", "not found"},
	} {
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			root := copyCoreFixture(t)
			installWorkerSummarizer(t, root)
			replaceFile(t, templatePath(root), "    cumulativeBudget: 20000\n", "    cumulativeBudget: 20000\n"+test.line+"\n")
			snapshot, err := LoadUnionReadOnly(root, filepath.Join(t.TempDir(), "managed"))
			if err == nil || snapshot != nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("Load() = (%v, %v), want error containing %q", snapshot, err, test.want)
			}
		})
	}
}

func TestWorkerSummarizerResolvedIncludeReasoningRequiresExplicitBoolean(t *testing.T) {
	t.Parallel()
	root := copyCoreFixture(t)
	installWorkerSummarizer(t, root)
	snapshot := mustLoad(t, root)
	workflow, err := snapshot.ResolveRunWorkflow(
		t.Context(), "artifact-copy@1", developmentCredentialLookup(t, snapshot),
	)
	if err != nil {
		t.Fatal(err)
	}
	encoded, err := resolved.EncodeResolvedWorkflowSnapshot(workflow)
	if err != nil {
		t.Fatal(err)
	}
	member := []byte(`"includeReasoning":false`)
	if !bytes.Contains(encoded, member) {
		t.Fatal("normalized false must be serialized explicitly")
	}
	reloaded, err := resolved.DecodeResolvedWorkflowSnapshot(encoded)
	if err != nil || reloaded.Stages["copy"].Agents["builder"].Template.Summarizer.IncludeReasoning {
		t.Fatalf("snapshot false round trip: %v", err)
	}
	for name, replacement := range map[string]string{
		"missing":     "",
		"null":        `,"includeReasoning":null`,
		"string":      `,"includeReasoning":"false"`,
		"number":      `,"includeReasoning":0`,
		"case alias":  `,"IncludeReasoning":false`,
		"snake alias": `,"include_reasoning":false`,
		"duplicate":   `,"includeReasoning":false,"includeReasoning":false`,
	} {
		t.Run(name, func(t *testing.T) {
			candidate := bytes.Replace(encoded, append([]byte(","), member...), []byte(replacement), 1)
			if bytes.Equal(candidate, encoded) {
				t.Fatal("snapshot fixture has no replaceable boolean member")
			}
			if _, err := resolved.DecodeResolvedWorkflowSnapshot(candidate); err == nil {
				t.Fatal("invalid resolved summarizer was accepted")
			}
		})
	}
}
