package config

import (
	"bytes"
	"encoding/json"
	"errors"
	"maps"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/configtest"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
	"github.com/grauwolf32/contractor/internal/securefile"
)

// coreFixtureRoot is the frozen catalog that behavioral tests load or copy. On
// a copy, configtest adds its test-* policies, artifact builder and
// artifact-copy Workflow; testdata/<topic> directories add focused slices.
const coreFixtureRoot = "testdata/valid"

func TestLoadResolvesAndPinsCatalogReferences(t *testing.T) {
	t.Parallel()

	snapshot := mustLoad(t, coreFixtureRoot)

	policy, err := snapshot.ModelPolicy("worker@1")
	if err != nil {
		t.Fatalf("resolve ModelPolicy: %v", err)
	}
	assertDigest(t, policy.Ref.Digest)
	if policy.Model != "worker-model" || policy.MaxOutputTokens != 4096 ||
		policy.MaxModelCalls != 8 || policy.MaxToolCalls != 16 || policy.MaxTotalTokens != 32768 ||
		policy.Temperature == nil || *policy.Temperature != 0.1 {
		t.Fatalf("unexpected resolved ModelPolicy: %+v", policy)
	}

	template, err := snapshot.AgentTemplate("artifact_builder@1")
	if err != nil {
		t.Fatalf("resolve AgentTemplate: %v", err)
	}
	assertDigest(t, template.Ref.Digest)
	assertDigest(t, template.Instructions.Digest)
	if template.ModelPolicy.Ref.Digest != policy.Ref.Digest {
		t.Fatalf("template policy digest = %q, want %q", template.ModelPolicy.Ref.Digest, policy.Ref.Digest)
	}
	if len(template.Toolsets) != 1 || template.Toolsets[0].Ref != (contracts.ToolsetRef{ToolsetID: "run-artifacts", Version: "1"}) ||
		!equalStrings(template.Toolsets[0].Tools, []string{"list_artifacts", "read_artifact", "write_artifact"}) {
		t.Fatalf("selected tools = %+v", template.Toolsets)
	}

	workflow, err := snapshot.Workflow("artifact-copy@1")
	if err != nil {
		t.Fatalf("resolve Workflow: %v", err)
	}
	stage := workflow.Stages[workflow.EntryStage]
	if stage.Planner != (PlannerRef{PlannerID: "passthrough", Version: "1"}) {
		t.Fatalf("planner = %+v", stage.Planner)
	}
	if stage.Agents["builder"].Template.Ref != template.Ref {
		t.Fatalf("Workflow did not pin exact AgentTemplate: %+v", stage.Agents["builder"].Template.Ref)
	}
	if stage.WorkflowOutputs["result"] != "copied" || stage.On.Succeeded.Kind != TransitionSucceed {
		t.Fatalf("unexpected resolved Stage: %+v", stage)
	}
	from := stage.Result.Artifacts["copied"].From
	if from == nil || *from != (ArtifactBinding{Namespace: "builder", Name: "copied"}) {
		t.Fatalf("Stage result binding = %+v", from)
	}

	instructions, err := snapshot.Instructions("instructions/copy-planner.md")
	if err != nil {
		t.Fatalf("resolve instructions: %v", err)
	}
	if instructions != stage.Instructions {
		t.Fatalf("resolved instructions differ: %+v != %+v", instructions, stage.Instructions)
	}
}

// Visible-name collisions between Toolsets are TestToolsetVisibleNameCollision.
func TestTaintAnnotationsToolSelectionIsStrict(t *testing.T) {
	t.Parallel()

	current := &loader{}
	selected, err := current.resolveToolsets(&[]toolsetSelectionSource{{
		Ref: "taint-annotations@1", Tools: []string{"annotate_validate", "annotate_trace"},
	}})
	if err != nil {
		t.Fatalf("valid taint annotation selection: %v", err)
	}
	if got, want := selected[0].Tools, []string{"annotate_trace", "annotate_validate"}; !equalStrings(got, want) {
		t.Fatalf("normalized taint annotation tools = %v, want %v", got, want)
	}

	if _, err := current.resolveToolsets(&[]toolsetSelectionSource{{
		Ref: "taint-annotations@1", Tools: []string{"annotate"},
	}}); err == nil || !strings.Contains(err.Error(), "does not export selected tool") {
		t.Fatalf("unknown taint annotation tool error = %v", err)
	}
}

func TestCodeAnalysisToolSelectionIsStrict(t *testing.T) {
	t.Parallel()

	current := &loader{}
	selected, err := current.resolveToolsets(&[]toolsetSelectionSource{{
		Ref: "code-analysis@1", Tools: []string{"search_def", "find_callers"},
	}})
	if err != nil {
		t.Fatalf("valid code-analysis selection: %v", err)
	}
	if got, want := selected[0].Tools, []string{"find_callers", "search_def"}; !equalStrings(got, want) {
		t.Fatalf("normalized code-analysis tools = %v, want %v", got, want)
	}

	if _, err := current.resolveToolsets(&[]toolsetSelectionSource{{
		Ref: "code-analysis@1", Tools: []string{"search_definition"},
	}}); err == nil || !strings.Contains(err.Error(), "does not export selected tool") {
		t.Fatalf("unknown code-analysis tool error = %v", err)
	}
}

func TestCodeAnalysisSelectionLoadsAndDigestsInTemplate(t *testing.T) {
	t.Parallel()

	root := copyCoreFixture(t)
	writeFile(t, filepath.Join(root, "agent-templates/source_graph_analyst.yaml"), []byte(codeAnalysisTemplateYAML))
	snapshot := mustLoad(t, root)
	template, err := snapshot.AgentTemplate("source_graph_analyst@1")
	if err != nil {
		t.Fatal(err)
	}
	assertDigest(t, template.Ref.Digest)
	found := false
	for _, toolset := range template.Toolsets {
		if toolset.Ref.ToolsetID != "code-analysis" {
			continue
		}
		found = true
		want := []string{
			"attack_surface", "complexity_hotspots", "entrypoint_paths_to",
			"find_callees", "find_callers", "find_symbol", "functions_that_raise",
			"graph_summary", "list_symbols", "paths_between", "search_def",
		}
		if !equalStrings(toolset.Tools, want) {
			t.Fatalf("normalized code-analysis tools = %v, want %v", toolset.Tools, want)
		}
	}
	if !found {
		t.Fatal("resolved template omitted code-analysis@1")
	}
}

// codeAnalysisTemplateYAML selects every code-analysis@1 tool in authoring
// order so the loader must normalize the selection before digesting.
const codeAnalysisTemplateYAML = `apiVersion: contractor/v1alpha1
kind: AgentTemplate
metadata:
  name: source_graph_analyst
  version: "1"
spec:
  description: Inspects the source graph of a workspace
  runtime: adk@1
  instructions:
    ref: instructions/artifact-builder.md
  modelPolicy: test-worker@1
  toolsets:
    - ref: code-analysis@1
      tools:
        - search_def
        - list_symbols
        - graph_summary
        - find_symbol
        - find_callers
        - find_callees
        - paths_between
        - entrypoint_paths_to
        - attack_surface
        - complexity_hotspots
        - functions_that_raise
  sandboxProfile: local-workdir@1
`

// TestRequiredRuntimeAdaptersFollowToolChannels checks how the channels the
// descriptors declare become placement requirements: every http-tools@1 tool
// uses the direct or optionally proxied HTTP client and needs no adapter,
// while every Caido tool needs the typed Caido client. The tool lists and
// their channels are the ones TestSharedToolsetDescriptorParity pins.
func TestRequiredRuntimeAdaptersFollowToolChannels(t *testing.T) {
	t.Parallel()
	for ref, want := range map[contracts.ToolsetRef][]contracts.RuntimeAdapterRef{
		{ToolsetID: "http-tools", Version: "1"}: nil,
		{ToolsetID: "caido", Version: "1"}:      {contracts.RuntimeAdapterCaidoGraphQL},
	} {
		tools := registeredDescriptors().Toolsets[ref.ToolsetID+"@"+ref.Version].Tools
		if len(tools) == 0 {
			t.Fatalf("Toolset %s has no tools", ref.ToolsetID)
		}
		for _, tool := range tools {
			selected := contracts.ResolvedAgentTemplate{Toolsets: []contracts.ToolsetSelection{{Ref: ref, Tools: []string{tool}}}}
			adapters, err := RequiredRuntimeAdaptersForTemplate(selected)
			if err != nil || !slices.Equal(adapters, want) {
				t.Errorf("%s adapter requirements = (%v, %v), want %v", tool, adapters, err, want)
			}
		}
	}
}

func TestToolsetDescriptorRejectsInvalidInfrastructureChannels(t *testing.T) {
	t.Parallel()

	for name, channels := range map[string]map[string][]infrastructureChannel{
		"unknown tool": {"missing": {channelRuntimeHTTPClient}},
		"empty":        {"read_artifact": {}},
		"unknown channel": {
			"read_artifact": {infrastructureChannel("ambient-network")},
		},
		"duplicate": {"read_artifact": {channelRuntimeHTTPClient, channelRuntimeHTTPClient}},
	} {
		t.Run(name, func(t *testing.T) {
			descriptors := builtInDescriptors()
			descriptor := descriptors.Toolsets["run-artifacts@1"]
			descriptor.InfrastructureChannels = channels
			descriptors.Toolsets["run-artifacts@1"] = descriptor
			if _, err := normalizeDescriptors(descriptors); err == nil {
				t.Fatal("normalizeDescriptors accepted invalid infrastructure channels")
			}
		})
	}
}

// toolsetDescriptorFixture is one Toolset of api/testdata/toolset-descriptors.json.
type toolsetDescriptorFixture struct {
	Tools                  []string                           `json:"tools"`
	InfrastructureChannels map[string][]infrastructureChannel `json:"infrastructureChannels"`
	RequiresWorkspace      bool                               `json:"requiresWorkspace"`
	RequiredSandboxProfile *string                            `json:"requiredSandboxProfile"`
	ActiveCheckTools       []string                           `json:"activeCheckTools"`
	FindingProposalTools   []string                           `json:"findingProposalTools"`
	RequiredWorkflowInputs map[string]struct {
		Input     string `json:"input"`
		MediaType string `json:"mediaType"`
	} `json:"requiredWorkflowInputs"`
}

// TestSharedToolsetDescriptorParity holds every Server Toolset descriptor to
// api/testdata/toolset-descriptors.json, which the Runtime's factories are
// held to as well (runtime/tests/test_capabilities.py), so a tool or Toolset
// added on one side only fails.
func TestSharedToolsetDescriptorParity(t *testing.T) {
	t.Parallel()
	var fixture struct {
		Description string                              `json:"description"`
		Toolsets    map[string]toolsetDescriptorFixture `json:"toolsets"`
	}
	decoder := json.NewDecoder(bytes.NewReader(contractstest.ReadFile(t, "api", "testdata", "toolset-descriptors.json")))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&fixture); err != nil {
		t.Fatal(err)
	}
	normalized := registeredDescriptors()
	if got, want := slices.Sorted(maps.Keys(normalized.Toolsets)), slices.Sorted(maps.Keys(fixture.Toolsets)); !slices.Equal(got, want) {
		t.Fatalf("Toolset descriptors = %v, shared fixture = %v", got, want)
	}
	for ref, shared := range fixture.Toolsets {
		profile := ""
		if shared.RequiredSandboxProfile != nil {
			profile = *shared.RequiredSandboxProfile
		}
		// The fixture spells an absent collection as an empty one.
		want := toolsetDescriptor{
			RequiredSandboxProfile: profile, RequiresWorkspace: shared.RequiresWorkspace,
			Tools: shared.Tools, InfrastructureChannels: shared.InfrastructureChannels,
		}
		if len(shared.ActiveCheckTools) > 0 {
			want.ActiveCheckTools = shared.ActiveCheckTools
		}
		if len(shared.FindingProposalTools) > 0 {
			want.FindingProposalTools = shared.FindingProposalTools
		}
		for tool, input := range shared.RequiredWorkflowInputs {
			if want.RequiredWorkflowInputs == nil {
				want.RequiredWorkflowInputs = map[string]workflowInputRequirement{}
			}
			want.RequiredWorkflowInputs[tool] = workflowInputRequirement{Input: input.Input, MediaType: input.MediaType}
		}
		if got := normalized.Toolsets[ref]; !reflect.DeepEqual(got, want) {
			t.Errorf("Toolset %s descriptor = %+v, shared fixture = %+v", ref, got, want)
		}
	}
}

func TestStreamlineRequiresExactlyOneLogicalAgent(t *testing.T) {
	root := copyCoreFixture(t, "workflow-examples")
	path := filepath.Join(root, "workflows", "streamline_review_workflow.yaml")
	example := readFile(t, path)
	snapshot := mustLoad(t, root)
	workflow, err := snapshot.Workflow("streamline-review@2")
	if err != nil {
		t.Fatal(err)
	}
	stage := workflow.Stages[workflow.EntryStage]
	stage.Agents["second"] = stage.Agents["reviewer"]
	workflow.Stages[workflow.EntryStage] = stage
	if err := ValidateWorkflowGraph(workflow); err == nil ||
		!strings.Contains(err.Error(), "streamline@1 requires exactly one logical Agent binding") {
		t.Fatalf("ValidateWorkflowGraph error = %v", err)
	}

	secondBinding := `        second:
          template: artifact_builder@1
          namespace: second
`
	invalid := strings.Replace(string(example), "      context:\n", secondBinding+"      context:\n", 1)
	writeFile(t, path, []byte(invalid))
	loaded, err := Load(root)
	if err == nil || loaded != nil ||
		!strings.Contains(err.Error(), "streamline@1 requires exactly one logical Agent binding") {
		t.Fatalf("Load() = (%v, %v), want Streamline cardinality error", loaded, err)
	}
}

func TestRouterRequiresAtLeastOneLogicalAgent(t *testing.T) {
	snapshot := mustLoad(t, copyCoreFixture(t, "workflow-examples"))
	workflow, err := snapshot.Workflow("router-openapi@2")
	if err != nil {
		t.Fatal(err)
	}
	stage := workflow.Stages[workflow.EntryStage]
	stage.Agents = map[string]ResolvedAgentBinding{}
	workflow.Stages[workflow.EntryStage] = stage
	if err := ValidateWorkflowGraph(workflow); err == nil ||
		!strings.Contains(err.Error(), "router@1 requires at least one logical Agent binding") {
		t.Fatalf("ValidateWorkflowGraph error = %v", err)
	}
}

func TestStoredFixtures(t *testing.T) {
	t.Parallel()

	valid := mustLoad(t, "testdata/valid")
	if got, want := valid.Counts(), (Counts{
		Workflows: 1, AgentTemplates: 1, ModelPolicies: 1, LLMGateways: 1,
		ExecutionConfigs: 1, AuditProfiles: 0, Instructions: 2,
	}); got != want {
		t.Fatalf("valid fixture Counts() = %+v, want %+v", got, want)
	}

	invalid := []string{
		"unknown-field-policy.yaml",
		"duplicate-key-policy.yaml",
		"multiple-documents-policy.yaml",
		"wrong-kind-policy.yaml",
	}
	for _, name := range invalid {
		t.Run(name, func(t *testing.T) {
			root := copyCoreFixture(t)
			fixture := readFile(t, filepath.Join("testdata/invalid", name))
			writeFile(t, filepath.Join(root, "model-policies/test-worker.yaml"), fixture)
			snapshot, err := Load(root)
			if err == nil || snapshot != nil {
				t.Fatalf("Load() = (%v, %v), want (nil, error)", snapshot, err)
			}
		})
	}
}

func TestModelPolicyWorkerBudgetsAreRequiredAndBounded(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name, old, replacement, want string
	}{
		{"missing output tokens", "  maxOutputTokens: 4096\n", "", "maxOutputTokens"},
		{"zero output tokens", "maxOutputTokens: 4096", "maxOutputTokens: 0", "spec.maxOutputTokens"},
		{"missing model calls", "  maxModelCalls: 8\n", "", "maxModelCalls"},
		{"zero model calls", "maxModelCalls: 8", "maxModelCalls: 0", "maxModelCalls"},
		{"negative tool calls", "maxToolCalls: 16", "maxToolCalls: -1", "spec.maxToolCalls"},
		{"fractional tool calls", "maxToolCalls: 16", "maxToolCalls: 1.5", "decode strict YAML"},
		{"too many model calls", "maxModelCalls: 8", "maxModelCalls: 1001", "spec.maxModelCalls"},
		{"too many tool calls", "maxToolCalls: 16", "maxToolCalls: 10001", "spec.maxToolCalls"},
		{"zero Planner Worker calls", "  maxToolCalls: 16\n", "  maxToolCalls: 16\n  maxWorkerCalls: 0\n", "spec.maxWorkerCalls"},
		{"missing total tokens", "  maxTotalTokens: 32768\n", "", "maxTotalTokens"},
		{"too many total tokens", "maxTotalTokens: 32768", "maxTotalTokens: 100000001", "spec.maxTotalTokens"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			root := copyCoreFixture(t)
			replaceFile(t, filepath.Join(root, "model-policies/test-worker.yaml"), test.old, test.replacement)
			if snapshot, err := Load(root); err == nil || snapshot != nil ||
				!strings.Contains(err.Error(), test.want) {
				t.Fatalf("Load() = (%v, %v), want error containing %q", snapshot, err, test.want)
			}
		})
	}
}

func TestStrictManifestFailuresReturnNoSnapshot(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name   string
		mutate func(*testing.T, string)
		want   []string
	}{
		{
			name: "unknown field",
			mutate: func(t *testing.T, root string) {
				appendFile(t, filepath.Join(root, "model-policies/test-worker.yaml"), "unknownField: true\n")
			},
			want: []string{"model-policies/test-worker.yaml", "unknownField"},
		},
		{
			name: "duplicate mapping key",
			mutate: func(t *testing.T, root string) {
				appendFile(t, filepath.Join(root, "model-policies/test-worker.yaml"), "kind: ModelPolicy\n")
			},
			want: []string{"model-policies/test-worker.yaml", "already defined"},
		},
		{
			name: "multiple documents",
			mutate: func(t *testing.T, root string) {
				appendFile(t, filepath.Join(root, "model-policies/test-worker.yaml"), `---
apiVersion: contractor/v1alpha1
kind: ModelPolicy
metadata: {name: other, version: "1"}
spec: {model: other, maxOutputTokens: 1}
`)
			},
			want: []string{"model-policies/test-worker.yaml", "exactly one"},
		},
		{
			name: "empty document",
			mutate: func(t *testing.T, root string) {
				writeFile(t, filepath.Join(root, "model-policies/test-worker.yaml"), nil)
			},
			want: []string{"model-policies/test-worker.yaml", "exactly one"},
		},
		{
			name: "wrong subtree kind",
			mutate: func(t *testing.T, root string) {
				replaceFile(t, filepath.Join(root, "model-policies/test-worker.yaml"), "kind: ModelPolicy", "kind: Workflow")
			},
			want: []string{"model-policies/test-worker.yaml", "does not match ModelPolicy subtree"},
		},
		{
			name: "duplicate identity",
			mutate: func(t *testing.T, root string) {
				source := readFile(t, filepath.Join(root, "model-policies/test-worker.yaml"))
				writeFile(t, filepath.Join(root, "model-policies/z/duplicate.yaml"), source)
			},
			want: []string{"model-policies/z/duplicate.yaml", "duplicate ModelPolicy identity test-worker@1"},
		},
		{
			name: "unknown runtime",
			mutate: func(t *testing.T, root string) {
				replaceFile(t, filepath.Join(root, "agent-templates/artifact_builder.yaml"), "runtime: adk@1", "runtime: missing@1")
			},
			want: []string{"agent-templates/artifact_builder.yaml", "unknown WorkerRuntime"},
		},
		{
			name: "unknown tool",
			mutate: func(t *testing.T, root string) {
				replaceFile(t, filepath.Join(root, "agent-templates/artifact_builder.yaml"), "read_artifact", "delete_artifact")
			},
			want: []string{"agent-templates/artifact_builder.yaml", "does not export selected tool"},
		},
		{
			name: "unknown planner",
			mutate: func(t *testing.T, root string) {
				replaceFile(t, filepath.Join(root, "workflows/artifact_copy.yaml"), "planner: passthrough@1", "planner: missing@1")
			},
			want: []string{"workflows/artifact_copy.yaml", "unknown PlannerFactory"},
		},
		{
			name: "missing required boolean",
			mutate: func(t *testing.T, root string) {
				replaceFile(t, filepath.Join(root, "workflows/artifact_copy.yaml"), "    source:\n      required: true\n      mediaTypes", "    source:\n      mediaTypes")
			},
			want: []string{"workflows/artifact_copy.yaml", "spec.inputs.source.required is required"},
		},
		{
			name: "incompatible output mapping",
			mutate: func(t *testing.T, root string) {
				replaceFile(t, filepath.Join(root, "workflows/artifact_copy.yaml"), "  outputs:\n    result:\n      required: true\n      mediaTypes: [text/plain]", "  outputs:\n    result:\n      required: true\n      mediaTypes: [application/json]")
			},
			want: []string{"workflows/artifact_copy.yaml", "incompatible"},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			root := copyCoreFixture(t)
			test.mutate(t, root)
			snapshot, err := Load(root)
			if err == nil || snapshot != nil {
				t.Fatalf("Load() = (%v, %v), want (nil, error)", snapshot, err)
			}
			for _, expected := range test.want {
				if !strings.Contains(err.Error(), expected) {
					t.Fatalf("error %q does not contain %q", err, expected)
				}
			}
		})
	}
}

func TestInstructionPathAndFilesystemContainment(t *testing.T) {
	t.Parallel()

	invalid := []string{
		"", "/instructions/a.md", "https://example.test/a", `instructions\a.md`,
		"instructions//a.md", "instructions/./a.md", "instructions/../a.md", "instructions",
	}
	for _, value := range invalid {
		if _, err := validateInstructionRef(value); err == nil {
			t.Errorf("validateInstructionRef(%q) succeeded", value)
		}
	}
	if got, err := validateInstructionRef("instructions/team/a.md"); err != nil || got != "instructions/team/a.md" {
		t.Fatalf("valid instruction ref = (%q, %v)", got, err)
	}

	t.Run("symlink escape", func(t *testing.T) {
		root := copyCoreFixture(t)
		outside := filepath.Join(filepath.Dir(root), "outside.md")
		writeFile(t, outside, []byte("outside\n"))
		target := filepath.Join(root, "instructions/artifact-builder.md")
		if err := os.Remove(target); err != nil {
			t.Fatal(err)
		}
		if err := os.Symlink(outside, target); err != nil {
			t.Fatal(err)
		}
		snapshot, err := Load(root)
		if err == nil || snapshot != nil || !strings.Contains(err.Error(), "is a symlink") {
			t.Fatalf("Load() = (%v, %v), want strict symlink error", snapshot, err)
		}
	})

	t.Run("invalid utf8", func(t *testing.T) {
		root := copyCoreFixture(t)
		writeFile(t, filepath.Join(root, "instructions/artifact-builder.md"), []byte{0xff, 0xfe})
		snapshot, err := Load(root)
		if err == nil || snapshot != nil || !strings.Contains(err.Error(), "strict UTF-8") {
			t.Fatalf("Load() = (%v, %v), want UTF-8 error", snapshot, err)
		}
	})
}

func TestManifestDiscoveryRejectsSymlinks(t *testing.T) {
	t.Parallel()

	root := copyCoreFixture(t)
	if err := os.Symlink("test-worker.yaml", filepath.Join(root, "model-policies/link.yaml")); err != nil {
		t.Fatal(err)
	}
	if snapshot, err := Load(root); err == nil || snapshot != nil ||
		!strings.Contains(err.Error(), "is a symlink") {
		t.Fatalf("Load() = (%v, %v), want strict symlink error", snapshot, err)
	}
}

// Every manifest and instruction resource is read with one file bound
// (LIM-CAT-5, CAT-R77): a file of exactly the bound loads, one more byte
// rejects the complete set.
func TestCatalogFileReadsAreBounded(t *testing.T) {
	t.Parallel()

	t.Run("instruction", func(t *testing.T) {
		root := copyCoreFixture(t)
		path := filepath.Join(root, "instructions", "padding.md")
		writeFile(t, path, bytes.Repeat([]byte("a"), maximumCatalogFileBytes))
		if _, err := Load(root); err != nil {
			t.Fatalf("instruction at the bound: %v", err)
		}
		writeFile(t, path, bytes.Repeat([]byte("a"), maximumCatalogFileBytes+1))
		snapshot, err := Load(root)
		if snapshot != nil || !errors.Is(err, securefile.ErrTooLarge) ||
			!strings.Contains(err.Error(), `read instruction "instructions/padding.md"`) {
			t.Fatalf("instruction over the bound = (%v, %v)", snapshot, err)
		}
	})
	t.Run("manifest", func(t *testing.T) {
		root := copyCoreFixture(t)
		path := filepath.Join(root, "model-policies", "test-worker.yaml")
		original := readFile(t, path)
		// A trailing YAML comment keeps the manifest valid at any length.
		padded := func(size int) []byte {
			data := append(append([]byte(nil), original...), "\n#"...)
			return append(data, bytes.Repeat([]byte("x"), size-len(data))...)
		}
		writeFile(t, path, padded(maximumCatalogFileBytes))
		if _, err := Load(root); err != nil {
			t.Fatalf("manifest at the bound: %v", err)
		}
		writeFile(t, path, padded(maximumCatalogFileBytes+1))
		snapshot, err := Load(root)
		if snapshot != nil || !errors.Is(err, securefile.ErrTooLarge) ||
			!strings.Contains(err.Error(), "model-policies/test-worker.yaml: read manifest") {
			t.Fatalf("manifest over the bound = (%v, %v)", snapshot, err)
		}
	})
}

// A persisted Workflow snapshot is read with the same complete graph check as
// a catalog load (CAT-R78), so no reader validates it again and none skips it.
func TestDecodeResolvedWorkflowSnapshotChecksTheWholeGraph(t *testing.T) {
	t.Parallel()

	snapshot := mustLoad(t, copyCoreFixture(t))
	workflow, err := snapshot.Workflow("artifact-copy@1")
	if err != nil {
		t.Fatal(err)
	}
	encoded, err := json.Marshal(workflow)
	if err != nil {
		t.Fatal(err)
	}
	if decoded, err := DecodeResolvedWorkflowSnapshot(encoded); err != nil || !reflect.DeepEqual(decoded, workflow) {
		t.Fatalf("valid snapshot = (%+v, %v)", decoded, err)
	}
	for name, tamper := range map[string]func(*ResolvedWorkflow){
		"missing entry Stage": func(workflow *ResolvedWorkflow) { workflow.EntryStage = "missing" },
		"undeclared Workflow output": func(workflow *ResolvedWorkflow) {
			workflow.Stages["copy"].WorkflowOutputs["undeclared"] = "copied"
		},
		"transition to a missing Stage": func(workflow *ResolvedWorkflow) {
			stage := workflow.Stages["copy"]
			stage.On.Succeeded = TransitionAction{Kind: TransitionNext, NextStage: "missing"}
			workflow.Stages["copy"] = stage
		},
		"unregistered Planner": func(workflow *ResolvedWorkflow) {
			stage := workflow.Stages["copy"]
			stage.Planner = PlannerRef{PlannerID: "passthrough", Version: "2"}
			workflow.Stages["copy"] = stage
		},
	} {
		tampered := cloneWorkflow(workflow)
		tamper(&tampered)
		encoded, err := json.Marshal(tampered)
		if err != nil {
			t.Fatal(err)
		}
		if decoded, err := DecodeResolvedWorkflowSnapshot(encoded); err == nil ||
			!strings.HasPrefix(err.Error(), "persisted Workflow: ") || !reflect.DeepEqual(decoded, ResolvedWorkflow{}) {
			t.Errorf("%s: DecodeResolvedWorkflowSnapshot = (%+v, %v)", name, decoded, err)
		}
	}
}

func TestToolsetVisibleNameCollision(t *testing.T) {
	t.Parallel()

	root := copyCoreFixture(t)
	path := filepath.Join(root, "agent-templates/artifact_builder.yaml")
	// Both registered finding publishers export the model-visible "finding".
	replaceFile(t, path, "  sandboxProfile: local-workdir@1", `    - ref: security-findings@1
      tools: [finding]
    - ref: security-findings-code@1
      tools: [finding]
  sandboxProfile: local-workdir@1`)
	snapshot, err := Load(root)
	if err == nil || snapshot != nil || !strings.Contains(err.Error(), "collides between Toolsets") {
		t.Fatalf("Load() = (%v, %v), want collision error", snapshot, err)
	}
}

func TestWorkflowGraphLoadsMultiStageAndBoundedRetry(t *testing.T) {
	t.Parallel()
	root := copyCoreFixture(t)
	writeFile(t, filepath.Join(root, "workflows/artifact_copy.yaml"), []byte(multiStageWorkflowYAML))

	snapshot := mustLoad(t, root)
	workflow, err := snapshot.Workflow("artifact-copy@1")
	if err != nil {
		t.Fatal(err)
	}
	retry := workflow.Stages["build"].On.Failed.Retry
	if len(workflow.Stages) != 2 || retry == nil || retry.MaxAttempts != 3 ||
		retry.Then.Kind != TransitionNext || retry.Then.NextStage != "review" {
		t.Fatalf("resolved multi-Stage Workflow = %+v", workflow)
	}
}

func TestWorkflowAutomaticRetryMayExceed1024Attempts(t *testing.T) {
	root := copyCoreFixture(t)
	writeFile(t, filepath.Join(root, "workflows/artifact_copy.yaml"), []byte(strings.Replace(
		multiStageWorkflowYAML, "maxAttempts: 3", "maxAttempts: 1025", 1,
	)))
	workflow, err := mustLoad(t, root).Workflow("artifact-copy@1")
	if err != nil {
		t.Fatal(err)
	}
	if retry := workflow.Stages["build"].On.Failed.Retry; retry == nil || retry.MaxAttempts != 1025 {
		t.Fatalf("automatic retry configuration was capped: %+v", retry)
	}
}

func TestWorkflowPrimaryMarkerIsAcceptedOnlyOnWorkflowOutputs(t *testing.T) {
	t.Run("output", func(t *testing.T) {
		root := copyCoreFixture(t)
		path := filepath.Join(root, "workflows/artifact_copy.yaml")
		replaceFile(t, path,
			"  outputs:\n    result:\n      required: true\n      mediaTypes: [text/plain]",
			"  outputs:\n    result:\n      required: true\n      primary: true\n      mediaTypes: [text/plain]",
		)
		workflow, err := mustLoad(t, root).Workflow("artifact-copy@1")
		if err != nil || !workflow.Outputs["result"].Primary || workflow.Inputs["source"].Primary ||
			workflow.Stages["copy"].Result.Artifacts["copied"].Primary {
			t.Fatalf("resolved primary output marker = (%+v, %v)", workflow, err)
		}
	})

	for _, test := range []struct {
		name, old, replacement, field string
	}{
		{
			name: "input",
			old:  "  inputs:\n    source:\n      required: true\n      mediaTypes: [text/plain]",
			replacement: "  inputs:\n    source:\n      required: true\n      primary: true\n" +
				"      mediaTypes: [text/plain]",
			field: "spec.inputs.source.primary",
		},
		{
			name: "Stage result",
			old:  "          copied:\n            required: true\n            mediaTypes: [text/plain]",
			replacement: "          copied:\n            required: true\n            primary: true\n" +
				"            mediaTypes: [text/plain]",
			field: "result.artifacts.copied.primary",
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			root := copyCoreFixture(t)
			path := filepath.Join(root, "workflows/artifact_copy.yaml")
			replaceFile(t, path, test.old, test.replacement)
			snapshot, err := Load(root)
			if err == nil || snapshot != nil || !strings.Contains(err.Error(), test.field) ||
				!strings.Contains(err.Error(), "allowed only for Workflow outputs") {
				t.Fatalf("Load() = (%v, %v), want output-only primary error", snapshot, err)
			}
		})
	}
}

func TestWorkflowGraphRejectsNextCycle(t *testing.T) {
	t.Parallel()
	root := copyCoreFixture(t)
	cyclic := strings.Replace(
		multiStageWorkflowYAML,
		"    review:\n"+reviewStageYAML,
		"    review:\n"+strings.Replace(reviewStageYAML, "succeed: {}", "next: build", 1),
		1,
	)
	writeFile(t, filepath.Join(root, "workflows/artifact_copy.yaml"), []byte(cyclic))

	snapshot, err := Load(root)
	if err == nil || snapshot != nil || !strings.Contains(err.Error(), "Cycle") {
		t.Fatalf("Load(cycle) = (%v, %v)", snapshot, err)
	}
}

func TestWorkflowGraphRejectsUnreachableStage(t *testing.T) {
	t.Parallel()
	root := copyCoreFixture(t)
	workflow := multiStageWorkflowYAML + "    orphan:\n" + reviewStageYAML
	writeFile(t, filepath.Join(root, "workflows/artifact_copy.yaml"), []byte(workflow))

	snapshot, err := Load(root)
	if err == nil || snapshot != nil || !strings.Contains(err.Error(), "unreachable") {
		t.Fatalf("Load(unreachable) = (%v, %v)", snapshot, err)
	}
}

func TestWorkflowTransitionRejectsSuccessPathWithoutRequiredOutput(t *testing.T) {
	t.Parallel()
	root := copyCoreFixture(t)
	workflow := strings.Replace(
		multiStageWorkflowYAML,
		"      workflowOutputs:\n        result: copied\n",
		"",
		1,
	)
	writeFile(t, filepath.Join(root, "workflows/artifact_copy.yaml"), []byte(workflow))

	snapshot, err := Load(root)
	if err == nil || snapshot != nil || !strings.Contains(err.Error(), "without required output") {
		t.Fatalf("Load(missing output path) = (%v, %v)", snapshot, err)
	}
}

func TestWorkflowTransitionRejectsOptionalResultAsRequiredOutput(t *testing.T) {
	t.Parallel()

	root := copyCoreFixture(t)
	workflow := strings.Replace(
		multiStageWorkflowYAML,
		"          copied: {required: true, mediaTypes: [text/plain], from: {namespace: builder, name: copied}}\n      workflowOutputs:",
		"          copied: {required: false, mediaTypes: [text/plain], from: {namespace: builder, name: copied}}\n      workflowOutputs:",
		1,
	)
	writeFile(t, filepath.Join(root, "workflows/artifact_copy.yaml"), []byte(workflow))
	if snapshot, err := Load(root); err == nil || snapshot != nil ||
		!strings.Contains(err.Error(), "without required output") {
		t.Fatalf("Load() = (%v, %v), want required output error", snapshot, err)
	}
}

func TestWorkflowResultBindingsAreTrustedAndVersionless(t *testing.T) {
	t.Parallel()

	path := "workflows/artifact_copy.yaml"
	for _, test := range []struct {
		name        string
		old         string
		replacement string
		fragment    string
	}{
		{
			name:        "missing",
			old:         "            from: {namespace: builder, name: copied}\n",
			replacement: "",
			fragment:    "result.artifacts.copied.from is required",
		},
		{
			name:        "unassigned namespace",
			old:         "from: {namespace: builder, name: copied}",
			replacement: "from: {namespace: reviewer, name: copied}",
			fragment:    "is not assigned to a Stage Agent",
		},
		{
			name:        "reserved namespace",
			old:         "from: {namespace: builder, name: copied}",
			replacement: "from: {namespace: outputs, name: copied}",
			fragment:    "Runtime-reserved binding",
		},
		{
			name:        "revision",
			old:         "from: {namespace: builder, name: copied}",
			replacement: "from: {namespace: builder, name: copied, revision: invented}",
			fragment:    "field revision not found",
		},
		{
			name:        "workflow input",
			old:         "      mediaTypes: [text/plain]\n\n  outputs:",
			replacement: "      mediaTypes: [text/plain]\n      from: {namespace: builder, name: copied}\n\n  outputs:",
			fragment:    "allowed only for Stage result artifacts",
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			root := copyCoreFixture(t)
			replaceFile(t, filepath.Join(root, path), test.old, test.replacement)
			snapshot, err := Load(root)
			if err == nil || snapshot != nil || !strings.Contains(err.Error(), test.fragment) {
				t.Fatalf("Load() = (%v, %v), want %q", snapshot, err, test.fragment)
			}
		})
	}
}

const multiStageWorkflowYAML = `apiVersion: contractor/v1alpha1
kind: Workflow
metadata:
  name: artifact-copy
  version: "1"
spec:
  parameters: {}
  inputs:
    source: {required: true, mediaTypes: [text/plain]}
  outputs:
    result: {required: true, mediaTypes: [text/plain]}
  executionConfig:
    workers: {llmGateway: local-litellm@1}
  entryStage: build
  stages:
    build:
      objective: Build a candidate
      instructions: {ref: instructions/copy-planner.md}
      planner: passthrough@1
      agents:
        builder: {template: artifact_builder@1}
      context:
        artifacts:
          source: {namespace: inputs, name: source, required: true}
      result:
        artifacts:
          copied: {required: true, mediaTypes: [text/plain], from: {namespace: builder, name: copied}}
      on:
        succeeded: {next: review}
        failed:
          retry:
            maxAttempts: 3
            then: {next: review}
        interrupted: {fail: {}}
    review:
` + reviewStageYAML

const reviewStageYAML = `      objective: Review the candidate
      instructions: {ref: instructions/copy-planner.md}
      planner: passthrough@1
      agents:
        reviewer: {template: artifact_builder@1, namespace: builder}
      context:
        artifacts:
          candidate: {namespace: builder, name: copied, required: true}
      result:
        artifacts:
          copied: {required: true, mediaTypes: [text/plain], from: {namespace: builder, name: copied}}
      workflowOutputs:
        result: copied
      on:
        succeeded: {succeed: {}}
        failed: {fail: {}}
        interrupted: {fail: {}}
`

func mustLoad(t *testing.T, root string) *Snapshot {
	t.Helper()
	snapshot, err := Load(root)
	if err != nil {
		t.Fatalf("Load(%q): %v", root, err)
	}
	return snapshot
}

func assertDigest(t *testing.T, value string) {
	t.Helper()
	if len(value) != len("sha256:")+64 || !strings.HasPrefix(value, "sha256:") {
		t.Fatalf("invalid digest %q", value)
	}
}

func equalStrings(left, right []string) bool {
	if len(left) != len(right) {
		return false
	}
	for index := range left {
		if left[index] != right[index] {
			return false
		}
	}
	return true
}

// copyCoreFixture copies the core fixture catalog with the shared test
// policies, then adds each named testdata/<topic> catalog slice. Topic files
// must not replace core files, so every fixture stays attributable.
func copyCoreFixture(t *testing.T, topics ...string) string {
	t.Helper()
	root := configtest.CopyWithPolicies(t, coreFixtureRoot)
	for _, topic := range topics {
		if err := os.CopyFS(root, os.DirFS(filepath.Join("testdata", topic))); err != nil {
			t.Fatal(err)
		}
	}
	return root
}

func readFile(t *testing.T, path string) []byte {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	return data
}

func writeFile(t *testing.T, path string, data []byte) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, data, 0o644); err != nil {
		t.Fatal(err)
	}
}

func appendFile(t *testing.T, path, suffix string) {
	t.Helper()
	data := append(readFile(t, path), []byte(suffix)...)
	writeFile(t, path, data)
}

func replaceFile(t *testing.T, path, old, replacement string) {
	t.Helper()
	data := string(readFile(t, path))
	if !strings.Contains(data, old) {
		t.Fatalf("%s does not contain replacement source %q", path, old)
	}
	writeFile(t, path, []byte(strings.Replace(data, old, replacement, 1)))
}
