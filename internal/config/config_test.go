package config

import (
	"io/fs"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts"
	"go.yaml.in/yaml/v4"
)

const repositoryConfigRoot = "../../configs"

func TestLoadRepositoryConfig(t *testing.T) {
	t.Parallel()

	snapshot := mustLoad(t, repositoryConfigRoot, MVPDescriptors())
	if got, want := snapshot.Counts(), (Counts{
		Workflows: 8, AgentTemplates: 13, ModelPolicies: 7, LLMGateways: 1, ExecutionConfigs: 1, Instructions: 22,
	}); got != want {
		t.Fatalf("Counts() = %+v, want %+v", got, want)
	}

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
	if got, want := template.Toolsets[0].Tools, []string{"list_artifacts", "read_artifact", "write_artifact"}; !equalStrings(got, want) {
		t.Fatalf("selected tools = %v, want %v", got, want)
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

func TestRepositoryLocalLiteLLMWorkflowsPinRoleCredentials(t *testing.T) {
	t.Parallel()

	for _, root := range []string{repositoryConfigRoot, filepath.Join(repositoryConfigRoot, "e2e")} {
		root := root
		t.Run(root, func(t *testing.T) {
			t.Parallel()
			snapshot := mustLoad(t, root, MVPDescriptors())
			for _, workflow := range snapshot.Workflows() {
				workflowRef := workflow.Ref.Name + "@" + workflow.Ref.Version
				for stageName, stage := range workflow.Stages {
					assertLocalLiteLLMCredential(
						t, workflowRef+" Stage "+stageName+" Planner",
						stage.ExecutionConfig.Planner, "development-planner",
					)
					for logicalName, selection := range stage.ExecutionConfig.Agents {
						selection := selection
						assertLocalLiteLLMCredential(
							t, workflowRef+" Stage "+stageName+" Agent "+logicalName,
							&selection, "development-worker",
						)
					}
				}
			}
		})
	}
}

func assertLocalLiteLLMCredential(
	t *testing.T,
	consumer string,
	selection *ResolvedConsumerExecutionConfig,
	wantCredential string,
) {
	t.Helper()
	if selection == nil || selection.LLMGateway == nil ||
		selection.LLMGateway.Ref.GatewayID != "local-litellm" {
		return
	}
	if selection.Credential == nil || selection.Credential.CredentialID != wantCredential {
		t.Fatalf(
			"%s selects local-litellm without role credential %q: %+v",
			consumer, wantCredential, selection.Credential,
		)
	}
}

func TestRepositoryInstructionsDoNotExposePrivateWorkerProtocol(t *testing.T) {
	t.Parallel()

	patterns := []string{
		filepath.Join(repositoryConfigRoot, "instructions", "*.md"),
		filepath.Join(repositoryConfigRoot, "e2e", "instructions", "*.md"),
	}
	for _, pattern := range patterns {
		paths, err := filepath.Glob(pattern)
		if err != nil {
			t.Fatal(err)
		}
		for _, path := range paths {
			text := string(readFile(t, path))
			for _, forbidden := range []string{
				"StageContentRequest",
				"StageContentResult",
				"contractor/v1alpha1",
				"result slot",
				"declared output",
				"Runtime records",
				"Runtime associates",
				"Runtime owns",
				"Scheduler",
			} {
				if strings.Contains(text, forbidden) {
					t.Errorf("%s exposes private Worker protocol phrase %q", path, forbidden)
				}
			}
		}
	}
}

func TestLoadValidatesBundledSkillsWithoutReadingManagedRoot(t *testing.T) {
	root := filepath.Join(t.TempDir(), "operator")
	if err := os.CopyFS(root, os.DirFS(repositoryConfigRoot)); err != nil {
		t.Fatal(err)
	}
	skillDirectory := filepath.Join(root, "skills", "review")
	if err := os.MkdirAll(skillDirectory, 0o755); err != nil {
		t.Fatal(err)
	}
	valid := []byte("---\nname: review\ndescription: Review guidance.\n---\n# Review\n")
	if err := os.WriteFile(filepath.Join(skillDirectory, "SKILL.md"), valid, 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := Load(root, MVPDescriptors()); err != nil {
		t.Fatalf("valid bundled skill rejected: %v", err)
	}

	secret := "PRIVATE-SKILL-INSTRUCTION"
	if err := os.WriteFile(filepath.Join(skillDirectory, "SKILL.md"), []byte(secret), 0o644); err != nil {
		t.Fatal(err)
	}
	_, err := Load(root, MVPDescriptors())
	if err == nil || strings.Contains(err.Error(), root) || strings.Contains(err.Error(), secret) || !strings.Contains(err.Error(), "skill_manifest_invalid") {
		t.Fatalf("unsafe bundled skill validation error: %v", err)
	}
}

func TestTextArtifactToolsetDescriptor(t *testing.T) {
	t.Parallel()

	descriptor, ok := MVPDescriptors().Toolsets["text-artifacts@1"]
	if !ok {
		t.Fatal("text-artifacts@1 descriptor is missing")
	}
	if got, want := descriptor.Tools, []string{"read_text_artifact", "write_text_artifact"}; !equalStrings(got, want) {
		t.Fatalf("text-artifacts@1 tools = %v, want %v", got, want)
	}
}

func TestFilesystemToolsetDescriptor(t *testing.T) {
	t.Parallel()

	descriptor, ok := MVPDescriptors().Toolsets["filesystem@1"]
	if !ok {
		t.Fatal("filesystem@1 descriptor is missing")
	}
	want := []string{"glob", "grep", "ls", "read_file"}
	if !equalStrings(descriptor.Tools, want) {
		t.Fatalf("filesystem@1 tools = %v, want %v", descriptor.Tools, want)
	}
	if len(descriptor.InfrastructureChannels) != 0 {
		t.Fatalf("filesystem@1 infrastructure channels = %v, want none", descriptor.InfrastructureChannels)
	}
}

func TestCodeAnalysisToolsetDescriptor(t *testing.T) {
	t.Parallel()

	descriptor, ok := MVPDescriptors().Toolsets["code-analysis@1"]
	if !ok {
		t.Fatal("code-analysis@1 descriptor is missing")
	}
	want := []string{
		"attack_surface", "complexity_hotspots", "entrypoint_paths_to",
		"find_callees", "find_callers", "find_symbol", "functions_that_raise",
		"graph_summary", "list_symbols", "paths_between", "search_def",
	}
	if !equalStrings(descriptor.Tools, want) {
		t.Fatalf("code-analysis@1 tools = %v, want %v", descriptor.Tools, want)
	}
	if len(descriptor.InfrastructureChannels) != 0 {
		t.Fatalf(
			"code-analysis@1 infrastructure channels = %v, want none",
			descriptor.InfrastructureChannels,
		)
	}
}

func TestTaintAnnotationsToolsetDescriptor(t *testing.T) {
	t.Parallel()

	descriptor, ok := MVPDescriptors().Toolsets["taint-annotations@1"]
	if !ok {
		t.Fatal("taint-annotations@1 descriptor is missing")
	}
	want := []string{"annotate_sink", "annotate_trace", "annotate_validate"}
	if !equalStrings(descriptor.Tools, want) {
		t.Fatalf("taint-annotations@1 tools = %v, want %v", descriptor.Tools, want)
	}
	if len(descriptor.InfrastructureChannels) != 0 {
		t.Fatalf(
			"taint-annotations@1 infrastructure channels = %v, want none",
			descriptor.InfrastructureChannels,
		)
	}
}

func TestTaintAnnotationsToolSelectionIsStrictAndCollisionSafe(t *testing.T) {
	t.Parallel()

	descriptors, err := normalizeDescriptors(MVPDescriptors())
	if err != nil {
		t.Fatal(err)
	}
	current := &loader{descriptors: descriptors}
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

	descriptors.Toolsets["alternate@1"] = ToolsetDescriptor{Tools: []string{"annotate_trace"}}
	current.descriptors = descriptors
	if _, err := current.resolveToolsets(&[]toolsetSelectionSource{
		{Ref: "taint-annotations@1", Tools: []string{"annotate_trace"}},
		{Ref: "alternate@1", Tools: []string{"annotate_trace"}},
	}); err == nil || !strings.Contains(err.Error(), "collides between Toolsets") {
		t.Fatalf("taint annotation visible-name collision error = %v", err)
	}
}

func TestCodeAnalysisToolSelectionIsStrictAndCollisionSafe(t *testing.T) {
	t.Parallel()

	descriptors, err := normalizeDescriptors(MVPDescriptors())
	if err != nil {
		t.Fatal(err)
	}
	current := &loader{descriptors: descriptors}
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

	descriptors.Toolsets["alternate@1"] = ToolsetDescriptor{Tools: []string{"search_def"}}
	current.descriptors = descriptors
	if _, err := current.resolveToolsets(&[]toolsetSelectionSource{
		{Ref: "code-analysis@1", Tools: []string{"search_def"}},
		{Ref: "alternate@1", Tools: []string{"search_def"}},
	}); err == nil || !strings.Contains(err.Error(), "collides between Toolsets") {
		t.Fatalf("code-analysis visible-name collision error = %v", err)
	}
}

func TestCodeAnalysisSelectionLoadsAndDigestsInWorkspaceTemplate(t *testing.T) {
	t.Parallel()

	snapshot := mustLoad(t, repositoryConfigRoot, MVPDescriptors())
	template, err := snapshot.AgentTemplate("workspace_source_graph_analyst@1")
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

func TestEditFilesToolsetDescriptor(t *testing.T) {
	t.Parallel()

	descriptor, ok := MVPDescriptors().Toolsets["edit-files@1"]
	if !ok {
		t.Fatal("edit-files@1 descriptor is missing")
	}
	want := []string{
		"append_file", "cp", "edit", "insert_line", "mkdir", "mv", "replace_range", "rm", "write_file",
	}
	if !equalStrings(descriptor.Tools, want) {
		t.Fatalf("edit-files@1 tools = %v, want %v", descriptor.Tools, want)
	}
	if len(descriptor.InfrastructureChannels) != 0 {
		t.Fatalf("edit-files@1 infrastructure channels = %v, want none", descriptor.InfrastructureChannels)
	}
}

func TestWorkspaceChangesToolsetDescriptor(t *testing.T) {
	t.Parallel()

	descriptor, ok := MVPDescriptors().Toolsets["workspace-changes@1"]
	if !ok {
		t.Fatal("workspace-changes@1 descriptor is missing")
	}
	want := []string{"changed_paths", "diff", "rollback_changes"}
	if !equalStrings(descriptor.Tools, want) {
		t.Fatalf("workspace-changes@1 tools = %v, want %v", descriptor.Tools, want)
	}
	if len(descriptor.InfrastructureChannels) != 0 {
		t.Fatalf("workspace-changes@1 infrastructure channels = %v, want none", descriptor.InfrastructureChannels)
	}
}

func TestMemoryToolsetDescriptor(t *testing.T) {
	t.Parallel()

	descriptor, ok := MVPDescriptors().Toolsets["memory-tools@1"]
	if !ok {
		t.Fatal("memory-tools@1 descriptor is missing")
	}
	want := []string{
		"append_memory", "list_memories", "list_memory_tags",
		"read_memory", "search_memory", "write_memory",
	}
	if !equalStrings(descriptor.Tools, want) {
		t.Fatalf("memory-tools@1 tools = %v, want %v", descriptor.Tools, want)
	}
	if len(descriptor.InfrastructureChannels) != 0 {
		t.Fatalf("memory-tools@1 infrastructure channels = %v, want none", descriptor.InfrastructureChannels)
	}
}

func TestHTTPToolsetDescriptorUsesOptionalTypedClient(t *testing.T) {
	t.Parallel()

	descriptor, ok := MVPDescriptors().Toolsets["http-tools@1"]
	if !ok {
		t.Fatal("http-tools@1 descriptor is missing")
	}
	want := []string{
		"http_history", "http_read_body", "http_request", "http_session_clear",
		"http_session_get", "http_session_set",
	}
	if !equalStrings(descriptor.Tools, want) {
		t.Fatalf("http-tools@1 tools = %v, want %v", descriptor.Tools, want)
	}
	channels := descriptor.InfrastructureChannels["http_request"]
	if len(channels) != 1 || channels[0] != RuntimeHTTPClient {
		t.Fatalf("http_request channels = %v, want RuntimeHTTPClient", channels)
	}
	selected := contracts.ResolvedAgentTemplate{Toolsets: []contracts.ToolsetSelection{{
		Ref:   contracts.ToolsetRef{ToolsetID: "http-tools", Version: "1"},
		Tools: []string{"http_request"},
	}}}
	adapters, err := RequiredRuntimeAdaptersForTemplate(selected)
	if err != nil || len(adapters) != 0 {
		t.Fatalf("HTTP direct/optional-proxy requirements = (%v, %v), want no hard adapter", adapters, err)
	}
}

func TestCaidoToolsetDescriptorRequiresTypedClientForEveryTool(t *testing.T) {
	t.Parallel()
	descriptor, ok := MVPDescriptors().Toolsets["caido@1"]
	if !ok {
		t.Fatal("caido@1 descriptor is missing")
	}
	want := []string{
		"caido_automate_results", "caido_automate_run", "caido_history", "caido_replay",
		"caido_request_detail", "caido_scope", "caido_sitemap", "caido_workflow_findings",
		"caido_workflow_list", "caido_workflow_run",
	}
	if !equalStrings(descriptor.Tools, want) {
		t.Fatalf("caido@1 tools = %v, want %v", descriptor.Tools, want)
	}
	for _, tool := range want {
		channels := descriptor.InfrastructureChannels[tool]
		if len(channels) != 1 || channels[0] != CaidoGraphQLClient {
			t.Fatalf("%s channels = %v, want CaidoGraphQLClient", tool, channels)
		}
	}
	selected := contracts.ResolvedAgentTemplate{Toolsets: []contracts.ToolsetSelection{{
		Ref:   contracts.ToolsetRef{ToolsetID: "caido", Version: "1"},
		Tools: []string{"caido_history", "caido_scope"},
	}}}
	adapters, err := RequiredRuntimeAdaptersForTemplate(selected)
	if err != nil || !reflect.DeepEqual(adapters, []contracts.RuntimeAdapterRef{contracts.RuntimeAdapterCaidoGraphQL}) {
		t.Fatalf("Caido tool adapter requirements = (%v, %v)", adapters, err)
	}
}

func TestSourceAnalysisToolsetDescriptor(t *testing.T) {
	t.Parallel()

	descriptor, ok := MVPDescriptors().Toolsets["source-analysis@1"]
	if !ok {
		t.Fatal("source-analysis@1 descriptor is missing")
	}
	want := []string{"list_source_files", "open_source_archive", "read_source", "search_source"}
	if !equalStrings(descriptor.Tools, want) {
		t.Fatalf("source-analysis@1 tools = %v, want %v", descriptor.Tools, want)
	}
}

func TestOpenAPIToolsetDescriptor(t *testing.T) {
	t.Parallel()

	descriptor, ok := MVPDescriptors().Toolsets["openapi@1"]
	if !ok {
		t.Fatal("openapi@1 descriptor is missing")
	}
	want := []string{
		"get_openapi_component", "get_openapi_info", "get_openapi_path",
		"initialize_openapi", "list_openapi_components", "list_openapi_paths",
		"list_openapi_servers", "list_openapi_tags", "load_openapi", "read_openapi_document",
		"remove_openapi_component", "remove_openapi_path", "set_openapi_info",
		"set_openapi_servers", "set_openapi_tags", "upsert_openapi_component", "upsert_openapi_path",
		"validate_openapi",
	}
	if !equalStrings(descriptor.Tools, want) {
		t.Fatalf("openapi@1 tools = %v, want %v", descriptor.Tools, want)
	}
	if got := descriptor.InfrastructureChannels["validate_openapi"]; len(got) != 1 || got[0] != RuntimeSubprocessLauncher {
		t.Fatalf("validate_openapi channels = %v, want RuntimeSubprocessLauncher", got)
	}
}

func TestLikeC4ToolsetDescriptor(t *testing.T) {
	t.Parallel()

	descriptor, ok := MVPDescriptors().Toolsets["likec4@1"]
	if !ok {
		t.Fatal("likec4@1 descriptor is missing")
	}
	want := []string{
		"append_likec4", "load_likec4", "read_likec4", "replace_likec4",
		"validate_likec4", "write_likec4",
	}
	if !equalStrings(descriptor.Tools, want) {
		t.Fatalf("likec4@1 tools = %v, want %v", descriptor.Tools, want)
	}
	if got := descriptor.InfrastructureChannels["validate_likec4"]; len(got) != 1 || got[0] != RuntimeSubprocessLauncher {
		t.Fatalf("validate_likec4 channels = %v, want RuntimeSubprocessLauncher", got)
	}
}

func TestToolsetDescriptorRejectsInvalidInfrastructureChannels(t *testing.T) {
	t.Parallel()

	for name, channels := range map[string]map[string][]ToolInfrastructureChannel{
		"unknown tool": {"missing": {RuntimeHTTPClient}},
		"empty":        {"read_artifact": {}},
		"unknown channel": {
			"read_artifact": {ToolInfrastructureChannel("ambient-network")},
		},
		"duplicate": {"read_artifact": {RuntimeHTTPClient, RuntimeHTTPClient}},
	} {
		t.Run(name, func(t *testing.T) {
			descriptors := MVPDescriptors()
			descriptor := descriptors.Toolsets["run-artifacts@1"]
			descriptor.InfrastructureChannels = channels
			descriptors.Toolsets["run-artifacts@1"] = descriptor
			if _, err := normalizeDescriptors(descriptors); err == nil {
				t.Fatal("normalizeDescriptors accepted invalid infrastructure channels")
			}
		})
	}
}

func TestToolsetInfrastructureChannelParityFixture(t *testing.T) {
	t.Parallel()

	raw, err := os.ReadFile("../../api/descriptor-parity/toolset-infrastructure-channels.yaml")
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		SchemaVersion string                                            `yaml:"schemaVersion"`
		Toolsets      map[string]map[string][]ToolInfrastructureChannel `yaml:"toolsets"`
	}
	if err := yaml.Unmarshal(raw, &fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.SchemaVersion != "1.0" {
		t.Fatalf("descriptor parity schemaVersion = %q", fixture.SchemaVersion)
	}
	normalized, err := normalizeDescriptors(MVPDescriptors())
	if err != nil {
		t.Fatal(err)
	}
	descriptors := normalized.Toolsets
	if len(fixture.Toolsets) != len(descriptors) {
		t.Fatalf("descriptor parity Toolsets = %d, want %d", len(fixture.Toolsets), len(descriptors))
	}
	for ref, descriptor := range descriptors {
		if !reflect.DeepEqual(fixture.Toolsets[ref], descriptor.InfrastructureChannels) {
			t.Fatalf("descriptor parity channels for %s = %v, want %v", ref, fixture.Toolsets[ref], descriptor.InfrastructureChannels)
		}
	}
}

func TestWorkflowExamplesLoad(t *testing.T) {
	for _, name := range []string{
		"bounded_retry_workflow.yaml",
		"multi_stage_workflow.yaml",
		"router_openapi_workflow.yaml",
		"streamline_review_workflow.yaml",
	} {
		t.Run(name, func(t *testing.T) {
			root := copyConfigTree(t)
			example := readFile(t, filepath.Join(repositoryConfigRoot, "examples", name))
			writeFile(t, filepath.Join(root, "workflows", name), example)
			snapshot := mustLoad(t, root, MVPDescriptors())
			if snapshot.Counts().Workflows != 9 {
				t.Fatalf("example Workflow count = %d", snapshot.Counts().Workflows)
			}
		})
	}
}

func TestStreamlineRequiresExactlyOneLogicalAgent(t *testing.T) {
	root := copyConfigTree(t)
	path := filepath.Join(root, "workflows", "streamline_review_workflow.yaml")
	example := readFile(t, filepath.Join(repositoryConfigRoot, "examples", "streamline_review_workflow.yaml"))
	writeFile(t, path, example)
	snapshot := mustLoad(t, root, MVPDescriptors())
	workflow, err := snapshot.Workflow("streamline-review@1")
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
	loaded, err := Load(root, MVPDescriptors())
	if err == nil || loaded != nil ||
		!strings.Contains(err.Error(), "streamline@1 requires exactly one logical Agent binding") {
		t.Fatalf("Load() = (%v, %v), want Streamline cardinality error", loaded, err)
	}
}

func TestRouterRequiresAtLeastOneLogicalAgent(t *testing.T) {
	root := copyConfigTree(t)
	path := filepath.Join(root, "workflows", "router_openapi_workflow.yaml")
	example := readFile(t, filepath.Join(repositoryConfigRoot, "examples", "router_openapi_workflow.yaml"))
	writeFile(t, path, example)
	snapshot := mustLoad(t, root, MVPDescriptors())
	workflow, err := snapshot.Workflow("router-openapi@1")
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

	valid := mustLoad(t, "testdata/valid", MVPDescriptors())
	if got, want := valid.Counts(), (Counts{
		Workflows: 1, AgentTemplates: 1, ModelPolicies: 1, LLMGateways: 1, ExecutionConfigs: 1, Instructions: 2,
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
			root := copyConfigTree(t)
			fixture := readFile(t, filepath.Join("testdata/invalid", name))
			writeFile(t, filepath.Join(root, "model-policies/worker.yaml"), fixture)
			snapshot, err := Load(root, MVPDescriptors())
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
			root := copyConfigTree(t)
			replaceFile(t, filepath.Join(root, "model-policies/worker.yaml"), test.old, test.replacement)
			if snapshot, err := Load(root, MVPDescriptors()); err == nil || snapshot != nil ||
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
				appendFile(t, filepath.Join(root, "model-policies/worker.yaml"), "unknownField: true\n")
			},
			want: []string{"model-policies/worker.yaml", "unknownField"},
		},
		{
			name: "duplicate mapping key",
			mutate: func(t *testing.T, root string) {
				appendFile(t, filepath.Join(root, "model-policies/worker.yaml"), "kind: ModelPolicy\n")
			},
			want: []string{"model-policies/worker.yaml", "already defined"},
		},
		{
			name: "multiple documents",
			mutate: func(t *testing.T, root string) {
				appendFile(t, filepath.Join(root, "model-policies/worker.yaml"), `---
apiVersion: contractor/v1alpha1
kind: ModelPolicy
metadata: {name: other, version: "1"}
spec: {model: other, maxOutputTokens: 1}
`)
			},
			want: []string{"model-policies/worker.yaml", "exactly one"},
		},
		{
			name: "empty document",
			mutate: func(t *testing.T, root string) {
				writeFile(t, filepath.Join(root, "model-policies/worker.yaml"), nil)
			},
			want: []string{"model-policies/worker.yaml", "exactly one"},
		},
		{
			name: "wrong subtree kind",
			mutate: func(t *testing.T, root string) {
				replaceFile(t, filepath.Join(root, "model-policies/worker.yaml"), "kind: ModelPolicy", "kind: Workflow")
			},
			want: []string{"model-policies/worker.yaml", "does not match ModelPolicy subtree"},
		},
		{
			name: "duplicate identity",
			mutate: func(t *testing.T, root string) {
				source := readFile(t, filepath.Join(root, "model-policies/worker.yaml"))
				writeFile(t, filepath.Join(root, "model-policies/z/duplicate.yaml"), source)
			},
			want: []string{"model-policies/z/duplicate.yaml", "duplicate ModelPolicy identity worker@1"},
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
			root := copyConfigTree(t)
			test.mutate(t, root)
			snapshot, err := Load(root, MVPDescriptors())
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
		root := copyConfigTree(t)
		outside := filepath.Join(filepath.Dir(root), "outside.md")
		writeFile(t, outside, []byte("outside\n"))
		target := filepath.Join(root, "instructions/artifact-builder.md")
		if err := os.Remove(target); err != nil {
			t.Fatal(err)
		}
		if err := os.Symlink(outside, target); err != nil {
			t.Fatal(err)
		}
		snapshot, err := Load(root, MVPDescriptors())
		if err == nil || snapshot != nil || !strings.Contains(err.Error(), "escapes configuration root") {
			t.Fatalf("Load() = (%v, %v), want symlink escape error", snapshot, err)
		}
	})

	t.Run("invalid utf8", func(t *testing.T) {
		root := copyConfigTree(t)
		writeFile(t, filepath.Join(root, "instructions/artifact-builder.md"), []byte{0xff, 0xfe})
		snapshot, err := Load(root, MVPDescriptors())
		if err == nil || snapshot != nil || !strings.Contains(err.Error(), "strict UTF-8") {
			t.Fatalf("Load() = (%v, %v), want UTF-8 error", snapshot, err)
		}
	})
}

func TestManifestDiscoveryIgnoresSymlinks(t *testing.T) {
	t.Parallel()

	root := copyConfigTree(t)
	if err := os.Symlink("worker.yaml", filepath.Join(root, "model-policies/link.yaml")); err != nil {
		t.Fatal(err)
	}
	snapshot := mustLoad(t, root, MVPDescriptors())
	if snapshot.Counts().ModelPolicies != 7 {
		t.Fatalf("ModelPolicies = %d, want 7", snapshot.Counts().ModelPolicies)
	}
}

func TestToolsetVisibleNameCollision(t *testing.T) {
	t.Parallel()

	root := copyConfigTree(t)
	path := filepath.Join(root, "agent-templates/artifact_builder.yaml")
	replaceFile(t, path, "  sandboxProfile: local-workdir@1", `    - ref: alternate@1
      tools: [read_artifact]
  sandboxProfile: local-workdir@1`)
	descriptors := MVPDescriptors()
	descriptors.Toolsets["alternate@1"] = ToolsetDescriptor{Tools: []string{"read_artifact"}}
	snapshot, err := Load(root, descriptors)
	if err == nil || snapshot != nil || !strings.Contains(err.Error(), "collides between Toolsets") {
		t.Fatalf("Load() = (%v, %v), want collision error", snapshot, err)
	}
}

func TestWorkflowGraphLoadsMultiStageAndBoundedRetry(t *testing.T) {
	t.Parallel()
	root := copyConfigTree(t)
	writeFile(t, filepath.Join(root, "workflows/artifact_copy.yaml"), []byte(multiStageWorkflowYAML))

	snapshot := mustLoad(t, root, MVPDescriptors())
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

func TestWorkflowPrimaryMarkerIsAcceptedOnlyOnWorkflowOutputs(t *testing.T) {
	t.Run("output", func(t *testing.T) {
		root := copyConfigTree(t)
		path := filepath.Join(root, "workflows/artifact_copy.yaml")
		replaceFile(t, path,
			"  outputs:\n    result:\n      required: true\n      mediaTypes: [text/plain]",
			"  outputs:\n    result:\n      required: true\n      primary: true\n      mediaTypes: [text/plain]",
		)
		workflow, err := mustLoad(t, root, MVPDescriptors()).Workflow("artifact-copy@1")
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
			root := copyConfigTree(t)
			path := filepath.Join(root, "workflows/artifact_copy.yaml")
			replaceFile(t, path, test.old, test.replacement)
			snapshot, err := Load(root, MVPDescriptors())
			if err == nil || snapshot != nil || !strings.Contains(err.Error(), test.field) ||
				!strings.Contains(err.Error(), "allowed only for Workflow outputs") {
				t.Fatalf("Load() = (%v, %v), want output-only primary error", snapshot, err)
			}
		})
	}
}

func TestWorkflowGraphRejectsNextCycle(t *testing.T) {
	t.Parallel()
	root := copyConfigTree(t)
	cyclic := strings.Replace(
		multiStageWorkflowYAML,
		"    review:\n"+reviewStageYAML,
		"    review:\n"+strings.Replace(reviewStageYAML, "succeed: {}", "next: build", 1),
		1,
	)
	writeFile(t, filepath.Join(root, "workflows/artifact_copy.yaml"), []byte(cyclic))

	snapshot, err := Load(root, MVPDescriptors())
	if err == nil || snapshot != nil || !strings.Contains(err.Error(), "Cycle") {
		t.Fatalf("Load(cycle) = (%v, %v)", snapshot, err)
	}
}

func TestWorkflowGraphRejectsUnreachableStage(t *testing.T) {
	t.Parallel()
	root := copyConfigTree(t)
	workflow := multiStageWorkflowYAML + "    orphan:\n" + reviewStageYAML
	writeFile(t, filepath.Join(root, "workflows/artifact_copy.yaml"), []byte(workflow))

	snapshot, err := Load(root, MVPDescriptors())
	if err == nil || snapshot != nil || !strings.Contains(err.Error(), "unreachable") {
		t.Fatalf("Load(unreachable) = (%v, %v)", snapshot, err)
	}
}

func TestWorkflowTransitionRejectsSuccessPathWithoutRequiredOutput(t *testing.T) {
	t.Parallel()
	root := copyConfigTree(t)
	workflow := strings.Replace(
		multiStageWorkflowYAML,
		"      workflowOutputs:\n        result: copied\n",
		"",
		1,
	)
	writeFile(t, filepath.Join(root, "workflows/artifact_copy.yaml"), []byte(workflow))

	snapshot, err := Load(root, MVPDescriptors())
	if err == nil || snapshot != nil || !strings.Contains(err.Error(), "without required output") {
		t.Fatalf("Load(missing output path) = (%v, %v)", snapshot, err)
	}
}

func TestWorkflowTransitionRejectsOptionalResultAsRequiredOutput(t *testing.T) {
	t.Parallel()

	root := copyConfigTree(t)
	workflow := strings.Replace(
		multiStageWorkflowYAML,
		"          copied: {required: true, mediaTypes: [text/plain], from: {namespace: builder, name: copied}}\n      workflowOutputs:",
		"          copied: {required: false, mediaTypes: [text/plain], from: {namespace: builder, name: copied}}\n      workflowOutputs:",
		1,
	)
	writeFile(t, filepath.Join(root, "workflows/artifact_copy.yaml"), []byte(workflow))
	if snapshot, err := Load(root, MVPDescriptors()); err == nil || snapshot != nil ||
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
			root := copyConfigTree(t)
			replaceFile(t, filepath.Join(root, path), test.old, test.replacement)
			snapshot, err := Load(root, MVPDescriptors())
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

func mustLoad(t *testing.T, root string, descriptors Descriptors) *Snapshot {
	t.Helper()
	snapshot, err := Load(root, descriptors)
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

func copyConfigTree(t *testing.T) string {
	t.Helper()
	destination := filepath.Join(t.TempDir(), "configs")
	err := filepath.WalkDir(repositoryConfigRoot, func(path string, entry fs.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		relative, err := filepath.Rel(repositoryConfigRoot, path)
		if err != nil {
			return err
		}
		target := filepath.Join(destination, relative)
		if entry.IsDir() {
			return os.MkdirAll(target, 0o755)
		}
		data, err := os.ReadFile(path)
		if err != nil {
			return err
		}
		return os.WriteFile(target, data, 0o644)
	})
	if err != nil {
		t.Fatalf("copy config tree: %v", err)
	}
	return destination
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
