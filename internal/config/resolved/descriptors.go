package resolved

import (
	"fmt"
	"maps"
	"slices"
	"sort"
	"sync"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/runtimesettings"
)

type infrastructureChannel string

const (
	channelRuntimeHTTPClient         infrastructureChannel = "runtime-http-client"
	channelRuntimeSubprocessLauncher infrastructureChannel = "runtime-subprocess-launcher"
	channelCaidoGraphQLClient        infrastructureChannel = "caido-graphql-client"
	channelSandboxExecution          infrastructureChannel = "sandbox-execution"
)

// plannerDescriptor is the Server-visible part of one registered
// PlannerFactory.
type plannerDescriptor struct {
	// SingleAgent marks a Planner that drives exactly one logical Agent
	// binding (LIM-PLN-1); every other Planner accepts one or more
	// (LIM-PLN-2).
	SingleAgent bool
	// ModelFree marks a Planner that runs without a model: its Stage has no
	// Planner executionConfig, and Workflow-wide Planner defaults skip it.
	ModelFree bool
}

// toolsetDescriptor is the Server-visible part of one runtime ToolsetFactory.
// Tools are the final model-visible names exported by that exact version.
// InfrastructureChannels contains only tools with a non-empty fixed channel
// set; an absent tool is local/artifact-only. ActiveCheckTools and
// FindingProposalTools are closed Server-side effect classifications used by
// Audit compatibility checks and finding intake; model text and tool names are
// never trusted to infer those effects. RequiredWorkflowInputs names, per tool,
// the ordinary Workflow input that the runtime toolset reads during allocation
// preparation; ValidateWorkflowGraph requires it of every Workflow whose agent
// selects that tool. RequiresWorkspace marks a Toolset that reads or changes
// the allocation workspace, so its Stage needs context.workspace. The Runtime
// ToolsetFactory of every built-in Toolset declares the same tools, channels,
// workspace need and SandboxProfile; api/testdata/toolset-descriptors.json
// pins both sides and the Server-only classifications.
type toolsetDescriptor struct {
	RequiredSandboxProfiles []string
	RequiresWorkspace       bool
	Tools                   []string
	InfrastructureChannels  map[string][]infrastructureChannel
	ActiveCheckTools        []string
	SandboxActiveCheckTools map[string][]string
	FindingProposalTools    []string
	RequiredWorkflowInputs  map[string]workflowInputRequirement
}

// workflowInputRequirement is one Workflow input a selected tool depends on:
// the Workflow must declare Input as required and accept MediaType exactly.
type workflowInputRequirement struct {
	Input     string
	MediaType string
}

// sandboxProfileDescriptor describes authoring and binding-specific placement
// requirements, not proof that a Runtime has installed the implementation.
type sandboxProfileDescriptor struct {
	WorkspaceMode    contracts.WorkspaceMode
	WorkspaceStorage contracts.WorkspaceStorage
}

// descriptorTable enumerates the registered code contracts that
// configuration is allowed to select. Installation and readiness are checked
// separately against Runtime discovery. Map keys use exact <id>@<version>
// selectors.
type descriptorTable struct {
	PlannerFactories map[string]plannerDescriptor
	WorkerRuntimes   map[string]struct{}
	Toolsets         map[string]toolsetDescriptor
	SandboxProfiles  map[string]sandboxProfileDescriptor
}

// registeredDescriptors is the one descriptor registry: the built-in table,
// normalized once. Every check of a selection against registered code reads
// it, whether it runs while the catalog loads, while a persisted snapshot is
// decoded or on the placement hot path, so no reader can see another table.
// The table is static, so a failure to normalize it is a programming error
// that TestRegisteredDescriptorsNormalize reports first.
var registeredDescriptors = sync.OnceValue(func() descriptorTable {
	table, err := normalizeDescriptors(builtInDescriptors())
	if err != nil {
		panic(fmt.Sprintf("built-in descriptor table: %v", err))
	}
	return table
})

// builtInDescriptors returns the contracts available to strict Workflow
// configuration. Podman/execution descriptors are registered before their
// Runtime factories; their presence here must never be interpreted as fleet
// capability. Only registeredDescriptors reads it.
func builtInDescriptors() descriptorTable {
	return descriptorTable{
		PlannerFactories: map[string]plannerDescriptor{
			"passthrough@1": {SingleAgent: true, ModelFree: true},
			"streamline@1":  {SingleAgent: true},
			"router@1":      {},
			"scan-plan@1":   {ModelFree: true},
		},
		WorkerRuntimes: map[string]struct{}{
			"adk@1":  {},
			"tool@1": {},
		},
		Toolsets: map[string]toolsetDescriptor{
			"code-execution@1": {
				Tools:                   []string{"exec_command"},
				RequiredSandboxProfiles: []string{"podman-network@1", "podman@1"},
				SandboxActiveCheckTools: map[string][]string{"podman-network@1": {"exec_command"}},
				RequiresWorkspace:       true,
				InfrastructureChannels: map[string][]infrastructureChannel{
					"exec_command": {channelSandboxExecution},
				},
			},
			AuditResultsToolsetID + "@" + AuditResultsToolsetVersion: {
				Tools: []string{ReadAuditTaskTool, SubmitCheckResultTool},
			},
			"security-findings@1": {
				Tools:                []string{"finding", "list_findings"},
				FindingProposalTools: []string{"finding"},
				RequiredWorkflowInputs: map[string]workflowInputRequirement{
					"list_findings": {Input: "findings", MediaType: FindingCollectionMediaType},
				},
			},
			"security-findings-code@1": {
				Tools:                []string{"finding"},
				FindingProposalTools: []string{"finding"},
			},
			"security-findings-http@1": {
				Tools:                []string{"finding"},
				FindingProposalTools: []string{"finding"},
			},
			"caido@1": {
				Tools: []string{
					"caido_automate_results", "caido_automate_run", "caido_history",
					"caido_replay", "caido_request_detail", "caido_scope", "caido_sitemap",
					"caido_workflow_findings", "caido_workflow_list", "caido_workflow_run",
				},
				InfrastructureChannels: map[string][]infrastructureChannel{
					"caido_automate_results":  {channelCaidoGraphQLClient},
					"caido_automate_run":      {channelCaidoGraphQLClient},
					"caido_history":           {channelCaidoGraphQLClient},
					"caido_replay":            {channelCaidoGraphQLClient},
					"caido_request_detail":    {channelCaidoGraphQLClient},
					"caido_scope":             {channelCaidoGraphQLClient},
					"caido_sitemap":           {channelCaidoGraphQLClient},
					"caido_workflow_findings": {channelCaidoGraphQLClient},
					"caido_workflow_list":     {channelCaidoGraphQLClient},
					"caido_workflow_run":      {channelCaidoGraphQLClient},
				},
				ActiveCheckTools: []string{
					"caido_automate_run", "caido_replay", "caido_workflow_run",
				},
			},
			"code-analysis@1": {
				RequiresWorkspace: true,
				Tools: []string{
					"attack_surface", "complexity_hotspots", "entrypoint_paths_to",
					"find_callees", "find_callers", "find_symbol", "functions_that_raise",
					"graph_summary", "list_symbols", "paths_between", "search_def",
				},
			},
			"edit-files@1": {
				RequiresWorkspace: true,
				Tools: []string{
					"append_file", "cp", "edit", "insert_line", "mkdir", "mv", "replace_range", "rm", "write_file",
				},
			},
			"filesystem@1": {
				RequiresWorkspace: true,
				Tools:             []string{"glob", "grep", "ls", "read_file"},
			},
			"http-tools@1": {
				Tools: []string{
					"http_history", "http_read_body", "http_request", "http_session_clear",
					"http_session_get", "http_session_set",
				},
				InfrastructureChannels: map[string][]infrastructureChannel{
					"http_request": {channelRuntimeHTTPClient},
				},
				ActiveCheckTools: []string{"http_request"},
			},
			"memory-tools@1": {
				Tools: []string{
					"append_memory", "list_memories", "list_memory_tags",
					"read_memory", "search_memory", "write_memory",
				},
			},
			"likec4@1": {
				Tools: []string{
					"append_likec4", "load_likec4", "read_likec4", "replace_likec4",
					"validate_likec4", "write_likec4",
				},
				InfrastructureChannels: map[string][]infrastructureChannel{
					"validate_likec4": {channelRuntimeSubprocessLauncher},
				},
			},
			"openapi@1": {
				Tools: []string{
					"get_openapi_component", "get_openapi_info", "get_openapi_path",
					"initialize_openapi", "list_openapi_components", "list_openapi_paths",
					"list_openapi_servers", "list_openapi_tags", "load_openapi", "read_openapi_document",
					"remove_openapi_component", "remove_openapi_path", "set_openapi_info",
					"set_openapi_servers", "set_openapi_tags", "upsert_openapi_component", "upsert_openapi_path",
					"validate_openapi",
				},
				InfrastructureChannels: map[string][]infrastructureChannel{
					"validate_openapi": {channelRuntimeSubprocessLauncher},
				},
			},
			"run-artifacts@1": {
				Tools: []string{"list_artifacts", "read_artifact", "write_artifact"},
			},
			"scan@1": {
				Tools: []string{"scan_ffuf", "scan_katana", "scan_naabu", "scan_nuclei", "scan_sqlmap"},
				InfrastructureChannels: map[string][]infrastructureChannel{
					"scan_ffuf":   {channelRuntimeSubprocessLauncher},
					"scan_katana": {channelRuntimeSubprocessLauncher},
					"scan_naabu":  {channelRuntimeSubprocessLauncher},
					"scan_nuclei": {channelRuntimeSubprocessLauncher},
					"scan_sqlmap": {channelRuntimeSubprocessLauncher},
				},
				ActiveCheckTools: []string{"scan_ffuf", "scan_katana", "scan_naabu", "scan_nuclei", "scan_sqlmap"},
			},
			"source-analysis@1": {
				Tools: []string{"list_source_files", "open_source_archive", "read_source", "search_source"},
			},
			"text-artifacts@1": {
				Tools: []string{"read_text_artifact", "write_text_artifact"},
			},
			"taint-annotations@1": {
				RequiresWorkspace: true,
				Tools:             []string{"annotate_sink", "annotate_trace", "annotate_validate"},
			},
			"workspace-changes@1": {
				RequiresWorkspace: true,
				Tools:             []string{"changed_paths", "diff", "rollback_changes"},
			},
		},
		SandboxProfiles: map[string]sandboxProfileDescriptor{
			"podman-network@1": {WorkspaceMode: contracts.WorkspaceModeDirect, WorkspaceStorage: contracts.WorkspaceStorageLocal},
			"podman@1":         {WorkspaceMode: contracts.WorkspaceModeDirect, WorkspaceStorage: contracts.WorkspaceStorageLocal},
			"local-workdir@1":  {},
		},
	}
}

// KnownPlanner reports whether ref selects a registered PlannerFactory.
func KnownPlanner(ref PlannerRef) bool {
	_, ok := registeredDescriptors().PlannerFactories[ref.PlannerID+"@"+ref.Version]
	return ok
}

// IsModelFreePlanner classifies exact implementations, independently of any
// authored model selection. Workflow-wide defaults skip these implementations.
func IsModelFreePlanner(ref PlannerRef) bool {
	return registeredDescriptors().PlannerFactories[ref.PlannerID+"@"+ref.Version].ModelFree
}

// ValidatePlannerAgents checks that a Stage selecting planner binds as many
// logical Agents as its registered PlannerFactory accepts.
func ValidatePlannerAgents(planner PlannerRef, agents int) error {
	reference := planner.PlannerID + "@" + planner.Version
	descriptor, ok := registeredDescriptors().PlannerFactories[reference]
	switch {
	case !ok:
		return fmt.Errorf("planner selects unknown PlannerFactory %q", reference)
	case descriptor.SingleAgent && agents != 1:
		return fmt.Errorf("%s requires exactly one logical Agent binding; use router@1 for multiple bindings", reference)
	case agents < 1:
		return fmt.Errorf("%s requires at least one logical Agent binding", reference)
	}
	return nil
}

// KnownWorkerRuntime reports whether selector names a registered
// WorkerRuntime.
func KnownWorkerRuntime(selector string) bool {
	_, ok := registeredDescriptors().WorkerRuntimes[selector]
	return ok
}

// KnownSandboxProfile reports whether selector names a registered
// SandboxProfile.
func KnownSandboxProfile(selector string) bool {
	_, ok := registeredDescriptors().SandboxProfiles[selector]
	return ok
}

// ToolsetTools returns the sorted model-visible tools the registered Toolset
// selector exports, or false for an unregistered Toolset. The slice is the
// caller's.
func ToolsetTools(selector string) ([]string, bool) {
	descriptor, ok := registeredDescriptors().Toolsets[selector]
	if !ok {
		return nil, false
	}
	return append([]string(nil), descriptor.Tools...), true
}

// SelectsActiveCheckTool reports whether selection selects a tool that its
// registered Toolset classifies as an active check.
func SelectsActiveCheckTool(selection contracts.ToolsetSelection, profile contracts.SandboxProfileRef) bool {
	descriptor := registeredDescriptors().Toolsets[selection.Ref.ToolsetID+"@"+selection.Ref.Version]
	return selectsAny(selection.Tools, descriptor.ActiveCheckTools) || selectsAny(selection.Tools, descriptor.SandboxActiveCheckTools[profile.SandboxProfileID+"@"+profile.Version])
}

// SelectsFindingProposalTool reports whether selection selects a tool that
// its registered Toolset classifies as proposing findings.
func SelectsFindingProposalTool(selection contracts.ToolsetSelection) bool {
	descriptor := registeredDescriptors().Toolsets[selection.Ref.ToolsetID+"@"+selection.Ref.Version]
	return selectsAny(selection.Tools, descriptor.FindingProposalTools)
}

func selectsAny(selected, classified []string) bool {
	for _, tool := range selected {
		if _, found := slices.BinarySearch(classified, tool); found {
			return true
		}
	}
	return false
}

// RequiredRuntimeAdaptersForTemplate maps mandatory typed infrastructure
// channels to the adapter capabilities that must be resolved for placement.
// Direct HTTP and subprocess channels remain optional routes; Caido has no
// ambient/direct fallback and therefore contributes a hard requirement.
func RequiredRuntimeAdaptersForTemplate(
	template contracts.ResolvedAgentTemplate,
) ([]runtimesettings.RuntimeAdapterRef, error) {
	toolsets := registeredDescriptors().Toolsets
	selected := make(map[runtimesettings.RuntimeAdapterRef]struct{})
	for _, toolset := range template.Toolsets {
		ref := toolset.Ref.ToolsetID + "@" + toolset.Ref.Version
		descriptor, ok := toolsets[ref]
		if !ok {
			return nil, fmt.Errorf("unknown Toolset descriptor %q", ref)
		}
		for _, tool := range toolset.Tools {
			channels, ok := descriptor.InfrastructureChannels[tool]
			if !ok {
				continue
			}
			for _, channel := range channels {
				if channel == channelCaidoGraphQLClient {
					selected[runtimesettings.RuntimeAdapterCaidoGraphQL] = struct{}{}
				}
			}
		}
	}
	result := make([]runtimesettings.RuntimeAdapterRef, 0, len(selected))
	for ref := range selected {
		result = append(result, ref)
	}
	sort.Slice(result, func(i, j int) bool { return result[i] < result[j] })
	return result, nil
}

func normalizeDescriptors(input descriptorTable) (descriptorTable, error) {
	result := descriptorTable{
		PlannerFactories: make(map[string]plannerDescriptor, len(input.PlannerFactories)),
		WorkerRuntimes:   make(map[string]struct{}, len(input.WorkerRuntimes)),
		Toolsets:         make(map[string]toolsetDescriptor, len(input.Toolsets)),
		SandboxProfiles:  make(map[string]sandboxProfileDescriptor, len(input.SandboxProfiles)),
	}

	for raw, descriptor := range input.PlannerFactories {
		if _, err := ParseSelector(raw); err != nil {
			return descriptorTable{}, fmt.Errorf("invalid PlannerFactory descriptor %q: %w", raw, err)
		}
		result.PlannerFactories[raw] = descriptor
	}
	for raw := range input.WorkerRuntimes {
		if _, err := ParseSelector(raw); err != nil {
			return descriptorTable{}, fmt.Errorf("invalid WorkerRuntime descriptor %q: %w", raw, err)
		}
		result.WorkerRuntimes[raw] = struct{}{}
	}
	for raw, descriptor := range input.SandboxProfiles {
		value, err := normalizeSandboxProfileDescriptor(raw, descriptor)
		if err != nil {
			return descriptorTable{}, err
		}
		result.SandboxProfiles[raw] = value
	}
	for raw, descriptor := range input.Toolsets {
		value, err := normalizeToolsetDescriptor(raw, descriptor, result.SandboxProfiles)
		if err != nil {
			return descriptorTable{}, err
		}
		result.Toolsets[raw] = value
	}

	return result, nil
}

func validExecutionDescriptor(descriptor toolsetDescriptor) bool {
	channels := descriptor.InfrastructureChannels["exec_command"]
	return len(descriptor.Tools) == 1 && descriptor.Tools[0] == "exec_command" &&
		slices.Equal(descriptor.RequiredSandboxProfiles, []string{"podman-network@1", "podman@1"}) &&
		len(descriptor.InfrastructureChannels) == 1 && len(channels) == 1 &&
		channels[0] == channelSandboxExecution
}

func normalizeToolClassification(
	selector, classification string,
	values []string,
	exported map[string]struct{},
) ([]string, error) {
	result := append([]string(nil), values...)
	sort.Strings(result)
	for index, tool := range result {
		if _, exists := exported[tool]; !exists {
			return nil, fmt.Errorf(
				"Toolset descriptor %q classifies unknown %s tool %q",
				selector, classification, tool,
			)
		}
		if index > 0 && result[index-1] == tool {
			return nil, fmt.Errorf(
				"Toolset descriptor %q has duplicate %s tool %q",
				selector, classification, tool,
			)
		}
	}
	return result, nil
}

func normalizeSandboxProfileDescriptor(raw string, descriptor sandboxProfileDescriptor) (sandboxProfileDescriptor, error) {
	if _, err := ParseSelector(raw); err != nil {
		return sandboxProfileDescriptor{}, fmt.Errorf("invalid SandboxProfile descriptor %q: %w", raw, err)
	}
	if descriptor.WorkspaceMode != "" && descriptor.WorkspaceMode != contracts.WorkspaceModeDirect && descriptor.WorkspaceMode != contracts.WorkspaceModeOverlay {
		return sandboxProfileDescriptor{}, fmt.Errorf("invalid workspace mode for SandboxProfile %q", raw)
	}
	if descriptor.WorkspaceStorage != "" && (descriptor.WorkspaceMode == "" || (descriptor.WorkspaceStorage != contracts.WorkspaceStorageLocal && descriptor.WorkspaceStorage != contracts.WorkspaceStorageMemory)) {
		return sandboxProfileDescriptor{}, fmt.Errorf("invalid workspace storage for SandboxProfile %q", raw)
	}
	return descriptor, nil
}

func normalizeToolsetDescriptor(raw string, descriptor toolsetDescriptor, sandboxProfiles map[string]sandboxProfileDescriptor) (toolsetDescriptor, error) {
	if raw == "code-execution@1" && !validExecutionDescriptor(descriptor) {
		return toolsetDescriptor{}, fmt.Errorf("sandbox execution requires its exact isolated Toolset channel")
	}
	for _, profile := range descriptor.RequiredSandboxProfiles {
		if _, ok := sandboxProfiles[profile]; !ok {
			return toolsetDescriptor{}, fmt.Errorf("Toolset %q requires an unknown SandboxProfile", raw)
		}
	}
	if _, err := ParseSelector(raw); err != nil {
		return toolsetDescriptor{}, fmt.Errorf("invalid Toolset descriptor %q: %w", raw, err)
	}
	seen := make(map[string]struct{}, len(descriptor.Tools))
	tools := append([]string(nil), descriptor.Tools...)
	for _, tool := range tools {
		// Runtime Agents advertise and Operations shows exported tools in the
		// shared name grammar, so a descriptor cannot export another name.
		if !contracts.ValidName(tool) {
			return toolsetDescriptor{}, fmt.Errorf("invalid Toolset descriptor %q: tool %q is not a name of at most %d characters", raw, tool, contracts.MaxNameLength)
		}
		// The Agent Skills toolset owns these names whenever a template
		// selects skills, so no registered Toolset may export one.
		if contracts.IsNativeSkillToolName(tool) {
			return toolsetDescriptor{}, fmt.Errorf("Toolset descriptor %q exports tool %q, which is reserved by Agent Skills", raw, tool)
		}
		if _, exists := seen[tool]; exists {
			return toolsetDescriptor{}, fmt.Errorf("Toolset descriptor %q exports duplicate tool %q", raw, tool)
		}
		seen[tool] = struct{}{}
	}
	sort.Strings(tools)
	channels := make(map[string][]infrastructureChannel, len(descriptor.InfrastructureChannels))
	for tool, rawChannels := range descriptor.InfrastructureChannels {
		if _, ok := seen[tool]; !ok {
			return toolsetDescriptor{}, fmt.Errorf("Toolset descriptor %q describes channels for unknown tool %q", raw, tool)
		}
		if len(rawChannels) == 0 {
			return toolsetDescriptor{}, fmt.Errorf("Toolset descriptor %q has empty channels for tool %q", raw, tool)
		}
		selected := append([]infrastructureChannel(nil), rawChannels...)
		sort.Slice(selected, func(i, j int) bool { return selected[i] < selected[j] })
		for index, channel := range selected {
			if channel != channelRuntimeHTTPClient && channel != channelRuntimeSubprocessLauncher &&
				channel != channelCaidoGraphQLClient && channel != channelSandboxExecution {
				return toolsetDescriptor{}, fmt.Errorf("Toolset descriptor %q has invalid channel for tool %q", raw, tool)
			}
			if index > 0 && channel == selected[index-1] {
				return toolsetDescriptor{}, fmt.Errorf("Toolset descriptor %q has duplicate channel for tool %q", raw, tool)
			}
		}
		for _, channel := range selected {
			if channel == channelSandboxExecution && (raw != "code-execution@1" || tool != "exec_command" || len(selected) != 1 || !slices.Equal(descriptor.RequiredSandboxProfiles, []string{"podman-network@1", "podman@1"})) {
				return toolsetDescriptor{}, fmt.Errorf("sandbox execution requires its exact isolated Toolset channel")
			}
		}
		channels[tool] = selected
	}
	active, err := normalizeToolClassification(raw, "active-check", descriptor.ActiveCheckTools, seen)
	if err != nil {
		return toolsetDescriptor{}, err
	}
	var sandboxActive map[string][]string
	for _, profile := range slices.Sorted(maps.Keys(descriptor.SandboxActiveCheckTools)) {
		if _, known := sandboxProfiles[profile]; !known {
			return toolsetDescriptor{}, fmt.Errorf("Toolset %q classifies an unknown SandboxProfile", raw)
		}
		tools, err := normalizeToolClassification(raw, "sandbox-active-check", descriptor.SandboxActiveCheckTools[profile], seen)
		if err != nil {
			return toolsetDescriptor{}, err
		}
		if len(tools) == 0 {
			return toolsetDescriptor{}, fmt.Errorf("Toolset %q has an empty sandbox active-check classification", raw)
		}
		if sandboxActive == nil {
			sandboxActive = map[string][]string{}
		}
		sandboxActive[profile] = tools
	}
	findings, err := normalizeToolClassification(raw, "finding-proposal", descriptor.FindingProposalTools, seen)
	if err != nil {
		return toolsetDescriptor{}, err
	}
	inputs, err := normalizeWorkflowInputRequirements(raw, descriptor.RequiredWorkflowInputs, seen)
	if err != nil {
		return toolsetDescriptor{}, err
	}
	return toolsetDescriptor{
		RequiredSandboxProfiles: append([]string(nil), descriptor.RequiredSandboxProfiles...),
		RequiresWorkspace:       descriptor.RequiresWorkspace,
		Tools:                   tools, InfrastructureChannels: channels,
		ActiveCheckTools: active, SandboxActiveCheckTools: sandboxActive, FindingProposalTools: findings,
		RequiredWorkflowInputs: inputs,
	}, nil
}

func normalizeWorkflowInputRequirements(
	selector string,
	requirements map[string]workflowInputRequirement,
	exported map[string]struct{},
) (map[string]workflowInputRequirement, error) {
	if len(requirements) == 0 {
		return nil, nil
	}
	result := make(map[string]workflowInputRequirement, len(requirements))
	for tool, requirement := range requirements {
		if _, exists := exported[tool]; !exists {
			return nil, fmt.Errorf("Toolset descriptor %q requires a Workflow input for unknown tool %q", selector, tool)
		}
		if err := ValidateArtifactComponent("Toolset descriptor "+selector+" Workflow input", requirement.Input); err != nil {
			return nil, err
		}
		if !contracts.ValidMediaType(requirement.MediaType) {
			return nil, fmt.Errorf("Toolset descriptor %q requires non-canonical media type %q for tool %q", selector, requirement.MediaType, tool)
		}
		result[tool] = requirement
	}
	return result, nil
}
