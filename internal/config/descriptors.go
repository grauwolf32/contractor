package config

import (
	"fmt"
	"sort"

	"github.com/grauwolf32/contractor/internal/contracts"
)

type ToolInfrastructureChannel string

const (
	RuntimeHTTPClient         ToolInfrastructureChannel = "runtime-http-client"
	RuntimeSubprocessLauncher ToolInfrastructureChannel = "runtime-subprocess-launcher"
	CaidoGraphQLClient        ToolInfrastructureChannel = "caido-graphql-client"
)

// ToolsetDescriptor is the Server-visible part of one runtime ToolsetFactory.
// Tools are the final model-visible names exported by that exact version.
// InfrastructureChannels contains only tools with a non-empty fixed channel
// set; an absent tool is local/artifact-only. ActiveCheckTools and
// FindingProposalTools are closed Server-side effect classifications used by
// Audit compatibility checks; model text and tool names are never trusted to
// infer those effects at start time.
type ToolsetDescriptor struct {
	Tools                  []string
	InfrastructureChannels map[string][]ToolInfrastructureChannel
	ActiveCheckTools       []string
	FindingProposalTools   []string
}

// Descriptors enumerates code-backed factories that configuration is allowed
// to select. Map keys use exact <id>@<version> selectors.
type Descriptors struct {
	PlannerFactories map[string]struct{}
	WorkerRuntimes   map[string]struct{}
	Toolsets         map[string]ToolsetDescriptor
	SandboxProfiles  map[string]struct{}
}

// MVPDescriptors returns every code-backed implementation available to strict
// Workflow configuration in the current executable slice.
func MVPDescriptors() Descriptors {
	return Descriptors{
		PlannerFactories: map[string]struct{}{
			"passthrough@1": {},
			"streamline@1":  {},
			"router@1":      {},
		},
		WorkerRuntimes: map[string]struct{}{
			"adk@1": {},
		},
		Toolsets: map[string]ToolsetDescriptor{
			"caido@1": {
				Tools: []string{
					"caido_automate_results", "caido_automate_run", "caido_history",
					"caido_replay", "caido_request_detail", "caido_scope", "caido_sitemap",
					"caido_workflow_findings", "caido_workflow_list", "caido_workflow_run",
				},
				InfrastructureChannels: map[string][]ToolInfrastructureChannel{
					"caido_automate_results":  {CaidoGraphQLClient},
					"caido_automate_run":      {CaidoGraphQLClient},
					"caido_history":           {CaidoGraphQLClient},
					"caido_replay":            {CaidoGraphQLClient},
					"caido_request_detail":    {CaidoGraphQLClient},
					"caido_scope":             {CaidoGraphQLClient},
					"caido_sitemap":           {CaidoGraphQLClient},
					"caido_workflow_findings": {CaidoGraphQLClient},
					"caido_workflow_list":     {CaidoGraphQLClient},
					"caido_workflow_run":      {CaidoGraphQLClient},
				},
				ActiveCheckTools: []string{
					"caido_automate_run", "caido_replay", "caido_workflow_run",
				},
			},
			"code-analysis@1": {
				Tools: []string{
					"attack_surface", "complexity_hotspots", "entrypoint_paths_to",
					"find_callees", "find_callers", "find_symbol", "functions_that_raise",
					"graph_summary", "list_symbols", "paths_between", "search_def",
				},
			},
			"edit-files@1": {
				Tools: []string{
					"append_file", "cp", "edit", "insert_line", "mkdir", "mv", "replace_range", "rm", "write_file",
				},
			},
			"filesystem@1": {
				Tools: []string{"glob", "grep", "ls", "read_file"},
			},
			"http-tools@1": {
				Tools: []string{
					"http_history", "http_read_body", "http_request", "http_session_clear",
					"http_session_get", "http_session_set",
				},
				InfrastructureChannels: map[string][]ToolInfrastructureChannel{
					"http_request": {RuntimeHTTPClient},
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
				InfrastructureChannels: map[string][]ToolInfrastructureChannel{
					"validate_likec4": {RuntimeSubprocessLauncher},
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
				InfrastructureChannels: map[string][]ToolInfrastructureChannel{
					"validate_openapi": {RuntimeSubprocessLauncher},
				},
			},
			"run-artifacts@1": {
				Tools: []string{"list_artifacts", "read_artifact", "write_artifact"},
			},
			"source-analysis@1": {
				Tools: []string{"list_source_files", "open_source_archive", "read_source", "search_source"},
			},
			"text-artifacts@1": {
				Tools: []string{"read_text_artifact", "write_text_artifact"},
			},
			"taint-annotations@1": {
				Tools: []string{"annotate_sink", "annotate_trace", "annotate_validate"},
			},
			"workspace-changes@1": {
				Tools: []string{"changed_paths", "diff", "rollback_changes"},
			},
		},
		SandboxProfiles: map[string]struct{}{
			"local-workdir@1": {},
		},
	}
}

func normalizeDescriptors(input Descriptors) (Descriptors, error) {
	result := Descriptors{
		PlannerFactories: make(map[string]struct{}, len(input.PlannerFactories)),
		WorkerRuntimes:   make(map[string]struct{}, len(input.WorkerRuntimes)),
		Toolsets:         make(map[string]ToolsetDescriptor, len(input.Toolsets)),
		SandboxProfiles:  make(map[string]struct{}, len(input.SandboxProfiles)),
	}

	for raw := range input.PlannerFactories {
		if _, err := ParseSelector(raw); err != nil {
			return Descriptors{}, fmt.Errorf("invalid PlannerFactory descriptor %q: %w", raw, err)
		}
		result.PlannerFactories[raw] = struct{}{}
	}
	for raw := range input.WorkerRuntimes {
		if _, err := ParseSelector(raw); err != nil {
			return Descriptors{}, fmt.Errorf("invalid WorkerRuntime descriptor %q: %w", raw, err)
		}
		result.WorkerRuntimes[raw] = struct{}{}
	}
	for raw := range input.SandboxProfiles {
		if _, err := ParseSelector(raw); err != nil {
			return Descriptors{}, fmt.Errorf("invalid SandboxProfile descriptor %q: %w", raw, err)
		}
		result.SandboxProfiles[raw] = struct{}{}
	}
	for raw, descriptor := range input.Toolsets {
		if _, err := ParseSelector(raw); err != nil {
			return Descriptors{}, fmt.Errorf("invalid Toolset descriptor %q: %w", raw, err)
		}
		seen := make(map[string]struct{}, len(descriptor.Tools))
		tools := append([]string(nil), descriptor.Tools...)
		for _, tool := range tools {
			if err := validateIdentifier("Toolset tool", tool); err != nil {
				return Descriptors{}, fmt.Errorf("invalid Toolset descriptor %q: %w", raw, err)
			}
			if _, exists := seen[tool]; exists {
				return Descriptors{}, fmt.Errorf("Toolset descriptor %q exports duplicate tool %q", raw, tool)
			}
			seen[tool] = struct{}{}
		}
		sort.Strings(tools)
		channels := make(map[string][]ToolInfrastructureChannel, len(descriptor.InfrastructureChannels))
		for tool, rawChannels := range descriptor.InfrastructureChannels {
			if _, ok := seen[tool]; !ok {
				return Descriptors{}, fmt.Errorf("Toolset descriptor %q describes channels for unknown tool %q", raw, tool)
			}
			if len(rawChannels) == 0 {
				return Descriptors{}, fmt.Errorf("Toolset descriptor %q has empty channels for tool %q", raw, tool)
			}
			selected := append([]ToolInfrastructureChannel(nil), rawChannels...)
			sort.Slice(selected, func(i, j int) bool { return selected[i] < selected[j] })
			for index, channel := range selected {
				if channel != RuntimeHTTPClient && channel != RuntimeSubprocessLauncher &&
					channel != CaidoGraphQLClient {
					return Descriptors{}, fmt.Errorf("Toolset descriptor %q has invalid channel for tool %q", raw, tool)
				}
				if index > 0 && channel == selected[index-1] {
					return Descriptors{}, fmt.Errorf("Toolset descriptor %q has duplicate channel for tool %q", raw, tool)
				}
			}
			channels[tool] = selected
		}
		active, err := normalizeToolClassification(raw, "active-check", descriptor.ActiveCheckTools, seen)
		if err != nil {
			return Descriptors{}, err
		}
		findings, err := normalizeToolClassification(raw, "finding-proposal", descriptor.FindingProposalTools, seen)
		if err != nil {
			return Descriptors{}, err
		}
		result.Toolsets[raw] = ToolsetDescriptor{
			Tools: tools, InfrastructureChannels: channels,
			ActiveCheckTools: active, FindingProposalTools: findings,
		}
	}

	return result, nil
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

// RequiredRuntimeAdaptersForTemplate maps mandatory typed infrastructure
// channels to the adapter capabilities that must be resolved for placement.
// Direct HTTP and subprocess channels remain optional routes; Caido has no
// ambient/direct fallback and therefore contributes a hard requirement.
func RequiredRuntimeAdaptersForTemplate(
	template contracts.ResolvedAgentTemplate,
) ([]contracts.RuntimeAdapterRef, error) {
	descriptors := MVPDescriptors()
	selected := make(map[contracts.RuntimeAdapterRef]struct{})
	for _, toolset := range template.Toolsets {
		ref := toolset.Ref.ToolsetID + "@" + toolset.Ref.Version
		descriptor, ok := descriptors.Toolsets[ref]
		if !ok {
			return nil, fmt.Errorf("unknown Toolset descriptor %q", ref)
		}
		for _, tool := range toolset.Tools {
			channels, ok := descriptor.InfrastructureChannels[tool]
			if !ok {
				continue
			}
			for _, channel := range channels {
				if channel == CaidoGraphQLClient {
					selected[contracts.RuntimeAdapterCaidoGraphQL] = struct{}{}
				}
			}
		}
	}
	result := make([]contracts.RuntimeAdapterRef, 0, len(selected))
	for ref := range selected {
		result = append(result, ref)
	}
	sort.Slice(result, func(i, j int) bool { return result[i] < result[j] })
	return result, nil
}
