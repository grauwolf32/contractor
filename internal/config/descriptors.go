package config

import (
	"fmt"
	"sort"
)

// ToolsetDescriptor is the Server-visible part of one runtime ToolsetFactory.
// Tools are the final model-visible names exported by that exact version.
type ToolsetDescriptor struct {
	Tools []string
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
			"likec4@1": {
				Tools: []string{
					"append_likec4", "load_likec4", "read_likec4", "replace_likec4",
					"validate_likec4", "write_likec4",
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
		result.Toolsets[raw] = ToolsetDescriptor{Tools: tools}
	}

	return result, nil
}
