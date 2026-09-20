package runtimeconfig

import (
	"bytes"
	"encoding/json"
	"sort"

	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
)

// ToolsetConfig is a non-secret, typed atomic value. Implementations are
// registered in code, never by a configuration document or remote endpoint.
type ToolsetConfig interface {
	Validate() error
	CloneToolsetConfig() ToolsetConfig
	CredentialReferences() []contracts.RuntimeCredentialRef
}

type ToolsetPatches map[string]AtomicPatch[ToolsetConfig]
type ToolsetConfigurations map[string]ToolsetConfig
type toolsetSchemas map[string]func(json.RawMessage) (ToolsetConfig, error)

func registeredToolsetSchemas() toolsetSchemas {
	return toolsetSchemas{contracts.MCPToolsetRef: normalizeMCPConfig}
}

type MCPConfig struct {
	Endpoint              string `json:"endpoint"`
	Credential            string `json:"credential,omitempty"`
	CABundlePEM           string `json:"caBundlePem,omitempty"`
	ConnectTimeoutSeconds int    `json:"connectTimeoutSeconds"`
	RequestTimeoutSeconds int    `json:"requestTimeoutSeconds"`
}

func (c MCPConfig) Validate() error {
	settings := contracts.MCPToolsetSettings{Endpoint: c.Endpoint, ConnectTimeoutSeconds: c.ConnectTimeoutSeconds, RequestTimeoutSeconds: c.RequestTimeoutSeconds}
	if c.CABundlePEM != "" {
		settings.CABundlePEM = &c.CABundlePEM
	}
	if err := settings.Validate(); err != nil {
		return invalid("invalid MCP configuration")
	}
	if c.Credential != "" && validateID("MCP credential", c.Credential, 128) != nil {
		return invalid("invalid MCP credential ref")
	}
	return nil
}
func (c MCPConfig) CloneToolsetConfig() ToolsetConfig { return c }
func (c MCPConfig) CredentialReferences() []contracts.RuntimeCredentialRef {
	if c.Credential == "" {
		return nil
	}
	return []contracts.RuntimeCredentialRef{{CredentialID: c.Credential, Kind: contracts.RuntimeCredentialMCPHeaders}}
}

func normalizeMCPConfig(raw json.RawMessage) (ToolsetConfig, error) {
	var source struct {
		Endpoint              string           `json:"endpoint"`
		Credential            optional[string] `json:"credential"`
		CABundlePEM           optional[string] `json:"caBundlePem"`
		ConnectTimeoutSeconds optional[int]    `json:"connectTimeoutSeconds"`
		RequestTimeoutSeconds optional[int]    `json:"requestTimeoutSeconds"`
	}
	if decodeStrict(raw, &source) != nil || source.Credential.null || source.CABundlePEM.null || source.ConnectTimeoutSeconds.null || source.RequestTimeoutSeconds.null {
		return nil, invalid("invalid MCP configuration shape")
	}
	result := MCPConfig{Endpoint: source.Endpoint, Credential: source.Credential.value, CABundlePEM: source.CABundlePEM.value, ConnectTimeoutSeconds: contracts.DefaultMCPConnectTimeoutSeconds, RequestTimeoutSeconds: contracts.DefaultMCPRequestTimeoutSeconds}
	if source.Credential.present && source.Credential.value == "" || source.CABundlePEM.present && source.CABundlePEM.value == "" {
		return nil, invalid("MCP optional values must be nonempty")
	}
	if source.ConnectTimeoutSeconds.present {
		result.ConnectTimeoutSeconds = source.ConnectTimeoutSeconds.value
	}
	if source.RequestTimeoutSeconds.present {
		result.RequestTimeoutSeconds = source.RequestTimeoutSeconds.value
	}
	if err := result.Validate(); err != nil {
		return nil, err
	}
	return result, nil
}

func normalizeToolsetPatches(source optional[map[string]json.RawMessage], schemas toolsetSchemas) (ToolsetPatches, map[string]any, error) {
	if source.null || len(source.value) > contracts.MaximumConfiguredToolsets {
		return nil, nil, invalid("invalid worker.toolsets map")
	}
	if len(source.value) == 0 {
		return nil, nil, nil
	}
	result := make(ToolsetPatches, len(source.value))
	canonical := make(map[string]any, len(source.value))
	for _, ref := range sortedToolsetKeys(source.value) {
		normalize, exists := schemas[ref]
		if !exists || validateSelector(ref) != nil {
			return nil, nil, invalid("unregistered toolset configuration")
		}
		raw := source.value[ref]
		patch := AtomicPatch[ToolsetConfig]{Present: true}
		if bytes.Equal(bytes.TrimSpace(raw), []byte("null")) {
			patch.Clear = true
			canonical[ref] = nil
		} else {
			value, err := normalize(raw)
			if err != nil || value == nil {
				return nil, nil, invalid("invalid worker.toolsets.%s", ref)
			}
			if err := value.Validate(); err != nil {
				return nil, nil, invalid("invalid worker.toolsets.%s", ref)
			}
			patch.Value = value
			canonical[ref] = value
		}
		result[ref] = patch
	}
	return result, canonical, nil
}

func (c ToolsetConfigurations) Clone() ToolsetConfigurations {
	if c == nil {
		return nil
	}
	result := make(ToolsetConfigurations, len(c))
	for ref, value := range c {
		result[ref] = value.CloneToolsetConfig()
	}
	return result
}

func (c *ToolsetConfigurations) UnmarshalJSON(raw []byte) error {
	var source optional[map[string]json.RawMessage]
	if err := source.UnmarshalJSON(raw); err != nil {
		return invalid("invalid toolset configurations")
	}
	patches, _, err := normalizeToolsetPatches(source, registeredToolsetSchemas())
	if err != nil {
		return err
	}
	result := make(ToolsetConfigurations, len(patches))
	for ref, patch := range patches {
		if patch.Clear {
			return invalid("materialized toolset configuration cannot be clear")
		}
		result[ref] = patch.Value
	}
	*c = result
	return nil
}

func (p ToolsetPatches) Clone() ToolsetPatches {
	if p == nil {
		return nil
	}
	result := make(ToolsetPatches, len(p))
	for ref, patch := range p {
		if patch.Value != nil {
			patch.Value = patch.Value.CloneToolsetConfig()
		}
		result[ref] = patch
	}
	return result
}

func sortedToolsetKeys[T any](values map[string]T) []string {
	result := make([]string, 0, len(values))
	for ref := range values {
		result = append(result, ref)
	}
	sort.Strings(result)
	return result
}

// ResolveRunToolsets is pure: validate/merge every selected label before an
// allocation projects to its selected refs or looks up any credential material.
func ResolveRunToolsets(defaultConfig PinnedRuntimeConfig, labels []PinnedRuntimeConfig) (ToolsetConfigurations, map[string]RuntimeFieldOrigin, error) {
	if len(labels) > MaximumRunLabels {
		return nil, nil, resolutionError(ResolutionInvalid, "run_labels", ErrInvalid)
	}
	if err := validatePinned(defaultConfig, DefaultLabel); err != nil {
		return nil, nil, err
	}
	toolsetPins := make([]PinnedRuntimeConfig, len(labels))
	for i, pin := range labels {
		toolsetPins[i] = pin
		toolsetPins[i].Spec = Spec{Worker: WorkerPatch{Toolsets: pin.Spec.Worker.Toolsets}}
	}
	merged, _, pins, err := mergePinnedLayer(toolsetPins, LayerRunLabels, false, false)
	if err != nil {
		return nil, nil, err
	}
	values := make(ToolsetConfigurations)
	origins := make(map[string]RuntimeFieldOrigin)
	apply := func(patches ToolsetPatches, layer RuntimeLayer, entries []PinnedRuntimeConfig) error {
		for _, ref := range sortedToolsetKeys(patches) {
			patch := patches[ref]
			if !patch.Present {
				return resolutionError(ResolutionInvalid, "worker.toolsets."+ref, ErrInvalid)
			}
			if patch.Clear {
				delete(values, ref)
			} else {
				if patch.Value == nil || patch.Value.Validate() != nil {
					return resolutionError(ResolutionInvalid, "worker.toolsets."+ref, ErrInvalid)
				}
				values[ref] = patch.Value.CloneToolsetConfig()
			}
			sources := map[Ref]struct{}{}
			for _, entry := range entries {
				if entry.Spec.Worker.Toolsets[ref].Present {
					sources[entry.Config] = struct{}{}
				}
			}
			refs := make([]Ref, 0, len(sources))
			for source := range sources {
				refs = append(refs, source)
			}
			sort.Slice(refs, func(i, j int) bool { return refs[i].String() < refs[j].String() })
			origins[ref] = RuntimeFieldOrigin{Layer: layer, Configs: refs}
		}
		if len(origins) > contracts.MaximumConfiguredToolsets {
			return resolutionError(ResolutionInvalid, "worker.toolsets", ErrInvalid)
		}
		return nil
	}
	if err := apply(defaultConfig.Spec.Worker.Toolsets, LayerDefault, []PinnedRuntimeConfig{defaultConfig}); err != nil {
		return nil, nil, err
	}
	if err := apply(merged.Worker.Toolsets, LayerRunLabels, pins); err != nil {
		return nil, nil, err
	}
	if len(values) == 0 {
		values = nil
	}
	if len(origins) == 0 {
		origins = nil
	}
	return values, origins, nil
}

// ResolveAllocationToolsets uses the Run merge and then projects to the exact
// template refs. Callers must use this result, not label patches, for credential
// lookup: unselected and overridden configurations create no allocation access.
func ResolveAllocationToolsets(defaultConfig PinnedRuntimeConfig, labels []PinnedRuntimeConfig, selections []contracts.ToolsetSelection) (ToolsetConfigurations, map[string]RuntimeFieldOrigin, error) {
	values, origins, err := ResolveRunToolsets(defaultConfig, labels)
	if err != nil {
		return nil, nil, err
	}
	selected := make(ToolsetConfigurations)
	selectedOrigins := make(map[string]RuntimeFieldOrigin)
	for _, selection := range selections {
		ref := selection.Ref.ToolsetID + "@" + selection.Ref.Version
		value := values[ref]
		if value == nil && config.RequiredToolsetConfiguration(ref) {
			return nil, nil, resolutionError(ResolutionIncomplete, "worker.toolsets."+ref, ErrInvalid)
		}
		if value != nil {
			selected[ref] = value
		}
		if origin, ok := origins[ref]; ok {
			selectedOrigins[ref] = origin
		}
	}
	if len(selected) == 0 {
		selected = nil
	}
	if len(selectedOrigins) == 0 {
		selectedOrigins = nil
	}
	return selected, selectedOrigins, nil
}

func withoutToolsetPatches(value PinnedRuntimeConfig) PinnedRuntimeConfig {
	value.Spec.Worker.Toolsets = nil
	return value
}
