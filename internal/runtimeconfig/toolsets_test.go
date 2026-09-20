package runtimeconfig

import (
	"context"
	"encoding/json"
	"errors"
	"reflect"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts"
)

func toolsetDocument(name string, worker any) []byte {
	raw, _ := json.Marshal(map[string]any{"apiVersion": APIVersion, "kind": Kind, "metadata": map[string]string{"name": name, "version": "1"}, "spec": map[string]any{"worker": worker}})
	return raw
}
func normalizedToolsetSpec(t *testing.T, name string, toolsets any) Spec {
	t.Helper()
	p, err := PreparePublication(toolsetDocument(name, map[string]any{"toolsets": toolsets}))
	if err != nil {
		t.Fatal(err)
	}
	version, err := p.Resolve(context.Background(), nil)
	if err != nil {
		t.Fatal(err)
	}
	decoded, err := DecodeStoredDocument(version.CanonicalDocument)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(version.Spec, decoded.Spec) {
		t.Fatal("typed stored value changed")
	}
	return version.Spec
}

func TestToolsetsNormalizeBoundsDefaultsAndEmptyMap(t *testing.T) {
	mcp := map[string]any{"endpoint": "https://knowledge.example/mcp"}
	worker := map[string]any{"toolsets": map[string]any{"mcp@1": mcp}}
	first, err := PreparePublication(toolsetDocument("knowledge", worker))
	if err != nil {
		t.Fatal(err)
	}
	mcp["connectTimeoutSeconds"] = 10
	mcp["requestTimeoutSeconds"] = 60
	second, err := PreparePublication(toolsetDocument("knowledge", worker))
	if err != nil {
		t.Fatal(err)
	}
	if first.RequestDigest() != second.RequestDigest() {
		t.Fatal("explicit defaults changed digest")
	}
	version, err := first.Resolve(context.Background(), nil)
	if err != nil {
		t.Fatal(err)
	}
	value := version.Spec.Worker.Toolsets["mcp@1"].Value.(MCPConfig)
	if value.ConnectTimeoutSeconds != 10 || value.RequestTimeoutSeconds != 60 {
		t.Fatal("missing normalized defaults")
	}
	for _, invalidValue := range []any{nil, []any{}, map[string]any{}, map[string]any{"sse@1": nil}, map[string]any{"mcp@1": map[string]any{}}, map[string]any{"mcp@1": map[string]any{"endpoint": "https://x/mcp", "headers": map[string]string{"Authorization": "secret"}}}} {
		if _, err := PreparePublication(toolsetDocument("invalid", map[string]any{"toolsets": invalidValue})); err == nil {
			t.Fatalf("accepted invalid map %T", invalidValue)
		}
	}
	for field, invalid := range map[string][]any{
		"endpoint":   {nil, "", "https://user:secret@x/mcp", "https://x/mcp?", "https://x/mcp#", "file:///tmp/x"},
		"credential": {nil, "", 42}, "caBundlePem": {nil, "", "not a certificate"},
		"connectTimeoutSeconds": {nil, 0, 61, "10"}, "requestTimeoutSeconds": {nil, 0, 601}, "transport": {"sse", "stdio"},
	} {
		for _, bad := range invalid {
			candidate := map[string]any{"endpoint": "https://knowledge.example/mcp"}
			candidate[field] = bad
			if _, err := PreparePublication(toolsetDocument("invalid", map[string]any{"toolsets": map[string]any{"mcp@1": candidate}})); err == nil {
				t.Fatalf("accepted invalid %s", field)
			}
		}
	}
	raw := []byte(`{"apiVersion":"contractor/v1alpha1","kind":"RuntimeConfig","metadata":{"name":"empty-map","version":"1"},"spec":{"worker":{"toolsets":{}},"planner":{"telemetry":null}}}`)
	p, err := PreparePublication(raw)
	if err != nil {
		t.Fatal(err)
	}
	normalized, err := p.Resolve(context.Background(), nil)
	if err != nil {
		t.Fatal(err)
	}
	if string(normalized.CanonicalDocument) != `{"apiVersion":"contractor/v1alpha1","kind":"RuntimeConfig","metadata":{"name":"empty-map","version":"1"},"spec":{"planner":{"telemetry":null}}}` {
		t.Fatal("empty map not omitted")
	}
	if len(normalized.Spec.Worker.Toolsets) != 0 {
		t.Fatal("empty map retained")
	}
}

func TestRunToolsetsAtomicMergeAndSources(t *testing.T) {
	first := normalizedToolsetSpec(t, "first", map[string]any{"mcp@1": map[string]any{"endpoint": "https://one.example/mcp", "credential": "access-one"}})
	second := normalizedToolsetSpec(t, "second", map[string]any{"mcp@1": map[string]any{"endpoint": "https://two.example/mcp"}})
	clear := normalizedToolsetSpec(t, "clear", map[string]any{"mcp@1": nil})
	defaultPin := testPin("default", "1", first)
	a, b := testPin("a", "2", second), testPin("b", "3", second)
	for _, pins := range [][]PinnedRuntimeConfig{{a, b}, {b, a}} {
		values, origins, err := ResolveRunToolsets(defaultPin, pins)
		if err != nil {
			t.Fatal(err)
		}
		if values["mcp@1"].(MCPConfig).Credential != "" {
			t.Fatal("atomic replacement inherited credential")
		}
		if origins["mcp@1"].Layer != LayerRunLabels || len(origins["mcp@1"].Configs) != 2 {
			t.Fatal("equal values lost origins")
		}
	}
	inherited, origins, err := ResolveRunToolsets(defaultPin, nil)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(inherited["mcp@1"], first.Worker.Toolsets["mcp@1"].Value) || origins["mcp@1"].Layer != LayerDefault {
		t.Fatal("default not inherited")
	}
	values, origins, err := ResolveRunToolsets(defaultPin, []PinnedRuntimeConfig{testPin("clear", "2", clear)})
	if err != nil {
		t.Fatal(err)
	}
	if len(values) != 0 || origins["mcp@1"].Layer != LayerRunLabels {
		t.Fatal("clear must remove value and retain origin")
	}
	for _, other := range []Spec{first, clear} {
		conflict := testPin("c", "4", other)
		for _, pins := range [][]PinnedRuntimeConfig{{a, conflict}, {conflict, a}} {
			_, _, err := ResolveRunToolsets(defaultPin, pins)
			var reason *ResolutionError
			if !errors.As(err, &reason) || reason.Code != ResolutionConflict || reason.Path != "worker.toolsets.mcp@1" {
				t.Fatalf("conflict = %v", err)
			}
		}
	}
	input := resolverInput()
	input.AgentLabels = []PinnedRuntimeConfig{testPin("agent", "2", first)}
	if _, err := ResolveRuntimeConfig(input); err == nil {
		t.Fatal("physical agent override accepted")
	}
	input.AgentLabels[0].Spec = clear
	if _, err := ResolveRuntimeConfig(input); err == nil {
		t.Fatal("physical agent clear accepted")
	}
}

type nativeTestConfiguration struct {
	Region  string   `json:"region"`
	Limit   int      `json:"limit"`
	Options []string `json:"options,omitempty"`
}

func (v nativeTestConfiguration) Validate() error {
	if v.Region == "" || v.Limit < 1 || v.Limit > 10 {
		return ErrInvalid
	}
	return nil
}
func (v nativeTestConfiguration) CloneToolsetConfig() ToolsetConfig {
	v.Options = append([]string(nil), v.Options...)
	return v
}
func (v nativeTestConfiguration) CredentialReferences() []contracts.RuntimeCredentialRef { return nil }

func TestToolsetSchemaRegistryNativeUsesSameNormalizationMergeAndClone(t *testing.T) {
	schemas := registeredToolsetSchemas()
	schemas["test-native@2"] = func(raw json.RawMessage) (ToolsetConfig, error) {
		var v nativeTestConfiguration
		if decodeStrict(raw, &v) != nil {
			return nil, ErrInvalid
		}
		if v.Limit == 0 {
			v.Limit = 5
		}
		return v, v.Validate()
	}
	raw := toolsetDocument("native", map[string]any{"toolsets": map[string]any{"test-native@2": map[string]any{"region": "local", "options": []string{"a"}}, "mcp@1": map[string]any{"endpoint": "https://x.example/mcp"}}})
	if _, err := PreparePublication(raw); err == nil {
		t.Fatal("test registry leaked into production")
	}
	p, err := preparePublication(raw, schemas)
	if err != nil {
		t.Fatal(err)
	}
	version, err := p.Resolve(context.Background(), nil)
	if err != nil {
		t.Fatal(err)
	}
	replay, err := preparePublication(raw, schemas)
	if err != nil || replay.RequestDigest() != p.RequestDigest() {
		t.Fatal("native normalized publication replay changed")
	}
	replayed, err := replay.Resolve(context.Background(), nil)
	if err != nil || replayed.Ref != version.Ref {
		t.Fatal("native immutable publication identity changed")
	}
	stored, err := decodeStoredDocument(version.CanonicalDocument, schemas)
	if err != nil {
		t.Fatal(err)
	}
	if stored.Spec.Worker.Toolsets["test-native@2"].Value.(nativeTestConfiguration).Limit != 5 {
		t.Fatal("native defaults not normalized")
	}
	merged, err := MergeSameLayer([]LayerEntry{{Ref: version.Ref, Spec: version.Spec}})
	if err != nil {
		t.Fatal(err)
	}
	cloned := merged.Worker.Toolsets.Clone()
	cloned["test-native@2"].Value.(nativeTestConfiguration).Options[0] = "changed"
	if version.Spec.Worker.Toolsets["test-native@2"].Value.(nativeTestConfiguration).Options[0] != "a" {
		t.Fatal("clone aliases typed native data")
	}
	empty := testPin("default", "1", Spec{})
	values, origins, err := ResolveRunToolsets(empty, []PinnedRuntimeConfig{testPin("native", "2", stored.Spec)})
	if err != nil {
		t.Fatal(err)
	}
	if len(values) != 2 || len(origins) != 2 {
		t.Fatal("generic resolution lost a ref")
	}
}
