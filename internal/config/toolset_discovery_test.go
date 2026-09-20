package config

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts"
)

func TestToolsetDiscoverySharedPythonGoldenCompatibility(t *testing.T) {
	raw, err := os.ReadFile("../../api/testdata/toolset-runtime-configuration/compatibility.json")
	if err != nil {
		t.Fatal(err)
	}
	var cases []struct {
		Name, Ref, Runtime string
		Selected           []string
		Capability         json.RawMessage
		Expected           bool
	}
	if err := json.Unmarshal(raw, &cases); err != nil {
		t.Fatal(err)
	}
	for _, test := range cases {
		t.Run(test.Name, func(t *testing.T) {
			var capability contracts.ToolsetCapability
			got := false
			if err := json.Unmarshal(test.Capability, &capability); err == nil {
				ref, err := ParseSelector(test.Ref)
				if err != nil {
					t.Fatal(err)
				}
				got = ToolsetCapabilitySupports(test.Runtime, contracts.ToolsetSelection{Ref: contracts.ToolsetRef{ToolsetID: ref.ID, Version: ref.Version}, Tools: test.Selected}, capability)
			}
			if got != test.Expected {
				t.Fatalf("compatible=%v want %v", got, test.Expected)
			}
		})
	}
}

func TestMCPTemplateAuthoringRetainsShapeAndExactNames(t *testing.T) {
	for _, test := range []struct {
		name, tools string
		valid       bool
	}{{"valid", "search, _Fetch", true}, {"wildcard", "'*'", false}, {"invalid name", "remote-search", false}, {"duplicate", "search, search", false}, {"too long", strings.Repeat("x", 65), false}} {
		t.Run(test.name, func(t *testing.T) {
			root := copyConfigTree(t)
			path := filepath.Join(root, "agent-templates/artifact_builder.yaml")
			replaceFile(t, path, "ref: run-artifacts@1", "ref: mcp@1")
			replaceFile(t, path, "tools:\n        - list_artifacts\n        - read_artifact\n        - write_artifact", "tools: ["+test.tools+"]")
			snapshot, err := Load(root, MVPDescriptors())
			if (err == nil) != test.valid {
				t.Fatalf("valid=%v error=%v", test.valid, err)
			}
			if test.valid {
				template, err := snapshot.AgentTemplate("artifact_builder@1")
				if err != nil {
					t.Fatal(err)
				}
				if len(template.Toolsets) != 1 || template.Toolsets[0].Ref.ToolsetID != "mcp" {
					t.Fatal("selection shape changed")
				}
			}
		})
	}
	t.Run("native skill collision", func(t *testing.T) {
		root := copyConfigTree(t)
		addTemplateSkills(t, root, "  skills: [{namespace: skills, name: review}]\n")
		path := filepath.Join(root, "agent-templates/artifact_builder.yaml")
		replaceFile(t, path, "ref: run-artifacts@1", "ref: mcp@1")
		replaceFile(t, path, "        - list_artifacts", "        - load_skill")
		if _, err := Load(root, MVPDescriptors()); err == nil || !strings.Contains(err.Error(), "reserved by Agent Skills") {
			t.Fatalf("reserved MCP name accepted: %v", err)
		}
	})
	root := copyConfigTree(t)
	path := filepath.Join(root, "agent-templates/artifact_builder.yaml")
	replaceFile(t, path, "  sandboxProfile: local-workdir@1", "    - ref: mcp@1\n      tools: [read_artifact]\n  sandboxProfile: local-workdir@1")
	if _, err := Load(root, MVPDescriptors()); err == nil || !strings.Contains(err.Error(), "collides") {
		t.Fatalf("cross-toolset collision accepted: %v", err)
	}
	root = copyConfigTree(t)
	path = filepath.Join(root, "agent-templates/artifact_builder.yaml")
	writeFile(t, path, []byte(`apiVersion: contractor/v1alpha1
kind: AgentTemplate
metadata: {name: artifact_builder, version: '1'}
spec:
  description: Executes a tool
  runtime: tool@1
  toolsets: [{ref: mcp@1, tools: [search]}]
  sandboxProfile: local-workdir@1
  execution:
    tool: search
    arguments: {}
    resultArtifact: result
    timeoutSeconds: 60
`))
	if _, err := Load(root, MVPDescriptors()); err == nil || !strings.Contains(err.Error(), "does not support") {
		t.Fatalf("tool@1 MCP selection accepted: %v", err)
	}
}

func TestAllocationToolsetDescriptorMetadataIsValidatedAndDetached(t *testing.T) {
	descriptors := MVPDescriptors()
	normalized, err := normalizeDescriptors(descriptors)
	if err != nil {
		t.Fatal(err)
	}
	descriptor := normalized.Toolsets["mcp@1"]
	if descriptor.ToolDiscovery != "allocation" || !descriptor.Configuration.Required || descriptor.Configuration.Schema != "mcp-streamable-http@1" || !descriptor.UnknownExternalEffects || !descriptor.SupportsWorker("adk@1") || descriptor.SupportsWorker("tool@1") {
		t.Fatal("MCP metadata lost")
	}
	if channels := descriptor.ChannelsFor("arbitrary"); len(channels) != 1 || channels[0] != RuntimeHTTPClient {
		t.Fatal("dynamic common channel lost")
	}
	descriptor.Configuration.Schema = "changed"
	descriptor.CommonInfrastructureChannels[0] = "changed"
	if descriptors.Toolsets["mcp@1"].Configuration.Schema == "changed" || descriptors.Toolsets["mcp@1"].CommonInfrastructureChannels[0] == "changed" {
		t.Fatal("descriptor clone aliases metadata")
	}
	for _, mutate := range []func(*ToolsetDescriptor){func(d *ToolsetDescriptor) { d.ToolDiscovery = "remote" }, func(d *ToolsetDescriptor) { d.Tools = []string{"*"} }, func(d *ToolsetDescriptor) { d.Configuration = nil }, func(d *ToolsetDescriptor) {
		d.CommonInfrastructureChannels = []ToolInfrastructureChannel{SandboxExecution}
	}} {
		descriptors := MVPDescriptors()
		d := descriptors.Toolsets["mcp@1"]
		mutate(&d)
		descriptors.Toolsets["mcp@1"] = d
		if _, err := normalizeDescriptors(descriptors); err == nil {
			t.Fatal("invalid dynamic descriptor accepted")
		}
	}
}
