package contracts

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestToolsetConfigurationSharedPythonGolden(t *testing.T) {
	root := filepath.Join("..", "..", "api", "testdata", "toolset-runtime-configuration")
	index, err := os.ReadFile(filepath.Join(root, "cases.json"))
	if err != nil {
		t.Fatal(err)
	}
	var cases []struct {
		File  string
		Kind  string
		Valid bool
	}
	if err := json.Unmarshal(index, &cases); err != nil {
		t.Fatal(err)
	}
	for _, test := range cases {
		if test.Kind == "origins" {
			continue
		} // Server projection tested by runtimeconfig.
		t.Run(test.File, func(t *testing.T) {
			raw, err := os.ReadFile(filepath.Join(root, test.File))
			if err != nil {
				t.Fatal(err)
			}
			var canonical []byte
			switch test.Kind {
			case "settings":
				canonical, err = privateRoundTrip[RuntimeSettings](raw)
			case "capability":
				canonical, err = privateRoundTrip[ToolsetCapability](raw)
			case "provenance":
				canonical, err = privateRoundTrip[ResolvedRuntimeConfigProvenance](raw)
			default:
				t.Fatalf("unhandled fixture kind %q", test.Kind)
			}
			if (err == nil) != test.Valid {
				t.Fatalf("valid=%v, error=%v", test.Valid, err)
			}
			if test.Valid && !bytes.Equal(canonical, bytes.TrimSpace(raw)) {
				t.Fatal("canonical bytes changed")
			}
			if strings.Contains(fmt.Sprintf("%v %+v %#v", err, err, err), "mcp-secret-canary") {
				t.Fatal("secret in error")
			}
			if test.Valid && test.Kind == "settings" {
				value, _ := DecodePrivateStrict[RuntimeSettings](raw)
				if strings.Contains(fmt.Sprintf("%v %+v %#v", value, value, value), "mcp-secret-canary") {
					t.Fatal("secret in formatting")
				}
				clone := value.Toolsets.Clone()
				if mcp, ok := clone[MCPToolsetRef].(MCPToolsetSettings); ok {
					delete(mcp.Headers, "Authorization")
					if len(value.Toolsets[MCPToolsetRef].(MCPToolsetSettings).Headers) != bytes.Count(raw, []byte("Authorization")) {
						t.Fatal("clone shares secret map")
					}
				}
			}
		})
	}
}

func TestAllocationDiscoveryRegistrationNormalization(t *testing.T) {
	registration, err := DecodePrivateStrict[AgentRegistration](readFixture(t, "valid", "agent-registration.json"))
	if err != nil {
		t.Fatal(err)
	}
	registration.SupportedToolsets = append(registration.SupportedToolsets, ToolsetCapability{Ref: MCPToolsetRef, Tools: []string{}, ToolDiscovery: "allocation"})
	normalized := NormalizeAgentRegistration(registration)
	if err := normalized.Validate(); err != nil {
		t.Fatal(err)
	}
	if _, err := AgentRegistrationFingerprint(normalized); err != nil {
		t.Fatal(err)
	}
	raw, err := MarshalPrivateCanonical(normalized)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Contains(raw, []byte(`"toolDiscovery":"allocation","tools":[]`)) {
		t.Fatal("discovery lost its non-null empty inventory")
	}
}
