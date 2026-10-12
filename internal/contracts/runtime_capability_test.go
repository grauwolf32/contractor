package contracts_test

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strings"
	"testing"

	jsonschema "github.com/santhosh-tekuri/jsonschema/v6"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
	"github.com/grauwolf32/contractor/internal/contracts/control"
)

// runtimeCapabilityCases mirrors api/testdata/v1alpha1/runtime-capability-cases.json,
// which the Runtime reads as well.
type runtimeCapabilityCases struct {
	ToolName struct {
		MaxLength int      `json:"maxLength"`
		Valid     []string `json:"valid"`
		Invalid   []string `json:"invalid"`
	} `json:"toolName"`
	ToolsPerToolset struct {
		MaxItems int `json:"maxItems"`
	} `json:"toolsPerToolset"`
	CapabilityRefs struct {
		MaxItems  int `json:"maxItems"`
		MaxLength int `json:"maxLength"`
	} `json:"capabilityRefs"`
	Identity struct {
		MaxLength int      `json:"maxLength"`
		Valid     []string `json:"valid"`
		Invalid   []string `json:"invalid"`
	} `json:"identity"`
}

func readRuntimeCapabilityCases(t *testing.T) runtimeCapabilityCases {
	t.Helper()
	var cases runtimeCapabilityCases
	if err := json.Unmarshal(contractstest.ReadFile(t, "api", "testdata", "v1alpha1", "runtime-capability-cases.json"), &cases); err != nil {
		t.Fatal(err)
	}
	if cases.ToolName.MaxLength == 0 || cases.ToolsPerToolset.MaxItems == 0 || cases.CapabilityRefs.MaxItems == 0 ||
		cases.CapabilityRefs.MaxLength == 0 || cases.Identity.MaxLength == 0 ||
		len(cases.ToolName.Valid) == 0 || len(cases.ToolName.Invalid) == 0 ||
		len(cases.Identity.Valid) == 0 || len(cases.Identity.Invalid) == 0 {
		t.Fatal("runtime capability cases are incomplete")
	}
	return cases
}

// wireCase decodes one golden fixture into a mutable JSON object.
func wireCase(t *testing.T, fixture string) map[string]any {
	t.Helper()
	var value map[string]any
	if err := json.Unmarshal(contractstest.ReadFixture(t, "valid", fixture), &value); err != nil {
		t.Fatal(err)
	}
	return value
}

// TestRuntimeCapabilityGoldenCases holds Go registration and heartbeat
// validation and the private JSON Schemas to the shared Runtime capability
// cases: both must accept or reject every case alike.
func TestRuntimeCapabilityGoldenCases(t *testing.T) {
	t.Parallel()
	cases := readRuntimeCapabilityCases(t)
	compiler, ids := privateSchemaCompiler(t)
	registrationSchema, err := compiler.Compile(ids["agent-registration.schema.json"])
	if err != nil {
		t.Fatal(err)
	}
	heartbeatSchema, err := compiler.Compile(ids["agent-heartbeat.schema.json"])
	if err != nil {
		t.Fatal(err)
	}
	check := func(label string, valid bool, schema *jsonschema.Schema, value map[string]any, decode func([]byte) error) {
		t.Helper()
		encoded, err := json.Marshal(value)
		if err != nil {
			t.Fatal(err)
		}
		if err := decode(encoded); (err == nil) != valid {
			t.Errorf("%s: Go decode valid=%v, error=%v", label, valid, err)
		}
		document, err := jsonschema.UnmarshalJSON(bytes.NewReader(encoded))
		if err != nil {
			t.Fatal(err)
		}
		if err := schema.Validate(document); (err == nil) != valid {
			t.Errorf("%s: schema valid=%v, error=%v", label, valid, err)
		}
	}
	decodeRegistration := func(data []byte) error {
		_, err := contracts.DecodePrivateStrict[control.AgentRegistration](data)
		return err
	}
	decodeHeartbeat := func(data []byte) error {
		_, err := contracts.DecodePrivateStrict[control.AgentHeartbeat](data)
		return err
	}
	registration := func(mutate func(map[string]any)) map[string]any {
		value := wireCase(t, "agent-registration.json")
		mutate(value)
		return value
	}
	withTools := func(tools []string) map[string]any {
		return registration(func(value map[string]any) {
			value["supportedToolsets"] = []any{map[string]any{"ref": "run-artifacts@1", "tools": tools}}
		})
	}
	numbered := func(count int, format string) []string {
		values := make([]string, count)
		for index := range values {
			values[index] = fmt.Sprintf(format, index)
		}
		return values
	}

	for _, group := range []struct {
		values []string
		valid  bool
	}{{cases.ToolName.Valid, true}, {cases.ToolName.Invalid, false}} {
		for _, tool := range group.values {
			check(fmt.Sprintf("tool %q", tool), group.valid, registrationSchema, withTools([]string{tool}), decodeRegistration)
		}
	}
	longestTool := strings.Repeat("a", cases.ToolName.MaxLength)
	check("longest tool name", true, registrationSchema, withTools([]string{longestTool}), decodeRegistration)
	check("oversized tool name", false, registrationSchema, withTools([]string{longestTool + "a"}), decodeRegistration)

	// A logical Agent name uses the same grammar: Operations shows it as the
	// allocation's logicalWorker (WFL-R84).
	allocationSchema, err := compiler.Compile(ids["allocation.schema.json"])
	if err != nil {
		t.Fatal(err)
	}
	decodeAllocation := func(data []byte) error {
		_, err := contracts.DecodePrivateStrict[control.AllocationSpec](data)
		return err
	}
	for _, group := range []struct {
		values []string
		valid  bool
	}{
		{append([]string{longestTool}, cases.ToolName.Valid...), true},
		{append([]string{longestTool + "a"}, cases.ToolName.Invalid...), false},
	} {
		for _, name := range group.values {
			allocation := wireCase(t, "allocation-spec.json")
			allocation["logicalAgentName"] = name
			check(fmt.Sprintf("logicalAgentName %q", name), group.valid, allocationSchema, allocation, decodeAllocation)
		}
	}
	check("most tools", true, registrationSchema, withTools(numbered(cases.ToolsPerToolset.MaxItems, "tool_%03d")), decodeRegistration)
	check("too many tools", false, registrationSchema, withTools(numbered(cases.ToolsPerToolset.MaxItems+1, "tool_%03d")), decodeRegistration)

	longestRef := strings.Repeat("a", cases.CapabilityRefs.MaxLength-2) + "@1"
	for _, field := range []string{"supportedRuntimes", "supportedSandboxProfiles", "supportedToolsets"} {
		refs := func(values []string) map[string]any {
			return registration(func(value map[string]any) {
				if field != "supportedToolsets" {
					value[field] = values
					return
				}
				toolsets := make([]any, len(values))
				for index, ref := range values {
					toolsets[index] = map[string]any{"ref": ref, "tools": []string{"read_artifact"}}
				}
				value[field] = toolsets
			})
		}
		check(field+" longest ref", true, registrationSchema, refs([]string{longestRef}), decodeRegistration)
		check(field+" oversized ref", false, registrationSchema, refs([]string{"a" + longestRef}), decodeRegistration)
		check(field+" most refs", true, registrationSchema, refs(numbered(cases.CapabilityRefs.MaxItems, "ref-%03d@1")), decodeRegistration)
		check(field+" too many refs", false, registrationSchema, refs(numbered(cases.CapabilityRefs.MaxItems+1, "ref-%03d@1")), decodeRegistration)
	}

	identities := append([]string{strings.Repeat("a", cases.Identity.MaxLength)}, cases.Identity.Valid...)
	invalidIdentities := append([]string{strings.Repeat("a", cases.Identity.MaxLength+1)}, cases.Identity.Invalid...)
	for _, group := range []struct {
		values []string
		valid  bool
	}{{identities, true}, {invalidIdentities, false}} {
		for _, identity := range group.values {
			check(fmt.Sprintf("registration instanceId %q", identity), group.valid, registrationSchema,
				registration(func(value map[string]any) { value["instanceId"] = identity }), decodeRegistration)
			check(fmt.Sprintf("registration allocationId %q", identity), group.valid, registrationSchema,
				registration(func(value map[string]any) {
					value["observedState"] = "allocated"
					value["allocationId"] = identity
				}), decodeRegistration)
			heartbeat := wireCase(t, "agent-heartbeat.json")
			heartbeat["instanceId"] = identity
			check(fmt.Sprintf("heartbeat instanceId %q", identity), group.valid, heartbeatSchema, heartbeat, decodeHeartbeat)
			heartbeat = wireCase(t, "agent-heartbeat.json")
			heartbeat["allocationId"] = identity
			check(fmt.Sprintf("heartbeat allocationId %q", identity), group.valid, heartbeatSchema, heartbeat, decodeHeartbeat)
		}
	}
}
