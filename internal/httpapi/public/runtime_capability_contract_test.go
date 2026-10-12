package public

import (
	"encoding/json"
	"fmt"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
)

// TestSharedPublicRuntimeAgentObservationSchemaMatchesSharedCapabilityCases holds
// the public RuntimeAgentObservation schema to the Runtime capability cases
// that registration applies, so the Operations lists accept every
// registration the Server accepts and nothing else.
func TestSharedPublicRuntimeAgentObservationSchemaMatchesSharedCapabilityCases(t *testing.T) {
	t.Parallel()
	schema := loadPublicOpenAPI(t).Components.Schemas["RuntimeAgentObservation"].Value
	data := contractstest.ReadFile(t, "api", "testdata", "v1alpha1", "runtime-capability-cases.json")
	var cases struct {
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
	if err := json.Unmarshal(data, &cases); err != nil {
		t.Fatal(err)
	}
	numbered := func(count int, format string) []string {
		values := make([]string, count)
		for index := range values {
			values[index] = fmt.Sprintf(format, index)
		}
		return values
	}
	check := func(label string, valid bool, mutate func(map[string]any)) {
		t.Helper()
		var observation map[string]any
		if err := json.Unmarshal([]byte(`{
			"instanceId": "runtime-1", "softwareVersion": "0.1.0",
			"supportedRuntimes": ["adk@1"], "supportedSandboxProfiles": ["local-workdir@1"],
			"supportedToolsets": [{"ref": "run-artifacts@1", "tools": ["read_artifact"]}],
			"supportedRuntimeAdapters": [], "observedState": "allocated", "slotState": "busy",
			"currentAllocationId": "allocation-1"
		}`), &observation); err != nil {
			t.Fatal(err)
		}
		mutate(observation)
		// Round-trip so the schema visits JSON values, not Go slices.
		encoded, err := json.Marshal(observation)
		if err != nil {
			t.Fatal(err)
		}
		var value any
		if err := json.Unmarshal(encoded, &value); err != nil {
			t.Fatal(err)
		}
		if err := schema.VisitJSON(value); (err == nil) != valid {
			t.Errorf("RuntimeAgentObservation %s: valid=%v, error=%v", label, valid, err)
		}
	}
	tools := func(values []string) func(map[string]any) {
		return func(observation map[string]any) {
			observation["supportedToolsets"] = []any{map[string]any{"ref": "run-artifacts@1", "tools": values}}
		}
	}
	for _, group := range []struct {
		values []string
		valid  bool
	}{
		{append([]string{strings.Repeat("a", cases.ToolName.MaxLength)}, cases.ToolName.Valid...), true},
		{append([]string{strings.Repeat("a", cases.ToolName.MaxLength+1)}, cases.ToolName.Invalid...), false},
	} {
		for _, tool := range group.values {
			check(fmt.Sprintf("tool %q", tool), group.valid, tools([]string{tool}))
		}
	}
	check("most tools", true, tools(numbered(cases.ToolsPerToolset.MaxItems, "tool_%03d")))
	check("too many tools", false, tools(numbered(cases.ToolsPerToolset.MaxItems+1, "tool_%03d")))
	longestRef := strings.Repeat("a", cases.CapabilityRefs.MaxLength-2) + "@1"
	for _, field := range []string{"supportedRuntimes", "supportedSandboxProfiles"} {
		check(field+" longest ref", true, func(observation map[string]any) { observation[field] = []string{longestRef} })
		check(field+" oversized ref", false, func(observation map[string]any) { observation[field] = []string{"a" + longestRef} })
		check(field+" most refs", true, func(observation map[string]any) {
			observation[field] = numbered(cases.CapabilityRefs.MaxItems, "ref-%03d@1")
		})
		check(field+" too many refs", false, func(observation map[string]any) {
			observation[field] = numbered(cases.CapabilityRefs.MaxItems+1, "ref-%03d@1")
		})
	}
	for _, group := range []struct {
		values []string
		valid  bool
	}{
		{append([]string{strings.Repeat("a", cases.Identity.MaxLength)}, cases.Identity.Valid...), true},
		{append([]string{strings.Repeat("a", cases.Identity.MaxLength+1)}, cases.Identity.Invalid...), false},
	} {
		for _, identity := range group.values {
			for _, field := range []string{"instanceId", "currentAllocationId"} {
				check(fmt.Sprintf("%s %q", field, identity), group.valid, func(observation map[string]any) {
					observation[field] = identity
				})
			}
		}
	}
}
