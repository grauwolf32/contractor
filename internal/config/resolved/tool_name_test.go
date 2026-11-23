package resolved

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
)

// TestSharedToolsetDescriptorToolNamesGoldenCases holds the Server's Toolset
// descriptor registry to the tool-name grammar Runtime registration and
// Operations apply, so every selectable tool can be advertised and shown.
func TestSharedToolsetDescriptorToolNamesGoldenCases(t *testing.T) {
	t.Parallel()
	data := contractstest.ReadFile(t, "api", "testdata", "v1alpha1", "runtime-capability-cases.json")
	var cases struct {
		ToolName struct {
			MaxLength int      `json:"maxLength"`
			Valid     []string `json:"valid"`
			Invalid   []string `json:"invalid"`
		} `json:"toolName"`
	}
	if err := json.Unmarshal(data, &cases); err != nil {
		t.Fatal(err)
	}
	if cases.ToolName.MaxLength == 0 || len(cases.ToolName.Valid) == 0 || len(cases.ToolName.Invalid) == 0 {
		t.Fatal("tool name cases are incomplete")
	}
	accepts := func(tool string) bool {
		_, err := normalizeDescriptors(descriptorTable{Toolsets: map[string]toolsetDescriptor{
			"named-tools@1": {Tools: []string{tool}},
		}})
		return err == nil
	}
	valid := append([]string{strings.Repeat("a", cases.ToolName.MaxLength)}, cases.ToolName.Valid...)
	invalid := append([]string{strings.Repeat("a", cases.ToolName.MaxLength+1)}, cases.ToolName.Invalid...)
	for _, tool := range valid {
		if !accepts(tool) {
			t.Errorf("Toolset descriptor rejects valid tool name %q", tool)
		}
	}
	for _, tool := range invalid {
		if accepts(tool) {
			t.Errorf("Toolset descriptor accepts invalid tool name %q", tool)
		}
	}
	if _, err := normalizeDescriptors(builtInDescriptors()); err != nil {
		t.Fatalf("registered Toolset descriptors: %v", err)
	}
}
