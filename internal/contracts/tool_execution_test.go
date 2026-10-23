package contracts

import (
	"encoding/json"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
)

func TestToolBindingRejectsAmbiguousOrNullSources(t *testing.T) {
	for _, raw := range []string{
		`{"source":"parameter","name":"target","value":null}`,
		`{"source":"literal","name":null,"value":10}`,
		`{"source":"literal","value":null}`,
		`{"source":"parameter"}`,
	} {
		var value ToolArgumentBinding
		if err := json.Unmarshal([]byte(raw), &value); err == nil {
			execution := ToolExecutionConfig{Tool: "scan_nuclei", Arguments: map[string]ToolArgumentBinding{"url": value}, ResultArtifact: "report", TimeoutSeconds: 60}
			if err := execution.Validate(); err == nil {
				t.Fatalf("accepted invalid binding: %s", raw)
			}
		}
	}
}

func TestSharedToolParameterBindings(t *testing.T) {
	var cases struct {
		Bindings []struct {
			Name    string
			Binding json.RawMessage
			Valid   bool
		}
		Projections []struct {
			Name       string
			Binding    json.RawMessage
			Parameters map[string]string
			Valid      bool
			Expected   json.RawMessage
		}
	}
	if err := json.Unmarshal(contractstest.ReadFile(t, "api", "testdata", "tool-parameter-binding-cases.json"), &cases); err != nil {
		t.Fatal(err)
	}
	if len(cases.Bindings) == 0 || len(cases.Projections) == 0 {
		t.Fatal("empty binding cases")
	}
	for _, item := range cases.Bindings {
		t.Run("binding/"+item.Name, func(t *testing.T) {
			var binding ToolArgumentBinding
			err := json.Unmarshal(item.Binding, &binding)
			if err == nil {
				err = (ToolExecutionConfig{Tool: "scan_nuclei", Arguments: map[string]ToolArgumentBinding{"rate_limit": binding}, ResultArtifact: "report", TimeoutSeconds: 60}).Validate()
			}
			if (err == nil) != item.Valid {
				t.Fatalf("accepted=%v, want %v: %v", err == nil, item.Valid, err)
			}
			if err == nil {
				encoded, err := json.Marshal(binding)
				if err != nil {
					t.Fatal(err)
				}
				contractstest.AssertSemanticJSONEqual(t, encoded, item.Binding)
			}
		})
	}
	for _, item := range cases.Projections {
		t.Run("projection/"+item.Name, func(t *testing.T) {
			var binding ToolArgumentBinding
			if err := json.Unmarshal(item.Binding, &binding); err != nil {
				t.Fatal(err)
			}
			before, _ := json.Marshal(item.Parameters)
			value, err := binding.ResolveParameter(item.Parameters)
			if (err == nil) != item.Valid {
				t.Fatalf("accepted=%v, want %v: %v", err == nil, item.Valid, err)
			}
			after, _ := json.Marshal(item.Parameters)
			if string(before) != string(after) {
				t.Fatal("parameter projection mutated supplied strings")
			}
			if err == nil {
				encoded, err := json.Marshal(value)
				if err != nil {
					t.Fatal(err)
				}
				contractstest.AssertSemanticJSONEqual(t, encoded, item.Expected)
			}
		})
	}
}

func TestToolParameterBindingStrictMembers(t *testing.T) {
	for _, raw := range []string{
		`{"source":"parameter","name":"rate","parse":"integer","parse":"integer"}`,
		`{"source":"parameter","name":"rate","Parse":"integer"}`,
		`{"source":"parameter","name":"rate","default":null}`,
	} {
		var binding ToolArgumentBinding
		if err := json.Unmarshal([]byte(raw), &binding); err == nil {
			t.Fatalf("accepted %s", raw)
		}
	}
}
