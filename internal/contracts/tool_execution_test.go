package contracts

import (
	"encoding/json"
	"testing"
)

func TestToolWorkerRejectsNullModelFields(t *testing.T) {
	for _, path := range [][]string{
		{"modelPolicy"}, {"agentTemplate", "modelPolicy"},
		{"agentTemplate", "instructions"}, {"agentTemplate", "summarizer"},
		{"runtimeSettings", "llmGatewayUrl"},
	} {
		t.Run(path[len(path)-1], func(t *testing.T) {
			var raw map[string]any
			if err := json.Unmarshal(readFixture(t, "valid", "allocation-spec-tool.json"), &raw); err != nil {
				t.Fatal(err)
			}
			target := raw
			if len(path) > 1 {
				target = raw[path[0]].(map[string]any)
			}
			target[path[len(path)-1]] = nil
			data, err := json.Marshal(raw)
			if err != nil {
				t.Fatal(err)
			}
			if _, err := DecodePrivateStrict[AllocationSpec](data); err == nil {
				t.Fatal("explicit null accepted as model absence")
			}
		})
	}
}

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
