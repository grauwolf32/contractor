package contracts

import (
	"encoding/json"
	"testing"
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
