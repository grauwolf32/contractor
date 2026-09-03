package contracts

import (
	"encoding/json"
	"testing"
)

func TestAgentStateRejectsMissingNestedRequiredFields(t *testing.T) {
	var baseline map[string]any
	if err := json.Unmarshal(readFixture(t, "valid", "agent-state-snapshot.json"), &baseline); err != nil {
		t.Fatal(err)
	}
	tests := map[string]func(map[string]any){
		"invocation counter": func(value map[string]any) {
			state := value["state"].(map[string]any)
			current := state["currentInvocation"].(map[string]any)
			delete(current["metrics"].(map[string]any), "modelCalls")
		},
		"latest prompt tokens": func(value map[string]any) {
			state := value["state"].(map[string]any)
			current := state["currentInvocation"].(map[string]any)
			delete(current["metrics"].(map[string]any), "latestPromptTokens")
		},
		"summarizer state": func(value map[string]any) {
			state := value["state"].(map[string]any)
			current := state["currentInvocation"].(map[string]any)
			delete(current, "summarizer")
		},
		"summarizer counter": func(value map[string]any) {
			state := value["state"].(map[string]any)
			current := state["currentInvocation"].(map[string]any)
			delete(current["summarizer"].(map[string]any), "modelCalls")
		},
		"tool aggregate counter": func(value map[string]any) {
			state := value["state"].(map[string]any)
			completed := state["lastCompletedInvocation"].(map[string]any)
			tools := completed["metrics"].(map[string]any)["tools"].(map[string]any)
			delete(tools["read_file"].(map[string]any), "failures")
		},
		"workspace completeness": func(value map[string]any) {
			state := value["state"].(map[string]any)
			current := state["currentInvocation"].(map[string]any)
			delete(current["workspace"].(map[string]any), "scopeComplete")
		},
		"interaction counter": func(value map[string]any) {
			state := value["state"].(map[string]any)
			completed := state["lastCompletedInvocation"].(map[string]any)
			workspace := completed["workspace"].(map[string]any)
			interaction := workspace["interactions"].([]any)[0].(map[string]any)
			delete(interaction, "matchCalls")
		},
		"tool boolean": func(value map[string]any) {
			state := value["state"].(map[string]any)
			metrics := state["metrics"].(map[string]any)
			delete(metrics["toolCalls"].([]any)[0].(map[string]any), "argumentsTruncated")
		},
	}
	for name, mutate := range tests {
		t.Run(name, func(t *testing.T) {
			encoded, _ := json.Marshal(baseline)
			var candidate map[string]any
			_ = json.Unmarshal(encoded, &candidate)
			mutate(candidate)
			encoded, _ = json.Marshal(candidate)
			if _, err := DecodeStrict[AgentStateSnapshot](encoded); err == nil {
				t.Fatal("Agent State with a missing nested field was accepted")
			}
		})
	}
}

func TestAgentStateRejectsInconsistentSummarizerState(t *testing.T) {
	var baseline map[string]any
	if err := json.Unmarshal(readFixture(t, "valid", "agent-state-snapshot.json"), &baseline); err != nil {
		t.Fatal(err)
	}
	tests := map[string]func(map[string]any){
		"inactive usage": func(summary map[string]any) {
			summary["modelCalls"] = float64(1)
		},
		"requested without revision": func(summary map[string]any) {
			summary["phase"] = "requested"
		},
		"inactive with revision": func(summary map[string]any) {
			summary["requestStateRevision"] = float64(2)
		},
		"succeeded without call": func(summary map[string]any) {
			summary["phase"] = "succeeded"
			summary["requestStateRevision"] = float64(2)
		},
		"failed without code": func(summary map[string]any) {
			summary["phase"] = "failed"
			summary["requestStateRevision"] = float64(2)
		},
		"invalid failure code": func(summary map[string]any) {
			summary["phase"] = "failed"
			summary["requestStateRevision"] = float64(2)
			summary["failureCode"] = "bad-code"
		},
	}
	for name, mutate := range tests {
		t.Run(name, func(t *testing.T) {
			encoded, _ := json.Marshal(baseline)
			var candidate map[string]any
			_ = json.Unmarshal(encoded, &candidate)
			state := candidate["state"].(map[string]any)
			current := state["currentInvocation"].(map[string]any)
			mutate(current["summarizer"].(map[string]any))
			encoded, _ = json.Marshal(candidate)
			if _, err := DecodeStrict[AgentStateSnapshot](encoded); err == nil {
				t.Fatal("Agent State with inconsistent summarizer state was accepted")
			}
		})
	}
}
