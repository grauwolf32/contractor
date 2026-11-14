package reporting

import (
	"encoding/json"
	"testing"
)

func TestCompactionProgressRejectsInconsistentOutcomesAndMeasurements(t *testing.T) {
	before, after := uint64(600), uint64(80)
	valid := WorkerCompactionProgress{InvocationID: "worker-1", Generation: 1, Phase: "applied", Attempts: 1, InputBytes: 100, SummaryBytes: 20, BeforePrompt: &CompactionPromptMeasurement{Generation: 0, PromptTokens: &before}, AfterPrompt: &CompactionPromptMeasurement{Generation: 1, PromptTokens: &after}}
	if err := valid.Validate(); err != nil {
		t.Fatal(err)
	}
	for name, mutate := range map[string]func(*WorkerCompactionProgress){
		"outcomes":   func(p *WorkerCompactionProgress) { p.Failures = 1 },
		"generation": func(p *WorkerCompactionProgress) { p.Generation = 2 },
		"before":     func(p *WorkerCompactionProgress) { p.BeforePrompt.Generation = 1 },
		"after":      func(p *WorkerCompactionProgress) { p.AfterPrompt.Generation = 0 },

		"reasoning": func(p *WorkerCompactionProgress) { p.ReasoningIncluded = 1 },
	} {
		t.Run(name, func(t *testing.T) {
			p := valid.Clone()
			mutate(&p)
			if p.Validate() == nil {
				t.Fatal("inconsistent observation accepted")
			}
		})
	}
	copy := valid.Clone()
	*copy.AfterPrompt.PromptTokens = 0
	if *valid.AfterPrompt.PromptTokens != 80 {
		t.Fatal("clone aliases measurement")
	}
}

func TestNumericModelPhasesUseExistingCostPhaseVocabulary(t *testing.T) {
	data, err := json.Marshal(WorkerModelPhases{})
	if err != nil {
		t.Fatal(err)
	}
	var phases map[string]json.RawMessage
	if err := json.Unmarshal(data, &phases); err != nil {
		t.Fatal(err)
	}
	for phase := range phases {
		if !validCostGroup(LLMCostGroup{Model: "fixture", Phase: phase, Source: "provider", Calls: 1, Amount: "0"}) {
			t.Fatalf("numeric phase differs from cost vocabulary: %s", phase)
		}
	}
}
