package resolved

import (
	"reflect"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts"
)

func TestNucleiRulepackInputsFollowsStageAliasesWithoutChangingScope(t *testing.T) {
	worker := func(tool, alias string) ResolvedAgentBinding {
		return ResolvedAgentBinding{Template: contracts.ResolvedAgentTemplate{
			Runtime: contracts.WorkerRuntimeRef{RuntimeID: "tool", Version: "1"},
			Execution: &contracts.ToolExecutionConfig{Tool: tool, Arguments: map[string]contracts.ToolArgumentBinding{
				"rulepack_ref": {Source: "artifact", Name: alias},
			}},
		}}
	}
	workflow := ResolvedWorkflow{
		Inputs: map[string]ArtifactSlot{"custom_rules": {SourceScope: "user"}},
		Stages: map[string]ResolvedStage{
			"ordinary": {
				Agents:  map[string]ResolvedAgentBinding{"one": worker("scan_nuclei", "rules_alias"), "two": worker("scan_nuclei", "rules_alias")},
				Context: StageContext{Artifacts: map[string]ContextArtifact{"rules_alias": {Namespace: "inputs", Name: "custom_rules"}}},
			},
			"later": {
				Agents:  map[string]ResolvedAgentBinding{"one": worker("scan_nuclei", "rules_alias")},
				Context: StageContext{Artifacts: map[string]ContextArtifact{"rules_alias": {Namespace: "outputs", Name: "generated_rules"}}},
			},
			"other-tool": {
				Agents:  map[string]ResolvedAgentBinding{"one": worker("scan_sqlmap", "rules_alias")},
				Context: StageContext{Artifacts: map[string]ContextArtifact{"rules_alias": {Namespace: "inputs", Name: "unrelated"}}},
			},
		},
	}
	if got := NucleiRulepackInputs(workflow); !reflect.DeepEqual(got, []string{"custom_rules"}) {
		t.Fatalf("rulepack inputs = %v", got)
	}
	if workflow.Inputs["custom_rules"].SourceScope != "user" {
		t.Fatal("slot discovery changed source authority")
	}
}
