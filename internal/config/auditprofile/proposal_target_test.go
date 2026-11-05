package auditprofile_test

import (
	"github.com/grauwolf32/contractor/internal/config/auditprofile"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/auditcontract"
	"testing"
)

func TestProposalRolesUseThePinnedGraphAndRejectIncompatibleTargets(t *testing.T) {
	catalog := loadCatalog(t, copyCatalog(t, "audit-common-catalog"))
	original, err := catalog.Workflow("audit-hypothesis-generation@1")
	if err != nil {
		t.Fatal(err)
	}
	for _, kind := range []string{"configured", "missing-context", "missing-output", "invisible-context", "set-context", "wrong-worker", "misplaced-tool"} {
		t.Run(kind, func(t *testing.T) {
			workflow := original.Clone()
			binding := auditcontract.ProgramWorkflow{Inputs: map[string]auditcontract.InputMapping{"analysis": {Source: "prior-results"}}, Outputs: map[string]string{"proposals": "proposals"}}
			stage := workflow.Stages["generate"]
			agent := stage.Agents["generator"]
			agent.Template.Toolsets = []contracts.ToolsetSelection{{Ref: contracts.ToolsetRef{ToolsetID: "audit-proposals", Version: "1"}, Tools: []string{"submit_hypothesis_proposals"}}}
			stage.Agents["generator"] = agent
			switch kind {
			case "missing-context":
				delete(binding.Inputs, "analysis")
			case "missing-output":
				binding.Outputs = map[string]string{}
			case "invisible-context":
				stage.Context.Artifacts = nil
			case "set-context":
				slot := workflow.Inputs["analysis"]
				slot.MaterialSet = true
				workflow.Inputs["analysis"] = slot
			case "wrong-worker":
				agent.Namespace = "other"
				stage.Agents["generator"] = agent
			case "misplaced-tool":
				other := agent
				other.Namespace = "other"
				stage.Agents["other"] = other
			}
			workflow.Stages["generate"] = stage
			target, err := auditprofile.ResolveProposalTarget(workflow, binding)
			if kind != "configured" {
				if err == nil {
					t.Fatal("incompatible role accepted", target)
				}
				return
			}
			if err != nil || target.ContextSlot != "analysis" || target.Result.Namespace != "generator" || target.Result.Name != "proposals" {
				t.Fatal(target, err)
			}
		})
	}
}
