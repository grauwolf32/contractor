package auditservice

import (
	"slices"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
)

func TestFindingsReaderDoesNotRequireEmissionPolicy(t *testing.T) {
	for _, selected := range [][]string{{"list_findings"}, {"finding"}, {"finding", "list_findings"}} {
		t.Run(strings.Join(selected, "+"), func(t *testing.T) {
			profile := mustAuditServiceProfile(t)
			profile.Interaction.FindingConfirmation = config.AuditFindingDisabled
			binding := profile.Workflows["check"]
			stage := binding.Workflow.Stages["check"]
			agent := stage.Agents["worker"]
			agent.Template.Toolsets = []contracts.ToolsetSelection{{
				Ref: contracts.ToolsetRef{ToolsetID: "security-findings", Version: "2"}, Tools: selected,
			}}
			stage.Agents["worker"] = agent
			binding.Workflow.Stages["check"] = stage
			profile.Workflows["check"] = binding
			emits := slices.Contains(selected, "finding")
			got := ProfileCompatibility(profile)
			if profileCanEmitFindings(profile) != emits || got.ServerCompatible == emits ||
				slices.Contains(got.Reasons, ReasonFindingConfirmationUnsupported) != emits {
				t.Fatalf("selected %v, emits = %v, compatibility = %+v", selected, emits, got)
			}
		})
	}
}
