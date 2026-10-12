package auditprofile

import (
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/config/resolved"
)

func TestAuditProfileRestrictsRetainedOutputsToAvailableRolePhases(t *testing.T) {
	t.Parallel()

	outputWorkflow := resolved.ResolvedWorkflow{Outputs: map[string]resolved.ArtifactSlot{
		"result": {Required: true, MediaTypes: []string{"application/json"}},
	}}
	consumerWorkflow := resolved.ResolvedWorkflow{Inputs: map[string]resolved.ArtifactSlot{
		"context": {Required: true, MediaTypes: []string{"application/json"}},
	}}
	dependency := func(kind AuditWorkflowRoleKind, source string) ResolvedAuditWorkflowBinding {
		return ResolvedAuditWorkflowBinding{
			Kind: kind, Workflow: consumerWorkflow,
			Inputs: map[string]AuditWorkflowInputMapping{
				"context": {Source: AuditInputFromRetainedOutput, Role: source, Name: "result"},
			},
		}
	}

	t.Run("later phase", func(t *testing.T) {
		workflows := map[string]ResolvedAuditWorkflowBinding{
			"discover": dependency(AuditWorkflowDiscovery, "assess"),
			"assess":   {Kind: AuditWorkflowAssessment, Workflow: outputWorkflow, Outputs: map[string]string{"result": "result"}},
		}
		err := validateAuditWorkflowDependencies(workflows)
		if err == nil || !strings.Contains(err.Error(), "later execution phase") {
			t.Fatalf("later-phase retained output error = %v", err)
		}
	})

	t.Run("per-item check output", func(t *testing.T) {
		workflows := map[string]ResolvedAuditWorkflowBinding{
			"check":  {Kind: AuditWorkflowCheck, Workflow: outputWorkflow, Outputs: map[string]string{"result": "result"}},
			"assess": dependency(AuditWorkflowAssessment, "check"),
		}
		err := validateAuditWorkflowDependencies(workflows)
		if err == nil || !strings.Contains(err.Error(), "cannot consume retained output from check role") {
			t.Fatalf("check retained output error = %v", err)
		}
	})

	t.Run("discovery into check", func(t *testing.T) {
		workflows := map[string]ResolvedAuditWorkflowBinding{
			"discover": {Kind: AuditWorkflowDiscovery, Workflow: outputWorkflow, Outputs: map[string]string{"result": "result"}},
			"check":    dependency(AuditWorkflowCheck, "discover"),
		}
		if err := validateAuditWorkflowDependencies(workflows); err != nil {
			t.Fatalf("valid discovery dependency = %v", err)
		}
	})
}
