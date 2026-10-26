package auditprofile

import (
	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contracts"
	"slices"
)

// WorkflowRoleSelectsClassifiedTool checks actual selected tools of a pinned
// binding; an operator-chosen role name carries no capability semantics.
func WorkflowRoleSelectsClassifiedTool(
	profile ResolvedAuditProfile, workflowRole string, active bool,
) bool {
	binding, exists := profile.Workflows[workflowRole]
	if !exists {
		return false
	}
	for _, stage := range binding.Workflow.Stages {
		for _, agent := range stage.Agents {
			if slices.ContainsFunc(agent.Template.Toolsets, func(selection contracts.ToolsetSelection) bool {
				if active {
					return resolved.SelectsActiveCheckTool(selection, agent.Template.SandboxProfile)
				}
				return resolved.SelectsFindingProposalTool(selection)
			}) {
				return true
			}
		}
	}
	return false
}
