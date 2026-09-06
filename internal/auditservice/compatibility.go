package auditservice

import (
	"sort"

	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/config"
)

var reasonOrder = []CompatibilityReason{
	ReasonDiscoveryUnsupported,
	ReasonAssessmentUnsupported,
	ReasonMultipleRoundsUnsupported,
	ReasonBatchingUnsupported,
	ReasonAutomaticActiveChecksUnsupported,
	ReasonActiveCheckApprovalUnsupported,
	ReasonFindingConfirmationUnsupported,
	ReasonManualApplicabilityUnsupported,
	ReasonReportAcceptanceUnsupported,
	ReasonManualItemUnsupported,
}

func ProfileCompatibility(profile config.ResolvedAuditProfile) Compatibility {
	reasons := make(map[CompatibilityReason]struct{})
	if profile.Execution.BatchSize != 1 {
		reasons[ReasonBatchingUnsupported] = struct{}{}
	}
	if profile.Inventory.Implementation == "finding-candidates@1" {
		reasons[ReasonAssessmentUnsupported] = struct{}{}
	}
	switch profile.Interaction.ActiveChecks {
	case config.AuditActiveChecksAutomatic:
		reasons[ReasonAutomaticActiveChecksUnsupported] = struct{}{}
	case config.AuditActiveChecksApprovalRequired:
		// Exact action approval is enforced by the shared Audit review ledger
		// and rechecked in CreateExecutionIntent.
	case config.AuditActiveChecksProhibited:
		if profileSelectsClassifiedTool(profile, true) {
			reasons[ReasonAutomaticActiveChecksUnsupported] = struct{}{}
		}
	}
	if profile.Interaction.FindingConfirmation == config.AuditFindingDisabled && profileCanEmitFindings(profile) {
		reasons[ReasonFindingConfirmationUnsupported] = struct{}{}
	}
	ordered := orderedReasons(reasons)
	return Compatibility{
		ServerCompatible:        len(ordered) == 0,
		RequiresInputValidation: profile.Inventory.Implementation == "checklist@1",
		Reasons:                 ordered,
	}
}

func InventoryCompatibility(inventory auditdomain.Inventory) []CompatibilityReason {
	return nil
}

func profileSelectsClassifiedTool(profile config.ResolvedAuditProfile, active bool) bool {
	for role := range profile.Workflows {
		if workflowRoleSelectsClassifiedTool(profile, role, active) {
			return true
		}
	}
	return false
}

func workflowRoleSelectsClassifiedTool(
	profile config.ResolvedAuditProfile, workflowRole string, active bool,
) bool {
	descriptors := config.MVPDescriptors().Toolsets
	binding, exists := profile.Workflows[workflowRole]
	if !exists {
		return false
	}
	for _, stage := range binding.Workflow.Stages {
		for _, agent := range stage.Agents {
			for _, selected := range agent.Template.Toolsets {
				descriptor, ok := descriptors[selected.Ref.ToolsetID+"@"+selected.Ref.Version]
				if !ok {
					continue
				}
				classified := descriptor.FindingProposalTools
				if active {
					classified = descriptor.ActiveCheckTools
				}
				if intersects(selected.Tools, classified) {
					return true
				}
			}
		}
	}
	return false
}

func profileCanEmitFindings(profile config.ResolvedAuditProfile) bool {
	if profileSelectsClassifiedTool(profile, false) {
		return true
	}
	for _, binding := range profile.Workflows {
		for logical := range binding.Outputs {
			// "proposals" is the reserved Audit-profile output surface from
			// section 6 of the contract. Do not infer authority from arbitrary
			// substrings chosen by an operator.
			if logical == "proposals" {
				return true
			}
		}
	}
	return false
}

func intersects(left, right []string) bool {
	if len(left) == 0 || len(right) == 0 {
		return false
	}
	set := make(map[string]struct{}, len(right))
	for _, value := range right {
		set[value] = struct{}{}
	}
	for _, value := range left {
		if _, ok := set[value]; ok {
			return true
		}
	}
	return false
}

func orderedReasons(source map[CompatibilityReason]struct{}) []CompatibilityReason {
	result := make([]CompatibilityReason, 0, len(source))
	for _, reason := range reasonOrder {
		if _, ok := source[reason]; ok {
			result = append(result, reason)
		}
	}
	// Keep future reason additions deterministic even before reasonOrder is
	// updated; current public reasons always take the explicit order above.
	unknown := make([]string, 0)
	for reason := range source {
		found := false
		for _, known := range reasonOrder {
			found = found || known == reason
		}
		if !found {
			unknown = append(unknown, string(reason))
		}
	}
	sort.Strings(unknown)
	for _, reason := range unknown {
		result = append(result, CompatibilityReason(reason))
	}
	return result
}
