package auditservice

import (
	"fmt"
	"slices"
	"sort"

	"github.com/grauwolf32/contractor/internal/config/auditprofile"
	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contracts"
)

var reasonOrder = []CompatibilityReason{
	ReasonPreparationUnsupported,
	ReasonDiscoveryUnsupported,
	ReasonAssessmentUnsupported,
	ReasonMultipleRoundsUnsupported,
	ReasonBatchingUnsupported,
	ReasonAutomaticActiveChecksUnsupported,
	ReasonProjectActiveChecksProhibited,
	ReasonActiveCheckApprovalUnsupported,
	ReasonFindingConfirmationUnsupported,
	ReasonManualApplicabilityUnsupported,
	ReasonReportAcceptanceUnsupported,
	ReasonManualItemUnsupported,
}

func ProfileCompatibility(profile auditprofile.ResolvedAuditProfile) Compatibility {
	reasons := make(map[CompatibilityReason]struct{})
	if profile.Inventory.Implementation == "finding-candidates@1" {
		reasons[ReasonAssessmentUnsupported] = struct{}{}
	}
	switch profile.Interaction.ActiveChecks {
	case auditprofile.AuditActiveChecksAutomatic:
	case auditprofile.AuditActiveChecksApprovalRequired:
		// Exact action approval is enforced by the shared Audit review ledger
		// and rechecked in CreateExecutionIntent.
	case auditprofile.AuditActiveChecksProhibited:
		if profileSelectsClassifiedTool(profile, true) {
			reasons[ReasonAutomaticActiveChecksUnsupported] = struct{}{}
		}
	}
	if profile.Interaction.FindingConfirmation == auditprofile.AuditFindingDisabled && profileCanEmitFindings(profile) {
		reasons[ReasonFindingConfirmationUnsupported] = struct{}{}
	}
	ordered := orderedReasons(reasons)
	return Compatibility{
		ServerCompatible: len(ordered) == 0,
		RequiresInputValidation: profile.Inventory.Implementation == "checklist@1" ||
			profile.Inventory.Implementation == "standard-mappings@1" || profile.Inventory.Implementation == auditprofile.AuditInventoryOpenAPIScans,
		Reasons: ordered,
	}
}

// activeCheckPolicy resolves Project approval authority without modifying the
// retained AuditProfile identity. A profile prohibition cannot be broadened.
func activeCheckPolicy(profile auditprofile.ResolvedAuditProfile, projectPolicy contracts.ActiveChecksPolicy) (contracts.ActiveChecksPolicy, error) {
	if !projectPolicy.Valid() {
		return "", fmt.Errorf("%w: Project active checks policy is invalid", ErrInvalid)
	}
	policy := projectPolicy
	if profile.Interaction.ActiveChecks == contracts.ActiveChecksProhibited {
		policy = contracts.ActiveChecksProhibited
	}
	if policy == contracts.ActiveChecksProhibited && profileSelectsClassifiedTool(profile, true) {
		return "", unsupported([]CompatibilityReason{ReasonProjectActiveChecksProhibited})
	}
	if policy == contracts.ActiveChecksApprovalRequired {
		for role, binding := range profile.Workflows {
			if binding.Kind == auditprofile.AuditWorkflowPrepare && workflowRoleSelectsClassifiedTool(profile, role, true) {
				return "", unsupported([]CompatibilityReason{ReasonActiveCheckApprovalUnsupported})
			}
		}
	}
	return policy, nil
}

func profileSelectsClassifiedTool(profile auditprofile.ResolvedAuditProfile, active bool) bool {
	for role := range profile.Workflows {
		if workflowRoleSelectsClassifiedTool(profile, role, active) {
			return true
		}
	}
	return false
}

func workflowRoleSelectsClassifiedTool(
	profile auditprofile.ResolvedAuditProfile, workflowRole string, active bool,
) bool {
	binding, exists := profile.Workflows[workflowRole]
	if !exists {
		return false
	}
	classified := resolved.SelectsFindingProposalTool
	if active {
		classified = resolved.SelectsActiveCheckTool
	}
	for _, stage := range binding.Workflow.Stages {
		for _, agent := range stage.Agents {
			if slices.ContainsFunc(agent.Template.Toolsets, classified) {
				return true
			}
		}
	}
	return false
}

func profileCanEmitFindings(profile auditprofile.ResolvedAuditProfile) bool {
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
