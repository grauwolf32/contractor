package auditservice

import (
	"fmt"
	"strings"

	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
)

// provisionalRevision stands in for artifact revisions that a later Round
// has not written yet when its budget is measured. The artifact store assigns
// 36-byte revisions; the longer placeholder keeps the budget an upper bound.
var provisionalRevision = strings.Repeat("0", 64)

// maximumReportedUnschedulable bounds the proposals one stop reason names.
const maximumReportedUnschedulable = 8

// newNextRoundBudget measures a later Round with the same inventory options
// and dispatched Workflow inputs that PrepareNextRound later builds it from.
func newNextRoundBudget(
	auditID, namespace string,
	roundOrdinal, capacity int,
	profile config.ResolvedAuditProfile,
	baseline BaselineSnapshot,
	selection DraftSelection,
	approval auditdomain.ApprovalRequirement,
) (*auditdomain.FindingRoundBudget, error) {
	binding, exists := profile.Workflows[profile.Inventory.ItemWorkflowRole]
	if !exists {
		return nil, fmt.Errorf("next Round item Workflow role is not pinned")
	}
	sourceRef := contracts.ArtifactRef{
		Namespace: namespace,
		Name:      auditdomain.DeterministicID("proposal-inventory", auditID),
		Revision:  &provisionalRevision,
	}
	return auditdomain.NewFindingRoundBudget(capacity, auditdomain.FindingRoundShape{
		Inventory:     nextRoundInventoryOptions(roundOrdinal, profile, baseline, approval, sourceRef),
		TaskNamespace: namespace, TaskRevision: provisionalRevision,
		ExecutionInputs: workflowInputs(binding, selection),
	})
}

func nextRoundInventoryOptions(
	roundOrdinal int,
	profile config.ResolvedAuditProfile,
	baseline BaselineSnapshot,
	approval auditdomain.ApprovalRequirement,
	sourceRef contracts.ArtifactRef,
) auditdomain.InventoryOptions {
	return auditdomain.InventoryOptions{
		Round: roundOrdinal, ProfileMode: string(profile.Mode), WorkflowRole: profile.Inventory.ItemWorkflowRole,
		SourceInputName: "proposal_inventory", SourceRef: sourceRef,
		Scope: baseline.Scope.Values(), ApprovalRequirement: approval,
	}
}

func nextRoundApproval(profile config.ResolvedAuditProfile) auditdomain.ApprovalRequirement {
	if profile.Interaction.ActiveChecks == config.AuditActiveChecksApprovalRequired &&
		workflowRoleSelectsClassifiedTool(profile, profile.Inventory.ItemWorkflowRole, true) {
		// Proposed-check methods are operator/model data, not a safe classifier.
		// Requiring approval for the whole later-round set is conservative and
		// cannot weaken a profile that requests an active-check gate.
		return auditdomain.ApprovalActiveCheck
	}
	return auditdomain.ApprovalNone
}

// unschedulableStopReason names the held proposals that no later Round can
// verify because even alone they exceed a Round inventory limit.
func unschedulableStopReason(proposals []unschedulableProposal) *auditstore.StopReason {
	named := make([]string, 0, min(len(proposals), maximumReportedUnschedulable))
	for _, proposal := range proposals[:min(len(proposals), maximumReportedUnschedulable)] {
		named = append(named, fmt.Sprintf("%s (%s)", proposal.ReceiptID, proposal.Limit))
	}
	message := "Finding proposals exceed a Round inventory limit and cannot be verified: " + strings.Join(named, ", ")
	if omitted := len(proposals) - len(named); omitted > 0 {
		message += fmt.Sprintf(", and %d more", omitted)
	}
	return &auditstore.StopReason{Code: "proposal_inventory_limit_exceeded", Message: message + "."}
}
