package auditservice

import (
	"context"
	"fmt"
	"sort"

	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/findingintake"
)

func (s *Service) selectNextProposalChecks(
	ctx context.Context,
	audit auditstore.Audit,
	budget *auditdomain.FindingRoundBudget,
) (proposalCheckSelection, error) {
	return selectNextProposalChecksFromInbox(
		ctx, audit, budget, s.findings.ListAuditHeldInbox, s.scheduledProposalChecks,
	)
}

// selectNextProposalChecksFromInbox admits unscheduled proposal checks in
// inbox order until the Round budget is full. A proposal whose first check
// cannot fit even an empty Round is reported as unschedulable instead of
// counting as schedulable work.
func selectNextProposalChecksFromInbox(
	ctx context.Context,
	audit auditstore.Audit,
	budget *auditdomain.FindingRoundBudget,
	list func(context.Context, string, string, findingintake.ListQuery) ([]findingintake.Receipt, error),
	scheduled func(context.Context, string, []string) (map[string]bool, error),
) (proposalCheckSelection, error) {
	selection := proposalCheckAccumulator{budget: budget}
	query := findingintake.ListQuery{}
	for selection.scanned < maxNextRoundInboxScan {
		query.Limit = min(nextRoundInboxPage, maxNextRoundInboxScan-selection.scanned)
		receipts, err := list(ctx, audit.OwnerID, audit.AuditID, query)
		if err != nil {
			return proposalCheckSelection{}, err
		}
		if len(receipts) == 0 {
			break
		}
		ids := make([]string, len(receipts))
		for index := range receipts {
			ids[index] = receipts[index].ReceiptID
		}
		used, err := scheduled(ctx, audit.AuditID, ids)
		if err != nil {
			return proposalCheckSelection{}, err
		}
		if err := selection.addPage(audit, receipts, used); err != nil {
			return proposalCheckSelection{}, err
		}
		last := receipts[len(receipts)-1]
		query.AfterCreatedAt, query.AfterReceiptID = &last.CreatedAt, last.ReceiptID
		// A full Round with admitted checks is complete. Without capacity,
		// one schedulable check proves that work remains.
		if len(receipts) < query.Limit || budget.Full() && (budget.Checks() > 0 || selection.eligible) {
			break
		}
	}
	scanExhausted := false
	if selection.scanned >= maxNextRoundInboxScan && len(selection.selected) == 0 && !selection.eligible {
		// A full 10,000-receipt scan is complete when no later receipt exists.
		// Only an additional receipt proves that the scan budget hid work.
		query.Limit = 1
		more, err := list(ctx, audit.OwnerID, audit.AuditID, query)
		if err != nil {
			return proposalCheckSelection{}, err
		}
		scanExhausted = len(more) != 0
	}
	result := selection.selected
	sort.Slice(result, func(i, j int) bool { return result[i].ReceiptID < result[j].ReceiptID })
	return proposalCheckSelection{
		Proposals: result, Eligible: selection.eligible, ScanExhausted: scanExhausted,
		Unschedulable: selection.unschedulable,
	}, nil
}

type proposalCheckSelection struct {
	Proposals     []auditdomain.FindingInventoryProposal
	Eligible      bool
	ScanExhausted bool
	Unschedulable []unschedulableProposal
}

// unschedulableProposal is a held proposal whose first unscheduled check
// exceeds Limit even in an otherwise empty Round.
type unschedulableProposal struct {
	ReceiptID string
	Limit     string
}

type proposalCheckAccumulator struct {
	budget        *auditdomain.FindingRoundBudget
	selected      []auditdomain.FindingInventoryProposal
	unschedulable []unschedulableProposal
	eligible      bool
	scanned       int
}

func (selection *proposalCheckAccumulator) addPage(audit auditstore.Audit, receipts []findingintake.Receipt, used map[string]bool) error {
	for _, receipt := range receipts {
		selection.scanned++
		hold, found := exactAuditHold(receipt, audit.AuditID, audit.ProjectID)
		if !found {
			return inconsistentRound("an Audit inbox receipt has no exact hold",
				fmt.Errorf("receipt %q", receipt.ReceiptID))
		}
		ordinals := make([]int, 0, len(receipt.Document.ProposedChecks))
		for ordinal := range receipt.Document.ProposedChecks {
			if !used[auditdomain.ReceiptCheckIdentity(receipt.ReceiptID, ordinal)] {
				ordinals = append(ordinals, ordinal)
			}
		}
		if len(ordinals) != 0 {
			if err := selection.admit(receipt, hold, ordinals); err != nil {
				return err
			}
		}
		if selection.scanned >= maxNextRoundInboxScan {
			break
		}
	}
	return nil
}

func (selection *proposalCheckAccumulator) admit(
	receipt findingintake.Receipt, hold findingintake.AuditHold, ordinals []int,
) error {
	if selection.budget.Full() {
		selection.eligible = true
		return nil
	}
	admission, err := selection.budget.Admit(auditdomain.FindingInventoryProposal{
		ReceiptID: receipt.ReceiptID,
		Proposal: auditdomain.FindingInventoryArtifact{
			Ref: hold.Proposal.Ref, Digest: hold.Proposal.Digest,
			MediaType: hold.Proposal.MediaType, SizeBytes: hold.Proposal.SizeBytes,
		},
		Document: receipt.Document, SelectedCheckOrdinals: ordinals,
	})
	if err != nil {
		return err
	}
	if admission.Unschedulable != "" {
		selection.unschedulable = append(selection.unschedulable, unschedulableProposal{
			ReceiptID: receipt.ReceiptID, Limit: admission.Unschedulable,
		})
		return nil
	}
	selection.eligible = true
	if admission.Checks != 0 {
		selection.selected = append(selection.selected, admission.Proposal)
	}
	return nil
}

func (s *Service) scheduledProposalChecks(
	ctx context.Context, auditID string, receiptIDs []string,
) (map[string]bool, error) {
	result := make(map[string]bool)
	if len(receiptIDs) == 0 {
		return result, nil
	}
	rows, err := s.pool.Query(ctx, `
SELECT receipt_id, proposed_check_ordinal
  FROM audit_proposal_items
 WHERE audit_id = $1 AND receipt_id = ANY($2::text[])`, auditID, receiptIDs)
	if err != nil {
		return nil, fmt.Errorf("list scheduled Audit proposal checks: %w", err)
	}
	defer rows.Close()
	for rows.Next() {
		var receiptID string
		var ordinal int
		if err := rows.Scan(&receiptID, &ordinal); err != nil {
			return nil, err
		}
		result[auditdomain.ReceiptCheckIdentity(receiptID, ordinal)] = true
	}
	return result, rows.Err()
}

func exactAuditHold(receipt findingintake.Receipt, auditID, projectID string) (findingintake.AuditHold, bool) {
	for _, hold := range receipt.AuditHolds {
		if hold.AuditID == auditID && hold.ProjectID == projectID {
			return hold, true
		}
	}
	return findingintake.AuditHold{}, false
}
