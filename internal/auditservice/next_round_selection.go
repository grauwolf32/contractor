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
	capacity int,
) (proposalCheckSelection, error) {
	selection := proposalCheckAccumulator{selected: make(map[string]auditdomain.FindingInventoryProposal)}
	query := findingintake.ListQuery{Limit: nextRoundInboxPage}
	for selection.scanned < maxNextRoundInboxScan {
		receipts, err := s.findings.ListAuditInbox(ctx, audit.OwnerID, audit.AuditID, query)
		if err != nil {
			return proposalCheckSelection{}, err
		}
		ids := make([]string, len(receipts))
		for index := range receipts {
			ids[index] = receipts[index].ReceiptID
		}
		used, err := s.scheduledProposalChecks(ctx, audit.AuditID, ids)
		if err != nil {
			return proposalCheckSelection{}, err
		}
		if err := selection.addPage(audit, receipts, used, capacity); err != nil {
			return proposalCheckSelection{}, err
		}
		if len(receipts) < query.Limit || (selection.selectedCount >= capacity && capacity > 0) ||
			(selection.eligible && capacity <= 0) {
			break
		}
		last := receipts[len(receipts)-1]
		query.AfterCreatedAt, query.AfterReceiptID = &last.CreatedAt, last.ReceiptID
	}
	result := make([]auditdomain.FindingInventoryProposal, 0, len(selection.selected))
	for _, proposal := range selection.selected {
		result = append(result, proposal)
	}
	sort.Slice(result, func(i, j int) bool { return result[i].ReceiptID < result[j].ReceiptID })
	return proposalCheckSelection{Proposals: result, Eligible: selection.eligible, ScanExhausted: selection.scanned >= maxNextRoundInboxScan}, nil
}

type proposalCheckSelection struct {
	Proposals     []auditdomain.FindingInventoryProposal
	Eligible      bool
	ScanExhausted bool
}

type proposalCheckAccumulator struct {
	selected      map[string]auditdomain.FindingInventoryProposal
	selectedCount int
	eligible      bool
	scanned       int
}

func (selection *proposalCheckAccumulator) addPage(audit auditstore.Audit, receipts []findingintake.Receipt, used map[string]bool, capacity int) error {
	for _, receipt := range receipts {
		selection.scanned++
		hold, found := exactAuditHold(receipt, audit.AuditID, audit.ProjectID)
		if !found {
			return fmt.Errorf("Audit inbox receipt %q has no exact hold", receipt.ReceiptID)
		}
		for ordinal := range receipt.Document.ProposedChecks {
			if used[auditdomain.ReceiptCheckIdentity(receipt.ReceiptID, ordinal)] {
				continue
			}
			selection.eligible = true
			if capacity <= 0 || selection.selectedCount >= capacity {
				continue
			}
			candidate, exists := selection.selected[receipt.ReceiptID]
			if !exists {
				candidate = auditdomain.FindingInventoryProposal{
					ReceiptID: receipt.ReceiptID,
					Proposal: auditdomain.FindingInventoryArtifact{
						Ref: hold.Proposal.Ref, Digest: hold.Proposal.Digest,
						MediaType: hold.Proposal.MediaType, SizeBytes: hold.Proposal.SizeBytes,
					},
					Document: receipt.Document, SelectedCheckOrdinals: []int{},
				}
			}
			candidate.SelectedCheckOrdinals = append(candidate.SelectedCheckOrdinals, ordinal)
			selection.selected[receipt.ReceiptID] = candidate
			selection.selectedCount++
		}
		if selection.scanned >= maxNextRoundInboxScan {
			break
		}
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
