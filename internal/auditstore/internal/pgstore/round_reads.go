package pgstore

import (
	"context"
	"fmt"

	"github.com/grauwolf32/contractor/internal/auditdomain"
)

// CountItems counts the items of every Round of an Audit.
func (s *QueryStore) CountItems(ctx context.Context, auditID string) (int, error) {
	var total int
	if err := s.db.QueryRow(ctx, `SELECT count(*) FROM audit_items WHERE audit_id = $1`, auditID).Scan(&total); err != nil {
		return 0, fmt.Errorf("count Audit items: %w", err)
	}
	return total, nil
}

// ScheduledProposalChecks reports which proposed checks of the given
// receipts already have an item in an Audit, keyed by
// auditdomain.ReceiptCheckIdentity.
func (s *QueryStore) ScheduledProposalChecks(
	ctx context.Context, auditID string, receiptIDs []string,
) (map[string]bool, error) {
	result := make(map[string]bool)
	if len(receiptIDs) == 0 {
		return result, nil
	}
	rows, err := s.db.Query(ctx, `
SELECT proposal_receipt_id, proposal_check_ordinal
  FROM audit_items
 WHERE audit_id = $1 AND proposal_receipt_id = ANY($2::text[])`, auditID, receiptIDs)
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
