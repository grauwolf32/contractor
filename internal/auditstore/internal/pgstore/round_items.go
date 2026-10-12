package pgstore

import (
	"context"
	"errors"
	"fmt"

	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/config/auditprofile"
)

// ListRoundItems reads every item of one Round in ordinal order. A Round
// holds at most auditprofile.MaxAuditItemsPerRound items (LIM-AUD-26), so the read
// is bounded; a stored Round beyond that bound is an integrity failure rather
// than a silently truncated list.
func (s *QueryStore) ListRoundItems(ctx context.Context, auditID, roundID string) ([]auditstore.Item, error) {
	if err := validateID("auditID", auditID); err != nil {
		return nil, err
	}
	if err := validateID("roundID", roundID); err != nil {
		return nil, err
	}
	rows, err := s.db.Query(ctx, `
SELECT `+prefixedItemColumns("item")+`
  FROM audit_items AS item
 WHERE item.audit_id = $1 AND item.round_id = $2
 ORDER BY item.ordinal, item.item_id
 LIMIT $3`, auditID, roundID, auditprofile.MaxAuditItemsPerRound+1)
	if err != nil {
		return nil, fmt.Errorf("list Audit Round items: %w", err)
	}
	defer rows.Close()
	result := make([]auditstore.Item, 0)
	for rows.Next() {
		item, scanErr := scanItem(rows)
		if scanErr != nil {
			return nil, fmt.Errorf("scan Audit Round item: %w", scanErr)
		}
		result = append(result, item)
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	if len(result) > auditprofile.MaxAuditItemsPerRound {
		return nil, errors.New("stored Audit Round exceeds its item bound")
	}
	return result, nil
}
