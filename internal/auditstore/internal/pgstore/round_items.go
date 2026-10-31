package pgstore

import (
	"context"
	"errors"
	"fmt"

	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/config/auditprofile"
	"github.com/jackc/pgx/v5"
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
   AND `+acceptedItemSQL("item")+`
 ORDER BY item.ordinal, item.item_id
 LIMIT $3`, auditID, roundID, auditprofile.MaxAuditItemsPerRound+1)
	if err != nil {
		return nil, fmt.Errorf("list Audit Round items: %w", err)
	}
	result, err := pgx.CollectRows(rows, func(row pgx.CollectableRow) (auditstore.Item, error) {
		return scanItem(row)
	})
	if err != nil {
		return nil, err
	}
	if len(result) > auditprofile.MaxAuditItemsPerRound {
		return nil, errors.New("stored Audit Round exceeds its item bound")
	}
	return result, nil
}
