package pgstore

import (
	"context"
	"errors"
	"fmt"
	"math"
	"regexp"

	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/helpers/pagelimit"
)

// ListItems reads every item of one Audit in Round and item order, without
// the owner fence of ListItemsPage, so a test can inspect all of them.
// Production reads items only by bounded page, selection or reconcile
// window.
func (s *QueryStore) ListItems(ctx context.Context, auditID string) ([]auditstore.Item, error) {
	if err := validateID("auditID", auditID); err != nil {
		return nil, err
	}
	rows, err := s.db.Query(ctx, `
SELECT `+prefixedItemColumns("item")+`
  FROM audit_items AS item
  JOIN audit_rounds AS round USING (round_id, audit_id)
 WHERE item.audit_id = $1
 ORDER BY round.ordinal, item.ordinal, item.item_id`, auditID)
	if err != nil {
		return nil, fmt.Errorf("list Audit items: %w", err)
	}
	defer rows.Close()
	result := make([]auditstore.Item, 0)
	for rows.Next() {
		item, scanErr := scanItem(rows)
		if scanErr != nil {
			return nil, fmt.Errorf("scan Audit item: %w", scanErr)
		}
		result = append(result, item)
	}
	return result, rows.Err()
}

var eventKindPattern = regexp.MustCompile(`^[a-z][a-z0-9_.-]{0,127}$`)

// ListEvents returns a forward, bounded page of one Audit's events above
// after and checks each stored event's sequence, kind and revision. Sequence
// zero starts at the first immutable event.
func (s *QueryStore) ListEvents(
	ctx context.Context, auditID string, after uint64, limit int,
) ([]auditstore.Event, error) {
	if err := validateID("auditID", auditID); err != nil {
		return nil, err
	}
	if after > math.MaxInt64 || !pagelimit.ValidFetch(limit) {
		return nil, invalidf("event page limit is invalid")
	}
	rows, err := s.db.Query(ctx, `
SELECT audit_id, sequence_number, kind, entity_id, entity_revision, summary, created_at
  FROM audit_events
 WHERE audit_id = $1 AND sequence_number > $2
 ORDER BY sequence_number LIMIT $3`, auditID, int64(after), limit)
	if err != nil {
		return nil, fmt.Errorf("list Audit events: %w", err)
	}
	defer rows.Close()
	result := make([]auditstore.Event, 0, limit)
	for rows.Next() {
		var event auditstore.Event
		var sequence int64
		var revision *int64
		if err := rows.Scan(
			&event.AuditID, &sequence, &event.Kind, &event.EntityID,
			&revision, &event.Summary, &event.CreatedAt,
		); err != nil {
			return nil, fmt.Errorf("scan Audit event: %w", err)
		}
		if sequence <= 0 || !eventKindPattern.MatchString(event.Kind) {
			return nil, errors.New("stored Audit event is invalid")
		}
		event.Sequence = uint64(sequence)
		if revision != nil {
			if *revision <= 0 {
				return nil, errors.New("stored Audit event revision is invalid")
			}
			value := uint64(*revision)
			event.EntityRevision = &value
		}
		result = append(result, event)
	}
	return result, rows.Err()
}
