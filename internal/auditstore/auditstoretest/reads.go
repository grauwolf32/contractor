package auditstoretest

import (
	"context"
	"slices"

	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/pagelimit"
)

// EventsAfter reads up to limit events of the owner's Audit numbered above
// after, oldest first. It pages through the owner-fenced event pages that the
// public API serves, newest first, back to after; events are numbered without
// gaps (AUD-R223).
func (s *Store) EventsAfter(
	ctx context.Context, ownerID, auditID string, after uint64, limit int,
) ([]auditstore.Event, error) {
	params := auditstore.EventListParams{OwnerID: ownerID, AuditID: auditID, Limit: pagelimit.MaxRows}
	newestFirst := make([]auditstore.Event, 0)
	for {
		through, page, err := s.ListEventsPage(ctx, params)
		if err != nil {
			return nil, err
		}
		done := len(page) < params.Limit
		for _, event := range page {
			if event.Sequence <= after {
				done = true
				break
			}
			newestFirst = append(newestFirst, event)
		}
		if done {
			break
		}
		before := page[len(page)-1].Sequence
		params.ThroughSequence, params.BeforeSequence = &through, &before
	}
	slices.Reverse(newestFirst)
	return newestFirst[:min(limit, len(newestFirst))], nil
}

// Items reads every item of the owner's Audit in Round and item order. It
// pages through the owner-fenced item pages that the public API serves.
func (s *Store) Items(ctx context.Context, ownerID, auditID string) ([]auditstore.Item, error) {
	params := auditstore.ListItemsParams{OwnerID: ownerID, AuditID: auditID, Limit: pagelimit.MaxRows}
	items := make([]auditstore.Item, 0)
	for {
		page, err := s.ListItemsPage(ctx, params)
		if err != nil {
			return nil, err
		}
		items = append(items, page...)
		if len(page) < params.Limit {
			return items, nil
		}
		last := page[len(page)-1]
		round, err := s.GetRound(ctx, auditID, last.RoundID)
		if err != nil {
			return nil, err
		}
		params.AfterRoundOrdinal, params.AfterItemOrdinal, params.AfterItemID = &round.Ordinal, &last.Ordinal, last.ItemID
	}
}
