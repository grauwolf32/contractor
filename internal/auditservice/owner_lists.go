package auditservice

import (
	"context"

	"github.com/grauwolf32/contractor/internal/auditstore"
)

// ListOwnerFindings lists the owner's findings across the Audits of listed
// Projects, newest first. Hydration reads every receipt through its finding's
// own Audit hold, for all Audits of the page at once; no unowned receipt can
// enter the projection.
func (s *Service) ListOwnerFindings(ctx context.Context, params auditstore.OwnerFindingListParams) ([]Finding, error) {
	rows, err := s.reads().ListOwnerFindings(ctx, params)
	if err != nil {
		return nil, err
	}
	return s.hydrateFindings(ctx, params.OwnerID, rows)
}

// ListOwnerReviews lists the owner's review requests newest first. The
// pending filter lists only requests that can still be decided.
func (s *Service) ListOwnerReviews(ctx context.Context, params auditstore.OwnerReviewListParams) ([]auditstore.ReviewRequest, error) {
	return s.reads().ListOwnerReviews(ctx, params)
}
