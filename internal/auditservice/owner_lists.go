package auditservice

import (
	"context"

	"github.com/grauwolf32/contractor/internal/auditstore"
)

// ListOwnerFindings examines one owner-scoped window of canonical Project
// summaries. It never hydrates source history or depends on Audit execution.
func (s *Service) ListOwnerFindings(ctx context.Context, params auditstore.OwnerFindingListParams) (auditstore.FindingWindow, error) {
	rows, err := s.reads().ListOwnerFindings(ctx, params)
	if err != nil {
		return auditstore.FindingWindow{}, err
	}
	return rows, nil
}

// ListOwnerReviews lists the owner's review requests newest first. The
// pending filter lists only requests that can still be decided.
func (s *Service) ListOwnerReviews(ctx context.Context, params auditstore.OwnerReviewListParams) ([]auditstore.ReviewRequest, error) {
	return s.reads().ListOwnerReviews(ctx, params)
}
