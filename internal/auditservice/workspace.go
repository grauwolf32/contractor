package auditservice

import (
	"context"

	"github.com/grauwolf32/contractor/internal/auditstore"
)

type FindingPage struct {
	Items []Finding
	auditstore.PageBasis
}
type ReviewPage struct {
	Items []auditstore.ReviewRequest
	auditstore.PageBasis
}

func (s *Service) GetWorkspace(ctx context.Context, ownerID, auditID string) (auditstore.WorkspaceSummary, error) {
	return s.reads().GetWorkspace(ctx, ownerID, auditID)
}

// A revision fence includes hydration and the whole-filter count. Audit mutations
// increment its revision atomically, so a changing view is rejected rather than
// returning rows/counts/evidence from different revisions. No transaction is held
// while hydration acquires a pool connection (including single-connection pools).
func (s *Service) checkPageRevision(ctx context.Context, ownerID, auditID string, revision uint64) error {
	audit, err := s.Get(ctx, ownerID, auditID)
	if err != nil {
		return err
	}
	if audit.Revision != revision {
		return auditstore.ErrConflict
	}
	return nil
}

func (s *Service) ListFindingsPage(ctx context.Context, params auditstore.FindingListParams) (FindingPage, error) {
	var result FindingPage
	basis, err := s.reads().FindingPageBasis(ctx, params)
	if err != nil {
		return result, err
	}
	result.PageBasis = basis
	if params.AuditRevision != nil && *params.AuditRevision != result.AuditRevision {
		return result, auditstore.ErrConflict
	}
	result.Items, err = s.ListFindings(ctx, params)
	if err != nil {
		return result, err
	}
	return result, s.checkPageRevision(ctx, params.OwnerID, params.AuditID, result.AuditRevision)
}

func (s *Service) ListReviewsPage(ctx context.Context, params auditstore.ReviewListParams) (ReviewPage, error) {
	var result ReviewPage
	basis, err := s.reads().ReviewPageBasis(ctx, params)
	if err != nil {
		return result, err
	}
	result.PageBasis = basis
	if params.AuditRevision != nil && *params.AuditRevision != result.AuditRevision {
		return result, auditstore.ErrConflict
	}
	result.Items, err = s.ListReviews(ctx, params)
	if err != nil {
		return result, err
	}
	return result, s.checkPageRevision(ctx, params.OwnerID, params.AuditID, result.AuditRevision)
}
