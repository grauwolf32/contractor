package auditservice

import (
	"context"

	"github.com/grauwolf32/contractor/internal/auditstore"
)

func (s *Service) ListFindings(ctx context.Context, params auditstore.FindingListParams) ([]auditstore.ProjectFindingSummary, error) {
	return s.reads().ListFindings(ctx, params)
}

// GetFinding authorizes the Audit filter and reads the canonical Project record.
func (s *Service) GetFinding(ctx context.Context, ownerID, auditID, findingID string) (Finding, error) {
	return s.GetAuditProjectFinding(ctx, ownerID, auditID, findingID)
}
