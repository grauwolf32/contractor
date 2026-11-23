package auditservice

import (
	"context"

	"github.com/grauwolf32/contractor/internal/auditstore"
)

func (s *Service) Materials(ctx context.Context, p auditstore.MaterialListParams) (auditstore.MaterialPage, error) {
	return s.reads().ListMaterials(ctx, p)
}

func (s *Service) TaskAttempts(ctx context.Context, p auditstore.TaskAttemptListParams) (auditstore.TaskAttemptPage, error) {
	return s.reads().ListTaskAttempts(ctx, p)
}
