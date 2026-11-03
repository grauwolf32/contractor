package auditservice

import (
	"context"

	"github.com/grauwolf32/contractor/internal/auditstore"
)

// List returns one bounded owner-authorized Audit window without hydrating task history.
func (s *Service) List(ctx context.Context, params auditstore.ListParams) ([]auditstore.Audit, error) {
	return s.reads().List(ctx, params)
}
