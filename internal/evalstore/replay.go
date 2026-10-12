package evalstore

import (
	"context"
	"errors"

	pg "github.com/grauwolf32/contractor/internal/persistence/postgres"

	"github.com/grauwolf32/contractor/internal/evaldomain"
)

// Replay is an authenticated fast path before mutable catalog preflight. Create
// still arbitrates concurrent callers under the transactional operation lock.
func (s *Store) Replay(ctx context.Context, scope Scope, resource, operation string, m evaldomain.MutationIdentity) (*Receipt, error) {
	if _, err := s.project(ctx, scope, false); err != nil {
		return nil, err
	}
	receipt, found, err := pg.FindReceipt(ctx, s.db,
		pg.EvalReceiptKey(scope.OwnerID, scope.ProjectID, resource, operation, m.Key), m.RequestSHA256)
	if errors.Is(err, pg.ErrReceiptMismatch) {
		return nil, evaldomain.Failure("eval_idempotency_conflict")
	}
	if err != nil || !found {
		return nil, err
	}
	return &Receipt{Response: receipt.Response, Replayed: true}, nil
}
