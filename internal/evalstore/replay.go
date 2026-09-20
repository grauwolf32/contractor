package evalstore

import (
	"context"
	"errors"

	"github.com/grauwolf32/contractor/internal/evaldomain"
	"github.com/jackc/pgx/v5"
)

// Replay is an authenticated fast path before mutable catalog preflight. Create
// still arbitrates concurrent callers under the transactional operation lock.
func (s *Store) Replay(ctx context.Context, scope Scope, resource, operation string, m evaldomain.MutationIdentity) (*Receipt, error) {
	if _, err := s.project(ctx, scope, false); err != nil {
		return nil, err
	}
	var digest string
	var response []byte
	err := s.db.QueryRow(ctx, `
SELECT request_sha256, response
FROM eval_mutation_receipts
WHERE owner_id=$1
    AND project_id=$2
    AND resource_id=$3
    AND operation=$4
    AND operation_key=$5
`, scope.OwnerID, scope.ProjectID, resource, operation, m.Key).Scan(&digest, &response)
	if errors.Is(err, pgx.ErrNoRows) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	if digest != m.RequestSHA256 {
		return nil, evaldomain.Failure("eval_idempotency_conflict")
	}
	return &Receipt{Response: response, Replayed: true}, nil
}
