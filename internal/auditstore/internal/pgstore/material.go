package pgstore

import (
	"context"
	"errors"

	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/jackc/pgx/v5"
)

// Material describes one committed Audit-owned output. Staged forks have no
// accepted link and cannot be published merely by guessing their address.
func (s *QueryStore) Material(ctx context.Context, auditID string, ref contracts.ArtifactRef) (auditstore.ExactArtifact, error) {
	if validateID("auditID", auditID) != nil || ref.ValidateExact() != nil {
		return auditstore.ExactArtifact{}, auditstore.ErrInvalid
	}
	result := auditstore.ExactArtifact{Ref: ref.Clone()}
	err := s.db.QueryRow(ctx, materialSQL, auditID, ref.Namespace, ref.Name, *ref.Revision).Scan(&result.Digest, &result.MediaType, &result.SizeBytes)
	if errors.Is(err, pgx.ErrNoRows) {
		return auditstore.ExactArtifact{}, auditstore.ErrNotFound
	}
	if err != nil {
		return auditstore.ExactArtifact{}, err
	}
	if result.Validate() != nil {
		return auditstore.ExactArtifact{}, auditstore.ErrInvalid
	}
	return result, nil
}
