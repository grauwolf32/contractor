package postgres

import (
	"context"
	"errors"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/artifacts/blob"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/helpers/strictjson"
	"github.com/jackc/pgx/v5"
)

// ExecutionOrigin reads trusted origin and the immutable version identity for
// an authorized exact binding. Independent writes of identical bytes have
// different identities; forks keep theirs without recursive lineage reads.
func (r *Repository) ExecutionOrigin(ctx context.Context, scope artifacts.Scope, ref contracts.ArtifactRef) (*artifacts.ExecutionOrigin, string, error) {
	if artifacts.ValidateScope(scope) != nil || ref.ValidateExact() != nil {
		return nil, "", artifacts.ErrInvalidName
	}
	var raw []byte
	var identity string
	err := r.db.QueryRow(ctx, executionOriginSQL, scope.Kind(), scope.ID(), ref.Namespace, ref.Name, *ref.Revision).Scan(&identity, &raw)
	if errors.Is(err, pgx.ErrNoRows) {
		return nil, "", artifacts.ErrArtifactNotFound
	}
	if err != nil {
		return nil, "", err
	}
	if raw == nil {
		return nil, identity, nil
	}
	var source artifacts.ExecutionOrigin
	if len(raw) > 2048 || strictjson.Decode(raw, &source) != nil || source.Artifact.ValidateExact() != nil ||
		!contracts.ValidResourceID(source.RunID) || !contracts.ValidResourceID(source.StageExecutionID) ||
		!contracts.ValidResourceID(source.AllocationID) || contracts.ValidateArtifactName(source.LogicalAgentName) != nil {
		return nil, "", blob.ErrIntegrity
	}
	return &source, identity, nil
}
