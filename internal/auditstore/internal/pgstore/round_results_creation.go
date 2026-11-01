package pgstore

import (
	"context"
	"errors"

	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/helpers/strictjson"
	"github.com/jackc/pgx/v5"
)

// GetAcceptedRoundResults is the trusted Run-creation authority for a
// declared assessment input. Staged pages and mutable current artifacts are
// never eligible; the returned exact root is immutable after publication.
func (s *ClaimStore) GetAcceptedRoundResults(ctx context.Context, auditID, roundID string) (auditstore.ExactArtifact, error) {
	if validateID("Audit", auditID) != nil || validateID("Round", roundID) != nil {
		return auditstore.ExactArtifact{}, invalidf("Round results selection is invalid")
	}
	var data []byte
	err := s.db.QueryRow(ctx, `SELECT results_manifest FROM audit_rounds WHERE audit_id=$1 AND round_id=$2 AND results_sealed AND results_manifest IS NOT NULL
AND EXISTS (SELECT 1 FROM audits WHERE audit_id=$1 AND state='active' AND dispatch_state='open' AND (deadline_at IS NULL OR deadline_at>clock_timestamp()))`, auditID, roundID).Scan(&data)
	if errors.Is(err, pgx.ErrNoRows) {
		return auditstore.ExactArtifact{}, auditstore.ErrNotFound
	}
	if err != nil {
		return auditstore.ExactArtifact{}, err
	}
	var result auditstore.ExactArtifact
	if strictjson.Decode(data, &result) != nil || result.Validate() != nil || !result.HasMetadata() {
		return result, invalidf("accepted Round results root is invalid")
	}
	return result, nil
}
