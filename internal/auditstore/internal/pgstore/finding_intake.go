package pgstore

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"

	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/jackc/pgx/v5"
)

// LockFindingIntakeAudit locks owner ownerID's Audit auditID FOR UPDATE until
// the caller's transaction ends and returns its state; a non-empty projectID
// also requires the Audit to belong to that Project. Finding intake takes this
// lock before it creates an Audit hold, accepts a direct verification or
// records a proposal rejection, after the source Run lock, in the order Run
// deletion and Audit purge use. ErrNotFound when no such Audit exists.
func (s *FindingEventsTx) LockFindingIntakeAudit(
	ctx context.Context, ownerID, auditID, projectID string,
) (auditstore.AuditState, error) {
	var state auditstore.AuditState
	err := s.tx.QueryRow(ctx, `
SELECT state FROM audits
 WHERE audit_id = $1 AND owner_id = $2 AND ($3::text = '' OR project_id = $3)
 FOR UPDATE`, auditID, ownerID, projectID).Scan(&state)
	if errors.Is(err, pgx.ErrNoRows) {
		return "", auditstore.ErrNotFound
	}
	if err != nil {
		return "", fmt.Errorf("lock Audit for finding intake: %w", err)
	}
	return state, nil
}

// CommitDirectFindingAssessment links the result and contract Artifacts to the
// Audit, records the assessment for the finding that the receipt contributed
// to and makes it the finding's current assessment, then charges the retained
// bytes to the Audit evidence quota and records finding.assessed. It runs in
// the caller's transaction, which holds LockFindingIntakeAudit's lock. It
// returns pgx.ErrNoRows when the receipt has no contribution in the Audit or
// the quota would be exceeded; the caller must then roll back.
func (s *FindingEventsTx) CommitDirectFindingAssessment(
	ctx context.Context, assessment auditstore.DirectFindingAssessment,
) error {
	resultRef, err := json.Marshal(assessment.Result.Artifact.Ref)
	if err != nil {
		return fmt.Errorf("encode direct verification result: %w", err)
	}
	contractRef, err := json.Marshal(assessment.Contract.Artifact.Ref)
	if err != nil {
		return fmt.Errorf("encode direct verification contract: %w", err)
	}
	result, contract := assessment.Result, assessment.Contract
	if _, err := s.tx.Exec(ctx, linkDirectFindingAssessmentSQL, assessment.AuditID,
		result.LogicalKey, resultRef, result.Artifact.Digest, result.Artifact.MediaType,
		result.Artifact.SizeBytes, []byte(result.SourceProvenance), result.DisplayRef,
		contract.LogicalKey, contractRef, contract.Artifact.Digest, contract.Artifact.MediaType,
		contract.Artifact.SizeBytes, []byte(contract.SourceProvenance), contract.DisplayRef,
	); err != nil {
		return err
	}
	var findingID string
	if err := s.tx.QueryRow(ctx, insertDirectFindingAssessmentSQL,
		assessment.AssessmentID, assessment.SemanticAssessment, resultRef, result.Artifact.Digest,
		contractRef, contract.Artifact.Digest, assessment.AuditID, assessment.ReceiptID,
	).Scan(&findingID); err != nil {
		return err
	}
	var findingRevision int64
	if err := s.tx.QueryRow(ctx, assessFindingDirectlySQL,
		assessment.AssessmentID, assessment.AuditID, findingID,
	).Scan(&findingRevision); err != nil {
		return err
	}
	return s.recordDirectFindingAssessment(ctx, directFindingAssessmentEvent{
		AuditID: assessment.AuditID, FindingID: findingID,
		FindingRevision: findingRevision, AssessmentID: assessment.AssessmentID,
		RetainedBytes: assessment.RetainedBytes,
	})
}
