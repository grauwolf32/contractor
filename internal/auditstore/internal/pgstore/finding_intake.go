package pgstore

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"

	"github.com/grauwolf32/contractor/internal/artifactpolicy"
	"github.com/grauwolf32/contractor/internal/artifacts"
	artifactpostgres "github.com/grauwolf32/contractor/internal/artifacts/postgres"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/contracts"
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

// LockFindingCorrelationFence takes Project projectID's finding correlation
// fence (FND-R212) until the caller's transaction ends, without waiting: the
// Project row FOR SHARE, then the fence row that only finding writers lock.
// Finding intake takes it after LockFindingIntakeAudit and before any
// retained artifact admission, so a Project deletion that holds the Project
// row aborts the intake instead of waiting on its Audit. Another finding
// writer of the Project, or a Project deletion or update, is a transaction
// conflict (SQLSTATE 40001); every ordinary Project receives its fence at
// creation, so a missing fence is a permanent integrity failure.
func (s *FindingEventsTx) LockFindingCorrelationFence(ctx context.Context, projectID string) error {
	if _, err := s.tx.Exec(ctx, `SELECT contractor_lock_finding_correlation_fence($1)`, projectID); err != nil {
		return fmt.Errorf("lock Project finding correlation fence: %w", err)
	}
	return nil
}

// CommitDirectFindingAssessment retains a historical source conclusion as
// an ordinary Project Artifact. It changes no current Finding state.
func (s *FindingEventsTx) CommitDirectFindingAssessment(ctx context.Context, assessment auditstore.DirectFindingAssessment) error {
	result, contract := assessment.Result, assessment.Contract
	resultRef, err := json.Marshal(result.Artifact.Ref)
	if err != nil {
		return err
	}
	contractRef, err := json.Marshal(contract.Artifact.Ref)
	if err != nil {
		return err
	}
	if _, err := s.tx.Exec(ctx, linkDirectFindingAssessmentSQL, assessment.AuditID,
		result.LogicalKey, resultRef, result.Artifact.Digest, result.Artifact.MediaType, result.Artifact.SizeBytes, []byte(result.SourceProvenance), result.DisplayRef,
		contract.LogicalKey, contractRef, contract.Artifact.Digest, contract.Artifact.MediaType, contract.Artifact.SizeBytes, []byte(contract.SourceProvenance), contract.DisplayRef); err != nil {
		return err
	}
	body, err := json.Marshal(struct {
		Schema    string         `json:"schema"`
		ReceiptID string         `json:"receiptId"`
		Kind      string         `json:"kind"`
		Original  map[string]any `json:"original"`
		Materials map[string]any `json:"materials"`
	}{auditdomain.FindingSourceHistorySchema, assessment.ReceiptID, "direct-verification",
		map[string]any{"auditId": assessment.AuditID, "recordId": assessment.AssessmentID, "semanticAssessment": assessment.SemanticAssessment},
		map[string]any{"result": result.Artifact, "contract": contract.Artifact}})
	if err != nil {
		return err
	}
	scope, err := artifacts.ProjectScope(assessment.ProjectID)
	if err != nil {
		return err
	}
	service := artifacts.NewService(artifactpostgres.NewRepository(s.tx))
	if _, err := service.CreateFrozen(ctx, scope, contracts.ArtifactRef{Namespace: artifactpolicy.FindingHistoryNamespace,
		Name: auditdomain.FindingSourceHistoryName(assessment.ReceiptID, "direct-verification", assessment.AssessmentID)},
		artifacts.Payload{MediaType: "application/json", Data: body}); err != nil {
		return err
	}
	if err := s.recordDirectFindingAssessment(ctx, directFindingAssessmentEvent{AuditID: assessment.AuditID,
		ReceiptID: assessment.ReceiptID, AssessmentID: assessment.AssessmentID, RetainedBytes: assessment.RetainedBytes}); err != nil {
		return err
	}
	_, err = s.tx.Exec(ctx, "SELECT contractor_advance_finding_correlation(ARRAY[$1::text])", assessment.ProjectID)
	return err
}
