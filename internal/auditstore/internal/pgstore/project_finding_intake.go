package pgstore

import (
	"context"
	"errors"
	"fmt"

	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/jackc/pgx/v5"
)

// AdmitProjectFinding attaches an explicitly published, retained receipt to
// its Project's canonical problem. The caller holds the existing Project
// correlation fence; neither a source nor a replay changes analyst state.
func (s *FindingEventsTx) AdmitProjectFinding(ctx context.Context, projectID, receiptID, findingID string) (string, error) {
	var existing string
	err := s.tx.QueryRow(ctx, `SELECT finding_id FROM project_finding_sources WHERE receipt_id=$1 AND project_id=$2`, receiptID, projectID).Scan(&existing)
	if err == nil {
		return existing, nil
	}
	if !errors.Is(err, pgx.ErrNoRows) {
		return "", err
	}
	if err := s.LockFindingCorrelationFence(ctx, projectID); err != nil {
		return "", err
	}
	err = s.tx.QueryRow(ctx, admitProjectFindingSQL, projectID, receiptID, findingID).Scan(&existing)
	if errors.Is(err, pgx.ErrNoRows) {
		return "", auditstore.ErrInvalid
	}
	if err != nil {
		return "", fmt.Errorf("admit Project Finding: %w", err)
	}
	if _, err := s.tx.Exec(ctx, `INSERT INTO project_finding_sources(project_id,finding_id,receipt_id,created_at)
        SELECT $1,$2,receipt_id,created_at FROM finding_proposal_receipts WHERE receipt_id=$3`, projectID, existing, receiptID); err != nil {
		return "", fmt.Errorf("retain Project Finding source: %w", err)
	}
	if err := s.linkFindingSourceAudit(ctx, receiptID, ""); err != nil {
		return "", err
	}

	if _, err := s.tx.Exec(ctx, `SELECT contractor_advance_finding_correlation(ARRAY[$1::text])`, projectID); err != nil {
		return "", err
	}
	return existing, nil
}

// linkFindingSourceAudit maintains only the indexed membership used by Audit
// pages. A receipt without an admitted Finding produces no membership.
func (s *FindingEventsTx) linkFindingSourceAudit(ctx context.Context, receiptID, auditID string) error {
	_, err := s.tx.Exec(ctx, linkFindingSourceAuditSQL, receiptID, auditID)
	return err
}
