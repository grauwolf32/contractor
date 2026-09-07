package auditservice

import (
	"context"
	"errors"
	"fmt"

	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/jackc/pgx/v5"
)

func (s *Service) ListFindings(
	ctx context.Context, params FindingListParams,
) ([]Finding, error) {
	if err := validateFindingList(params); err != nil {
		return nil, err
	}
	var state, verdict, severity *string
	if params.State != nil {
		value := string(*params.State)
		state = &value
	}
	if params.Verdict != nil {
		value := string(*params.Verdict)
		verdict = &value
	}
	if params.Severity != nil {
		value := string(*params.Severity)
		severity = &value
	}
	rows, err := s.pool.Query(ctx, `
SELECT `+findingRowColumns+`
  FROM audit_findings AS finding
  JOIN audits AS audit USING (audit_id)
  LEFT JOIN audit_review_decisions AS decision
    ON decision.decision_id = finding.current_decision_id
 WHERE audit.owner_id = $1 AND finding.audit_id = $2
   AND ($3::text IS NULL OR finding.state = $3)
   AND ($4::text IS NULL OR decision.verdict = $4)
   AND (NOT $5::boolean OR finding.current_decision_id IS NULL)
   AND ($6::text IS NULL OR decision.severity = $6)
   AND ($7::timestamptz IS NULL OR
        (finding.created_at, finding.finding_id) > ($7, $8))
 ORDER BY finding.created_at, finding.finding_id
 LIMIT $9`, params.OwnerID, params.AuditID, state, verdict, params.Unreviewed,
		severity, params.AfterCreatedAt, params.AfterFindingID, params.Limit)
	if err != nil {
		return nil, fmt.Errorf("list Audit findings: %w", err)
	}
	defer rows.Close()
	base := make([]findingRow, 0, params.Limit)
	for rows.Next() {
		row, scanErr := scanFindingRow(rows)
		if scanErr != nil {
			return nil, scanErr
		}
		base = append(base, row)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("iterate Audit findings: %w", err)
	}
	rows.Close()
	if len(base) == 0 {
		if _, err := auditstore.NewPostgresStore(s.pool).Get(ctx, params.OwnerID, params.AuditID); err != nil {
			return nil, err
		}
	}
	return s.hydrateFindings(ctx, params.OwnerID, params.AuditID, base)
}

func (s *Service) GetFinding(
	ctx context.Context, ownerID, auditID, findingID string,
) (Finding, error) {
	if !validReviewIdentity(ownerID, 256) || !validReviewIdentity(auditID, 256) ||
		!validReviewIdentity(findingID, 256) {
		return Finding{}, auditstore.ErrInvalid
	}
	row, err := scanFindingRow(s.pool.QueryRow(ctx, `
SELECT `+findingRowColumns+`
  FROM audit_findings AS finding
  JOIN audits AS audit USING (audit_id)
 WHERE audit.owner_id = $1 AND finding.audit_id = $2 AND finding.finding_id = $3`,
		ownerID, auditID, findingID))
	if errors.Is(err, pgx.ErrNoRows) {
		return Finding{}, auditstore.ErrNotFound
	}
	if err != nil {
		return Finding{}, fmt.Errorf("read Audit finding: %w", err)
	}
	return s.hydrateFinding(ctx, ownerID, row)
}

func (s *Service) hydrateFinding(ctx context.Context, ownerID string, row findingRow) (Finding, error) {
	values, err := s.hydrateFindings(ctx, ownerID, row.auditID, []findingRow{row})
	if err != nil {
		return Finding{}, err
	}
	return values[0], nil
}

func (s *Service) hydrateFindings(ctx context.Context, ownerID, auditID string, rows []findingRow) ([]Finding, error) {
	result := make([]Finding, len(rows))
	if len(rows) == 0 {
		return result, nil
	}
	receiptIDs := make([]string, len(rows))
	var decisionIDs, assessmentIDs []string
	for index, row := range rows {
		receiptIDs[index] = row.firstReceiptID
		if row.currentDecisionID != nil {
			decisionIDs = append(decisionIDs, *row.currentDecisionID)
		}
		if row.currentAssessmentID != nil {
			assessmentIDs = append(assessmentIDs, *row.currentAssessmentID)
		}
	}
	receipts, err := s.findings.GetAuditReceipts(ctx, ownerID, auditID, receiptIDs)
	if err != nil {
		return nil, err
	}
	decisions, err := s.readFindingDecisions(ctx, ownerID, auditID, decisionIDs)
	if err != nil {
		return nil, err
	}
	assessments, err := s.readFindingAssessments(ctx, ownerID, auditID, assessmentIDs)
	if err != nil {
		return nil, err
	}
	for index, row := range rows {
		finding := Finding{
			FindingID: row.findingID, AuditID: row.auditID, State: row.state,
			RejectionReason: row.rejectionReason, DuplicateTargetID: row.duplicateTargetID,
			FirstProposal: receipts[index], Revision: row.revision,
			CreatedAt: row.createdAt, UpdatedAt: row.updatedAt,
		}
		if row.currentDecisionID != nil {
			decision, ok := decisions[*row.currentDecisionID]
			if !ok {
				return nil, pgx.ErrNoRows
			}
			finding.AnalystDecision = &decision
			finding.AnalystVerdict = &decision.Verdict
			finding.AnalystSeverity = decision.Severity
		}
		if row.currentAssessmentID != nil {
			assessment, ok := assessments[*row.currentAssessmentID]
			if !ok {
				return nil, pgx.ErrNoRows
			}
			finding.CurrentAssessment = &assessment
		}
		result[index] = finding
	}
	return result, nil
}

func (s *Service) readFindingDecisions(ctx context.Context, ownerID, auditID string, ids []string) (map[string]ReviewDecision, error) {
	result := make(map[string]ReviewDecision, len(ids))
	if len(ids) == 0 {
		return result, nil
	}
	rows, err := s.pool.Query(ctx, decisionSelect+`
  JOIN audits AS audit ON audit.audit_id = decision.audit_id
 WHERE audit.owner_id = $1 AND audit.audit_id = $2
   AND decision.decision_id = ANY($3::text[])`, ownerID, auditID, ids)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	for rows.Next() {
		decision, err := readDecision(rows)
		if err != nil {
			return nil, err
		}
		result[decision.DecisionID] = decision
	}
	return result, rows.Err()
}

func (s *Service) readFindingAssessments(ctx context.Context, ownerID, auditID string, ids []string) (map[string]FindingAssessment, error) {
	result := make(map[string]FindingAssessment, len(ids))
	if len(ids) == 0 {
		return result, nil
	}
	rows, err := s.pool.Query(ctx, assessmentSelect+`
  JOIN audits AS audit ON audit.audit_id = assessment.audit_id
 WHERE audit.owner_id = $1 AND audit.audit_id = $2
   AND assessment.assessment_id = ANY($3::text[])`, ownerID, auditID, ids)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	for rows.Next() {
		assessment, err := readAssessment(rows)
		if err != nil {
			return nil, err
		}
		result[assessment.AssessmentID] = assessment
	}
	return result, rows.Err()
}
