package auditservice

import (
	"context"
	"errors"
	"fmt"

	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/findingintake"
	"github.com/jackc/pgx/v5"
)

const findingFilterSQL = `
  FROM audit_findings AS finding
  JOIN audits AS audit USING (audit_id)
  LEFT JOIN audit_review_decisions AS decision
    ON decision.decision_id = finding.current_decision_id
 WHERE audit.owner_id = $1 AND ($2::text IS NULL OR finding.audit_id = $2)
   AND ($3::text IS NULL OR finding.state = $3)
   AND ($4::text IS NULL OR decision.verdict = $4)
   AND (NOT $5::boolean OR finding.current_decision_id IS NULL)
   AND ($6::text IS NULL OR decision.severity = $6)`

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
`+findingFilterSQL+`
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
	return s.hydrateFindings(ctx, params.OwnerID, base)
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
	values, err := s.hydrateFindings(ctx, ownerID, []findingRow{row})
	if err != nil {
		return Finding{}, err
	}
	return values[0], nil
}

// hydrateFindings completes a page of finding rows, which may belong to
// several of the owner's Audits, with their first proposal receipts, analyst
// decisions and current assessments. Each related record is read through its
// finding's own Audit, and the page costs the same statements whatever the
// number of Audits on it: one batch of receipts per MaxAuditReceiptBatchSize
// rows, one decision read and one assessment read.
func (s *Service) hydrateFindings(ctx context.Context, ownerID string, rows []findingRow) ([]Finding, error) {
	result := make([]Finding, len(rows))
	if len(rows) == 0 {
		return result, nil
	}
	receiptKeys := make([]findingintake.AuditReceiptKey, len(rows))
	var decisions, assessments auditRecordKeys
	for index, row := range rows {
		receiptKeys[index] = findingintake.AuditReceiptKey{AuditID: row.auditID, ReceiptID: row.firstReceiptID}
		if row.currentDecisionID != nil && !decisions.add(row.auditID, *row.currentDecisionID) {
			return nil, pgx.ErrNoRows
		}
		if row.currentAssessmentID != nil && !assessments.add(row.auditID, *row.currentAssessmentID) {
			return nil, pgx.ErrNoRows
		}
	}
	// A full public page includes a cursor sentinel. Keep receipt hydration
	// within its independent batch bound, including ownership and exact reads
	// for that extra row before the HTTP handler trims it from the response.
	receipts := make([]findingintake.Receipt, 0, len(receiptKeys))
	for start := 0; start < len(receiptKeys); start += findingintake.MaxAuditReceiptBatchSize {
		end := min(start+findingintake.MaxAuditReceiptBatchSize, len(receiptKeys))
		batch, err := s.findings.GetOwnerAuditReceipts(ctx, ownerID, receiptKeys[start:end])
		if err != nil {
			return nil, err
		}
		receipts = append(receipts, batch...)
	}
	decisionsByID, err := s.readFindingDecisions(ctx, ownerID, decisions)
	if err != nil {
		return nil, err
	}
	assessmentsByID, err := s.readFindingAssessments(ctx, ownerID, assessments)
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
			decision, ok := decisionsByID[*row.currentDecisionID]
			if !ok {
				return nil, pgx.ErrNoRows
			}
			finding.AnalystDecision = &decision
			finding.AnalystVerdict = &decision.Verdict
			finding.AnalystSeverity = decision.Severity
		}
		if row.currentAssessmentID != nil {
			assessment, ok := assessmentsByID[*row.currentAssessmentID]
			if !ok {
				return nil, pgx.ErrNoRows
			}
			finding.CurrentAssessment = &assessment
		}
		result[index] = finding
	}
	return result, nil
}

// auditRecordKeys are the parallel Audit and record identities of the
// decisions or assessments one finding page reads, each through its own Audit.
// A read returns a record only through the Audit it belongs to, so a record
// identity names one Audit on the page.
type auditRecordKeys struct {
	auditIDs, recordIDs []string
	auditByRecord       map[string]string
}

// add records that a finding of auditID references recordID and reports
// false when a finding of another Audit references the same record: each
// decision and assessment belongs to one Audit's finding, so one of the two
// references cannot resolve.
func (k *auditRecordKeys) add(auditID, recordID string) bool {
	if k.auditByRecord == nil {
		k.auditByRecord = make(map[string]string)
	}
	if previous, seen := k.auditByRecord[recordID]; seen {
		return previous == auditID
	}
	k.auditByRecord[recordID] = auditID
	k.auditIDs = append(k.auditIDs, auditID)
	k.recordIDs = append(k.recordIDs, recordID)
	return true
}

func (s *Service) readFindingDecisions(ctx context.Context, ownerID string, keys auditRecordKeys) (map[string]ReviewDecision, error) {
	result := make(map[string]ReviewDecision, len(keys.recordIDs))
	if len(keys.recordIDs) == 0 {
		return result, nil
	}
	rows, err := s.pool.Query(ctx, decisionSelect+`
  JOIN audits AS audit ON audit.audit_id = decision.audit_id
 WHERE audit.owner_id = $1
   AND (decision.audit_id, decision.decision_id) IN (
       SELECT * FROM unnest($2::text[], $3::text[]))`, ownerID, keys.auditIDs, keys.recordIDs)
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

func (s *Service) readFindingAssessments(ctx context.Context, ownerID string, keys auditRecordKeys) (map[string]FindingAssessment, error) {
	result := make(map[string]FindingAssessment, len(keys.recordIDs))
	if len(keys.recordIDs) == 0 {
		return result, nil
	}
	rows, err := s.pool.Query(ctx, assessmentSelect+`
  JOIN audits AS audit ON audit.audit_id = assessment.audit_id
 WHERE audit.owner_id = $1
   AND (assessment.audit_id, assessment.assessment_id) IN (
       SELECT * FROM unnest($2::text[], $3::text[]))`, ownerID, keys.auditIDs, keys.recordIDs)
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
