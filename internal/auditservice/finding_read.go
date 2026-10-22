package auditservice

import (
	"context"

	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/findingintake"
	"github.com/jackc/pgx/v5"
)

func (s *Service) ListFindings(
	ctx context.Context, params auditstore.FindingListParams,
) ([]Finding, error) {
	rows, err := s.reads().ListFindings(ctx, params)
	if err != nil {
		return nil, err
	}
	return s.hydrateFindings(ctx, params.OwnerID, rows)
}

func (s *Service) GetFinding(
	ctx context.Context, ownerID, auditID, findingID string,
) (Finding, error) {
	row, err := s.reads().GetFinding(ctx, ownerID, auditID, findingID)
	if err != nil {
		return Finding{}, err
	}
	return s.hydrateFinding(ctx, ownerID, row)
}

func (s *Service) hydrateFinding(ctx context.Context, ownerID string, row auditstore.FindingRecord) (Finding, error) {
	values, err := s.hydrateFindings(ctx, ownerID, []auditstore.FindingRecord{row})
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
func (s *Service) hydrateFindings(ctx context.Context, ownerID string, rows []auditstore.FindingRecord) ([]Finding, error) {
	result := make([]Finding, len(rows))
	if len(rows) == 0 {
		return result, nil
	}
	receiptKeys := make([]findingintake.AuditReceiptKey, len(rows))
	var decisions, assessments auditRecordKeys
	for index, row := range rows {
		receiptKeys[index] = findingintake.AuditReceiptKey{AuditID: row.AuditID, ReceiptID: row.FirstReceiptID}
		if row.CurrentDecisionID != nil && !decisions.add(row.AuditID, *row.CurrentDecisionID) {
			return nil, pgx.ErrNoRows
		}
		if row.CurrentAssessmentID != nil && !assessments.add(row.AuditID, *row.CurrentAssessmentID) {
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
	reads := s.reads()
	decisionsByID, err := reads.ReadFindingDecisions(ctx, ownerID, decisions.auditIDs, decisions.recordIDs)
	if err != nil {
		return nil, err
	}
	assessmentsByID, err := reads.ReadFindingAssessments(ctx, ownerID, assessments.auditIDs, assessments.recordIDs)
	if err != nil {
		return nil, err
	}
	for index, row := range rows {
		finding := Finding{
			Fingerprint: &receipts[index].Fingerprint,
			Anchors:     &receipts[index].Anchors,
			FindingID:   row.FindingID, AuditID: row.AuditID, State: row.State,
			RejectionReason: row.RejectionReason, DuplicateTargetID: row.DuplicateTargetID,
			FirstProposal: receipts[index], Revision: row.Revision,
			CreatedAt: row.CreatedAt, UpdatedAt: row.UpdatedAt,
		}
		if row.CurrentDecisionID != nil {
			decision, ok := decisionsByID[*row.CurrentDecisionID]
			if !ok {
				return nil, pgx.ErrNoRows
			}
			finding.AnalystDecision = &decision
			finding.AnalystVerdict = &decision.Verdict
			finding.AnalystSeverity = decision.Severity
		}
		if row.CurrentAssessmentID != nil {
			assessment, ok := assessmentsByID[*row.CurrentAssessmentID]
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
