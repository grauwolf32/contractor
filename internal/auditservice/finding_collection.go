package auditservice

import (
	"context"
	"errors"
	"sort"

	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/findingintake"
	"github.com/jackc/pgx/v5"
)

// SelectCollectionFindings resolves explicit finding revisions to every source
// contribution in the publisher's snapshot. No page or FirstProposal shortcut
// may silently drop supporting receipts.
func (s *Service) SelectCollectionFindings(ctx context.Context, tx pgx.Tx, ownerID, auditID string, selected []findingintake.CollectionFindingSelection) ([]string, error) {
	ids := map[string]bool{}
	for _, selection := range selected {
		var revision uint64
		var first string
		err := tx.QueryRow(ctx, `
SELECT f.revision, f.first_receipt_id FROM audit_findings AS f JOIN audits AS a USING(audit_id)
 WHERE a.owner_id=$1 AND a.audit_id=$2 AND f.finding_id=$3`, ownerID, auditID, selection.FindingID).Scan(&revision, &first)
		if errors.Is(err, pgx.ErrNoRows) {
			return nil, findingintake.ErrNotFound
		}
		if err != nil {
			return nil, err
		}
		if revision != selection.Revision {
			return nil, findingintake.ErrConflict
		}
		ids[first] = true
		rows, err := tx.Query(ctx, `SELECT receipt_id FROM audit_finding_contributions WHERE audit_id=$1 AND finding_id=$2 ORDER BY receipt_id LIMIT $3`, auditID, selection.FindingID, auditdomain.MaximumCollectionEntries+1)
		if err != nil {
			return nil, err
		}
		for rows.Next() {
			var id string
			if err := rows.Scan(&id); err != nil {
				rows.Close()
				return nil, err
			}
			ids[id] = true
			if len(ids) > auditdomain.MaximumCollectionEntries {
				rows.Close()
				return nil, findingintake.ErrInvalid
			}
		}
		if err := rows.Err(); err != nil {
			rows.Close()
			return nil, err
		}
		rows.Close()
	}
	result := make([]string, 0, len(ids))
	for id := range ids {
		result = append(result, id)
	}
	sort.Strings(result)
	return result, nil
}

// ReadCollectionReviews captures only existing owner-authorized observations.
// It deliberately preserves missing decision IDs and does not create findings
// for unadmitted receipts or convert receipt retention into a review state.
func (s *Service) ReadCollectionReviews(ctx context.Context, tx pgx.Tx, ownerID string, receiptIDs []string) (map[string][]auditdomain.FindingCollectionReview, error) {
	result := map[string][]auditdomain.FindingCollectionReview{}
	if len(receiptIDs) == 0 {
		return result, nil
	}
	rows, err := tx.Query(ctx, `
WITH membership AS (
    SELECT c.receipt_id, c.finding_id, c.audit_id FROM audit_finding_contributions AS c WHERE c.receipt_id=ANY($2::text[])
    UNION
    SELECT f.first_receipt_id, f.finding_id, f.audit_id FROM audit_findings AS f WHERE f.first_receipt_id=ANY($2::text[])
)
SELECT m.receipt_id, f.audit_id, f.finding_id, f.revision, f.state,
       COALESCE(f.current_decision_id,''), COALESCE(f.current_assessment_id,''), COALESCE(f.duplicate_target_id,'')
  FROM membership AS m JOIN audit_findings AS f ON f.audit_id=m.audit_id AND f.finding_id=m.finding_id
  JOIN audits AS a ON a.audit_id=f.audit_id
 WHERE a.owner_id=$1 ORDER BY m.receipt_id, f.audit_id, f.finding_id LIMIT $3`,
		ownerID, receiptIDs, auditdomain.MaximumCollectionEntries*auditdomain.MaximumCollectionReviews+1)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	for rows.Next() {
		var receiptID string
		var review auditdomain.FindingCollectionReview
		if err := rows.Scan(&receiptID, &review.AuditID, &review.FindingID, &review.Revision, &review.State, &review.DecisionID, &review.AssessmentID, &review.DuplicateTargetID); err != nil {
			return nil, err
		}
		if len(result[receiptID]) >= auditdomain.MaximumCollectionReviews {
			return nil, findingintake.ErrInvalid
		}
		result[receiptID] = append(result[receiptID], review)
	}
	return result, rows.Err()
}
