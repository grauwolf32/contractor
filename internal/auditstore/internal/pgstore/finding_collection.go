package pgstore

import (
	"context"
	"errors"
	"sort"

	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/jackc/pgx/v5"
)

// CollectionReceiptIDs resolves findings of an owner's Audit, each at the
// exact revision the caller read, to every receipt that contributed to them
// or first proposed them, sorted. A missing finding is ErrNotFound, a finding
// at another revision ErrConflict, and more than
// auditdomain.MaximumCollectionEntries receipts ErrInvalid. No page or first
// proposal shortcut drops a supporting receipt.
func (s *QueryStore) CollectionReceiptIDs(
	ctx context.Context, ownerID, auditID string, findings []auditstore.FindingRevision,
) ([]string, error) {
	ids := map[string]bool{}
	for _, finding := range findings {
		var revision uint64
		var first string
		err := s.db.QueryRow(ctx, `
SELECT f.revision, f.first_receipt_id FROM audit_findings AS f JOIN audits AS a USING(audit_id)
 WHERE a.owner_id=$1 AND a.audit_id=$2 AND f.finding_id=$3`, ownerID, auditID, finding.FindingID).Scan(&revision, &first)
		if errors.Is(err, pgx.ErrNoRows) {
			return nil, auditstore.ErrNotFound
		}
		if err != nil {
			return nil, err
		}
		if revision != finding.Revision {
			return nil, auditstore.ErrConflict
		}
		ids[first] = true
		if err := s.addContributions(ctx, auditID, finding.FindingID, ids); err != nil {
			return nil, err
		}
	}
	result := make([]string, 0, len(ids))
	for id := range ids {
		result = append(result, id)
	}
	sort.Strings(result)
	return result, nil
}

func (s *QueryStore) addContributions(ctx context.Context, auditID, findingID string, ids map[string]bool) error {
	rows, err := s.db.Query(ctx, `SELECT receipt_id FROM audit_finding_contributions WHERE audit_id=$1 AND finding_id=$2 ORDER BY receipt_id LIMIT $3`,
		auditID, findingID, auditdomain.MaximumCollectionEntries+1)
	if err != nil {
		return err
	}
	defer rows.Close()
	for rows.Next() {
		var id string
		if err := rows.Scan(&id); err != nil {
			return err
		}
		ids[id] = true
		if len(ids) > auditdomain.MaximumCollectionEntries {
			return auditstore.ErrInvalid
		}
	}
	return rows.Err()
}

// CollectionReviews reads, for each receipt, the findings of the owner's
// Audits it contributed to or first proposed, with each finding's revision,
// state and current decision, assessment and duplicate target. It preserves
// missing decision IDs and creates nothing. More than
// auditdomain.MaximumCollectionReviews findings for one receipt is
// ErrInvalid.
func (s *QueryStore) CollectionReviews(
	ctx context.Context, ownerID string, receiptIDs []string,
) (map[string][]auditdomain.FindingCollectionReview, error) {
	result := map[string][]auditdomain.FindingCollectionReview{}
	if len(receiptIDs) == 0 {
		return result, nil
	}
	rows, err := s.db.Query(ctx, readCollectionReviewsSQL,
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
			return nil, auditstore.ErrInvalid
		}
		result[receiptID] = append(result[receiptID], review)
	}
	return result, rows.Err()
}
