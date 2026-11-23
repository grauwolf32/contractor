package pgstore

import (
	"context"
	"errors"
	"sort"

	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
)

// CollectionReceiptIDs resolves findings of an owned Audit or Project, each at the
// exact revision the caller read, to every receipt that contributed to them
// or first proposed them, sorted. It reads them in one statement that
// returns at most one row per finding and auditdomain.MaximumCollectionEntries
// + 1 receipts (LIM-FND-25, FND-R159). The first finding in input order that
// is missing is ErrNotFound, or at another revision ErrConflict; only then
// are more than auditdomain.MaximumCollectionEntries receipts, or more
// findings than that, ErrInvalid. No page or first proposal shortcut drops a
// supporting receipt.
func (s *QueryStore) CollectionReceiptIDs(
	ctx context.Context, ownerID, sourceID string, findings []auditstore.FindingRevision, sourceKind string,
) ([]string, error) {
	if sourceKind != "project" && sourceKind != "audit" {
		return nil, auditstore.ErrInvalid
	}
	if len(findings) == 0 {
		return []string{}, nil
	}
	if len(findings) > auditdomain.MaximumCollectionEntries {
		return nil, auditstore.ErrInvalid
	}
	findingIDs := make([]string, len(findings))
	for index, finding := range findings {
		findingIDs[index] = finding.FindingID
	}
	rows, err := s.db.Query(ctx, readCollectionReceiptIDsSQL,
		ownerID, sourceID, findingIDs, auditdomain.MaximumCollectionEntries+1, sourceKind)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	revisions := make([]*uint64, len(findings))
	receipts := make([]string, 0, len(findings))
	for rows.Next() {
		var kind int
		var ordinal *int64
		var revision *uint64
		var receiptID *string
		if err := rows.Scan(&kind, &ordinal, &revision, &receiptID); err != nil {
			return nil, err
		}
		switch {
		case kind == 0 && ordinal != nil && *ordinal >= 1 && *ordinal <= int64(len(findings)):
			revisions[*ordinal-1] = revision
		case kind == 1 && receiptID != nil:
			receipts = append(receipts, *receiptID)
		default:
			return nil, errors.New("read Audit collection receipts: unexpected row")
		}
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	for index, finding := range findings {
		if revisions[index] == nil {
			return nil, auditstore.ErrNotFound
		}
		if *revisions[index] != finding.Revision {
			return nil, auditstore.ErrConflict
		}
	}
	if len(receipts) > auditdomain.MaximumCollectionEntries {
		return nil, auditstore.ErrInvalid
	}
	sort.Strings(receipts)
	return receipts, nil
}

// CollectionFindings reads, for each receipt, the findings of the owner's
// Audits it contributed to or first proposed, with each finding's revision,
// state and current decision, assessment and duplicate target. It preserves
// missing decision IDs and creates nothing. More than
// auditdomain.MaximumCollectionFindings findings for one receipt is
// ErrInvalid.
func (s *QueryStore) CollectionFindings(
	ctx context.Context, ownerID string, receiptIDs []string,
) (map[string]*auditdomain.FindingCollectionFinding, error) {
	result := map[string]*auditdomain.FindingCollectionFinding{}
	if len(receiptIDs) == 0 {
		return result, nil
	}
	rows, err := s.db.Query(ctx, readCollectionFindingsSQL,
		ownerID, receiptIDs, auditdomain.MaximumCollectionEntries+1)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	for rows.Next() {
		var receiptID string
		var review auditdomain.FindingCollectionFinding
		if err := rows.Scan(&receiptID, &review.ProjectID, &review.FindingID, &review.Revision, &review.State, &review.Verdict, &review.Severity, &review.DuplicateTargetID); err != nil {
			return nil, err
		}
		if result[receiptID] != nil {
			return nil, auditstore.ErrInvalid
		}
		result[receiptID] = &review
	}
	return result, rows.Err()
}
