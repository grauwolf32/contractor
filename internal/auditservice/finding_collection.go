package auditservice

import (
	"context"
	"errors"

	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/auditstore/querystore"
	"github.com/grauwolf32/contractor/internal/findingintake"
	"github.com/jackc/pgx/v5"
)

// SelectCollectionFindings resolves explicit finding revisions to every source
// contribution in the publisher's snapshot. No page or FirstProposal shortcut
// may silently drop supporting receipts.
func (s *Service) SelectCollectionFindings(ctx context.Context, tx pgx.Tx, ownerID, auditID string, selected []findingintake.CollectionFindingSelection) ([]string, error) {
	findings := make([]auditstore.FindingRevision, len(selected))
	for index, selection := range selected {
		findings[index] = auditstore.FindingRevision{FindingID: selection.FindingID, Revision: selection.Revision}
	}
	ids, err := querystore.New(tx).CollectionReceiptIDs(ctx, ownerID, auditID, findings)
	return ids, collectionError(err)
}

// ReadCollectionReviews captures only existing owner-authorized observations.
// It deliberately preserves missing decision IDs and does not create findings
// for unadmitted receipts or convert receipt retention into a review state.
func (s *Service) ReadCollectionReviews(ctx context.Context, tx pgx.Tx, ownerID string, receiptIDs []string) (map[string][]auditdomain.FindingCollectionReview, error) {
	reviews, err := querystore.New(tx).CollectionReviews(ctx, ownerID, receiptIDs)
	return reviews, collectionError(err)
}

// collectionError reports a store refusal as the finding intake error the
// collection publisher classifies.
func collectionError(err error) error {
	switch {
	case errors.Is(err, auditstore.ErrNotFound):
		return findingintake.ErrNotFound
	case errors.Is(err, auditstore.ErrConflict):
		return findingintake.ErrConflict
	case errors.Is(err, auditstore.ErrInvalid):
		return findingintake.ErrInvalid
	}
	return err
}
