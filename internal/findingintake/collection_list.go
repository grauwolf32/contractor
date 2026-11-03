package findingintake

import (
	"context"
	"errors"
)

// CollectionReceipt carries the exact source receipt and the destination
// Audit's durable work state. An already held receipt needs no proposal
// document for collection; a direct assessment may still be pending.
type CollectionReceipt struct {
	Receipt Receipt
	// A safe permanent failure of this proposal's exact bytes, never storage ambiguity.
	ReadFailure string
}

// ListAuditCollection pages only unretained, unrejected proposals of a child
// Run. Existing holds/rejection events are the durable progress mechanism;
// collection processes one page per controller step without replaying history.
// The source Run quota bounds anti-join probes to at most 4,096 receipts.
func (s *Service) ListAuditCollection(
	ctx context.Context, ownerID, auditID, runID string, query ListQuery,
) ([]CollectionReceipt, error) {
	if auditID == "" {
		return nil, ErrInvalid
	}
	if ownerID == "" || runID == "" || !validListQuery(query) {
		return nil, ErrInvalid
	}
	rows, err := s.pool.Query(ctx, listPendingAuditReceiptsSQL, ownerID, auditID, runID, query.AfterCreatedAt, query.AfterReceiptID, query.Limit)
	if err != nil {
		return nil, err
	}
	receipts, err := scanReceiptRows(rows)
	if err != nil || len(receipts) == 0 {
		return nil, err
	}
	result := make([]CollectionReceipt, len(receipts))
	positions := make([]int, len(receipts))
	for index, receipt := range receipts {
		result[index].Receipt = receipt
		positions[index] = index
	}
	if err := s.hydrateCollectionPage(ctx, receipts, positions, result); err != nil {
		return nil, err
	}
	return result, nil
}

// hydrateCollectionPage keeps the healthy path batched. Only deterministic
// exact-data failures split a batch until the offending proposal is isolated;
// storage and context failures remain retryable errors for the whole page.
func (s *Service) hydrateCollectionPage(
	ctx context.Context, receipts []Receipt, positions []int, result []CollectionReceipt,
) error {
	if len(receipts) == 0 {
		return nil
	}
	hydrated, err := s.hydrateAuditReceiptBatch(ctx, receipts)
	if err == nil {
		for i, position := range positions {
			result[position].Receipt = hydrated[i]
		}
		return nil
	}
	var unretainable *unretainableProposal
	if !errors.As(classifyProposalData(err), &unretainable) {
		return err
	}
	if len(receipts) == 1 {
		result[positions[0]].ReadFailure = unretainable.reason
		return nil
	}
	middle := len(receipts) / 2
	if err := s.hydrateCollectionPage(ctx, receipts[:middle], positions[:middle], result); err != nil {
		return err
	}
	return s.hydrateCollectionPage(ctx, receipts[middle:], positions[middle:], result)
}
