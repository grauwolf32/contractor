package findingintake

import (
	"context"
	"errors"
	"fmt"

	"github.com/jackc/pgx/v5"
)

// CollectionReceipt carries the exact source receipt and the destination
// Audit's durable work state. An already held receipt needs no proposal
// document for collection; a direct assessment may still be pending.
type CollectionReceipt struct {
	Receipt              Receipt
	Retained             bool
	PostTerminalRetained bool
	DirectAssessed       bool
	Rejected             bool
	// ReadFailure is a safe deterministic rejection reason for this proposal.
	// Transient storage failures fail the page and are never converted to it.
	ReadFailure string
}

// NeedsRetention reports whether collection must still pass this proposal to
// RetainAuditCollectionBatch. A rejected proposal stays out of the Audit. A
// retained one needs another pass only while its direct verification can
// still be accepted: the source Run succeeded, no direct assessment exists,
// and the hold was created before the Run finished.
func (c CollectionReceipt) NeedsRetention(sourceSucceeded bool) bool {
	if c.Rejected {
		return false
	}
	return !c.Retained || (sourceSucceeded && !c.PostTerminalRetained && !c.DirectAssessed)
}

// ListAuditCollection pages every proposal of a child Run but hydrates only
// proposals that this Audit has not already retained or rejected. A retry over
// completed receipts therefore needs two set-based queries per page and no
// proposal artifact reads.
func (s *Service) ListAuditCollection(
	ctx context.Context, ownerID, auditID, runID string, query ListQuery,
) ([]CollectionReceipt, error) {
	if auditID == "" {
		return nil, ErrInvalid
	}
	receipts, err := s.listRunReceiptRows(ctx, ownerID, runID, query)
	if err != nil || len(receipts) == 0 {
		return nil, err
	}
	ids := make([]string, len(receipts))
	for i := range receipts {
		ids[i] = receipts[i].ReceiptID
	}
	rows, err := s.pool.Query(ctx, listAuditCollectionSQL, ids, auditID, ownerID, runID)
	if err != nil {
		return nil, fmt.Errorf("read Audit collection proposal state: %w", err)
	}
	states, err := scanCollectionStates(rows)
	if err != nil {
		return nil, err
	}
	if len(states) != len(receipts) {
		return nil, ErrNotFound
	}
	result := make([]CollectionReceipt, len(receipts))
	pending := make([]Receipt, 0, len(receipts))
	positions := make([]int, 0, len(receipts))
	for i, receipt := range receipts {
		state, ok := states[receipt.ReceiptID]
		if !ok {
			return nil, ErrNotFound
		}
		result[i] = CollectionReceipt{Receipt: receipt, Retained: state.retained,
			PostTerminalRetained: state.postTerminalRetained,
			DirectAssessed:       state.directAssessed, Rejected: state.rejected}
		if !state.retained && !state.rejected {
			pending = append(pending, receipt)
			positions = append(positions, i)
		}
	}
	if err := s.hydrateCollectionPage(ctx, pending, positions, result); err != nil {
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

type collectionState struct{ retained, postTerminalRetained, directAssessed, rejected bool }

func scanCollectionStates(rows pgx.Rows) (map[string]collectionState, error) {
	defer rows.Close()
	result := map[string]collectionState{}
	for rows.Next() {
		var id string
		var state collectionState
		if err := rows.Scan(&id, &state.retained, &state.postTerminalRetained, &state.directAssessed, &state.rejected); err != nil {
			return nil, fmt.Errorf("scan Audit collection proposal state: %w", err)
		}
		result[id] = state
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("iterate Audit collection proposal state: %w", err)
	}
	return result, nil
}
