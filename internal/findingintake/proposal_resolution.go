package findingintake

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"

	"github.com/grauwolf32/contractor/internal/artifacts/blob"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/jackc/pgx/v5"
)

// ResolveAuditProposals converts Runtime-injected invocation-local keys into
// exact receipts already retained by the same Audit execution. It rejects
// ambiguous, foreign, missing, or repeated keys and never falls back to a
// Run-wide client-key guess.
//
// At most auditdomain.MaximumProposalsPerItem keys (LIM-AWC-4) are resolved
// in a number of statements that does not grow with the keys beyond the
// exact document batches (AUX-R98): an invalid identity (ErrInvalid) or a
// repeated key (ErrConflict) fails the call before any read; one statement
// then resolves every key with the Audit's hold of its receipt, and the first
// key in input order that matches no retained receipt (ErrNotFound), several
// (ErrConflict), or a hold whose proposal reference is not exact
// (ErrArtifactIntegrity) fails the call before any document is read. The
// proposal documents are then read in exact batches of at most
// artifacts.MaxExactReadBatchSize references and artifacts.MaxPayloadSize
// bytes (LIM-ART-8, LIM-ART-4), from the source Run while it exists and from
// this Audit's retained copy after its deletion.
func (s *Service) ResolveAuditProposals(
	ctx context.Context,
	ownerID, auditID, executionID, runID string,
	keys []ProposalKey,
) ([]ResolvedProposal, error) {
	if ownerID == "" || auditID == "" || executionID == "" || runID == "" ||
		len(keys) > auditdomain.MaximumProposalsPerItem {
		return nil, ErrInvalid
	}
	invocationIDs, clientKeys := make([]string, len(keys)), make([]string, len(keys))
	seen := make(map[ProposalKey]struct{}, len(keys))
	for index, key := range keys {
		if !validIdentity(key.InvocationID) || !validIdentity(key.ClientKey) {
			return nil, ErrInvalid
		}
		if _, duplicate := seen[key]; duplicate {
			return nil, ErrConflict
		}
		seen[key] = struct{}{}
		invocationIDs[index], clientKeys[index] = key.InvocationID, key.ClientKey
	}
	if len(keys) == 0 {
		return []ResolvedProposal{}, nil
	}
	rows, err := s.pool.Query(ctx, resolveAuditProposalsSQL,
		ownerID, auditID, executionID, runID, invocationIDs, clientKeys)
	if err != nil {
		return nil, fmt.Errorf("resolve Audit proposal selections: %w", err)
	}
	matches, err := scanResolvedProposals(rows, len(keys))
	if err != nil {
		return nil, err
	}
	receipts := make([]Receipt, len(keys))
	for index, matched := range matches {
		switch {
		case len(matched) == 0:
			return nil, ErrNotFound
		case len(matched) > 1:
			return nil, ErrConflict
		case matched[0].heldProposalInvalid:
			return nil, blob.ErrIntegrity
		}
		receipts[index] = matched[0].receipt
	}
	hydrated, err := s.hydrateAuditReceiptBatch(ctx, receipts)
	if err != nil {
		return nil, err
	}
	result := make([]ResolvedProposal, len(hydrated))
	for index, receipt := range hydrated {
		result[index] = ResolvedProposal{
			ReceiptID: receipt.ReceiptID, Proposal: receipt.AuditHolds[0].Proposal,
			Origin: receipt.Origin, Document: receipt.Document,
		}
	}
	return result, nil
}

// resolvedMatch is one receipt a selection matched, with the resolving
// Audit's hold as its only hold.
type resolvedMatch struct {
	receipt             Receipt
	heldProposalInvalid bool
}

// scanResolvedProposals groups the rows of resolveAuditProposalsSQL by the
// 1-based ordinal of the selection they match.
func scanResolvedProposals(rows pgx.Rows, size int) ([][]resolvedMatch, error) {
	defer rows.Close()
	result := make([][]resolvedMatch, size)
	for rows.Next() {
		var ordinal int64
		var hold AuditHold
		var heldProposal, heldEvidence []byte
		receipt, err := scanReceiptRow(trailingColumns{row: rows, dest: []any{
			&ordinal, &hold.AuditID, &hold.ProjectID, &heldProposal, &heldEvidence, &hold.CreatedAt,
		}})
		if err != nil {
			return nil, err
		}
		if ordinal < 1 || ordinal > int64(size) {
			return nil, fmt.Errorf("resolve Audit proposal selections: ordinal %d of %d", ordinal, size)
		}
		if json.Unmarshal(heldEvidence, &hold.Evidence) != nil {
			return nil, errors.New("decode finding proposal Audit hold")
		}
		invalid := json.Unmarshal(heldProposal, &hold.Proposal) != nil || hold.Proposal.Ref.ValidateExact() != nil
		receipt.AuditHolds = []AuditHold{hold}
		result[ordinal-1] = append(result[ordinal-1], resolvedMatch{receipt: receipt, heldProposalInvalid: invalid})
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("iterate Audit proposal selections: %w", err)
	}
	return result, nil
}

// trailingColumns scans a receipt row followed by the columns dest names.
type trailingColumns struct {
	row  pgx.Row
	dest []any
}

func (r trailingColumns) Scan(dest ...any) error {
	return r.row.Scan(append(dest, r.dest...)...)
}
