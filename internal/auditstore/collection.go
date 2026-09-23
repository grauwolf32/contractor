package auditstore

import (
	"context"
	"errors"
	"fmt"

	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/jackc/pgx/v5"
)

func (s *PostgresStore) Collect(
	ctx context.Context,
	params CollectParams,
) (CollectionReceipt, bool, error) {
	if err := validateCollect(params); err != nil {
		return CollectionReceipt{}, false, err
	}
	if replay, found, err := s.lookupReceiptReplay(
		ctx, params.Claim.AuditID, params.ExecutionID, params.RequestDigest,
	); err != nil || found {
		return replay, false, err
	}
	prepared := prepareCollectionWrite(params)
	receipt, err := scanReceipt(s.db.QueryRow(ctx, collectAuditExecutionSQL,
		params.Claim.AuditID, params.Claim.HolderID, params.Claim.Epoch,
		params.ExecutionID, params.ReceiptID, string(params.Disposition),
		prepared.sourceRef, prepared.sourceDigest, prepared.retained, prepared.items, prepared.links,
		prepared.retainedBytes, params.ErrorCode, params.RequestDigest,
	))
	if err == nil {
		return receipt, true, nil
	}
	sqlState := persistencepostgres.SQLState(err)
	if sqlState == "23505" || errors.Is(err, pgx.ErrNoRows) {
		if existing, found, replayErr := s.lookupReceiptReplay(
			ctx, params.Claim.AuditID, params.ExecutionID, params.RequestDigest,
		); replayErr != nil || found {
			return existing, false, replayErr
		}
		if live, liveErr := s.claimLive(ctx, params.Claim); liveErr != nil {
			return CollectionReceipt{}, false, liveErr
		} else if !live {
			return CollectionReceipt{}, false, ErrClaimLost
		}
		if sqlState == "23505" {
			return CollectionReceipt{}, false, ErrConflict
		}
		if exceeded, budgetErr := s.evidenceBudgetExceeded(
			ctx, params.Claim.AuditID, prepared.retainedBytes,
		); budgetErr != nil {
			return CollectionReceipt{}, false, budgetErr
		} else if exceeded {
			return CollectionReceipt{}, false, ErrEvidenceBudgetExhausted
		}
		return CollectionReceipt{}, false, ErrPrecondition
	}
	return CollectionReceipt{}, false, fmt.Errorf("collect Audit execution: %w", err)
}

// evidenceBudgetExceeded names the advanced_audit gate that rejected a
// collection. Retained bytes never shrink while the Audit exists, so a
// request over the budget now can never commit and must not be retried.
func (s *PostgresStore) evidenceBudgetExceeded(
	ctx context.Context, auditID string, retainedBytes int64,
) (bool, error) {
	if retainedBytes <= 0 {
		return false, nil
	}
	var exceeded bool
	err := s.db.QueryRow(ctx, `
SELECT retained_evidence_bytes + $2 > max_evidence_bytes
  FROM audits
 WHERE audit_id = $1`, auditID, retainedBytes).Scan(&exceeded)
	if err != nil {
		return false, fmt.Errorf("read Audit evidence budget: %w", err)
	}
	return exceeded, nil
}

func prefixedReceiptColumns(prefix string) string {
	return prefix + ".receipt_id, " + prefix + ".audit_id, " + prefix + ".execution_id, " +
		prefix + ".run_id, " + prefix + ".terminal_outcome, " + prefix + ".terminal_run_generation, " +
		prefix + ".terminal_run_sequence, " + prefix + ".disposition, " + prefix + ".source_output_ref, " +
		prefix + ".source_output_digest, " + prefix + ".retained_refs, " + prefix + ".error_code, " +
		prefix + ".request_digest, " + prefix + ".created_at"
}

func (s *PostgresStore) lookupReceiptReplay(
	ctx context.Context, auditID, executionID, digest string,
) (CollectionReceipt, bool, error) {
	receipt, err := scanReceipt(s.db.QueryRow(ctx, `
SELECT `+receiptColumns+` FROM audit_collection_receipts
 WHERE execution_id = $1`, executionID))
	if errors.Is(err, pgx.ErrNoRows) {
		return CollectionReceipt{}, false, nil
	}
	if err != nil {
		return CollectionReceipt{}, false, fmt.Errorf("resolve Audit collection replay: %w", err)
	}
	if receipt.AuditID != auditID || receipt.RequestDigest != digest {
		return CollectionReceipt{}, true, ErrConflict
	}
	return receipt, true, nil
}
