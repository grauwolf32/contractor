package pgstore

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"

	"github.com/grauwolf32/contractor/internal/auditstore"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/jackc/pgx/v5"
)

func (s *ClaimStore) Collect(
	ctx context.Context,
	params auditstore.CollectParams,
) (auditstore.CollectionReceipt, bool, error) {
	var err error
	params, err = withPreparationLinks(params)
	if err != nil {
		return auditstore.CollectionReceipt{}, false, err
	}
	if err := ValidateCollect(params); err != nil {
		return auditstore.CollectionReceipt{}, false, err
	}
	if replay, found, err := s.lookupReceiptReplay(
		ctx, params.Claim.AuditID, params.ExecutionID, params.RequestDigest,
	); err != nil || found {
		return replay, false, err
	}
	prepared := prepareCollectionWrite(params)
	outputs, _ := json.Marshal(append([]auditstore.PreparationOutput{}, params.PreparationOutputs...))
	receipt, err := scanReceipt(persistencepostgres.SavepointQueryRow(ctx, s.db, collectAuditExecutionSQL,
		params.Claim.AuditID, params.Claim.HolderID, params.Claim.Epoch,
		params.ExecutionID, params.ReceiptID, string(params.Disposition),
		prepared.sourceRef, prepared.sourceDigest, prepared.retained, prepared.items, prepared.links,
		prepared.retainedBytes, params.ErrorCode, params.RequestDigest, outputs,
	))
	if err == nil {
		return receipt, true, nil
	}
	failure := persistencepostgres.Classify(err)
	if failure == persistencepostgres.ClassRejected {
		return auditstore.CollectionReceipt{}, false, auditstore.ErrPrecondition
	}
	if failure == persistencepostgres.ClassDuplicate || errors.Is(err, pgx.ErrNoRows) {
		if existing, found, replayErr := s.lookupReceiptReplay(
			ctx, params.Claim.AuditID, params.ExecutionID, params.RequestDigest,
		); replayErr != nil || found {
			return existing, false, replayErr
		}
		if live, liveErr := s.claimLive(ctx, params.Claim); liveErr != nil {
			return auditstore.CollectionReceipt{}, false, liveErr
		} else if !live {
			return auditstore.CollectionReceipt{}, false, auditstore.ErrClaimLost
		}
		if failure == persistencepostgres.ClassDuplicate {
			return auditstore.CollectionReceipt{}, false, auditstore.ErrConflict
		}
		if exceeded, budgetErr := s.evidenceBudgetExceeded(
			ctx, params.Claim.AuditID, prepared.retainedBytes,
		); budgetErr != nil {
			return auditstore.CollectionReceipt{}, false, budgetErr
		} else if exceeded {
			return auditstore.CollectionReceipt{}, false, auditstore.ErrEvidenceBudgetExhausted
		}
		return auditstore.CollectionReceipt{}, false, auditstore.ErrPrecondition
	}
	return auditstore.CollectionReceipt{}, false, fmt.Errorf("collect Audit execution: %w", err)
}

// evidenceBudgetExceeded names the advanced_audit gate that rejected a
// collection. Retained bytes never shrink while the Audit exists, so a
// request over the budget now can never commit and must not be retried.
func (s *ClaimStore) evidenceBudgetExceeded(
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

func (s *ClaimStore) lookupReceiptReplay(
	ctx context.Context, auditID, executionID, digest string,
) (auditstore.CollectionReceipt, bool, error) {
	receipt, err := scanReceipt(s.db.QueryRow(ctx, `
SELECT `+collectionReceiptProjection("execution")+`
  FROM audit_executions AS execution
 WHERE execution_id = $1 AND collection_receipt_id IS NOT NULL`, executionID))
	if errors.Is(err, pgx.ErrNoRows) {
		return auditstore.CollectionReceipt{}, false, nil
	}
	if err != nil {
		return auditstore.CollectionReceipt{}, false, fmt.Errorf("resolve Audit collection replay: %w", err)
	}
	if receipt.AuditID != auditID || receipt.RequestDigest != digest {
		return auditstore.CollectionReceipt{}, true, auditstore.ErrConflict
	}
	return receipt, true, nil
}
