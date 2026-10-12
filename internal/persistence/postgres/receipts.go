package postgres

import (
	"context"
	"errors"

	"github.com/jackc/pgx/v5"
)

// A keyed mutation is a request its caller may repeat under an idempotency
// key: an owner's Audit mutation, a managed Eval mutation or a Runtime
// management operation. Each records one receipt in mutation_receipts in the
// transaction of its effect, through KeyedMutation (PER-R23): the key's
// advisory lock first, then the replay of a stored receipt, then the effect
// and the claim of the key with the new receipt.

var (
	// ErrReceiptMismatch reports that the key's receipt records another
	// request; the caller reports its idempotency conflict.
	ErrReceiptMismatch = errors.New("the idempotency key's receipt records another request")
	// ErrReceiptClaimed reports that the key already had a receipt when a
	// mutation claimed it, which only a writer that skipped the key's lock
	// can cause.
	ErrReceiptClaimed = errors.New("the idempotency key already has a receipt")
)

// ReceiptKey names the receipt of one keyed mutation: the scope its
// idempotency key is unique within, led by the mutation family, and the key.
type ReceiptKey struct {
	scope []string
	key   string
}

// AuditReceiptKey is the receipt key of an owner's Audit mutation. A key is
// unique per owner and operation; the receipt leaves with its Audit.
func AuditReceiptKey(ownerID, operation, idempotencyKey string) ReceiptKey {
	return ReceiptKey{scope: []string{"audit", ownerID, operation}, key: idempotencyKey}
}

// EvalReceiptKey is the receipt key of a managed Eval mutation. A key is
// unique per owner, Project, resource and operation; the receipt stays
// replayable after its experiment's purge until its Project's purge
// (LIFE-EVM-7).
func EvalReceiptKey(ownerID, projectID, resourceID, operation, idempotencyKey string) ReceiptKey {
	return ReceiptKey{scope: []string{"eval", ownerID, projectID, resourceID, operation}, key: idempotencyKey}
}

// RuntimeManagementReceiptKey is the receipt key of a Runtime management
// operation: one key space for every operation, holding only the digest of
// the operator's key. The receipt is never deleted.
func RuntimeManagementReceiptKey(keyDigest string) ReceiptKey {
	return ReceiptKey{scope: []string{"runtime-management"}, key: keyDigest}
}

// AdvisoryKey is the advisory key that serializes the requests under k.
func (k ReceiptKey) AdvisoryKey() AdvisoryKey {
	return NewAdvisoryKey("mutation-receipt", append(append([]string{}, k.scope...), k.key)...)
}

// Receipt is the record of one keyed mutation: what it did, to which
// resource, for which request, and the response a replay returns. OwnerID,
// ProjectID and AuditID are empty unless the receipt's family records them:
// an Audit receipt names its owner and Audit, an Eval receipt its owner and
// Project.
type Receipt struct {
	Operation     string
	ResourceID    string
	RequestDigest string
	Response      []byte
	OwnerID       string
	ProjectID     string
	AuditID       string
}

// LockReceipt takes the advisory lock of key until tx ends, so the requests
// under one key run one at a time. KeyedMutation takes it first; a caller
// that must check other state under the key before its mutation takes it
// earlier, and the second take in the same transaction does not wait.
func LockReceipt(ctx context.Context, tx pgx.Tx, key ReceiptKey) error {
	if err := LockAdvisory(ctx, tx, key.AdvisoryKey()); err != nil {
		return WrapError("lock mutation receipt key", err)
	}
	return nil
}

// FindReceipt reads the receipt stored under key. A receipt of another
// request than requestDigest is ErrReceiptMismatch.
func FindReceipt(ctx context.Context, db DBTX, key ReceiptKey, requestDigest string) (Receipt, bool, error) {
	var result Receipt
	var ownerID, projectID, auditID *string
	err := db.QueryRow(ctx, `
SELECT operation, resource_id, request_digest, response, owner_id, project_id, audit_id
  FROM mutation_receipts
 WHERE key_scope = $1 AND idempotency_key = $2`, key.scope, key.key).Scan(
		&result.Operation, &result.ResourceID, &result.RequestDigest, &result.Response,
		&ownerID, &projectID, &auditID,
	)
	if errors.Is(err, pgx.ErrNoRows) {
		return Receipt{}, false, nil
	}
	if err != nil {
		return Receipt{}, false, WrapError("read mutation receipt", err)
	}
	if result.RequestDigest != requestDigest {
		return Receipt{}, true, ErrReceiptMismatch
	}
	result.OwnerID, result.ProjectID, result.AuditID = valueOrEmpty(ownerID), valueOrEmpty(projectID), valueOrEmpty(auditID)
	return result, true, nil
}

// claimReceipt stores receipt under key. A key that already has a receipt is
// ErrReceiptClaimed. Under a REPEATABLE READ snapshot taken before the key's
// lock, a receipt committed after the snapshot is a serialization failure
// instead, which the caller's retry turns into a replay.
func claimReceipt(ctx context.Context, db DBTX, key ReceiptKey, receipt Receipt) error {
	tag, err := db.Exec(ctx, `
INSERT INTO mutation_receipts (
    key_scope, idempotency_key, operation, request_digest, resource_id, response,
    owner_id, project_id, audit_id
) VALUES ($1, $2, $3, $4, $5, $6, NULLIF($7, ''), NULLIF($8, ''), NULLIF($9, ''))
ON CONFLICT DO NOTHING`,
		key.scope, key.key, receipt.Operation, receipt.RequestDigest, receipt.ResourceID, receipt.Response,
		receipt.OwnerID, receipt.ProjectID, receipt.AuditID,
	)
	if err != nil {
		return WrapError("claim mutation receipt", err)
	}
	if tag.RowsAffected() != 1 {
		return ErrReceiptClaimed
	}
	return nil
}

// KeyedMutation runs one keyed mutation in tx: it takes the key's lock,
// replays the receipt stored under key, and otherwise runs mutate and claims
// the key with the receipt mutate returns, completed with requestDigest. It
// reports the receipt and whether it was replayed. A receipt of another
// request is ErrReceiptMismatch; an error of mutate is returned as it is,
// and nothing is claimed.
func KeyedMutation(
	ctx context.Context, tx pgx.Tx, key ReceiptKey, requestDigest string, mutate func() (Receipt, error),
) (Receipt, bool, error) {
	if err := LockReceipt(ctx, tx, key); err != nil {
		return Receipt{}, false, err
	}
	stored, found, err := FindReceipt(ctx, tx, key, requestDigest)
	if err != nil || found {
		return stored, found, err
	}
	receipt, err := mutate()
	if err != nil {
		return Receipt{}, false, err
	}
	receipt.RequestDigest = requestDigest
	if err := claimReceipt(ctx, tx, key, receipt); err != nil {
		return Receipt{}, false, err
	}
	return receipt, false, nil
}

// EvalReceiptProject finds the Project of the owner's Eval receipt for
// operation on resourceID under key, for a request addressed by its resource
// after its experiment's purge (EVM-R226).
func EvalReceiptProject(
	ctx context.Context, db DBTX, ownerID, resourceID, operation, key string,
) (string, bool, error) {
	var projectID string
	err := db.QueryRow(ctx, `
SELECT project_id
  FROM mutation_receipts
 WHERE owner_id = $1 AND resource_id = $2 AND operation = $3 AND idempotency_key = $4
   AND key_scope[1] = 'eval'
 LIMIT 1`, ownerID, resourceID, operation, key).Scan(&projectID)
	if errors.Is(err, pgx.ErrNoRows) {
		return "", false, nil
	}
	if err != nil {
		return "", false, WrapError("read Eval mutation receipt", err)
	}
	return projectID, true, nil
}

// PurgeEvalReceipts deletes the Eval receipts of the owner's Project in the
// Project's purge, which runs with contractor.eval_purge set in tx; no other
// deletion of a receipt is allowed (LIFE-EVM-7).
func PurgeEvalReceipts(ctx context.Context, db DBTX, ownerID, projectID string) error {
	if _, err := db.Exec(ctx, `
DELETE FROM mutation_receipts
 WHERE owner_id = $1 AND project_id = $2 AND key_scope[1] = 'eval'`, ownerID, projectID); err != nil {
		return WrapError("purge Eval mutation receipts", err)
	}
	return nil
}

func valueOrEmpty(value *string) string {
	if value == nil {
		return ""
	}
	return *value
}
