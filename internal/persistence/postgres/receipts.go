package postgres

import (
	"context"
	"errors"

	"github.com/jackc/pgx/v5"
)

// A keyed mutation is a request its caller may repeat under an idempotency
// key: an owner's Audit or Project mutation, a managed Eval mutation or a
// Runtime management operation. Each records one receipt in mutation_receipts in the
// transaction of its effect, through KeyedMutation (PER-R24): the key's
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

// ScanSweepReceiptKey keeps owner-bound creation replayable after sweep purge.
// It retains only the original sweep identity and request digest.
func ScanSweepReceiptKey(ownerID, idempotencyKey string) ReceiptKey {
	return ReceiptKey{scope: []string{"scan-sweep", ownerID, "scan-sweep.create"}, key: idempotencyKey}
}

// ProjectReceiptKey binds a resource operation to its owner and Project. Its
// receipt survives resource changes and leaves only in Project purge.
func ProjectReceiptKey(ownerID, projectID, resourceID, operation, idempotencyKey string) ReceiptKey {
	return ReceiptKey{scope: []string{"project", ownerID, projectID, resourceID, operation}, key: idempotencyKey}
}

// DeleteProjectReceiptsBatch removes one bounded batch under the existing
// Project lifecycle fence. The caller must have locked its drained Project.
// Other receipt families retain their own lifecycle and are untouched.
func DeleteProjectReceiptsBatch(ctx context.Context, tx pgx.Tx, projectID string) (bool, error) {
	if tx == nil || projectID == "" {
		return false, errors.New("Project receipt purge arguments are invalid")
	}
	if _, err := tx.Exec(ctx, `SELECT set_config('contractor.lifecycle_purge', 'project', true)`); err != nil {
		return false, WrapError("authorize Project receipt purge", err)
	}
	budget := CleanupBudget{RowLimit: CleanupBatchRows, ByteLimit: 64 << 20}
	gone, err := DeleteWithinBudget(ctx, tx, &budget, BoundedDelete{Table: "mutation_receipts",
		Filter: `project_id=$5 AND key_scope[1]='project'`, RowBytes: receiptRowBytes, RowCost: 1, Args: []any{projectID}})
	if err != nil {
		return false, WrapError("purge Project mutation receipts", err)
	}
	// Commit receipt work before the controller starts an Artifact purge pass.
	return gone && budget.Rows == 0, nil
}

// AdvisoryKey is the advisory key that serializes the requests under k.
func (k ReceiptKey) AdvisoryKey() AdvisoryKey {
	return NewAdvisoryKey("mutation-receipt", append(append([]string{}, k.scope...), k.key)...)
}

// Receipt is the record of one keyed mutation: what it did, to which
// resource, for which request, and the response a replay returns. OwnerID,
// ProjectID and AuditID are empty unless the receipt's family records them:
// an Audit receipt names its owner and Audit, while Project and Eval receipts
// name their owner and Project.
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
	result, found, err := ReadReceipt(ctx, db, key)
	if err != nil || !found {
		return result, found, err
	}
	if result.RequestDigest != requestDigest {
		return Receipt{}, true, ErrReceiptMismatch
	}
	return result, true, nil
}

// ReadReceipt reads a receipt under its owner-bound key. A projected-request
// caller takes LockReceipt first and uses its saved safe projection policy to
// compute the replay digest, then still passes through KeyedMutation's digest
// comparison before accepting a replay. It never reads the old result payload.
func ReadReceipt(ctx context.Context, db DBTX, key ReceiptKey) (Receipt, bool, error) {
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

// receiptRowBytes is the stored size of one mutation_receipts row.
const receiptRowBytes = `pg_column_size(key_scope) + pg_column_size(idempotency_key) + pg_column_size(operation)
    + pg_column_size(request_digest) + pg_column_size(resource_id) + pg_column_size(response)
    + COALESCE(pg_column_size(owner_id), 0) + COALESCE(pg_column_size(project_id), 0)
    + COALESCE(pg_column_size(audit_id), 0) + pg_column_size(created_at)`

// DeleteEvalReceiptsWithinBudget deletes the next Eval receipts of the
// owner's Project that fit the Project purge pass's budget (EVM-R243). The
// pass runs with contractor.eval_purge set in db; no other deletion of an
// Eval receipt is allowed (LIFE-EVM-7). It reports whether none is left.
func DeleteEvalReceiptsWithinBudget(
	ctx context.Context, db DBTX, budget *CleanupBudget, ownerID, projectID string,
) (bool, error) {
	gone, err := DeleteWithinBudget(ctx, db, budget, BoundedDelete{
		Table: "mutation_receipts", Filter: `owner_id = $5 AND project_id = $6 AND key_scope[1] = 'eval'`,
		RowBytes: receiptRowBytes, RowCost: 1, Args: []any{ownerID, projectID},
	})
	if err != nil {
		return false, WrapError("purge Eval mutation receipts", err)
	}
	return gone, nil
}

func valueOrEmpty(value *string) string {
	if value == nil {
		return ""
	}
	return *value
}

// DeleteAuditReceiptsBatch removes at most limit receipts after the caller
// has locked a drained deleting Audit. The one creation receipt stays until
// the Audit DELETE, whose Eval tombstone trigger reads it. Other mutation
// families are untouched.
func DeleteAuditReceiptsBatch(ctx context.Context, tx pgx.Tx, auditID string, limit int) (int, error) {
	if tx == nil || auditID == "" || limit < 1 || limit > CleanupBatchRows {
		return 0, errors.New("Audit receipt purge arguments are invalid")
	}
	tag, err := tx.Exec(ctx, `WITH batch AS (
  SELECT ctid FROM mutation_receipts WHERE audit_id = $1 AND operation <> 'audit.create' LIMIT $2
)
DELETE FROM mutation_receipts AS receipt USING batch WHERE receipt.ctid = batch.ctid`, auditID, limit)
	if err != nil {
		return 0, WrapError("purge Audit mutation receipt batch", err)
	}
	return int(tag.RowsAffected()), nil
}
