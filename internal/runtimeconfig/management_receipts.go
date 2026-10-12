package runtimeconfig

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"time"

	"github.com/grauwolf32/contractor/internal/contracts"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/strictjson"
	"github.com/jackc/pgx/v5"
)

// ManagementReceipt identifies one idempotent Runtime management mutation in
// runtime_management_operations: the digest of the caller's idempotency key,
// the digest of its normalized request, the operation kind and the resource
// it changes. Label bindings and Runtime Agent principals record their
// mutations through the functions below, so the lock, replay and record
// rules exist once.
type ManagementReceipt struct {
	KeyDigest     string
	RequestDigest string
	Kind          string
	ResourceID    string
}

// ReadManagementReplay decodes into result the outcome recorded for the
// receipt's key and reports whether one exists; it takes no lock. The key
// bound to another request, operation or resource is the permanent
// contracts.ErrIdempotencyConflict.
func ReadManagementReplay(
	ctx context.Context, db persistencepostgres.DBTX, receipt ManagementReceipt, result any,
) (bool, error) {
	var storedRequest, storedKind, storedResource, stored string
	err := db.QueryRow(ctx, `
SELECT request_digest, operation_kind, resource_id, result::text
FROM runtime_management_operations
WHERE idempotency_key_digest = $1`, receipt.KeyDigest).Scan(
		&storedRequest, &storedKind, &storedResource, &stored,
	)
	if errors.Is(err, pgx.ErrNoRows) {
		return false, nil
	}
	if err != nil {
		return false, errors.New("get Runtime management replay")
	}
	if storedRequest != receipt.RequestDigest || storedKind != receipt.Kind || storedResource != receipt.ResourceID {
		return false, fmt.Errorf("%w: Runtime management key", contracts.ErrIdempotencyConflict)
	}
	if err := strictjson.Decode([]byte(stored), result); err != nil {
		return false, errors.New("invalid Runtime management replay")
	}
	return true, nil
}

// LockManagementReplay takes the key's transaction-scoped advisory lock and
// then reads the key like ReadManagementReplay, so the replay lookup and the
// mutation that follows are one linearizable operation even before the
// receipt row exists.
func LockManagementReplay(ctx context.Context, tx pgx.Tx, receipt ManagementReceipt, result any) (bool, error) {
	if _, err := tx.Exec(ctx, `SELECT pg_advisory_xact_lock(hashtextextended($1, 0))`, receipt.KeyDigest); err != nil {
		return false, errors.New("lock Runtime management idempotency key")
	}
	return ReadManagementReplay(ctx, tx, receipt, result)
}

// RecordManagementResult stores result as the receipt's outcome in the
// mutation's own transaction. A receipt already recorded for the key is
// ErrConflict; rows are immutable once written.
func RecordManagementResult(
	ctx context.Context, tx pgx.Tx, receipt ManagementReceipt, result any, actor string, at time.Time,
) error {
	if !ValidActor(actor) || at.IsZero() {
		return invalid("Runtime management audit is invalid")
	}
	encoded, err := json.Marshal(result)
	if err != nil {
		return invalid("Runtime management result cannot be encoded")
	}
	command, err := tx.Exec(ctx, `
INSERT INTO runtime_management_operations (
    idempotency_key_digest, request_digest, operation_kind, resource_id,
    result, actor_id, performed_at
) VALUES ($1, $2, $3, $4, $5::jsonb, $6, $7)
ON CONFLICT DO NOTHING`, receipt.KeyDigest, receipt.RequestDigest, receipt.Kind, receipt.ResourceID,
		string(encoded), actor, persistencepostgres.Timestamp(at))
	if err != nil {
		return errors.New("store Runtime management audit")
	}
	if command.RowsAffected() != 1 {
		return ErrConflict
	}
	return nil
}
