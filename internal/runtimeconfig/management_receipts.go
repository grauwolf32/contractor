package runtimeconfig

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"

	"github.com/grauwolf32/contractor/internal/contracts"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/strictjson"
	"github.com/jackc/pgx/v5"
)

// ManagementReceipt identifies one idempotent Runtime management mutation:
// the digest of the caller's idempotency key, the digest of its normalized
// request, the operation kind and the resource it changes. Label bindings and
// Runtime Agent principals run their mutations through the functions below.
// They are keyed mutations (PER-R24) in one key space: the receipt key is the
// digest of the operator's key, the shared keyed-mutation helper keeps the
// receipt, and a receipt of another request, operation kind or resource under
// the key is the idempotency conflict.
type ManagementReceipt struct {
	KeyDigest     string
	RequestDigest string
	Kind          string
	ResourceID    string
}

func (r ManagementReceipt) key() persistencepostgres.ReceiptKey {
	return persistencepostgres.RuntimeManagementReceiptKey(r.KeyDigest)
}

// ReadManagementReplay decodes into result the outcome recorded for the
// receipt's key and reports whether one exists; it takes no lock. The key
// bound to another request, operation or resource is the permanent
// contracts.ErrIdempotencyConflict.
func ReadManagementReplay(
	ctx context.Context, db persistencepostgres.DBTX, receipt ManagementReceipt, result any,
) (bool, error) {
	stored, found, err := persistencepostgres.FindReceipt(ctx, db, receipt.key(), receipt.RequestDigest)
	return decodeManagementReplay(receipt, stored, found, err, result)
}

// KeyedManagementMutation runs one Runtime management mutation in tx: it
// takes the key's lock, decodes the outcome recorded for the key into result
// and reports true, and otherwise runs mutate and records the outcome mutate
// returns in the same transaction. An error of mutate is returned as it is,
// and nothing is recorded.
func KeyedManagementMutation(
	ctx context.Context, tx pgx.Tx, receipt ManagementReceipt, result any, mutate func() (any, error),
) (bool, error) {
	stored, replayed, err := persistencepostgres.KeyedMutation(ctx, tx, receipt.key(), receipt.RequestDigest,
		func() (persistencepostgres.Receipt, error) {
			outcome, err := mutate()
			if err != nil {
				return persistencepostgres.Receipt{}, err
			}
			encoded, err := json.Marshal(outcome)
			if err != nil {
				return persistencepostgres.Receipt{}, invalid("Runtime management result cannot be encoded")
			}
			return persistencepostgres.Receipt{
				Operation: receipt.Kind, ResourceID: receipt.ResourceID, Response: encoded,
			}, nil
		})
	if err == nil && !replayed {
		return false, nil
	}
	return decodeManagementReplay(receipt, stored, replayed, err, result)
}

// decodeManagementReplay turns a stored receipt into the caller's replay, and
// a receipt of another request, operation or resource into the idempotency
// conflict.
func decodeManagementReplay(
	receipt ManagementReceipt, stored persistencepostgres.Receipt, found bool, err error, result any,
) (bool, error) {
	switch {
	case errors.Is(err, persistencepostgres.ErrReceiptMismatch):
		return false, fmt.Errorf("%w: Runtime management key", contracts.ErrIdempotencyConflict)
	case errors.Is(err, persistencepostgres.ErrReceiptClaimed):
		return false, ErrConflict
	case err != nil || !found:
		return false, err
	case stored.Operation != receipt.Kind || stored.ResourceID != receipt.ResourceID:
		return false, fmt.Errorf("%w: Runtime management key", contracts.ErrIdempotencyConflict)
	}
	if err := strictjson.Decode(stored.Response, result); err != nil {
		return false, errors.New("invalid Runtime management replay")
	}
	return true, nil
}
