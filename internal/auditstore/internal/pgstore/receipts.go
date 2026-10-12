package pgstore

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"

	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/contracts"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/jackc/pgx/v5"
)

// The owner's Audit mutations are keyed mutations (PER-R23): each records its
// receipt through persistencepostgres.KeyedMutation, and a repeated request
// replays with the Audit as it is now.

// auditReceiptKey is the receipt key of an owner's Audit mutation.
func auditReceiptKey(ownerID string, operation auditstore.MutationOperation, key string) persistencepostgres.ReceiptKey {
	return persistencepostgres.AuditReceiptKey(ownerID, string(operation), key)
}

// auditMutationResult is what one owner mutation changed: the Audit and the
// resource and response its receipt records.
type auditMutationResult struct {
	audit      auditstore.Audit
	resourceID string
	response   map[string]string
}

// keyedAuditMutation runs one owner mutation of an Audit on the store's pool
// or in the caller's transaction (PER-R18). It takes the key's lock, replays a
// receipt of the same request with the Audit as it is now, and otherwise runs
// mutate in the transaction and records its receipt. It reports whether the
// mutation ran.
func (s *OwnerStore) keyedAuditMutation(
	ctx context.Context, ownerID string, operation auditstore.MutationOperation, key, digest string,
	mutate func(tx pgx.Tx) (auditMutationResult, error),
) (auditstore.Audit, bool, error) {
	var result auditstore.Audit
	var inserted bool
	err := persistencepostgres.Within(ctx, s.db, pgx.TxOptions{}, func(tx pgx.Tx) error {
		result, inserted = auditstore.Audit{}, false
		var changed auditMutationResult
		receipt, replayed, err := persistencepostgres.KeyedMutation(
			ctx, tx, auditReceiptKey(ownerID, operation, key), digest,
			func() (persistencepostgres.Receipt, error) {
				var err error
				if changed, err = mutate(tx); err != nil {
					return persistencepostgres.Receipt{}, err
				}
				response, err := json.Marshal(changed.response)
				if err != nil {
					return persistencepostgres.Receipt{}, err
				}
				return persistencepostgres.Receipt{
					Operation: string(operation), ResourceID: changed.resourceID, Response: response,
					OwnerID: ownerID, AuditID: changed.audit.AuditID,
				}, nil
			},
		)
		if err != nil {
			return auditReceiptError(err, operation)
		}
		if replayed {
			result, err = (&store{db: tx}).get(ctx, ownerID, receipt.AuditID)
			return err
		}
		result, inserted = changed.audit, true
		return nil
	})
	if err != nil {
		return auditstore.Audit{}, false, err
	}
	return result, inserted, nil
}

// LookupMutationReplay is the owner-safe pre-resolution boundary used by the
// public API. A matching replay can be returned even when a mutable profile,
// credential, Runtime label binding, or Skill binding has since disappeared.
func (s *OwnerStore) LookupMutationReplay(
	ctx context.Context,
	ownerID string,
	operation auditstore.MutationOperation,
	key string,
	digest string,
) (auditstore.Audit, bool, error) {
	if err := validateMutationReplay(ownerID, operation, key, digest); err != nil {
		return auditstore.Audit{}, false, err
	}
	receipt, found, err := persistencepostgres.FindReceipt(ctx, s.db, auditReceiptKey(ownerID, operation, key), digest)
	if err != nil {
		return auditstore.Audit{}, found, auditReceiptError(err, operation)
	}
	if !found {
		return auditstore.Audit{}, false, nil
	}
	audit, err := s.get(ctx, ownerID, receipt.AuditID)
	return audit, true, err
}

// LockMutation takes the lock of an owner mutation's key until the caller
// commits, for a mutation that checks other state under the key before it
// runs; the mutation's own take of the lock then does not wait.
func (s *OwnerTxStore) LockMutation(
	ctx context.Context, ownerID string, operation auditstore.MutationOperation, key string,
) error {
	return persistencepostgres.LockReceipt(ctx, s.tx, auditReceiptKey(ownerID, operation, key))
}

// auditReceiptError reports a receipt of another request under the key as
// the idempotency conflict and a key another writer claimed as a conflict.
func auditReceiptError(err error, operation auditstore.MutationOperation) error {
	switch {
	case errors.Is(err, persistencepostgres.ErrReceiptMismatch):
		return fmt.Errorf("%w: Audit %s key", contracts.ErrIdempotencyConflict, operation)
	case errors.Is(err, persistencepostgres.ErrReceiptClaimed):
		return auditstore.ErrConflict
	}
	return err
}
