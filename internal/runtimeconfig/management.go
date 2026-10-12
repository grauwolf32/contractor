package runtimeconfig

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"time"

	"github.com/grauwolf32/contractor/internal/contentdigest"
	"github.com/grauwolf32/contractor/internal/contracts"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

const (
	bindingCreateOperation = "runtime-label.create"
	bindingRebindOperation = "runtime-label.rebind"
	bindingDeleteOperation = "runtime-label.delete"
)

type BindingMutationResult struct {
	Binding  *Binding
	Deleted  bool
	Replayed bool
}

type bindingMutationResultWire struct {
	Binding *Binding `json:"binding,omitempty"`
	Deleted bool     `json:"deleted,omitempty"`
}

// ManagementService is the public Operations-facing facade over immutable
// RuntimeConfig versions and CAS label bindings. It deliberately returns only
// non-secret documents and metadata.
type ManagementService struct {
	pool       *pgxpool.Pool
	repository *Repository
	publisher  *Publisher
	bindings   *BindingService
}

func NewManagementService(
	pool *pgxpool.Pool,
	publisher *Publisher,
	bindings *BindingService,
) (*ManagementService, error) {
	if pool == nil || publisher == nil || bindings == nil {
		return nil, errors.New("RuntimeConfig management dependencies are incomplete")
	}
	return &ManagementService{
		pool: pool, repository: NewRepository(pool), publisher: publisher, bindings: bindings,
	}, nil
}

func (s *ManagementService) Publish(
	ctx context.Context, document []byte, idempotencyKey, actor string,
) (PublishResult, error) {
	return s.publisher.Publish(ctx, document, idempotencyKey, actor)
}

func (s *ManagementService) ListVersions(
	ctx context.Context, afterName, afterVersion string, limit int,
) ([]Version, error) {
	return s.repository.ListVersions(ctx, afterName, afterVersion, limit)
}

func (s *ManagementService) GetVersion(ctx context.Context, name, version string) (Version, error) {
	return s.repository.GetVersion(ctx, name, version)
}

func (s *ManagementService) ListBindings(
	ctx context.Context, afterLabel string, limit int,
) ([]Binding, error) {
	return s.repository.ListBindings(ctx, afterLabel, limit)
}

func (s *ManagementService) GetBinding(ctx context.Context, label string) (Binding, error) {
	return s.repository.GetBinding(ctx, label)
}

func (s *ManagementService) CreateBinding(
	ctx context.Context,
	label string,
	ref Ref,
	idempotencyKey string,
	actor string,
	at time.Time,
) (BindingMutationResult, error) {
	requestDigest, keyDigest, err := prepareBindingMutation(
		bindingCreateOperation, label, ref, 0, idempotencyKey,
	)
	if err != nil {
		return BindingMutationResult{}, err
	}
	var result BindingMutationResult
	err = s.bindings.credentials.WithCredentialReferences(ctx, func() error {
		return persistencepostgres.InTx(ctx, s.pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
			var err error
			result, err = keyedBindingOperation(ctx, tx, keyDigest, requestDigest, bindingCreateOperation, label, actor, at,
				func() (BindingMutationResult, error) {
					if err := s.bindings.validateTargetInTransaction(ctx, tx, ref); err != nil {
						return BindingMutationResult{}, err
					}
					binding, err := NewRepository(tx).CreateBinding(ctx, label, ref, actor, at)
					if err != nil {
						return BindingMutationResult{}, err
					}
					return BindingMutationResult{Binding: &binding}, nil
				})
			return err
		})
	})
	return result, err
}

func (s *ManagementService) Rebind(
	ctx context.Context,
	label string,
	expectedRevision uint64,
	ref Ref,
	idempotencyKey string,
	actor string,
	at time.Time,
) (BindingMutationResult, error) {
	requestDigest, keyDigest, err := prepareBindingMutation(
		bindingRebindOperation, label, ref, expectedRevision, idempotencyKey,
	)
	if err != nil {
		return BindingMutationResult{}, err
	}
	if replay, found, replayErr := readManagementReplay(
		ctx, s.pool, keyDigest, requestDigest, bindingRebindOperation, label,
	); replayErr != nil {
		return BindingMutationResult{}, replayErr
	} else if found {
		return replay, nil
	}
	optimistic, err := NewPrincipalRepository(s.pool).ListByLabel(ctx, label)
	if err != nil {
		return BindingMutationResult{}, err
	}
	labelsToLock := []string{label}
	for _, principal := range optimistic {
		labelsToLock = append(labelsToLock, principal.Labels...)
	}
	labelsToLock = sortedUnion(labelsToLock)
	var result BindingMutationResult
	err = s.bindings.credentials.WithCredentialReferences(ctx, func() error {
		return persistencepostgres.InTx(ctx, s.pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
			var err error
			result, err = keyedBindingOperation(ctx, tx, keyDigest, requestDigest, bindingRebindOperation, label, actor, at,
				func() (BindingMutationResult, error) {
					return s.rebindInTransaction(ctx, tx, label, expectedRevision, ref, labelsToLock, optimistic, actor, at)
				})
			return err
		})
	})
	return result, err
}

// rebindInTransaction rebinds label in tx after the replay check: it locks
// every label of the optimistically listed principals, proves that listing
// current and validates each principal's label set under the new binding.
func (s *ManagementService) rebindInTransaction(
	ctx context.Context, tx pgx.Tx, label string, expectedRevision uint64, ref Ref,
	labelsToLock []string, optimistic []RuntimeAgentPrincipal, actor string, at time.Time,
) (BindingMutationResult, error) {
	repository := NewRepository(tx)
	if _, err := repository.LockBindingsForUpdate(ctx, labelsToLock); err != nil {
		if errors.Is(err, ErrNotFound) {
			if _, pathErr := repository.GetBinding(ctx, label); errors.Is(pathErr, ErrNotFound) {
				return BindingMutationResult{}, ErrLabelNotFound
			} else if pathErr != nil {
				return BindingMutationResult{}, pathErr
			}
			// The rebind target still exists, so the missing label is
			// one of the other labels held by an optimistically listed
			// principal that was removed concurrently. The snapshot is
			// stale; report a precondition failure rather than an
			// unknown label the request never selected.
			return BindingMutationResult{}, ErrPrecondition
		}
		return BindingMutationResult{}, err
	}
	principalRepository := NewPrincipalRepository(tx)
	current, err := principalRepository.ListByLabel(ctx, label)
	if err != nil {
		return BindingMutationResult{}, err
	}
	if !samePrincipalSnapshots(optimistic, current) {
		return BindingMutationResult{}, ErrPrecondition
	}
	principalIDs := make([]string, len(current))
	for index := range current {
		principalIDs[index] = current[index].RuntimeAgentID
	}
	locked, err := principalRepository.LockMany(ctx, principalIDs)
	if err != nil {
		return BindingMutationResult{}, err
	}
	if !samePrincipalSnapshots(current, locked) {
		return BindingMutationResult{}, ErrPrecondition
	}
	if err := s.bindings.validateTargetInTransaction(ctx, tx, ref); err != nil {
		return BindingMutationResult{}, err
	}
	binding, err := repository.Rebind(ctx, label, expectedRevision, ref, actor, at)
	if err != nil {
		if errors.Is(err, ErrNotFound) {
			return BindingMutationResult{}, ErrLabelNotFound
		}
		return BindingMutationResult{}, err
	}
	for _, principal := range locked {
		if _, err := validateAgentLabelSetFromLocked(ctx, tx, principal.Labels); err != nil {
			return BindingMutationResult{}, err
		}
	}
	return BindingMutationResult{Binding: &binding}, nil
}

func (s *ManagementService) DeleteBinding(
	ctx context.Context,
	label string,
	expectedRevision uint64,
	idempotencyKey string,
	actor string,
	at time.Time,
) (BindingMutationResult, error) {
	requestDigest, keyDigest, err := prepareBindingMutation(
		bindingDeleteOperation, label, Ref{}, expectedRevision, idempotencyKey,
	)
	if err != nil {
		return BindingMutationResult{}, err
	}
	var result BindingMutationResult
	err = s.bindings.credentials.WithCredentialReferences(ctx, func() error {
		return persistencepostgres.InTx(ctx, s.pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
			var err error
			result, err = keyedBindingOperation(ctx, tx, keyDigest, requestDigest, bindingDeleteOperation, label, actor, at,
				func() (BindingMutationResult, error) {
					repository := NewRepository(tx)
					if _, err := repository.LockBindingsForUpdate(ctx, []string{label}); err != nil {
						if errors.Is(err, ErrNotFound) {
							return BindingMutationResult{}, ErrLabelNotFound
						}
						return BindingMutationResult{}, err
					}
					principals, err := NewPrincipalRepository(tx).ListByLabel(ctx, label)
					if err != nil {
						return BindingMutationResult{}, err
					}
					if len(principals) != 0 {
						return BindingMutationResult{}, labelInUseError(principals)
					}
					if err := repository.DeleteBinding(ctx, label, expectedRevision); err != nil {
						if errors.Is(err, ErrNotFound) {
							return BindingMutationResult{}, ErrLabelNotFound
						}
						return BindingMutationResult{}, err
					}
					return BindingMutationResult{Deleted: true}, nil
				})
			return err
		})
	})
	return result, err
}

func prepareBindingMutation(
	kind string,
	label string,
	ref Ref,
	expectedRevision uint64,
	idempotencyKey string,
) (string, string, error) {
	if validateLabel(label) != nil ||
		(kind != bindingCreateOperation && expectedRevision == 0) ||
		(kind != bindingDeleteOperation && validateRef(ref) != nil) {
		return "", "", invalid("Runtime label mutation is invalid")
	}
	keyDigest, err := DigestIdempotencyKey(idempotencyKey)
	if err != nil {
		return "", "", err
	}
	request, err := json.Marshal(struct {
		Kind             string `json:"kind"`
		Label            string `json:"label"`
		ExpectedRevision uint64 `json:"expectedRevision"`
		Ref              Ref    `json:"ref"`
	}{Kind: kind, Label: label, ExpectedRevision: expectedRevision, Ref: ref})
	if err != nil {
		return "", "", invalid("Runtime label mutation cannot be normalized")
	}
	return contentdigest.Bytes(request), keyDigest, nil
}

// The Runtime management operations, label binding and Agent label set
// changes alike, are keyed mutations (PER-R23) in one key space: the receipt
// key is the digest of the operator's key, and a receipt of another operation
// kind or resource under it is the idempotency conflict, as is a receipt of
// another request.

// keyedManagementOperation runs one Runtime management operation in tx through
// the shared receipt helper and reports its stored or new response and
// whether it was replayed. keyName names the key in a conflict.
func keyedManagementOperation(
	ctx context.Context, tx pgx.Tx, keyDigest, requestDigest, kind, resourceID, keyName string,
	mutate func() ([]byte, error),
) ([]byte, bool, error) {
	receipt, replayed, err := persistencepostgres.KeyedMutation(ctx, tx,
		persistencepostgres.RuntimeManagementReceiptKey(keyDigest), requestDigest,
		func() (persistencepostgres.Receipt, error) {
			response, err := mutate()
			if err != nil {
				return persistencepostgres.Receipt{}, err
			}
			return persistencepostgres.Receipt{Operation: kind, ResourceID: resourceID, Response: response}, nil
		})
	if err == nil && !replayed {
		return receipt.Response, false, nil
	}
	return managementReplay(receipt, replayed, err, kind, resourceID, keyName)
}

// findManagementReceipt reads the response stored under an operator's key
// without its lock, for a replay before mutable dependencies are read.
func findManagementReceipt(
	ctx context.Context, db persistencepostgres.DBTX, keyDigest, requestDigest, kind, resourceID, keyName string,
) ([]byte, bool, error) {
	receipt, found, err := persistencepostgres.FindReceipt(ctx, db,
		persistencepostgres.RuntimeManagementReceiptKey(keyDigest), requestDigest)
	return managementReplay(receipt, found, err, kind, resourceID, keyName)
}

func managementReplay(
	receipt persistencepostgres.Receipt, found bool, err error, kind, resourceID, keyName string,
) ([]byte, bool, error) {
	switch {
	case errors.Is(err, persistencepostgres.ErrReceiptMismatch):
		return nil, false, fmt.Errorf("%w: %s", contracts.ErrIdempotencyConflict, keyName)
	case errors.Is(err, persistencepostgres.ErrReceiptClaimed):
		return nil, false, ErrConflict
	case err != nil || !found:
		return nil, false, err
	case receipt.Operation != kind || receipt.ResourceID != resourceID:
		return nil, false, fmt.Errorf("%w: %s", contracts.ErrIdempotencyConflict, keyName)
	}
	return receipt.Response, true, nil
}

// keyedBindingOperation runs one label binding operation in tx as a keyed
// mutation. The actor and time of the operation are recorded on the binding.
func keyedBindingOperation(
	ctx context.Context, tx pgx.Tx, keyDigest, requestDigest, kind, label, actor string, at time.Time,
	mutate func() (BindingMutationResult, error),
) (BindingMutationResult, error) {
	var changed BindingMutationResult
	response, replayed, err := keyedManagementOperation(ctx, tx, keyDigest, requestDigest, kind, label,
		"Runtime label management key", func() ([]byte, error) {
			if !validActor(actor) || at.IsZero() {
				return nil, invalid("Runtime management audit is invalid")
			}
			var err error
			if changed, err = mutate(); err != nil {
				return nil, err
			}
			if (changed.Binding == nil) == !changed.Deleted {
				return nil, invalid("Runtime management audit is invalid")
			}
			encoded, err := json.Marshal(bindingMutationResultWire{Binding: changed.Binding, Deleted: changed.Deleted})
			if err != nil {
				return nil, invalid("Runtime management result cannot be encoded")
			}
			return encoded, nil
		})
	if err != nil || !replayed {
		return changed, err
	}
	return decodeBindingReplay(response)
}

// readManagementReplay reads a label binding operation's replay outside its
// transaction.
func readManagementReplay(
	ctx context.Context, db persistencepostgres.DBTX, keyDigest, requestDigest, kind, label string,
) (BindingMutationResult, bool, error) {
	response, found, err := findManagementReceipt(ctx, db, keyDigest, requestDigest, kind, label,
		"Runtime label management key")
	if err != nil || !found {
		return BindingMutationResult{}, false, err
	}
	result, err := decodeBindingReplay(response)
	return result, err == nil, err
}

func decodeBindingReplay(response []byte) (BindingMutationResult, error) {
	var wire bindingMutationResultWire
	if err := json.Unmarshal(response, &wire); err != nil || (wire.Binding == nil) == !wire.Deleted {
		return BindingMutationResult{}, errors.New("invalid Runtime management replay")
	}
	return BindingMutationResult{Binding: wire.Binding, Deleted: wire.Deleted, Replayed: true}, nil
}
