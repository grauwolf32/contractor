package runtimeconfig

import (
	"context"
	"encoding/json"
	"errors"
	"time"

	"github.com/grauwolf32/contractor/internal/contentdigest"
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
	holders    LabelHolders
}

func NewManagementService(
	pool *pgxpool.Pool,
	publisher *Publisher,
	bindings *BindingService,
	holders LabelHolders,
) (*ManagementService, error) {
	if pool == nil || publisher == nil || bindings == nil || holders == nil {
		return nil, errors.New("RuntimeConfig management dependencies are incomplete")
	}
	return &ManagementService{
		pool: pool, repository: NewRepository(pool), publisher: publisher, bindings: bindings, holders: holders,
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
	receipt, err := prepareBindingMutation(bindingCreateOperation, label, ref, 0, idempotencyKey)
	if err != nil {
		return BindingMutationResult{}, err
	}
	var result BindingMutationResult
	err = s.bindings.credentials.WithCredentialReferences(ctx, func() error {
		return persistencepostgres.InTx(ctx, s.pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
			var err error
			result, err = keyedBindingMutation(ctx, tx, receipt, actor, at, func() (BindingMutationResult, error) {
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
	receipt, err := prepareBindingMutation(bindingRebindOperation, label, ref, expectedRevision, idempotencyKey)
	if err != nil {
		return BindingMutationResult{}, err
	}
	if replay, found, replayErr := readBindingReplay(ctx, s.pool, receipt); replayErr != nil {
		return BindingMutationResult{}, replayErr
	} else if found {
		return replay, nil
	}
	optimistic, err := s.holders.ListLabelHolders(ctx, s.pool, label)
	if err != nil {
		return BindingMutationResult{}, err
	}
	labelsToLock := []string{label}
	for _, holder := range optimistic {
		labelsToLock = append(labelsToLock, holder.Labels...)
	}
	labelsToLock = sortedUnion(labelsToLock)
	var result BindingMutationResult
	err = s.bindings.credentials.WithCredentialReferences(ctx, func() error {
		return persistencepostgres.InTx(ctx, s.pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
			var err error
			result, err = keyedBindingMutation(ctx, tx, receipt, actor, at, func() (BindingMutationResult, error) {
				return s.rebindInTransaction(ctx, tx, label, expectedRevision, ref, labelsToLock, optimistic, actor, at)
			})
			return err
		})
	})
	return result, err
}

// rebindInTransaction rebinds label in tx once the label and every label of
// its holders are locked and the holders are still those the caller listed.
func (s *ManagementService) rebindInTransaction(
	ctx context.Context, tx pgx.Tx, label string, expectedRevision uint64, ref Ref,
	labelsToLock []string, optimistic []LabelHolder, actor string, at time.Time,
) (BindingMutationResult, error) {
	repository := NewRepository(tx)
	if _, err := repository.LockBindingsForUpdate(ctx, labelsToLock); err != nil {
		if errors.Is(err, ErrNotFound) {
			if _, pathErr := repository.GetBinding(ctx, label); errors.Is(pathErr, ErrNotFound) {
				return BindingMutationResult{}, ErrLabelNotFound
			} else if pathErr != nil {
				return BindingMutationResult{}, pathErr
			}
			// The rebind target still exists, so the missing label is one of
			// the other labels held by an optimistically listed principal that
			// was removed concurrently. The snapshot is stale; report a
			// precondition failure rather than an unknown label the request
			// never selected.
			return BindingMutationResult{}, ErrPrecondition
		}
		return BindingMutationResult{}, err
	}
	current, err := s.holders.ListLabelHolders(ctx, tx, label)
	if err != nil {
		return BindingMutationResult{}, err
	}
	if !sameLabelHolders(optimistic, current) {
		return BindingMutationResult{}, ErrPrecondition
	}
	holderIDs := make([]string, len(current))
	for index := range current {
		holderIDs[index] = current[index].RuntimeAgentID
	}
	locked, err := s.holders.LockLabelHolders(ctx, tx, holderIDs)
	if err != nil {
		return BindingMutationResult{}, err
	}
	if !sameLabelHolders(current, locked) {
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
	for _, holder := range locked {
		if err := ValidateLockedAgentLabels(ctx, tx, holder.Labels); err != nil {
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
	receipt, err := prepareBindingMutation(bindingDeleteOperation, label, Ref{}, expectedRevision, idempotencyKey)
	if err != nil {
		return BindingMutationResult{}, err
	}
	var result BindingMutationResult
	err = s.bindings.credentials.WithCredentialReferences(ctx, func() error {
		return persistencepostgres.InTx(ctx, s.pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
			var err error
			result, err = keyedBindingMutation(ctx, tx, receipt, actor, at, func() (BindingMutationResult, error) {
				repository := NewRepository(tx)
				if _, err := repository.LockBindingsForUpdate(ctx, []string{label}); err != nil {
					if errors.Is(err, ErrNotFound) {
						return BindingMutationResult{}, ErrLabelNotFound
					}
					return BindingMutationResult{}, err
				}
				holders, err := s.holders.ListLabelHolders(ctx, tx, label)
				if err != nil {
					return BindingMutationResult{}, err
				}
				if len(holders) != 0 {
					return BindingMutationResult{}, labelInUseError(holders)
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
) (ManagementReceipt, error) {
	if ValidateLabel(label) != nil ||
		(kind != bindingCreateOperation && expectedRevision == 0) ||
		(kind != bindingDeleteOperation && validateRef(ref) != nil) {
		return ManagementReceipt{}, invalid("Runtime label mutation is invalid")
	}
	keyDigest, err := DigestIdempotencyKey(idempotencyKey)
	if err != nil {
		return ManagementReceipt{}, err
	}
	request, err := json.Marshal(struct {
		Kind             string `json:"kind"`
		Label            string `json:"label"`
		ExpectedRevision uint64 `json:"expectedRevision"`
		Ref              Ref    `json:"ref"`
	}{Kind: kind, Label: label, ExpectedRevision: expectedRevision, Ref: ref})
	if err != nil {
		return ManagementReceipt{}, invalid("Runtime label mutation cannot be normalized")
	}
	return ManagementReceipt{
		KeyDigest: keyDigest, RequestDigest: contentdigest.Bytes(request), Kind: kind, ResourceID: label,
	}, nil
}

func readBindingReplay(
	ctx context.Context, db persistencepostgres.DBTX, receipt ManagementReceipt,
) (BindingMutationResult, bool, error) {
	var wire bindingMutationResultWire
	found, err := ReadManagementReplay(ctx, db, receipt, &wire)
	return bindingReplay(wire, found, err)
}

func bindingReplay(wire bindingMutationResultWire, found bool, err error) (BindingMutationResult, bool, error) {
	if err != nil || !found {
		return BindingMutationResult{}, false, err
	}
	if (wire.Binding == nil) == !wire.Deleted {
		return BindingMutationResult{}, false, errors.New("invalid Runtime management replay")
	}
	return BindingMutationResult{Binding: wire.Binding, Deleted: wire.Deleted, Replayed: true}, true, nil
}

// keyedBindingMutation runs one label binding mutation in tx as a Runtime
// management keyed mutation (PER-R24): a recorded outcome under the key
// replays without running mutate. The actor and time are recorded on the
// binding, not in the receipt, and checked before the mutation runs.
func keyedBindingMutation(
	ctx context.Context, tx pgx.Tx, receipt ManagementReceipt, actor string, at time.Time,
	mutate func() (BindingMutationResult, error),
) (BindingMutationResult, error) {
	var wire bindingMutationResultWire
	var changed BindingMutationResult
	replayed, err := KeyedManagementMutation(ctx, tx, receipt, &wire, func() (any, error) {
		if !ValidActor(actor) || at.IsZero() {
			return nil, invalid("Runtime management audit is invalid")
		}
		var err error
		if changed, err = mutate(); err != nil {
			return nil, err
		}
		if (changed.Binding == nil) == !changed.Deleted {
			return nil, invalid("Runtime management audit is invalid")
		}
		return bindingMutationResultWire{Binding: changed.Binding, Deleted: changed.Deleted}, nil
	})
	if err != nil || !replayed {
		return changed, err
	}
	result, _, err := bindingReplay(wire, true, nil)
	return result, err
}
