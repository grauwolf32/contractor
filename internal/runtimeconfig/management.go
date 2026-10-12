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
			repository := NewRepository(tx)
			replayed, found, replayErr := lockBindingReplay(ctx, tx, receipt)
			if replayErr != nil {
				return replayErr
			}
			if found {
				result = replayed
				return nil
			}
			if err := s.bindings.validateTargetInTransaction(ctx, tx, ref); err != nil {
				return err
			}
			binding, err := repository.CreateBinding(ctx, label, ref, actor, at)
			if err != nil {
				return err
			}
			result = BindingMutationResult{Binding: &binding}
			return recordBindingResult(ctx, tx, receipt, result, actor, at)
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
			replayed, found, replayErr := lockBindingReplay(ctx, tx, receipt)
			if replayErr != nil {
				return replayErr
			}
			if found {
				result = replayed
				return nil
			}
			repository := NewRepository(tx)
			if _, err := repository.LockBindingsForUpdate(ctx, labelsToLock); err != nil {
				if errors.Is(err, ErrNotFound) {
					if _, pathErr := repository.GetBinding(ctx, label); errors.Is(pathErr, ErrNotFound) {
						return ErrLabelNotFound
					} else if pathErr != nil {
						return pathErr
					}
					// The rebind target still exists, so the missing label is
					// one of the other labels held by an optimistically listed
					// principal that was removed concurrently. The snapshot is
					// stale; report a precondition failure rather than an
					// unknown label the request never selected.
					return ErrPrecondition
				}
				return err
			}
			current, err := s.holders.ListLabelHolders(ctx, tx, label)
			if err != nil {
				return err
			}
			if !sameLabelHolders(optimistic, current) {
				return ErrPrecondition
			}
			holderIDs := make([]string, len(current))
			for index := range current {
				holderIDs[index] = current[index].RuntimeAgentID
			}
			locked, err := s.holders.LockLabelHolders(ctx, tx, holderIDs)
			if err != nil {
				return err
			}
			if !sameLabelHolders(current, locked) {
				return ErrPrecondition
			}
			if err := s.bindings.validateTargetInTransaction(ctx, tx, ref); err != nil {
				return err
			}
			binding, err := repository.Rebind(ctx, label, expectedRevision, ref, actor, at)
			if err != nil {
				if errors.Is(err, ErrNotFound) {
					return ErrLabelNotFound
				}
				return err
			}
			for _, holder := range locked {
				if err := ValidateLockedAgentLabels(ctx, tx, holder.Labels); err != nil {
					return err
				}
			}
			result = BindingMutationResult{Binding: &binding}
			return recordBindingResult(ctx, tx, receipt, result, actor, at)
		})
	})
	return result, err
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
			replayed, found, replayErr := lockBindingReplay(ctx, tx, receipt)
			if replayErr != nil {
				return replayErr
			}
			if found {
				result = replayed
				return nil
			}
			repository := NewRepository(tx)
			if _, err := repository.LockBindingsForUpdate(ctx, []string{label}); err != nil {
				if errors.Is(err, ErrNotFound) {
					return ErrLabelNotFound
				}
				return err
			}
			holders, err := s.holders.ListLabelHolders(ctx, tx, label)
			if err != nil {
				return err
			}
			if len(holders) != 0 {
				return labelInUseError(holders)
			}
			if err := repository.DeleteBinding(ctx, label, expectedRevision); err != nil {
				if errors.Is(err, ErrNotFound) {
					return ErrLabelNotFound
				}
				return err
			}
			result = BindingMutationResult{Deleted: true}
			return recordBindingResult(ctx, tx, receipt, result, actor, at)
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

func lockBindingReplay(
	ctx context.Context, tx pgx.Tx, receipt ManagementReceipt,
) (BindingMutationResult, bool, error) {
	var wire bindingMutationResultWire
	found, err := LockManagementReplay(ctx, tx, receipt, &wire)
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

func recordBindingResult(
	ctx context.Context, tx pgx.Tx, receipt ManagementReceipt,
	result BindingMutationResult, actor string, at time.Time,
) error {
	if (result.Binding == nil) == !result.Deleted {
		return invalid("Runtime management audit is invalid")
	}
	return RecordManagementResult(
		ctx, tx, receipt, bindingMutationResultWire{Binding: result.Binding, Deleted: result.Deleted}, actor, at,
	)
}
