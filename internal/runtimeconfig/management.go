package runtimeconfig

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"time"

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

type managementOperation struct {
	KeyDigest     string
	RequestDigest string
	Kind          string
	ResourceID    string
	Result        []byte
	ActorID       string
	PerformedAt   time.Time
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
			repository := NewRepository(tx)
			replayed, found, replayErr := lookupManagementReplay(
				ctx, tx, keyDigest, requestDigest, bindingCreateOperation, label,
			)
			if replayErr != nil {
				return replayErr
			}
			if found {
				result = replayed
				return nil
			}
			if err := s.bindings.validateTargetWith(ctx, repository, ref); err != nil {
				return err
			}
			binding, err := repository.CreateBinding(ctx, label, ref, actor, at)
			if err != nil {
				return err
			}
			result = BindingMutationResult{Binding: &binding}
			return insertManagementResult(
				ctx, tx, keyDigest, requestDigest, bindingCreateOperation, label, result, actor, at,
			)
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
			replayed, found, replayErr := lookupManagementReplay(
				ctx, tx, keyDigest, requestDigest, bindingRebindOperation, label,
			)
			if replayErr != nil {
				return replayErr
			}
			if found {
				result = replayed
				return nil
			}
			repository := NewRepository(tx)
			if _, err := repository.LockBindings(ctx, labelsToLock); err != nil {
				return err
			}
			principalRepository := NewPrincipalRepository(tx)
			current, err := principalRepository.ListByLabel(ctx, label)
			if err != nil {
				return err
			}
			if !samePrincipalSnapshots(optimistic, current) {
				return ErrPrecondition
			}
			principalIDs := make([]string, len(current))
			for index := range current {
				principalIDs[index] = current[index].RuntimeAgentID
			}
			locked, err := principalRepository.LockMany(ctx, principalIDs)
			if err != nil {
				return err
			}
			if !samePrincipalSnapshots(current, locked) {
				return ErrPrecondition
			}
			if err := s.bindings.validateTargetWith(ctx, repository, ref); err != nil {
				return err
			}
			binding, err := repository.Rebind(ctx, label, expectedRevision, ref, actor, at)
			if err != nil {
				return err
			}
			for _, principal := range locked {
				if _, err := validateAgentLabelSetFromLocked(ctx, tx, principal.Labels); err != nil {
					return err
				}
			}
			result = BindingMutationResult{Binding: &binding}
			return insertManagementResult(
				ctx, tx, keyDigest, requestDigest, bindingRebindOperation, label, result, actor, at,
			)
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
	requestDigest, keyDigest, err := prepareBindingMutation(
		bindingDeleteOperation, label, Ref{}, expectedRevision, idempotencyKey,
	)
	if err != nil {
		return BindingMutationResult{}, err
	}
	var result BindingMutationResult
	err = s.bindings.credentials.WithCredentialReferences(ctx, func() error {
		return persistencepostgres.InTx(ctx, s.pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
			replayed, found, replayErr := lookupManagementReplay(
				ctx, tx, keyDigest, requestDigest, bindingDeleteOperation, label,
			)
			if replayErr != nil {
				return replayErr
			}
			if found {
				result = replayed
				return nil
			}
			repository := NewRepository(tx)
			if _, err := repository.LockBindings(ctx, []string{label}); err != nil {
				return err
			}
			principals, err := NewPrincipalRepository(tx).ListByLabel(ctx, label)
			if err != nil {
				return err
			}
			if len(principals) != 0 {
				return labelInUseError(principals)
			}
			if err := repository.DeleteBinding(ctx, label, expectedRevision); err != nil {
				return err
			}
			result = BindingMutationResult{Deleted: true}
			return insertManagementResult(
				ctx, tx, keyDigest, requestDigest, bindingDeleteOperation, label, result, actor, at,
			)
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
	return digest(request), keyDigest, nil
}

func lookupManagementReplay(
	ctx context.Context,
	tx pgx.Tx,
	keyDigest string,
	requestDigest string,
	kind string,
	resourceID string,
) (BindingMutationResult, bool, error) {
	// A transaction-scoped advisory lock makes the replay lookup and mutation
	// one linearizable operation even before the immutable audit row exists.
	if _, err := tx.Exec(ctx, `SELECT pg_advisory_xact_lock(hashtextextended($1, 0))`, keyDigest); err != nil {
		return BindingMutationResult{}, false, errors.New("lock Runtime management idempotency key")
	}
	return readManagementReplay(ctx, tx, keyDigest, requestDigest, kind, resourceID)
}

func readManagementReplay(
	ctx context.Context,
	db persistencepostgres.DBTX,
	keyDigest string,
	requestDigest string,
	kind string,
	resourceID string,
) (BindingMutationResult, bool, error) {
	var operation managementOperation
	var resultText string
	err := db.QueryRow(ctx, `
SELECT idempotency_key_digest, request_digest, operation_kind, resource_id,
       result::text, actor_id, performed_at
FROM runtime_management_operations
WHERE idempotency_key_digest = $1`, keyDigest).Scan(
		&operation.KeyDigest, &operation.RequestDigest, &operation.Kind, &operation.ResourceID,
		&resultText, &operation.ActorID, &operation.PerformedAt,
	)
	if errors.Is(err, pgx.ErrNoRows) {
		return BindingMutationResult{}, false, nil
	}
	if err != nil {
		return BindingMutationResult{}, false, errors.New("get Runtime management replay")
	}
	if operation.RequestDigest != requestDigest || operation.Kind != kind || operation.ResourceID != resourceID {
		return BindingMutationResult{}, false, ErrConflict
	}
	var wire bindingMutationResultWire
	if err := json.Unmarshal([]byte(resultText), &wire); err != nil || (wire.Binding == nil) == !wire.Deleted {
		return BindingMutationResult{}, false, errors.New("invalid Runtime management replay")
	}
	return BindingMutationResult{Binding: wire.Binding, Deleted: wire.Deleted, Replayed: true}, true, nil
}

func insertManagementResult(
	ctx context.Context,
	tx pgx.Tx,
	keyDigest string,
	requestDigest string,
	kind string,
	resourceID string,
	result BindingMutationResult,
	actor string,
	at time.Time,
) error {
	if !validActor(actor) || at.IsZero() || (result.Binding == nil) == !result.Deleted {
		return invalid("Runtime management audit is invalid")
	}
	encoded, err := json.Marshal(bindingMutationResultWire{Binding: result.Binding, Deleted: result.Deleted})
	if err != nil {
		return invalid("Runtime management result cannot be encoded")
	}
	command, err := tx.Exec(ctx, `
INSERT INTO runtime_management_operations (
    idempotency_key_digest, request_digest, operation_kind, resource_id,
    result, actor_id, performed_at
) VALUES ($1, $2, $3, $4, $5::jsonb, $6, $7)
ON CONFLICT DO NOTHING`, keyDigest, requestDigest, kind, resourceID, string(encoded), actor, databaseTime(at))
	if err != nil {
		return fmt.Errorf("store Runtime management audit")
	}
	if command.RowsAffected() != 1 {
		return ErrConflict
	}
	return nil
}
