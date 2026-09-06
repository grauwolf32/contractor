package runtimeconfig

import (
	"context"
	"errors"
	"time"

	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

type BindingService struct {
	pool        *pgxpool.Pool
	repository  *Repository
	credentials RuntimeCredentialCatalog
}

func NewBindingService(
	pool *pgxpool.Pool,
	credentials RuntimeCredentialCatalog,
) (*BindingService, error) {
	if pool == nil || credentials == nil {
		return nil, errors.New("RuntimeConfig binding service dependencies are incomplete")
	}
	return &BindingService{pool: pool, repository: NewRepository(pool), credentials: credentials}, nil
}

func (s *BindingService) Create(
	ctx context.Context, label string, ref Ref, actor string, at time.Time,
) (Binding, error) {
	var result Binding
	err := s.credentials.WithCredentialReferences(ctx, func() error {
		if err := s.validateTarget(ctx, ref); err != nil {
			return err
		}
		created, err := s.repository.CreateBinding(ctx, label, ref, actor, at)
		result = created
		return err
	})
	return result, err
}

func (s *BindingService) Rebind(
	ctx context.Context, label string, expectedRevision uint64, ref Ref, actor string, at time.Time,
) (Binding, error) {
	optimistic, err := NewPrincipalRepository(s.pool).ListByLabel(ctx, label)
	if err != nil {
		return Binding{}, err
	}
	labelsToLock := []string{label}
	for _, principal := range optimistic {
		labelsToLock = append(labelsToLock, principal.Labels...)
	}
	labelsToLock = sortedUnion(labelsToLock)
	var result Binding
	err = s.credentials.WithCredentialReferences(ctx, func() error {
		return persistencepostgres.InTx(ctx, s.pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
			repository := NewRepository(tx)
			if _, err := repository.LockBindingsForUpdate(ctx, labelsToLock); err != nil {
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
			if err := s.validateTargetWith(ctx, repository, ref); err != nil {
				return err
			}
			result, err = repository.Rebind(ctx, label, expectedRevision, ref, actor, at)
			if err != nil {
				return err
			}
			for _, principal := range locked {
				if _, err := validateAgentLabelSetFromLocked(ctx, tx, principal.Labels); err != nil {
					return err
				}
			}
			return nil
		})
	})
	return result, err
}

func (s *BindingService) Delete(ctx context.Context, label string, expectedRevision uint64) error {
	return s.credentials.WithCredentialReferences(ctx, func() error {
		return persistencepostgres.InTx(ctx, s.pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
			repository := NewRepository(tx)
			if _, err := repository.LockBindingsForUpdate(ctx, []string{label}); err != nil {
				return err
			}
			principals, err := NewPrincipalRepository(tx).ListByLabel(ctx, label)
			if err != nil {
				return err
			}
			if len(principals) != 0 {
				return labelInUseError(principals)
			}
			return repository.DeleteBinding(ctx, label, expectedRevision)
		})
	})
}

func (s *BindingService) validateTarget(ctx context.Context, ref Ref) error {
	return s.validateTargetWith(ctx, s.repository, ref)
}

func (s *BindingService) validateTargetWith(ctx context.Context, repository *Repository, ref Ref) error {
	version, err := repository.GetVersionByRef(ctx, ref)
	if err != nil {
		return err
	}
	return validateSpecRuntimeCredentials(ctx, version.Spec, s.credentials)
}

func samePrincipalSnapshots(left, right []RuntimeAgentPrincipal) bool {
	if len(left) != len(right) {
		return false
	}
	for index := range left {
		if left[index].RuntimeAgentID != right[index].RuntimeAgentID ||
			left[index].LabelRevision != right[index].LabelRevision ||
			!equalStrings(left[index].Labels, right[index].Labels) {
			return false
		}
	}
	return true
}
