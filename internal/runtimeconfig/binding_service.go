package runtimeconfig

import (
	"context"
	"errors"
	"slices"
	"time"

	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

type BindingService struct {
	pool           *pgxpool.Pool
	credentials    TransactionRuntimeCredentialCatalog
	llmCredentials TransactionLLMCredentialLookupFactory
}

func NewBindingService(
	pool *pgxpool.Pool,
	credentials TransactionRuntimeCredentialCatalog,
	llmCredentials ...TransactionLLMCredentialLookupFactory,
) (*BindingService, error) {
	if pool == nil || credentials == nil || len(llmCredentials) > 1 {
		return nil, errors.New("RuntimeConfig binding service dependencies are incomplete")
	}
	service := &BindingService{pool: pool, credentials: credentials}
	if len(llmCredentials) == 1 {
		service.llmCredentials = llmCredentials[0]
	}
	return service, nil
}

func (s *BindingService) Create(
	ctx context.Context, label string, ref Ref, actor string, at time.Time,
) (Binding, error) {
	var result Binding
	err := s.credentials.WithCredentialReferences(ctx, func() error {
		return persistencepostgres.InTx(ctx, s.pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
			if err := s.validateTargetInTransaction(ctx, tx, ref); err != nil {
				return err
			}
			var err error
			result, err = NewRepository(tx).CreateBinding(ctx, label, ref, actor, at)
			return err
		})
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
			if err := s.validateTargetInTransaction(ctx, tx, ref); err != nil {
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

func (s *BindingService) validateTargetInTransaction(ctx context.Context, tx pgx.Tx, ref Ref) error {
	validator, err := s.credentials.ForRuntimeTransaction(tx)
	if err != nil {
		return err
	}
	if validator == nil {
		return errors.New("transaction Runtime credential validator is not configured")
	}
	version, err := NewRepository(tx).GetVersionByRef(ctx, ref)
	if err != nil {
		if errors.Is(err, ErrNotFound) {
			return ErrVersionNotFound
		}
		return err
	}
	if err := validateSpecRuntimeCredentials(ctx, version.Spec, validator); err != nil {
		return err
	}
	return validateSpecLLMCredentialInTransaction(ctx, tx, version.Spec, s.llmCredentials)
}

func samePrincipalSnapshots(left, right []RuntimeAgentPrincipal) bool {
	if len(left) != len(right) {
		return false
	}
	for index := range left {
		if left[index].RuntimeAgentID != right[index].RuntimeAgentID ||
			left[index].LabelRevision != right[index].LabelRevision ||
			!slices.Equal(left[index].Labels, right[index].Labels) {
			return false
		}
	}
	return true
}
