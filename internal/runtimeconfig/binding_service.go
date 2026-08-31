package runtimeconfig

import (
	"context"
	"errors"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"
)

type BindingService struct {
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
	return &BindingService{repository: NewRepository(pool), credentials: credentials}, nil
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
	var result Binding
	err := s.credentials.WithCredentialReferences(ctx, func() error {
		if err := s.validateTarget(ctx, ref); err != nil {
			return err
		}
		updated, err := s.repository.Rebind(ctx, label, expectedRevision, ref, actor, at)
		result = updated
		return err
	})
	return result, err
}

func (s *BindingService) Delete(ctx context.Context, label string, expectedRevision uint64) error {
	return s.credentials.WithCredentialReferences(ctx, func() error {
		return s.repository.DeleteBinding(ctx, label, expectedRevision)
	})
}

func (s *BindingService) validateTarget(ctx context.Context, ref Ref) error {
	version, err := s.repository.GetVersionByRef(ctx, ref)
	if err != nil {
		return err
	}
	return validateSpecRuntimeCredentials(ctx, version.Spec, s.credentials)
}
