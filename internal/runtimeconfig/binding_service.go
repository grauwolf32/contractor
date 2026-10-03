package runtimeconfig

import (
	"context"
	"errors"
	"slices"

	"github.com/jackc/pgx/v5"
)

type BindingService struct {
	credentials    TransactionRuntimeCredentialCatalog
	llmCredentials TransactionLLMCredentialLookupFactory
}

func NewBindingService(
	credentials TransactionRuntimeCredentialCatalog,
	llmCredentials ...TransactionLLMCredentialLookupFactory,
) (*BindingService, error) {
	if credentials == nil || len(llmCredentials) > 1 {
		return nil, errors.New("RuntimeConfig binding service dependencies are incomplete")
	}
	service := &BindingService{credentials: credentials}
	if len(llmCredentials) == 1 {
		service.llmCredentials = llmCredentials[0]
	}
	return service, nil
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
