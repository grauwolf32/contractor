package credentials

import (
	"context"
	"errors"
	"fmt"

	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
)

type CompositeProvider struct {
	providers []Resolver
}

func NewCompositeProvider(providers ...Resolver) (*CompositeProvider, error) {
	result := &CompositeProvider{}
	for _, provider := range providers {
		if provider != nil {
			result.providers = append(result.providers, provider)
		}
	}
	if len(result.providers) == 0 {
		return nil, errors.New("at least one credential provider is required")
	}
	return result, nil
}

func (p *CompositeProvider) LookupLLMCredential(
	ctx context.Context, id string,
) (config.CredentialMetadata, error) {
	var result config.CredentialMetadata
	found := false
	for _, provider := range p.providers {
		metadata, err := provider.LookupLLMCredential(ctx, id)
		if errors.Is(err, ErrNotFound) {
			continue
		}
		if err != nil {
			return config.CredentialMetadata{}, fmt.Errorf("lookup LLM credential")
		}
		if found {
			return config.CredentialMetadata{}, ErrConflict
		}
		result = metadata
		found = true
	}
	if !found {
		return config.CredentialMetadata{}, ErrNotFound
	}
	return result, nil
}

func (p *CompositeProvider) ResolveLLMCredential(
	ctx context.Context,
	ref contracts.LLMCredentialRef,
	gateway contracts.LLMGatewayConfigRef,
) (contracts.SecretString, error) {
	var selected Resolver
	for _, provider := range p.providers {
		metadata, err := provider.LookupLLMCredential(ctx, ref.CredentialID)
		if errors.Is(err, ErrNotFound) {
			continue
		}
		if err != nil {
			return contracts.SecretString{}, fmt.Errorf("resolve LLM credential")
		}
		if selected != nil {
			return contracts.SecretString{}, ErrConflict
		}
		if metadata.Ref != ref || metadata.LLMGateway != gateway {
			return contracts.SecretString{}, ErrNotFound
		}
		selected = provider
	}
	if selected == nil {
		return contracts.SecretString{}, ErrNotFound
	}
	result, err := selected.ResolveLLMCredential(ctx, ref, gateway)
	if err != nil {
		if errors.Is(err, ErrNotFound) || errors.Is(err, ErrConflict) ||
			errors.Is(err, ErrCrypto) || errors.Is(err, ErrKeyUnavailable) {
			return contracts.SecretString{}, err
		}
		return contracts.SecretString{}, errors.New("resolve LLM credential")
	}
	return result, nil
}

var _ Resolver = (*CompositeProvider)(nil)
