// Package credentials defines the narrow secret boundary shared by Run
// resolution, Planner construction, and Worker allocation preparation.
package credentials

import (
	"context"
	"errors"
	"fmt"
	"sync"

	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
)

var ErrNotFound = errors.New("LLM credential not found")

type Resolver interface {
	config.CredentialLookup
	ResolveLLMCredential(
		context.Context,
		contracts.LLMCredentialRef,
		contracts.LLMGatewayConfigRef,
	) (contracts.SecretString, error)
}

type StaticEntry struct {
	Metadata config.CredentialMetadata
	Token    contracts.SecretString
}

// StaticProvider is an explicit development bootstrap. It is immutable after
// construction and is replaced by the encrypted PostgreSQL provider in V4.
type StaticProvider struct {
	mu      sync.RWMutex
	entries map[string]StaticEntry
}

func NewStaticProvider(entries []StaticEntry) (*StaticProvider, error) {
	result := &StaticProvider{entries: make(map[string]StaticEntry, len(entries))}
	for _, entry := range entries {
		id := entry.Metadata.Ref.CredentialID
		if err := entry.Metadata.Ref.Validate(); err != nil {
			return nil, fmt.Errorf("development credential ID is invalid: %w", err)
		}
		if err := entry.Metadata.LLMGateway.ValidateRef(); err != nil {
			return nil, fmt.Errorf("development credential Gateway ref is invalid: %w", err)
		}
		if entry.Token.Reveal() == "" {
			return nil, errors.New("development credential token must not be empty")
		}
		if _, exists := result.entries[id]; exists {
			return nil, fmt.Errorf("duplicate development credential %q", id)
		}
		result.entries[id] = entry
	}
	return result, nil
}

func (p *StaticProvider) LookupLLMCredential(
	_ context.Context,
	id string,
) (config.CredentialMetadata, error) {
	if p == nil {
		return config.CredentialMetadata{}, ErrNotFound
	}
	p.mu.RLock()
	entry, ok := p.entries[id]
	p.mu.RUnlock()
	if !ok {
		return config.CredentialMetadata{}, ErrNotFound
	}
	return entry.Metadata, nil
}

func (p *StaticProvider) ResolveLLMCredential(
	ctx context.Context,
	ref contracts.LLMCredentialRef,
	gateway contracts.LLMGatewayConfigRef,
) (contracts.SecretString, error) {
	metadata, err := p.LookupLLMCredential(ctx, ref.CredentialID)
	if err != nil || metadata.Ref != ref || metadata.LLMGateway != gateway {
		return contracts.SecretString{}, ErrNotFound
	}
	p.mu.RLock()
	token := p.entries[ref.CredentialID].Token
	p.mu.RUnlock()
	return token, nil
}
