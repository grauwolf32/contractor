package credentials

import (
	"context"
	"errors"

	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
)

type RecordReader interface {
	GetCredential(context.Context, string) (Record, error)
}

type EncryptedProvider struct {
	records RecordReader
	cipher  *TokenCipher
}

func NewEncryptedProvider(records RecordReader, cipher *TokenCipher) (*EncryptedProvider, error) {
	if records == nil {
		return nil, errors.New("encrypted credential record reader is required")
	}
	return &EncryptedProvider{records: records, cipher: cipher}, nil
}

func (p *EncryptedProvider) LookupLLMCredential(
	ctx context.Context, id string,
) (config.CredentialMetadata, error) {
	if p == nil || p.records == nil {
		return config.CredentialMetadata{}, ErrNotFound
	}
	record, err := p.records.GetCredential(ctx, id)
	if err != nil {
		if errors.Is(err, ErrNotFound) {
			return config.CredentialMetadata{}, ErrNotFound
		}
		return config.CredentialMetadata{}, persistencepostgres.WrapError("lookup encrypted credential metadata", err)
	}
	return config.CredentialMetadata{
		Ref:           contracts.LLMCredentialRef{CredentialID: record.CredentialID},
		LLMGateway:    record.LLMGateway,
		ModelPolicies: append([]contracts.ModelPolicyRef(nil), record.EffectivePolicy.ModelPolicies...),
		Models:        append([]string(nil), record.EffectivePolicy.Models...),
	}, nil
}

func (p *EncryptedProvider) ResolveLLMCredential(
	ctx context.Context,
	ref contracts.LLMCredentialRef,
	gateway contracts.LLMGatewayConfigRef,
) (contracts.SecretString, error) {
	if p == nil || p.records == nil {
		return contracts.SecretString{}, ErrNotFound
	}
	record, err := p.records.GetCredential(ctx, ref.CredentialID)
	if errors.Is(err, ErrNotFound) {
		return contracts.SecretString{}, ErrNotFound
	}
	if err != nil {
		return contracts.SecretString{}, persistencepostgres.WrapError("resolve encrypted credential", err)
	}
	if record.CredentialID != ref.CredentialID || record.LLMGateway != gateway {
		return contracts.SecretString{}, ErrNotFound
	}
	if p.cipher == nil {
		return contracts.SecretString{}, ErrKeyUnavailable
	}
	token, err := p.cipher.Open(record.CredentialID, record.LLMGateway, record.Envelope)
	if err != nil {
		return contracts.SecretString{}, ErrCrypto
	}
	return contracts.NewSecretString(token.value), nil
}

var _ Resolver = (*EncryptedProvider)(nil)
