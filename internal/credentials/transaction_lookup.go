package credentials

import (
	"errors"

	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
	"github.com/jackc/pgx/v5"
)

// TransactionLookupFactory composes immutable process-local development
// credentials with managed credential metadata read through the caller-owned
// transaction. Secret decryption is intentionally unavailable on this lookup:
// Run creation needs only safe metadata and exact credential identities.
type TransactionLookupFactory struct {
	development *StaticProvider
}

var _ runtimeconfig.TransactionLLMCredentialLookupFactory = (*TransactionLookupFactory)(nil)

func NewTransactionLookupFactory(development *StaticProvider) (*TransactionLookupFactory, error) {
	if development == nil {
		return nil, errors.New("development credential provider is required")
	}
	return &TransactionLookupFactory{development: development}, nil
}

func (f *TransactionLookupFactory) ForTransaction(
	tx pgx.Tx,
) (config.CredentialLookup, error) {
	if f == nil || f.development == nil || tx == nil {
		return nil, errors.New("transaction credential lookup is not configured")
	}
	managed, err := NewEncryptedProvider(NewRepository(tx), nil)
	if err != nil {
		return nil, err
	}
	return NewCompositeProvider(f.development, managed)
}
