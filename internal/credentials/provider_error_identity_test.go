package credentials

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/jackc/pgx/v5/pgconn"
)

type failingCredentialRecords struct{ err error }

func (r failingCredentialRecords) GetCredential(context.Context, string) (Record, error) {
	return Record{}, r.err
}

func TestEncryptedProviderPreservesSafeDatabaseAndContextCauses(t *testing.T) {
	for _, cause := range []error{&pgconn.PgError{Code: "40001", Message: "private-token-sql"}, context.Canceled} {
		provider, err := NewEncryptedProvider(failingCredentialRecords{cause}, nil)
		if err != nil {
			t.Fatal(err)
		}
		_, err = provider.LookupLLMCredential(t.Context(), "credential")
		if !errors.Is(err, cause) || strings.Contains(err.Error(), "private-token-sql") {
			t.Fatalf("metadata identity/redaction = %v", err)
		}
		if cause != context.Canceled && persistencepostgres.SQLState(err) != "40001" {
			t.Fatal("SQLSTATE lost at provider boundary")
		}
	}
}

// resolutionFailingProvider reports metadata but fails secret resolution, so
// the composite reaches its selected-provider branch.
type resolutionFailingProvider struct {
	metadata config.CredentialMetadata
	err      error
}

func (p resolutionFailingProvider) LookupLLMCredential(context.Context, string) (config.CredentialMetadata, error) {
	return p.metadata, nil
}

func (p resolutionFailingProvider) ResolveLLMCredential(
	context.Context, contracts.LLMCredentialRef, contracts.LLMGatewayConfigRef,
) (contracts.SecretString, error) {
	return contracts.SecretString{}, p.err
}

func TestCompositeProviderPreservesSafeProviderCauses(t *testing.T) {
	const private = "private-token-sql"
	gateway := testPinnedGatewayRef("1", strings.Repeat("1", 64))
	ref := contracts.LLMCredentialRef{CredentialID: "credential"}
	development, err := NewStaticProvider(nil)
	if err != nil {
		t.Fatal(err)
	}
	for _, cause := range []error{
		&pgconn.PgError{Code: "40001", Message: private},
		&pgconn.PgError{Code: "40P01", Message: private},
		context.Canceled,
		ErrRecoveryRequired,
	} {
		managed, err := NewEncryptedProvider(failingCredentialRecords{cause}, nil)
		if err != nil {
			t.Fatal(err)
		}
		composite, err := NewCompositeProvider(development, managed)
		if err != nil {
			t.Fatal(err)
		}
		_, lookupErr := composite.LookupLLMCredential(t.Context(), ref.CredentialID)
		_, resolveErr := composite.ResolveLLMCredential(t.Context(), ref, gateway)

		selected := resolutionFailingProvider{
			metadata: config.CredentialMetadata{Ref: ref, LLMGateway: gateway},
			err:      fmt.Errorf("%s: %w", private, cause),
		}
		composite, err = NewCompositeProvider(development, selected)
		if err != nil {
			t.Fatal(err)
		}
		_, selectedErr := composite.ResolveLLMCredential(t.Context(), ref, gateway)

		for _, err := range []error{lookupErr, resolveErr, selectedErr} {
			if !errors.Is(err, cause) || strings.Contains(err.Error(), private) {
				t.Fatalf("composite identity/redaction for %v = %v", cause, err)
			}
			var pgErr *pgconn.PgError
			if errors.As(cause, &pgErr) {
				if persistencepostgres.SQLState(err) != pgErr.Code {
					t.Fatalf("SQLSTATE lost at composite boundary: %v", err)
				}
				if !persistencepostgres.IsTransactionConflict(err) {
					t.Fatalf("serialization failure is no longer retryable: %v", err)
				}
			}
		}
	}
}
