package credentials

import (
	"context"
	"errors"
	"strings"
	"testing"

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
