package credentials

import (
	"context"
	"errors"
	"os"
	"strings"
	"testing"

	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/jackc/pgx/v5"
)

type failedCredentialUsageDB struct{ persistencepostgres.DBTX }

func (db failedCredentialUsageDB) Query(ctx context.Context, _ string, _ ...any) (pgx.Rows, error) {
	return db.DBTX.Query(ctx, `SELECT credential_usage_failure()`)
}

func TestRuntimeCredentialDatabaseFailureIdentity(t *testing.T) {
	url := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if url == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	ctx := t.Context()
	pool := isolatedCredentialPool(t, ctx, url)
	if _, err := pool.Exec(ctx, `CREATE FUNCTION credential_usage_failure() RETURNS text LANGUAGE plpgsql AS $$
BEGIN
  RAISE EXCEPTION 'private SQL payload' USING ERRCODE = 'serialization_failure';
END $$`); err != nil {
		t.Fatal(err)
	}
	_, err := NewRuntimeCredentialRepository(failedCredentialUsageDB{pool}).InspectRuntimeCredentialUsage(ctx, "test-credential", 10)
	err = safeRuntimeCredentialStoreError(err)
	if !persistencepostgres.IsTransactionConflict(err) || strings.Contains(err.Error(), "private SQL payload") {
		t.Fatalf("usage failure lost identity or safe message: %v", err)
	}
	cancelled, cancel := context.WithCancel(ctx)
	cancel()
	repository := NewRuntimeCredentialRepository(pool)
	for name, read := range map[string]func() error{
		"usage": func() error {
			_, err := repository.InspectRuntimeCredentialUsage(cancelled, "test-credential", 10)
			return err
		},
		"count": func() error { _, err := repository.CountStored(cancelled); return err },
		"list":  func() error { _, err := repository.ListActiveMetadata(cancelled, "", 10); return err },
	} {
		if err := read(); !errors.Is(err, context.Canceled) {
			t.Errorf("%s lost cancellation: %v", name, err)
		}
	}
}
