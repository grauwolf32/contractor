package credentials

import (
	"context"
	"errors"
	"strings"
	"testing"

	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/jackc/pgx/v5"
)

const credentialFailureCanary = "private-credential-store-payload"

// Fail only the selected read after ordinary service and receipt reads have
// reached PostgreSQL. SQL failures come from PostgreSQL itself; cancellation
// happens immediately before the selected query uses its caller's context.
type credentialReadFailure struct {
	db     persistencepostgres.DBTX
	code   string
	cancel context.CancelFunc
}

func (f credentialReadFailure) row(ctx context.Context) pgx.Row {
	if f.cancel != nil {
		f.cancel()
		return f.db.QueryRow(ctx, `SELECT 'cancelled credential read'`)
	}
	return f.db.QueryRow(ctx, `SELECT credential_read_failure($1::text)`, f.code)
}

func (f credentialReadFailure) err(ctx context.Context) error {
	var ignored string
	return f.row(ctx).Scan(&ignored)
}

type credentialReadFailureDB struct {
	persistencepostgres.DBTX
	match   string
	failure credentialReadFailure
}

func (db credentialReadFailureDB) QueryRow(ctx context.Context, query string, args ...any) pgx.Row {
	if strings.Contains(query, db.match) {
		return db.failure.row(ctx)
	}
	return db.DBTX.QueryRow(ctx, query, args...)
}

func (f credentialReadFailure) InspectRuntimeCredentialUsage(ctx context.Context, _ string, _ int) (RuntimeCredentialUsage, error) {
	return RuntimeCredentialUsage{}, f.err(ctx)
}

func (f credentialReadFailure) ListProjectIDsByHTTPTargetCredential(ctx context.Context, _ string, _ int) ([]string, error) {
	return nil, f.err(ctx)
}

func (f credentialReadFailure) ListHeldAuditIDsByRuntimeCredential(ctx context.Context, _ string, _ int) ([]string, error) {
	return nil, f.err(ctx)
}

func installCredentialReadFailure(t *testing.T, ctx context.Context, db persistencepostgres.DBTX) {
	t.Helper()
	_, err := db.Exec(ctx, `CREATE FUNCTION credential_read_failure(failure_code text) RETURNS text LANGUAGE plpgsql AS $$
BEGIN
  RAISE EXCEPTION 'private-credential-store-payload' USING ERRCODE = failure_code;
END $$`)
	if err != nil {
		t.Fatal(err)
	}
}

func requireCredentialReadFailure(t *testing.T, err error, code string) {
	t.Helper()
	if code == "" {
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("credential read lost caller cancellation: %v", err)
		}
	} else if persistencepostgres.SQLState(err) != code || !persistencepostgres.IsTransactionConflict(err) {
		t.Fatalf("credential read lost PostgreSQL %s: %v", code, err)
	}
	if strings.Contains(err.Error(), credentialFailureCanary) {
		t.Fatal("credential failure rendered private database text")
	}
}

func TestCredentialReplayAndReconciliationKeepStoreFailures(t *testing.T) {
	for _, code := range []string{
		persistencepostgres.SQLStateSerializationFailure,
		persistencepostgres.SQLStateDeadlockDetected,
		"",
	} {
		name := code
		if name == "" {
			name = "cancelled"
		}
		for _, path := range []string{"LLM replay", "Runtime replay", "LLM record reconciliation", "LLM operation reconciliation"} {
			t.Run(name+"/"+path, func(t *testing.T) {
				f, fixtureContext := newAvailabilityFixture(t)
				installCredentialReadFailure(t, fixtureContext, f.pool)
				ctx, cancel := context.WithCancel(fixtureContext)
				defer cancel()
				failure := credentialReadFailure{db: f.pool, code: code}
				if code == "" {
					failure.cancel = cancel
				}
				db := credentialReadFailureDB{DBTX: f.pool, failure: failure}
				var err error
				switch path {
				case "LLM replay":
					db.match = "FROM llm_credentials"
					f.service.repository = NewRepository(db)
					_, err = f.service.Create(ctx, f.unrelated)
				case "Runtime replay":
					db.match = "FROM runtime_credentials c"
					f.runtimeCredentials.repository = NewRuntimeCredentialRepository(db)
					request := runtimeOTLPRequest(t, besideRuntimeCredential, "runtime-create-beside")
					defer request.Material.Destroy()
					_, err = f.runtimeCredentials.Create(ctx, request)
				case "LLM record reconciliation":
					record, readErr := f.service.GetCredential(ctx, f.unrelated.CredentialID)
					if readErr != nil {
						t.Fatal(readErr)
					}
					db.match = "FROM llm_credentials"
					err = NewRepository(db).InsertCredential(ctx, record)
				case "LLM operation reconciliation":
					operation, readErr := f.service.repository.GetOperationByIdempotency(ctx, OperationCreate, f.unrelated.IdempotencyKey)
					if readErr != nil {
						t.Fatal(readErr)
					}
					db.match = "WHERE operation_kind = $1 AND idempotency_key = $2"
					err = NewRepository(db).InsertOperation(ctx, operation)
				}
				requireCredentialReadFailure(t, err, code)
			})
		}
	}
}

func TestRuntimeCredentialDeleteKeepsEveryUsageStoreFailure(t *testing.T) {
	for _, code := range []string{
		persistencepostgres.SQLStateSerializationFailure,
		persistencepostgres.SQLStateDeadlockDetected,
		"",
	} {
		name := code
		if name == "" {
			name = "cancelled"
		}
		for _, source := range []string{"RuntimeConfig", "Project", "Audit"} {
			t.Run(name+"/"+source, func(t *testing.T) {
				f, fixtureContext := newAvailabilityFixture(t)
				installCredentialReadFailure(t, fixtureContext, f.pool)
				ctx, cancel := context.WithCancel(fixtureContext)
				defer cancel()
				failure := credentialReadFailure{db: f.pool, code: code}
				if code == "" {
					failure.cancel = cancel
				}
				switch source {
				case "RuntimeConfig":
					f.runtimeCredentials.usage = failure
				case "Project":
					f.runtimeCredentials.projects = failure
				case "Audit":
					f.runtimeCredentials.audits = failure
				}
				_, err := f.runtimeCredentials.Delete(ctx, besideRuntimeCredential, "operator")
				requireCredentialReadFailure(t, err, code)
				if _, err := f.runtimeCredentials.repository.GetTombstone(fixtureContext, besideRuntimeCredential); !errors.Is(err, ErrRuntimeCredentialNotFound) {
					t.Fatalf("failed usage read recorded a deletion: %v", err)
				}
				if _, err := f.runtimeCredentials.Get(fixtureContext, besideRuntimeCredential); err != nil {
					t.Fatalf("failed usage read removed the credential: %v", err)
				}
			})
		}
	}
}
