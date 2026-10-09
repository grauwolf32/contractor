package app

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	neturl "net/url"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifactpolicy"
	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/configload"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/credentials"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/runservice"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgconn"
	"github.com/jackc/pgx/v5/pgxpool"
)

func TestPostgresPublicRunCreationRecoversFreshPinsAndRolledBackResults(t *testing.T) {
	for _, name := range []string{"concurrent-binding-rebind", "rollback-after-create", "exhausted-deadlock"} {
		t.Run(name, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(t.Context(), 30*time.Second)
			defer cancel()
			pool := isolatedRecoveryPool(t, ctx)
			guard := recoveryCredentialGuard{}
			publisher, err := runtimeconfig.NewPublisher(runtimeconfig.PublisherOptions{
				Pool: pool, RuntimeCredentials: guard,
				PlannerTelemetryAdapters: runtimeconfig.PlannerTelemetryAdapterCatalogFunc(func(string) bool { return true }),
			})
			if err != nil {
				t.Fatal(err)
			}
			published, err := publisher.Publish(ctx, []byte(`{"apiVersion":"contractor/v1alpha1","kind":"RuntimeConfig","metadata":{"name":"retry-target","version":"1"},"spec":{"worker":{"telemetry":null}}}`), "publish-retry", "operator")
			if err != nil {
				t.Fatal(err)
			}
			provider, err := credentials.NewStaticProvider(nil)
			if err != nil {
				t.Fatal(err)
			}
			attempts := 0
			transactionLookup := runtimeconfig.TransactionLLMCredentialLookupFactoryFunc(
				func(tx pgx.Tx) (config.CredentialLookup, error) {
					attempts++
					if name == "exhausted-deadlock" {
						return nil, &pgconn.PgError{Code: "40P01", Message: "injected deadlock"}
					}
					if name == "concurrent-binding-rebind" && attempts == 1 {
						// Establish REPEATABLE READ, then commit a rebind on a
						// different connection before the first pin attempts SHARE.
						if _, err := runtimeconfig.NewRepository(tx).GetBinding(ctx, runtimeconfig.DefaultLabel); err != nil {
							return nil, err
						}
						if _, err := runtimeconfig.NewRepository(pool).Rebind(ctx, runtimeconfig.DefaultLabel, 1, published.Version.Ref, "operator", time.Now()); err != nil {
							return nil, err
						}
					}
					return provider, nil
				})
			manager, err := configload.NewManager(config.ManagerOptions{OperatorRoot: "../config/testdata/valid", ManagedRoot: t.TempDir(), Descriptors: config.MVPDescriptors()})
			if err != nil {
				t.Fatal(err)
			}
			serviceArtifacts := artifacts.NewService(artifacts.NewPostgresRepository(pool))
			user, _ := serviceArtifacts.User("owner-retry")
			source, err := user.Write(ctx, contracts.ArtifactRef{Namespace: "sources", Name: "source"}, artifacts.Payload{MediaType: "text/plain", Data: []byte("source")}, nil)
			if err != nil {
				t.Fatal(err)
			}
			if name == "rollback-after-create" {
				installRunCreationAbortAfterRepeatRequest(t, ctx, pool)
			}
			service, err := configureRunCreation(pool, manager, runCreationCredentials{
				provider: provider, guard: guard, runtime: guard, transactionLookup: transactionLookup,
			})
			if err != nil {
				t.Fatal(err)
			}
			generated := 0
			params := runservice.PublicCreateParams{
				OwnerID: "owner-retry", Workflow: "artifact-copy@1", IdempotencyKey: "create-retry", RequestDigest: "sha256:" + strings.Repeat("a", 64),
				Inputs:   map[string]contracts.ArtifactRef{"source": source.Ref},
				NewRunID: func() (string, error) { generated++; return "run-retry", nil },
			}
			result, err := service.CreatePublic(ctx, params)
			if name == "exhausted-deadlock" {
				if !persistencepostgres.IsTransactionConflict(err) || attempts != persistencepostgres.MaxTransactionAttempts || generated != 1 {
					t.Fatalf("exhausted creation=%+v attempts=%d ids=%d error=%v", result, attempts, generated, err)
				}
				var runs, refs int
				if err := pool.QueryRow(ctx, `SELECT (SELECT count(*) FROM workflow_runs), count(*) FROM artifact_binding_revisions WHERE scope_kind='run'`).Scan(&runs, &refs); err != nil || runs != 0 || refs != 0 {
					t.Fatalf("exhausted resources: runs=%d revisions=%d error=%v", runs, refs, err)
				}
				return
			}
			if err != nil || !result.Created || result.Replayed || result.Run.RunID != "run-retry" || attempts != 2 || generated != 1 {
				t.Fatalf("creation=%+v attempts=%d ids=%d error=%v", result, attempts, generated, err)
			}
			if name == "concurrent-binding-rebind" && (result.Run.RuntimeConfig.Default.Config != published.Version.Ref || result.Run.RuntimeConfig.Default.BindingRevision != 2) {
				t.Fatalf("retry persisted stale pin: %+v", result.Run.RuntimeConfig.Default)
			}
			replay, err := service.CreatePublic(ctx, params)
			if err != nil || !replay.Replayed || replay.Created || attempts != 2 || generated != 1 {
				t.Fatalf("replay=%+v, %v", replay, err)
			}
			// Run creation retains both the input and the protected repeat request.
			// Neither an aborted attempt nor an idempotent replay may add revisions.
			var runs, refs, inputs, repeatRequests int
			err = pool.QueryRow(ctx, `
SELECT (SELECT count(*) FROM workflow_runs), count(*),
       count(*) FILTER (WHERE scope_id=$1 AND namespace='inputs' AND name='source'),
       count(*) FILTER (WHERE scope_id=$1 AND namespace=$2 AND name=$3)
FROM artifact_binding_revisions WHERE scope_kind='run'`,
				result.Run.RunID, artifactpolicy.RunSystemNamespace, artifactpolicy.RunRepeatRequestName,
			).Scan(&runs, &refs, &inputs, &repeatRequests)
			if err != nil || runs != 1 || refs != 2 || inputs != 1 || repeatRequests != 1 {
				t.Fatalf("retry resource counts: runs=%d revisions=%d inputs=%d repeatRequests=%d error=%v", runs, refs, inputs, repeatRequests, err)
			}
		})
	}
}

func installRunCreationAbortAfterRepeatRequest(t *testing.T, ctx context.Context, pool *pgxpool.Pool) {
	t.Helper()
	// A sequence survives transaction rollback, so only the first attempt aborts
	// after Run and artifact writes through the production transaction callback.
	for _, statement := range []string{
		`CREATE SEQUENCE abort_after_repeat_request_sequence`,
		`CREATE FUNCTION abort_after_repeat_request_once() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
  IF NEW.scope_kind='run' AND NEW.namespace='contractor-system' AND NEW.name='repeat-request' THEN
    IF nextval('abort_after_repeat_request_sequence')=1 THEN
      RAISE EXCEPTION 'injected definite abort after writes' USING ERRCODE='40001';
    END IF;
  END IF;
  RETURN NEW;
END $$`,
		`CREATE TRIGGER abort_after_repeat_request_once AFTER INSERT ON artifact_binding_revisions
FOR EACH ROW EXECUTE FUNCTION abort_after_repeat_request_once()`,
	} {
		if _, err := pool.Exec(ctx, statement); err != nil {
			t.Fatal(err)
		}
	}
}

type recoveryCredentialGuard struct{}

func (recoveryCredentialGuard) WithRunCreation(_ context.Context, fn func() error) error { return fn() }
func (recoveryCredentialGuard) WithCredentialReferences(_ context.Context, fn func() error) error {
	return fn()
}
func (recoveryCredentialGuard) ValidateRuntimeCredential(context.Context, string, ...string) error {
	return nil
}
func (recoveryCredentialGuard) ValidateRuntimeCredentialUse(
	context.Context, credentials.RuntimeCredentialUser, string, ...string,
) error {
	return nil
}

func isolatedRecoveryPool(t *testing.T, ctx context.Context) *pgxpool.Pool {
	t.Helper()
	url := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if url == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	admin, err := pgxpool.New(ctx, url)
	if err != nil {
		t.Fatal(err)
	}
	var random [8]byte
	if _, err := rand.Read(random[:]); err != nil {
		t.Fatal(err)
	}
	schema := "run_recovery_" + hex.EncodeToString(random[:])
	quoted := pgx.Identifier{schema}.Sanitize()
	if _, err := admin.Exec(ctx, `CREATE SCHEMA `+quoted); err != nil {
		t.Fatal(err)
	}
	isolatedURL := url
	if strings.HasPrefix(url, "postgres://") || strings.HasPrefix(url, "postgresql://") {
		parsed, parseErr := neturl.Parse(url)
		if parseErr != nil {
			t.Fatal("invalid test database URL")
		}
		query := parsed.Query()
		query.Set("search_path", schema)
		parsed.RawQuery = query.Encode()
		isolatedURL = parsed.String()
	} else {
		isolatedURL += " search_path=" + schema
	}
	pool, err := persistencepostgres.OpenPool(ctx, isolatedURL, persistencepostgres.PoolOptions{})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		pool.Close()
		cleanup, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		if _, err := admin.Exec(cleanup, `DROP SCHEMA `+quoted+` CASCADE`); err != nil {
			t.Log(err)
		}
		admin.Close()
	})
	if _, err := persistencepostgres.ApplyMigrations(ctx, pool); err != nil {
		t.Fatal(err)
	}
	return pool
}
