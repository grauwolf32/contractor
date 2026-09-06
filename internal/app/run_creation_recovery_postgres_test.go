package app

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/credentials"
	publicapi "github.com/grauwolf32/contractor/internal/httpapi/public"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/grauwolf32/contractor/internal/runservice"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgconn"
	"github.com/jackc/pgx/v5/pgxpool"
)

func TestPostgresPublicRunCreationRecoversFreshPinsAndRolledBackResults(t *testing.T) {
	for _, afterCreate := range []bool{false, true} {
		name := "concurrent-binding-rebind"
		if afterCreate {
			name = "rollback-after-create"
		}
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
			uow := postgresPublicUnitOfWork{pool: pool, transactionLLMCredentials: runtimeconfig.TransactionLLMCredentialLookupFactoryFunc(
				func(tx pgx.Tx) (config.CredentialLookup, error) {
					attempts++
					if !afterCreate && attempts == 1 {
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
				}),
			}
			manager, err := config.NewManager(config.ManagerOptions{OperatorRoot: "../config/testdata/valid", ManagedRoot: t.TempDir(), Descriptors: config.MVPDescriptors()})
			if err != nil {
				t.Fatal(err)
			}
			serviceArtifacts := artifacts.NewService(artifacts.NewPostgresRepository(pool))
			user, _ := serviceArtifacts.User("owner-retry")
			source, err := user.Write(ctx, contracts.ArtifactRef{Namespace: "sources", Name: "source"}, artifacts.Payload{MediaType: "text/plain", Data: []byte("source")}, nil)
			if err != nil {
				t.Fatal(err)
			}
			service, err := runservice.New(runservice.Options{
				Runs: runstore.NewPostgresStore(pool), Workflows: manager, LLMCredentials: provider,
				CredentialGuard: guard, RuntimeCredentials: guard, Projects: projectstore.NewPostgresStore(pool),
				PublicTransaction: func(ctx context.Context, fn func(runservice.PublicRunWriter, *artifacts.Service) error) error {
					return uow.Do(ctx, func(writer publicapi.RunWriter, artifactService *artifacts.Service) error {
						if err := fn(writer, artifactService); err != nil {
							return err
						}
						if afterCreate && attempts == 1 {
							return &pgconn.PgError{Code: "40001", Message: "injected definite abort after durable writes"}
						}
						return nil
					})
				},
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
			if err != nil || !result.Created || result.Replayed || result.Run.RunID != "run-retry" || attempts != 2 || generated != 1 {
				t.Fatalf("creation=%+v attempts=%d ids=%d error=%v", result, attempts, generated, err)
			}
			if !afterCreate && (result.Run.RuntimeConfig.Default.Config != published.Version.Ref || result.Run.RuntimeConfig.Default.BindingRevision != 2) {
				t.Fatalf("retry persisted stale pin: %+v", result.Run.RuntimeConfig.Default)
			}
			replay, err := service.CreatePublic(ctx, params)
			if err != nil || !replay.Replayed || replay.Created || attempts != 2 || generated != 1 {
				t.Fatalf("replay=%+v, %v", replay, err)
			}
			var runs, refs int
			if err := pool.QueryRow(ctx, `SELECT (SELECT count(*) FROM workflow_runs), (SELECT count(*) FROM artifact_binding_revisions WHERE scope_kind='run')`).Scan(&runs, &refs); err != nil || runs != 1 || refs != 1 {
				t.Fatalf("retry left duplicated resources: runs=%d revisions=%d error=%v", runs, refs, err)
			}
		})
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
	configuration, err := pgxpool.ParseConfig(url)
	if err != nil {
		t.Fatal(err)
	}
	configuration.ConnConfig.RuntimeParams["search_path"] = schema
	pool, err := pgxpool.NewWithConfig(ctx, configuration)
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
