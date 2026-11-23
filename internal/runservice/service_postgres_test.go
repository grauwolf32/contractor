package runservice

import (
	"context"
	cryptorand "crypto/rand"
	"encoding/hex"
	"errors"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/auditstore/auditstoretest"
	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/credentials"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
	"github.com/grauwolf32/contractor/internal/runtimeconfig/runtimeconfigtest"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

type emptyCredentialLookup struct{}

func (emptyCredentialLookup) LookupLLMCredential(context.Context, string) (resolved.CredentialMetadata, error) {
	return resolved.CredentialMetadata{}, errors.New("credential unavailable")
}

type openCredentialGuard struct{}

func (openCredentialGuard) WithRunCreation(ctx context.Context, fn func() error) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	return fn()
}

type acceptRuntimeCredentials struct{}

func (acceptRuntimeCredentials) ValidateRuntimeCredentialUse(
	context.Context, credentials.RuntimeCredentialUser, string, ...string,
) error {
	return nil
}

func postgresPublicTransaction(pool *pgxpool.Pool) PublicTransaction {
	return func(ctx context.Context, fn func(PublicRunWriter, *artifacts.Service, RuntimeCredentialValidator) error) error {
		return persistencepostgres.InTx(ctx, pool, pgx.TxOptions{IsoLevel: pgx.RepeatableRead}, func(tx pgx.Tx) error {
			lookup, err := runtimeconfig.BindTransactionLLMCredentialLookup(
				tx,
				runtimeconfigtest.TransactionLLMCredentialLookupFactoryFunc(
					func(pgx.Tx) (resolved.CredentialLookup, error) { return emptyCredentialLookup{}, nil },
				),
			)
			if err != nil {
				return err
			}
			return fn(
				runstore.NewRunCreationPostgresStore(tx, lookup, credentials.NewRuntimeCredentialRepository(tx), artifactpgtest.NewStorage()),
				artifacts.NewService(artifactpgtest.NewStorage().Repository(tx)),
				credentials.NewRuntimeCredentialRepository(tx),
			)
		})
	}
}

func postgresAuditTransaction(pool *pgxpool.Pool) AuditTransaction {
	return func(ctx context.Context, fn func(AuditRunWriter, *artifacts.Service, AuditExecutionWriter) error) error {
		return persistencepostgres.InTx(ctx, pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
			return fn(
				runstore.NewPostgresTxStore(tx, artifactpgtest.NewStorage()), artifacts.NewService(artifactpgtest.NewStorage().Repository(tx)),
				auditstoretest.NewStore(tx, artifactpgtest.NewStorage()),
			)
		})
	}
}

func writeExact(
	t testing.TB,
	ctx context.Context,
	store artifacts.ScopedStore,
	namespace string,
	name string,
	mediaType string,
	payload []byte,
) auditstore.ExactArtifact {
	t.Helper()
	written, err := store.Write(
		ctx, contracts.ArtifactRef{Namespace: namespace, Name: name},
		artifacts.Payload{MediaType: mediaType, Data: payload}, nil,
	)
	if err != nil {
		t.Fatal(err)
	}
	metadata, err := store.Metadata(ctx, written.Ref)
	if err != nil {
		t.Fatal(err)
	}
	return auditstore.ExactArtifact{
		Ref: metadata.Ref, Digest: metadata.Digest, MediaType: metadata.MediaType, SizeBytes: metadata.Size,
	}
}

func testDigest(character string) string {
	return "sha256:" + repeat(character, 64)
}

func repeat(value string, count int) string {
	result := ""
	for range count {
		result += value
	}
	return result
}

func isolatedRunServicePool(
	t *testing.T,
	ctx context.Context,
	databaseURL string,
) *pgxpool.Pool {
	t.Helper()
	adminConfig, err := pgxpool.ParseConfig(databaseURL)
	if err != nil {
		t.Fatal(err)
	}
	admin, err := pgxpool.NewWithConfig(ctx, adminConfig)
	if err != nil {
		t.Fatal(err)
	}
	if err := admin.Ping(ctx); err != nil {
		admin.Close()
		t.Fatalf("CONTRACTOR_TEST_DATABASE_URL is set but PostgreSQL is unreachable: %v", err)
	}
	suffix := make([]byte, 6)
	if _, err := cryptorand.Read(suffix); err != nil {
		admin.Close()
		t.Fatal(err)
	}
	databaseName := "contractor_runservice_" + hex.EncodeToString(suffix)
	if _, err := admin.Exec(ctx, `CREATE DATABASE `+databaseName); err != nil {
		admin.Close()
		t.Fatal(err)
	}
	t.Cleanup(func() {
		cleanupContext, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		_, _ = admin.Exec(cleanupContext, `DROP DATABASE IF EXISTS `+databaseName+` WITH (FORCE)`)
		admin.Close()
	})
	poolConfig := adminConfig.Copy()
	poolConfig.ConnConfig.Database = databaseName
	pool, err := pgxpool.NewWithConfig(ctx, poolConfig)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(pool.Close)
	if _, err := persistencepostgres.ApplyMigrations(ctx, pool, persistencepostgres.MigrationBudgets{}); err != nil {
		t.Fatal(err)
	}
	return pool
}
