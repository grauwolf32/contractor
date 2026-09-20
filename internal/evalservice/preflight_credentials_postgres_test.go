package evalservice

import (
	"bytes"
	"context"
	"errors"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/credentials"
	"github.com/grauwolf32/contractor/internal/evaldomain"
	"github.com/grauwolf32/contractor/internal/evalstore"
	pg "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgconn"
)

func TestPostgresEvalRuntimeCredentialDatabaseFailureRetries(t *testing.T) {
	h := newHarness(t)
	ctx := t.Context()
	cipher, err := credentials.NewTokenCipher(bytes.Repeat([]byte{0x31}, 32))
	if err != nil {
		t.Fatal(err)
	}
	material, err := credentials.NewOTLPHeadersCredential(map[string]string{"Authorization": "fixture"})
	if err != nil {
		t.Fatal(err)
	}
	envelope, err := cipher.SealRuntimeCredential("review-otel", material)
	if err != nil {
		t.Fatal(err)
	}
	_, err = credentials.NewRuntimeCredentialRepository(h.pool).InsertRecord(ctx, credentials.RuntimeCredentialRecord{
		Metadata: credentials.RuntimeCredentialMetadata{CredentialID: "review-otel", Kind: material.Kind(), CreatedBy: "review", CreatedAt: time.Now()}, Envelope: envelope,
	})
	if err != nil {
		t.Fatal(err)
	}
	prepared, err := runtimeconfig.PreparePublication([]byte(`{"apiVersion":"contractor/v1alpha1","kind":"RuntimeConfig","metadata":{"name":"review-runtime","version":"1"},"spec":{"worker":{"telemetry":{"adapter":"otlp-http@1","endpoint":"https://collector.example/v1/traces","credential":"review-otel"}}}}`))
	if err != nil {
		t.Fatal(err)
	}
	version, err := prepared.Resolve(ctx, nil)
	if err != nil {
		t.Fatal(err)
	}
	version.ActorID, version.CreatedAt = "review", time.Now()
	repo := runtimeconfig.NewRepository(h.pool)
	if _, err = repo.InsertVersion(ctx, version); err != nil {
		t.Fatal(err)
	}
	if _, err = repo.Rebind(ctx, runtimeconfig.DefaultLabel, 1, version.Ref, "review", time.Now()); err != nil {
		t.Fatal(err)
	}
	// Confirm that the selected runtime config and credential are valid first.
	err = pg.InTx(ctx, h.pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		lookup, err := runtimeconfig.BindTransactionLLMCredentialLookup(tx, h.resolver.Credentials)
		if err != nil {
			return err
		}
		_, err = runtimeconfig.PinRunSnapshot(ctx, tx, nil, credentials.NewRuntimeCredentialRepository(tx), lookup)
		return err
	})
	if err != nil {
		t.Fatal(err)
	}
	e := h.create(t, "workflow")
	h.command(t, e, "prepare")
	if _, err = h.pool.Exec(ctx, `ALTER TABLE runtime_credentials RENAME TO review_unavailable_credentials`); err != nil {
		t.Fatal(err)
	}
	coordinator := h.coordinator(t, "review-runtime-failure")
	_, err = coordinator.RunOnce(ctx)
	var database *pgconn.PgError
	if !errors.As(err, &database) || database.Code != "42P01" {
		t.Fatalf("lost database failure: %v", err)
	}
	if _, err = h.pool.Exec(ctx, `ALTER TABLE review_unavailable_credentials RENAME TO runtime_credentials`); err != nil {
		t.Fatal(err)
	}
	e = h.get(t, e.ID)
	pending, err := evalstore.NewPostgresStore(h.pool).PendingCommands(ctx, e.OwnerID, e.ID)
	if err != nil {
		t.Fatal(err)
	}
	if e.State != evaldomain.StatePreparing || len(pending) != 1 {
		t.Fatalf("transient credential DB failure: state=%s diagnostic=%s pending=%d; want preparing and one pending command", e.State, e.Diagnostic, len(pending))
	}
	tick(t, coordinator)
	if e = h.get(t, e.ID); e.State != evaldomain.StateReady {
		t.Fatalf("no automatic retry after recovery: %s", e.State)
	}
}

type credentialLookupFunc func(context.Context, string) (config.CredentialMetadata, error)

func (f credentialLookupFunc) LookupLLMCredential(ctx context.Context, id string) (config.CredentialMetadata, error) {
	return f(ctx, id)
}

func TestPostgresEvalPinnedLLMCredentialFailureClassification(t *testing.T) {
	transient := errors.New("fixture: credential service temporarily unavailable")
	for _, test := range []struct {
		name     string
		cause    error
		mismatch bool
	}{
		{name: "infrastructure", cause: transient},
		{name: "missing", cause: credentials.ErrNotFound},
		{name: "identity", mismatch: true},
	} {
		t.Run(test.name, func(t *testing.T) {
			h := newHarness(t)
			ctx := t.Context()
			publication, err := runtimeconfig.PreparePublication([]byte(`{"apiVersion":"contractor/v1alpha1","kind":"RuntimeConfig","metadata":{"name":"credential-probe","version":"1"},"spec":{"worker":{"llmGateway":{"credential":"runtime-probe"}}}}`))
			if err != nil {
				t.Fatal(err)
			}
			version, err := publication.Resolve(ctx, nil)
			if err != nil {
				t.Fatal(err)
			}
			version.ActorID, version.CreatedAt = "review", time.Now()
			repo := runtimeconfig.NewRepository(h.pool)
			if _, err = repo.InsertVersion(ctx, version); err != nil {
				t.Fatal(err)
			}
			if _, err = repo.Rebind(ctx, runtimeconfig.DefaultLabel, 1, version.Ref, "review", time.Now()); err != nil {
				t.Fatal(err)
			}

			factory := h.resolver.Credentials
			failure := test.cause
			h.resolver.Credentials = runtimeconfig.TransactionLLMCredentialLookupFactoryFunc(func(tx pgx.Tx) (config.CredentialLookup, error) {
				original, err := factory.ForTransaction(tx)
				if err != nil {
					return nil, err
				}
				return credentialLookupFunc(func(ctx context.Context, id string) (config.CredentialMetadata, error) {
					if id != "runtime-probe" {
						return original.LookupLLMCredential(ctx, id)
					}
					if test.mismatch {
						id = "wrong-identity"
					}
					return config.CredentialMetadata{Ref: contracts.LLMCredentialRef{CredentialID: id}}, failure
				}), nil
			})
			e := h.create(t, "audit")
			h.command(t, e, "prepare")
			coordinator := h.coordinator(t, "runtime-lookup")
			_, err = coordinator.RunOnce(ctx)
			if err == nil || test.cause != nil && !errors.Is(err, test.cause) {
				t.Fatalf("lost lookup failure: %v", err)
			}
			e = h.get(t, e.ID)
			pending, err := evalstore.NewPostgresStore(h.pool).PendingCommands(ctx, e.OwnerID, e.ID)
			if err != nil {
				t.Fatal(err)
			}
			if test.cause != transient {
				if e.State != evaldomain.StateDraft || len(pending) != 0 {
					t.Fatalf("invalid credential did not return to draft: %s pending=%d", e.State, len(pending))
				}
				return
			}
			if e.State != evaldomain.StatePreparing || len(pending) != 1 {
				t.Fatalf("transient lookup was rejected: %s pending=%d", e.State, len(pending))
			}
			failure = nil
			tick(t, coordinator)
			if e = h.get(t, e.ID); e.State != evaldomain.StateReady || len(e.Diagnostic) != 0 {
				t.Fatalf("lookup recovery: %s %s", e.State, e.Diagnostic)
			}
		})
	}
}
