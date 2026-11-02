package postgres_test

import (
	"context"
	"errors"
	"testing"

	. "github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/artifacts/blob"
	"github.com/grauwolf32/contractor/internal/artifacts/blob/backends/filesystem"
	inline "github.com/grauwolf32/contractor/internal/artifacts/blob/backends/postgres"
	artifactpostgres "github.com/grauwolf32/contractor/internal/artifacts/postgres"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/jackc/pgx/v5"
)

func TestInjectedStorageSurvivesTransactionBindingAndFreshContexts(t *testing.T) {
	pool := isolatedArtifactPool(t, t.Context())
	files, err := filesystem.Open(t.Context(), t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer files.Close()
	policy, err := artifactpostgres.NewHistoryPolicy(2, artifactpostgres.DefaultHistoryMaxBytes)
	if err != nil {
		t.Fatal(err)
	}
	storage, err := artifactpostgres.NewStorage(blob.NewRuntime(files, nil), policy)
	if err != nil {
		t.Fatal(err)
	}
	scope, _ := UserScope("injected-dependencies")
	ref := ArtifactRef{Namespace: "notes", Name: "retained"}
	user, _ := NewService(storage.Repository(pool)).User(scope.ID())
	payload := func(value string) Payload { return Payload{MediaType: "text/plain", Data: []byte(value)} }
	first, err := user.Write(context.Background(), ref, payload("first"), nil)
	if err != nil {
		t.Fatal(err)
	}
	second, err := user.Write(context.Background(), ref, payload("second"), first.Ref.Revision)
	if err != nil {
		t.Fatal(err)
	}
	rolledBack := errors.New("rollback publication")
	err = persistencepostgres.InTx(t.Context(), pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		if _, err := storage.TxRepository(tx).Write(context.Background(), scope, ref, payload("rolled back"), second.Ref.Revision); err != nil {
			return err
		}
		return rolledBack
	})
	if !errors.Is(err, rolledBack) {
		t.Fatal(err)
	}
	if read, err := user.Read(context.Background(), first.Ref); err != nil || string(read.Payload.Data) != "first" {
		t.Fatalf("rollback changed retained bytes: %v", err)
	}
	if read, err := user.Read(context.Background(), ref); err != nil || string(read.Payload.Data) != "second" {
		t.Fatalf("rollback changed current: %v", err)
	}
	var admitted context.Context
	releases := make([]func(), 0, 4)
	for range 4 {
		lease, release, err := storage.AcquireTransfer(context.Background())
		if err != nil {
			t.Fatal(err)
		}
		admitted = lease
		releases = append(releases, release)
		defer release()
	}
	if _, err := user.Read(context.Background(), ref); !errors.Is(err, blob.ErrTransferCapacity) {
		t.Fatalf("fresh read bypassed admission: %v", err)
	}
	err = persistencepostgres.InTx(t.Context(), pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		bound := storage.TxRepository(tx)
		if _, err := bound.Read(context.Background(), scope, ref); !errors.Is(err, blob.ErrTransferCapacity) {
			t.Fatalf("transaction bypassed admission: %v", err)
		}
		// A nested read shares the outer lease even when all four slots are held.
		_, err := bound.Read(admitted, scope, ref)
		return err
	})
	if err != nil {
		t.Fatal(err)
	}
	for _, release := range releases {
		release()
	}
	err = persistencepostgres.InTx(t.Context(), pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		_, err := storage.TxRepository(tx).Write(context.Background(), scope, ref, payload("third"), second.Ref.Revision)
		return err
	})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := user.Read(context.Background(), first.Ref); !errors.Is(err, ErrArtifactNotFound) {
		t.Fatalf("transaction lost configured history: %v", err)
	}
	if read, err := user.Read(context.Background(), ref); err != nil || string(read.Payload.Data) != "third" {
		t.Fatalf("committed bytes: %v", err)
	}
	var count int
	if err := pool.QueryRow(t.Context(), `SELECT count(*) FROM artifact_blobs WHERE backend <> 'filesystem'`).Scan(&count); err != nil || count != 0 {
		t.Fatalf("transaction lost backend: %d %v", count, err)
	}
}

func TestStorageRejectsMissingProcessDependencies(t *testing.T) {
	for _, runtime := range []*blob.Runtime{nil, blob.NewRuntime(nil, nil)} {
		if storage, err := artifactpostgres.NewStorage(runtime, artifactpostgres.DefaultHistoryPolicy()); err == nil || storage != nil {
			t.Fatalf("missing physical store accepted: %v", err)
		}
	}
	runtime := blob.NewRuntime(inline.Store{}, nil)
	if storage, err := artifactpostgres.NewStorage(runtime, artifactpostgres.HistoryPolicy{}); !errors.Is(err, artifactpostgres.ErrInvalidHistoryPolicy) || storage != nil {
		t.Fatalf("zero policy accepted: %v", err)
	}
}
