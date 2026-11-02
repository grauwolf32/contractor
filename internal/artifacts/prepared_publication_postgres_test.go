package artifacts_test

import (
	"context"
	"errors"
	"testing"

	. "github.com/grauwolf32/contractor/internal/artifacts"
	postgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/jackc/pgx/v5"
)

type publicationNoRead struct{ BlobStore }

func (publicationNoRead) Read(context.Context, BlobObject) ([]byte, error) {
	return nil, errors.New("publication attempted a full payload read")
}

func TestPostgresPreparedPublicationDoesNotReadPayloadInsideTransaction(t *testing.T) {
	for _, changed := range []bool{false, true} {
		t.Run(map[bool]string{false: "verified-generation", true: "changed-generation"}[changed], func(t *testing.T) {
			f := newDedupFixture(t)
			prepared, err := PreparePayload(f.ctx, f.pool, f.payload)
			if err != nil {
				t.Fatal(err)
			}
			if changed {
				// A competing generation has the right recorded size but different
				// bytes. Publication must attach its own prepared candidate instead.
				other, err := f.files.Store(f.ctx, make([]byte, len(f.payload.Data)))
				if err != nil {
					t.Fatal(err)
				}
				if _, err := f.pool.Exec(f.ctx, `UPDATE artifact_blobs SET object_key=$1`, other.Key); err != nil {
					t.Fatal(err)
				}
			}
			ctx := WithBlobRuntime(f.ctx, NewBlobRuntime(publicationNoRead{BlobStore: f.files}, nil))
			err = postgres.InTx(ctx, f.pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
				store, err := NewService(NewPostgresRepository(tx)).User("user-1")
				if err != nil {
					return err
				}
				_, err = store.Write(ctx, ArtifactRef{Namespace: "docs", Name: "prepared-publication"}, prepared, nil)
				return err
			})
			if err != nil {
				t.Fatal(err)
			}
			read, err := f.store.Read(f.ctx, ArtifactRef{Namespace: "docs", Name: "prepared-publication"})
			if err != nil || string(read.Payload.Data) != string(f.payload.Data) {
				t.Fatalf("prepared bytes: %v", err)
			}
		})
	}
}
