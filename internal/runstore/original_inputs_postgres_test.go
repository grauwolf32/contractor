package runstore

import (
	"context"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/jackc/pgx/v5/pgxpool"
)

func TestPostgresDescribeOriginalInputsKeepsCreationRevisionsInCallerTransaction(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 30*time.Second)
	defer cancel()
	base := isolatedRunStorePool(t, ctx)
	config := base.Config()
	config.MaxConns = 1
	pool, err := pgxpool.NewWithConfig(ctx, config)
	if err != nil {
		t.Fatal(err)
	}
	defer pool.Close()
	const runID = "original-inputs"
	createTestRun(t, ctx, NewPostgresStore(pool, artifactpgtest.NewStorage()), runID)
	original := storeTestRunInput(t, ctx, pool, runID, "source")
	artifactStore, _ := artifacts.NewService(artifactpgtest.NewStorage().Repository(pool)).Run(runID)
	want, err := artifactStore.Metadata(ctx, original)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := artifactStore.Write(ctx, contracts.ArtifactRef{Namespace: "inputs", Name: "source"}, artifacts.Payload{MediaType: "text/plain", Data: []byte("edited input")}, original.Revision); err != nil {
		t.Fatal(err)
	}
	current, err := artifactStore.Metadata(ctx, contracts.ArtifactRef{Namespace: "inputs", Name: "source"})
	if err != nil || current.Digest == want.Digest || current.Ref.SameExact(original) {
		t.Fatalf("working copy did not advance: %+v %v", current, err)
	}
	// With the pool's only connection held, any accidental nested acquisition
	// cannot complete. The RunStore reader must use the provided transaction.
	tx, err := pool.Begin(ctx)
	if err != nil {
		t.Fatal(err)
	}
	defer tx.Rollback(ctx) //nolint:errcheck // transaction may already be committed
	got, err := NewPostgresStore(tx, artifactpgtest.NewStorage()).DescribeOriginalInputs(ctx, runID, []string{"missing", "source"})
	if err != nil || len(got) != 1 || !got[0].Ref.SameExact(original) || want.Exact().Verify(got[0]) != nil {
		t.Fatalf("original input changed: %+v %v", got, err)
	}
	if err := tx.Commit(ctx); err != nil {
		t.Fatal(err)
	}
}
