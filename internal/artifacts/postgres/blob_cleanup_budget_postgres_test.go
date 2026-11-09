package postgres_test

import (
	"context"
	"fmt"
	"testing"

	. "github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/artifacts/blob"
	"github.com/grauwolf32/contractor/internal/artifacts/blob/backends/filesystem"
	artifactpostgres "github.com/grauwolf32/contractor/internal/artifacts/postgres"
	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"
	"github.com/grauwolf32/contractor/internal/rundeletion"
	"github.com/grauwolf32/contractor/internal/runstore"
)

type cleanupContextProbe struct {
	blob.Storage
	contexts []context.Context
}

func (p *cleanupContextProbe) Delete(ctx context.Context, object blob.Object) error {
	p.contexts = append(p.contexts, ctx)
	return p.Storage.Delete(ctx, object)
}

// The real registry and filesystem path must pass one transaction budget to
// every object, rather than restarting a timeout for every unlink. The pure
// commit test separately exhausts an injected budget over 1000 callbacks.
func TestFilesystemPurgeSharesPostcommitCleanupBudget(t *testing.T) {
	ctx := t.Context()
	pool := isolatedArtifactPool(t, ctx)
	path := t.TempDir()
	files, err := filesystem.Open(ctx, path)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = files.Close() })
	probe := &cleanupContextProbe{Storage: files}
	storage := artifactpgtest.NewConfigured(blob.NewRuntime(probe, nil), artifactpostgres.DefaultHistoryPolicy())
	const runID = "shared-cleanup-budget"
	createArtifactRun(t, ctx, pool, runID, true)
	run, _ := NewService(storage.Repository(pool)).Run(runID)
	const count = 64
	for index := range count {
		if _, err := run.Write(ctx, ArtifactRef{Namespace: "scratch", Name: fmt.Sprintf("item-%d", index)},
			Payload{MediaType: "text/plain", Data: []byte(fmt.Sprintf("unique content %d", index))}, nil); err != nil {
			t.Fatal(err)
		}
	}
	if len(probe.contexts) != 0 {
		t.Fatal("unique writes unexpectedly unlinked payloads")
	}
	if _, err := runstore.NewPostgresStore(pool, storage).TransitionRun(ctx, runID, runstore.RunRunning, runstore.RunSucceeded, runstore.Reason{Code: "done"}); err != nil {
		t.Fatal(err)
	}
	if err := rundeletion.New(pool, storage).DeleteReleasedTerminalRun(ctx, "user-1", runID); err != nil {
		t.Fatal(err)
	}
	if len(probe.contexts) != count {
		t.Fatalf("unlinked %d objects, want %d", len(probe.contexts), count)
	}
	for _, cleanup := range probe.contexts {
		if cleanup != probe.contexts[0] {
			t.Fatal("postcommit unlink restarted its budget")
		}
		if _, bounded := cleanup.Deadline(); !bounded {
			t.Fatal("postcommit cleanup has no shared deadline")
		}
	}
	var remaining int
	if err := pool.QueryRow(ctx, `SELECT count(*) FROM artifact_blobs`).Scan(&remaining); err != nil || remaining != 0 {
		t.Fatalf("remaining registry objects: %d, %v", remaining, err)
	}
	report, err := artifactpostgres.CleanupFilesystemBlobs(ctx, pool, path, false)
	if err != nil || report.Orphans != 0 || report.Missing != 0 || report.Referenced != 0 {
		t.Fatalf("filesystem cleanup inventory: %+v, %v", report, err)
	}
	t.Logf("one committed Run purge unlinked %d filesystem objects with one shared context", count)
}
