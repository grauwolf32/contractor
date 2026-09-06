package artifacts_test

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"testing"
	"time"

	. "github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

func TestPostgresConcurrentPurgeSharedBlob(t *testing.T) {
	for _, rollback := range []bool{false, true} {
		t.Run(fmt.Sprintf("rollback=%t", rollback), func(t *testing.T) {
			ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
			defer cancel()
			pool := isolatedArtifactPool(t, ctx)
			refs := make([]ArtifactRef, 2)
			for i, id := range []string{"purge-a", "purge-b"} {
				createArtifactRun(t, ctx, pool, id, true)
				run, _ := NewService(NewPostgresRepository(pool)).Run(id)
				written, err := run.Write(ctx, ArtifactRef{Namespace: "scratch", Name: "same"}, Payload{MediaType: "text/plain", Data: []byte("shared")}, nil)
				if err != nil {
					t.Fatal(err)
				}
				refs[i] = written.Ref
				if _, err := runstore.NewPostgresStore(pool).TransitionRun(ctx, id, runstore.RunRunning, runstore.RunSucceeded, runstore.Reason{Code: "done"}); err != nil {
					t.Fatal(err)
				}
			}
			first, err := pool.Begin(ctx)
			if err != nil {
				t.Fatal(err)
			}
			defer first.Rollback(context.Background())
			if err := runstore.NewPostgresStore(first).DeleteReleasedTerminalRun(ctx, "user-1", "purge-a"); err != nil {
				t.Fatal(err)
			}
			second, err := pool.Begin(ctx)
			if err != nil {
				t.Fatal(err)
			}
			defer second.Rollback(context.Background())
			done := make(chan error, 1)
			go func() {
				err := runstore.NewPostgresStore(second).DeleteReleasedTerminalRun(ctx, "user-1", "purge-b")
				if err == nil {
					err = second.Commit(ctx)
				}
				done <- err
			}()
			waitArtifactLock(t, ctx, pool, second.Conn().PgConn().PID())
			if rollback {
				err = first.Rollback(ctx)
			} else {
				err = first.Commit(ctx)
			}
			if err != nil {
				t.Fatal(err)
			}
			if err := <-done; err != nil {
				t.Fatal(err)
			}
			var runs, versions, blobs int
			if err := pool.QueryRow(ctx, `SELECT (SELECT count(*) FROM workflow_runs), (SELECT count(*) FROM artifact_versions), (SELECT count(*) FROM artifact_blobs)`).Scan(&runs, &versions, &blobs); err != nil {
				t.Fatal(err)
			}
			want := 0
			if rollback {
				want = 1
			}
			if runs != want || versions != want || blobs != want {
				t.Fatalf("runs/versions/blobs = %d/%d/%d, want %d each", runs, versions, blobs, want)
			}
			if rollback {
				run, _ := NewService(NewPostgresRepository(pool)).Run("purge-a")
				if content, err := run.Read(ctx, refs[0]); err != nil || string(content.Payload.Data) != "shared" {
					t.Fatalf("retained content: %v %v", content, err)
				}
			}
		})
	}
}

func TestPostgresPurgeConcurrentIdenticalWrite(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	pool := isolatedArtifactPool(t, ctx)
	createArtifactRun(t, ctx, pool, "purge-write", true)
	run, _ := NewService(NewPostgresRepository(pool)).Run("purge-write")
	payload := Payload{MediaType: "text/plain", Data: []byte("shared")}
	if _, err := run.Write(ctx, ArtifactRef{Namespace: "scratch", Name: "same"}, payload, nil); err != nil {
		t.Fatal(err)
	}
	writer, err := pool.Begin(ctx)
	if err != nil {
		t.Fatal(err)
	}
	defer writer.Rollback(context.Background())
	user, _ := NewService(NewPostgresRepository(writer)).User("survivor")
	ref, err := user.Write(ctx, ArtifactRef{Namespace: "keep", Name: "same"}, payload, nil)
	if err != nil {
		t.Fatal(err)
	}
	purge, err := pool.Begin(ctx)
	if err != nil {
		t.Fatal(err)
	}
	defer purge.Rollback(context.Background())
	done := make(chan error, 1)
	go func() {
		p, _ := NewPostgresPurger(purge)
		err := p.PurgeRun(ctx, "purge-write")
		if err == nil {
			err = purge.Commit(ctx)
		}
		done <- err
	}()
	waitArtifactLock(t, ctx, pool, purge.Conn().PgConn().PID())
	if err := writer.Commit(ctx); err != nil {
		t.Fatal(err)
	}
	if err := <-done; err != nil {
		t.Fatal(err)
	}
	reader, _ := NewService(NewPostgresRepository(pool)).User("survivor")
	if got, err := reader.Read(ctx, ref.Ref); err != nil || string(got.Payload.Data) != "shared" {
		t.Fatalf("surviving write: %v %v", got, err)
	}
	var orphan int
	if err := pool.QueryRow(ctx, `SELECT count(*) FROM artifact_blobs b WHERE NOT EXISTS (SELECT 1 FROM artifact_versions v WHERE v.blob_sha256=b.sha256)`).Scan(&orphan); err != nil || orphan != 0 {
		t.Fatalf("orphan blobs %d: %v", orphan, err)
	}
}

func TestPostgresPurgeWaitsForRetainingPublication(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	pool := isolatedArtifactPool(t, ctx)
	projectID := "keep-project"
	if _, _, err := projectstore.NewPostgresStore(pool).Create(ctx, projectstore.CreateParams{
		ProjectID: projectID, OwnerID: "user-1", Kind: projectstore.KindProject, Name: "Keep",
		IdempotencyKey: "keep", RequestDigest: "sha256:" + strings.Repeat("a", 64),
	}); err != nil {
		t.Fatal(err)
	}
	runs := runstore.NewPostgresStore(pool)
	if _, err := runs.CreateRun(ctx, runstore.CreateRunParams{
		RunID: "publish-run", OwnerID: "user-1", ProjectID: &projectID, WorkflowName: "copy", WorkflowVersion: "1",
		WorkflowSchemaVersion: contracts.APIVersion, WorkflowSnapshot: json.RawMessage(`{}`), RuntimeConfig: runtimeconfig.BuiltInRunSnapshot(),
	}); err != nil {
		t.Fatal(err)
	}
	service := NewService(NewPostgresRepository(pool))
	run, _ := service.Run("publish-run")
	written, err := run.Write(ctx, ArtifactRef{Namespace: "scratch", Name: "source"}, Payload{MediaType: "text/plain", Data: []byte("retained")}, nil)
	if err != nil {
		t.Fatal(err)
	}
	bound, err := service.BindOutputExact(ctx, "publish-run", "result", written.Ref, nil)
	if err != nil {
		t.Fatal(err)
	}
	if err := service.FreezeRunOutputs(ctx, "publish-run"); err != nil {
		t.Fatal(err)
	}
	fork, err := pool.Begin(ctx)
	if err != nil {
		t.Fatal(err)
	}
	defer fork.Rollback(context.Background())
	publication, err := NewService(NewPostgresRepository(fork)).PublishRunOutput(ctx, "publish-run", projectID, "result", bound.TargetRef)
	if err != nil {
		t.Fatal(err)
	}
	purge, err := pool.Begin(ctx)
	if err != nil {
		t.Fatal(err)
	}
	defer purge.Rollback(context.Background())
	done := make(chan error, 1)
	go func() {
		p, _ := NewPostgresPurger(purge)
		err := p.PurgeRun(ctx, "publish-run")
		if err == nil {
			err = purge.Commit(ctx)
		}
		done <- err
	}()
	waitArtifactLock(t, ctx, pool, purge.Conn().PgConn().PID())
	if err := fork.Commit(ctx); err != nil {
		t.Fatal(err)
	}
	if err := <-done; err != nil {
		t.Fatal(err)
	}
	project, _ := service.Project(projectID)
	if got, err := project.Read(ctx, publication.TargetRef); err != nil || string(got.Payload.Data) != "retained" {
		t.Fatalf("retained publication = %v, %v", got, err)
	}
}

func TestPostgresPurgeRequiresFreshStatementSnapshots(t *testing.T) {
	ctx := context.Background()
	pool := isolatedArtifactPool(t, ctx)
	tx, err := pool.BeginTx(ctx, pgx.TxOptions{IsoLevel: pgx.RepeatableRead})
	if err != nil {
		t.Fatal(err)
	}
	defer tx.Rollback(ctx)
	p, _ := NewPostgresPurger(tx)
	if err := p.PurgeRun(ctx, "run-test"); err == nil {
		t.Fatal("repeatable-read purge accepted")
	}
}

func waitArtifactLock(t *testing.T, ctx context.Context, pool *pgxpool.Pool, pid uint32) {
	t.Helper()
	deadline := time.NewTimer(3 * time.Second)
	defer deadline.Stop()
	tick := time.NewTicker(5 * time.Millisecond)
	defer tick.Stop()
	for {
		var waiting bool
		if err := pool.QueryRow(ctx, `SELECT EXISTS(SELECT 1 FROM pg_stat_activity WHERE pid=$1 AND wait_event_type='Lock')`, pid).Scan(&waiting); err != nil {
			t.Fatal(err)
		}
		if waiting {
			return
		}
		select {
		case <-ctx.Done():
			t.Fatal(ctx.Err())
		case <-deadline.C:
			t.Fatal("operation never waited for content lock")
		case <-tick.C:
		}
	}
}
