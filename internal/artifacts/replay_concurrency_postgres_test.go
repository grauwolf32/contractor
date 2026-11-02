package artifacts_test

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"testing"
	"time"

	. "github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/runstore/runstoretest"
	"github.com/grauwolf32/contractor/internal/runtimeconfig/runtimeconfigtest"
)

// Force the second publication to wait for the identical uncommitted winner.
func TestPostgresConcurrentReplayFork(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()
	pool := isolatedArtifactPool(t, ctx)
	ctx = testBlobContext(t, ctx, pool)
	createArtifactRun(t, ctx, pool, "review-fork", false)
	service := NewService(NewPostgresRepository(pool))
	run, _ := service.Run("review-fork")
	written, err := run.Write(ctx, ArtifactRef{Namespace: "data", Name: "source"}, Payload{MediaType: "text/plain", Data: []byte("source")}, nil)
	if err != nil {
		t.Fatal(err)
	}
	scope, _ := RunScope("review-fork")
	fork := Fork{Kind: LineageInputFork, Source: scope, SourceRef: written.Ref, Target: scope, TargetRef: ArtifactRef{Namespace: "inputs", Name: "snapshot"}, Frozen: true, Replay: true}
	tx, err := pool.Begin(ctx)
	if err != nil {
		t.Fatal(err)
	}
	defer tx.Rollback(context.Background())
	first, err := NewService(NewPostgresRepository(tx)).Fork(ctx, fork)
	if err != nil {
		t.Fatal(err)
	}
	secondConn, err := pool.Acquire(ctx)
	if err != nil {
		t.Fatal(err)
	}
	defer secondConn.Release()
	pid := secondConn.Conn().PgConn().PID()
	done := make(chan error, 1)
	go func() {
		second, err := NewService(NewPostgresRepository(secondConn.Conn())).Fork(ctx, fork)
		if err == nil && !second.TargetRef.SameExact(first.TargetRef) {
			err = fmt.Errorf("different replay revisions")
		}
		done <- err
	}()
	for {
		var blocked bool
		if err := pool.QueryRow(ctx, `SELECT cardinality(pg_blocking_pids($1)) > 0`, pid).Scan(&blocked); err != nil {
			t.Fatal(err)
		}
		if blocked {
			break
		}
		select {
		case err := <-done:
			t.Fatalf("second finished before winner commit: %v", err)
		case <-ctx.Done():
			t.Fatal(ctx.Err())
		case <-time.After(time.Millisecond):
		}
	}
	if err := tx.Commit(ctx); err != nil {
		t.Fatal(err)
	}
	if err := <-done; err != nil {
		t.Fatalf("identical concurrent replay should converge: %v", err)
	}
	other, err := run.Write(ctx, ArtifactRef{Namespace: "data", Name: "other"}, Payload{MediaType: "text/plain", Data: []byte("source")}, nil)
	if err != nil {
		t.Fatal(err)
	}
	fork.SourceRef = other.Ref
	if _, err := service.Fork(ctx, fork); !errors.Is(err, ErrArtifactConflict) {
		t.Fatalf("a different exact source must conflict: %v", err)
	}
}

func TestPostgresConcurrentContextProjection(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()
	pool := isolatedArtifactPool(t, ctx)
	ctx = testBlobContext(t, ctx, pool)
	projectID := "review-project"
	_, _, err := projectstore.NewPostgresStore(pool).Create(ctx, projectstore.CreateParams{ProjectID: projectID, OwnerID: "user-1", Kind: projectstore.KindProject, Name: "Review", IdempotencyKey: "review-project", RequestDigest: "sha256:" + strings.Repeat("0", 64)})
	if err != nil {
		t.Fatal(err)
	}
	_, err = runstoretest.CreateRun(ctx, runstore.NewPostgresStore(pool), runstore.CreateRunParams{LaneClass: resolved.LaneClassModel, RunID: "review-projection", OwnerID: "user-1", ProjectID: &projectID, WorkflowName: "artifact-copy", WorkflowVersion: "1", WorkflowSchemaVersion: "contractor/v1alpha1", WorkflowSnapshot: json.RawMessage(`{"ref":{"name":"artifact-copy","version":"1"}}`), RuntimeConfig: runtimeconfigtest.BuiltInRunSnapshot()})
	if err != nil {
		t.Fatal(err)
	}
	service := NewService(NewPostgresRepository(pool))
	project, _ := service.Project(projectID)
	original, err := project.Write(ctx, ArtifactRef{Namespace: "data", Name: "source"}, Payload{MediaType: "text/plain", Data: []byte("source")}, nil)
	if err != nil {
		t.Fatal(err)
	}
	run, _ := service.Run("review-projection")
	_, err = run.Write(ctx, ArtifactRef{Namespace: "data", Name: "seed"}, Payload{MediaType: "text/plain", Data: []byte("seed")}, nil)
	if err != nil {
		t.Fatal(err)
	}
	source, _ := ProjectScope(projectID)
	target, _ := RunScope("review-projection")
	ref := ArtifactRef{Namespace: "context", Name: "snapshot"}
	payload := Payload{MediaType: "text/plain", Data: []byte("projection")}
	tx, err := pool.Begin(ctx)
	if err != nil {
		t.Fatal(err)
	}
	defer tx.Rollback(context.Background())
	_, _, err = NewPostgresTxRepository(tx).CreateContextProjection(ctx, source, original.Ref, target, ref, payload)
	if err != nil {
		t.Fatal(err)
	}
	secondConn, err := pool.Acquire(ctx)
	if err != nil {
		t.Fatal(err)
	}
	defer secondConn.Release()
	secondTx, err := secondConn.Begin(ctx)
	if err != nil {
		t.Fatal(err)
	}
	defer secondTx.Rollback(context.Background())
	done := make(chan error, 1)
	go func() {
		_, replay, err := NewPostgresTxRepository(secondTx).CreateContextProjection(ctx, source, original.Ref, target, ref, payload)
		if err == nil && !replay {
			err = fmt.Errorf("concurrent result was not marked as replay")
		}
		done <- err
	}()
	pid := secondConn.Conn().PgConn().PID()
	for {
		var blocked bool
		if err := pool.QueryRow(ctx, `SELECT cardinality(pg_blocking_pids($1))>0`, pid).Scan(&blocked); err != nil {
			t.Fatal(err)
		}
		if blocked {
			break
		}
		select {
		case err := <-done:
			t.Fatalf("second finished before winner commit: %v", err)
		case <-ctx.Done():
			t.Fatal(ctx.Err())
		case <-time.After(time.Millisecond):
		}
	}
	if err := tx.Commit(ctx); err != nil {
		t.Fatal(err)
	}
	if err := <-done; err != nil {
		t.Fatalf("identical concurrent projection should replay: %v", err)
	}
	if err := secondTx.Commit(ctx); err != nil {
		t.Fatal(err)
	}
	other, err := project.Write(ctx, ArtifactRef{Namespace: "data", Name: "other"}, Payload{MediaType: "text/plain", Data: []byte("source")}, nil)
	if err != nil {
		t.Fatal(err)
	}
	conflicting, err := pool.Begin(ctx)
	if err != nil {
		t.Fatal(err)
	}
	defer conflicting.Rollback(context.Background())
	if _, _, err := NewPostgresTxRepository(conflicting).CreateContextProjection(ctx, source, other.Ref, target, ref, payload); !errors.Is(err, contracts.ErrIdempotencyConflict) {
		t.Fatalf("a different projection source must conflict: %v", err)
	}
	var edges int
	if err := pool.QueryRow(ctx, `SELECT count(*) FROM artifact_lineage`).Scan(&edges); err != nil || edges != 1 {
		t.Fatalf("projection lineage count = %d, err = %v", edges, err)
	}
}
