package artifacts_test

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"
	"time"

	. "github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

func TestStorageReviewControlledCAS(t *testing.T) {
	for _, backend := range []string{"postgresql", "filesystem"} {
		t.Run(backend, func(t *testing.T) {
			t.Setenv("CONTRACTOR_TEST_ARTIFACT_BACKEND", backend)
			ctx, cancel := context.WithTimeout(t.Context(), 30*time.Second)
			defer cancel()
			pool := isolatedArtifactPool(t, ctx)
			ctx = testBlobContext(t, ctx, pool)
			store, _ := NewService(NewPostgresRepository(pool)).User("owner")
			target := ArtifactRef{Namespace: "docs", Name: "report"}
			initial, err := store.Write(ctx, target, Payload{MediaType: "text/plain", Data: []byte("initial")}, nil)
			if err != nil {
				t.Fatal(err)
			}
			first, err := pool.Begin(ctx)
			if err != nil {
				t.Fatal(err)
			}
			defer first.Rollback(context.Background())
			writer, _ := NewService(NewPostgresRepository(first)).User("owner")
			winning, err := writer.Write(ctx, target, Payload{MediaType: "text/plain", Data: []byte("winner")}, initial.Ref.Revision)
			if err != nil {
				t.Fatal(err)
			}
			second, err := pool.Begin(ctx)
			if err != nil {
				t.Fatal(err)
			}
			defer second.Rollback(context.Background())
			done := make(chan error, 1)
			go func() {
				loser, _ := NewService(NewPostgresRepository(second)).User("owner")
				_, err := loser.Write(ctx, target, Payload{MediaType: "text/plain", Data: []byte("loser")}, initial.Ref.Revision)
				done <- err
			}()
			waitArtifactLock(t, ctx, pool, second.Conn().PgConn().PID())
			if err := first.Commit(ctx); err != nil {
				t.Fatal(err)
			}
			if err := <-done; !errors.Is(err, ErrArtifactConflict) {
				t.Fatalf("stale writer: %v", err)
			}
			if err := second.Rollback(ctx); err != nil {
				t.Fatal(err)
			}
			for _, expected := range []struct {
				ref  ArtifactRef
				body string
			}{{initial.Ref, "initial"}, {winning.Ref, "winner"}, {target, "winner"}} {
				read, err := store.Read(ctx, expected.ref)
				if err != nil || string(read.Payload.Data) != expected.body {
					t.Fatalf("exact/current read = %q, %v", read.Payload.Data, err)
				}
			}
			var revisions int
			if err := pool.QueryRow(ctx, `SELECT count(*) FROM artifact_binding_revisions`).Scan(&revisions); err != nil || revisions != 2 {
				t.Fatalf("revisions=%d: %v", revisions, err)
			}
		})
	}
}

func TestStorageReviewProjectDeletionAdmission(t *testing.T) {
	for _, backend := range []string{"postgresql", "filesystem"} {
		for _, operation := range []string{"create", "update", "publication"} {
			for _, deletionFirst := range []bool{false, true} {
				winner := "mutation-first"
				if deletionFirst {
					winner = "deletion-first"
				}
				t.Run(backend+"/"+operation+"/"+winner, func(t *testing.T) {
					t.Setenv("CONTRACTOR_TEST_ARTIFACT_BACKEND", backend)
					ctx, cancel := context.WithTimeout(t.Context(), 30*time.Second)
					defer cancel()
					pool := isolatedArtifactPool(t, ctx)
					ctx = testBlobContext(t, ctx, pool)
					project, source := storageReviewProject(t, ctx, pool)
					store, _ := NewService(NewPostgresRepository(pool)).Project(project.ProjectID)
					target := ArtifactRef{Namespace: "docs", Name: "content"}
					var prior *string
					if operation == "update" {
						initial, err := store.Write(ctx, target, Payload{MediaType: "text/plain", Data: []byte("initial")}, nil)
						if err != nil {
							t.Fatal(err)
						}
						prior = initial.Ref.Revision
					}
					mutate := func(tx pgx.Tx) error {
						service := NewService(NewPostgresRepository(tx))
						if operation == "publication" {
							_, err := service.PublishRunOutput(ctx, "review-run", project.ProjectID, "result", source)
							return err
						}
						p, _ := service.Project(project.ProjectID)
						_, err := p.Write(ctx, target, Payload{MediaType: "text/plain", Data: []byte("accepted")}, prior)
						return err
					}
					remove := func(tx pgx.Tx) error {
						_, _, err := projectstore.NewPostgresStore(tx).BeginDeletion(ctx, projectstore.BeginDeletionParams{OwnerID: project.OwnerID, ProjectID: project.ProjectID, ExpectedRevision: project.Revision})
						return err
					}
					first, err := pool.Begin(ctx)
					if err != nil {
						t.Fatal(err)
					}
					defer first.Rollback(context.Background())
					second, err := pool.Begin(ctx)
					if err != nil {
						t.Fatal(err)
					}
					defer second.Rollback(context.Background())
					firstAction, secondAction := mutate, remove
					if deletionFirst {
						firstAction, secondAction = remove, mutate
					}
					if err := firstAction(first); err != nil {
						t.Fatal(err)
					}
					done := make(chan error, 1)
					go func() { done <- secondAction(second) }()
					waitArtifactLock(t, ctx, pool, second.Conn().PgConn().PID())
					if err := first.Commit(ctx); err != nil {
						t.Fatal(err)
					}
					secondErr := <-done
					if deletionFirst {
						if !errors.Is(secondErr, ErrScopeDeleting) {
							t.Fatalf("mutation after deletion: %v", secondErr)
						}
						if err := second.Rollback(ctx); err != nil {
							t.Fatal(err)
						}
					} else {
						if secondErr != nil {
							t.Fatal(secondErr)
						}
						if err := second.Commit(ctx); err != nil {
							t.Fatal(err)
						}
					}
					if operation == "publication" {
						target = ArtifactRef{Namespace: "outputs", Name: "result"}
					}
					read, err := store.Read(ctx, target)
					if deletionFirst && operation != "update" {
						if !errors.Is(err, ErrArtifactNotFound) {
							t.Fatalf("rejected mutation published data: %v", err)
						}
					} else {
						want := "accepted"
						if deletionFirst {
							want = "initial"
						}
						if err != nil || string(read.Payload.Data) != want {
							t.Fatalf("committed content=%q, want %q: %v", read.Payload.Data, want, err)
						}
					}
					actual, err := projectstore.NewPostgresStore(pool).Get(ctx, project.OwnerID, project.ProjectID)
					if err != nil || actual.Lifecycle != projectstore.LifecycleDeleting {
						t.Fatalf("deletion state: %+v %v", actual, err)
					}
				})
			}
		}
	}
}

func storageReviewProject(t *testing.T, ctx context.Context, pool *pgxpool.Pool) (projectstore.Project, ArtifactRef) {
	t.Helper()
	p, _, err := projectstore.NewPostgresStore(pool).Create(ctx, projectstore.CreateParams{ProjectID: "review-project", OwnerID: "user-1", Kind: projectstore.KindProject, Name: "Review", IdempotencyKey: "review", RequestDigest: "sha256:" + strings.Repeat("a", 64)})
	if err != nil {
		t.Fatal(err)
	}
	_, err = runstore.NewPostgresStore(pool).CreateRun(ctx, runstore.CreateRunParams{RunID: "review-run", OwnerID: p.OwnerID, ProjectID: &p.ProjectID, WorkflowName: "copy", WorkflowVersion: "1", WorkflowSchemaVersion: "contractor/v1alpha1", WorkflowSnapshot: json.RawMessage(`{}`), RuntimeConfig: runtimeconfig.BuiltInRunSnapshot()})
	if err != nil {
		t.Fatal(err)
	}
	s := NewService(NewPostgresRepository(pool))
	run, _ := s.Run("review-run")
	written, err := run.Write(ctx, ArtifactRef{Namespace: "scratch", Name: "source"}, Payload{MediaType: "text/plain", Data: []byte("accepted")}, nil)
	if err != nil {
		t.Fatal(err)
	}
	bound, err := s.BindOutputExact(ctx, "review-run", "result", written.Ref, nil)
	if err != nil {
		t.Fatal(err)
	}
	if err := s.FreezeRunOutputs(ctx, "review-run"); err != nil {
		t.Fatal(err)
	}
	return p, bound.TargetRef
}

func TestStorageReviewSingleConnectionCleanup(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 30*time.Second)
	defer cancel()
	base := isolatedArtifactPool(t, ctx)
	config := base.Config()
	config.MaxConns = 1
	pool, err := pgxpool.NewWithConfig(ctx, config)
	if err != nil {
		t.Fatal(err)
	}
	defer pool.Close()
	if err := ClaimBlobBackend(ctx, pool, BlobFilesystem); err != nil {
		t.Fatal(err)
	}
	path := t.TempDir()
	files, err := OpenFilesystemBlobStore(ctx, path)
	if err != nil {
		t.Fatal(err)
	}
	defer files.Close()
	ctx = WithBlobRuntime(ctx, NewBlobRuntime(files, nil))
	createArtifactRun(t, ctx, pool, "minimal-pool", true)
	run, _ := NewService(NewPostgresRepository(pool)).Run("minimal-pool")
	ref := ArtifactRef{Namespace: "scratch", Name: "content"}
	first, err := run.Write(ctx, ref, Payload{MediaType: "text/plain", Data: []byte("same bytes")}, nil)
	if err != nil {
		t.Fatal(err)
	}
	second, err := run.Write(ctx, ref, Payload{MediaType: "text/plain", Data: []byte("same bytes")}, first.Ref.Revision)
	if err != nil {
		t.Fatal(err)
	}
	for _, exact := range []ArtifactRef{first.Ref, second.Ref} {
		if _, err := run.Read(ctx, exact); err != nil {
			t.Fatal(err)
		}
	}
	if _, err := runstore.NewPostgresStore(pool).TransitionRun(ctx, "minimal-pool", runstore.RunRunning, runstore.RunSucceeded, runstore.Reason{Code: "done"}); err != nil {
		t.Fatal(err)
	}
	if err := runstore.NewPostgresStore(pool).DeleteReleasedTerminalRun(ctx, "user-1", "minimal-pool"); err != nil {
		t.Fatal(err)
	}
	report, err := CleanupFilesystemBlobs(ctx, pool, path, true)
	if err != nil || report.Referenced != 0 || report.Orphans != 0 {
		t.Fatalf("cleanup with one connection: %+v, %v", report, err)
	}
	if pool.Stat().AcquiredConns() != 0 {
		t.Fatal("cleanup retained a database connection")
	}
}
