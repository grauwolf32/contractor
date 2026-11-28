package postgres_test

import (
	"context"
	"errors"
	"testing"
	"time"

	. "github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/grauwolf32/contractor/internal/rundeletion"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

// These schedules correspond to V498-005's finite model. They exercise the
// actual SQL wait, not a timing assumption. Direct Purger cases isolate the
// ArtifactStore boundary; the composed test below checks lifecycle ownership.
func TestPostgresPublicationPilotSourcePurge(t *testing.T) {
	for _, firstAction := range []string{"publication", "purge"} {
		for _, outcome := range []string{"commit", "rollback"} {
			t.Run(firstAction+"/"+outcome, func(t *testing.T) {
				ctx, cancel := context.WithTimeout(t.Context(), 30*time.Second)
				defer cancel()
				pool := isolatedArtifactPool(t, ctx)
				storage := artifactpgtest.NewStorage()
				project, source := storageReviewProject(t, ctx, pool, storage)
				var published ForkResult
				publish := func(tx pgx.Tx) error {
					var err error
					published, err = publishOutput(ctx, NewService(storage.Repository(tx)), "review-run", project.ProjectID, "result", source)
					return err
				}
				purge := func(tx pgx.Tx) error {
					p, err := storage.Purger(tx)
					if err != nil {
						return err
					}
					return p.PurgeRun(ctx, "review-run")
				}
				first, second := publish, purge
				if firstAction == "purge" {
					first, second = purge, publish
				}
				secondErr := publicationPilotRace(t, ctx, pool, first, second, outcome == "rollback")
				if firstAction == "purge" && outcome == "commit" {
					if !errors.Is(secondErr, ErrArtifactNotFound) {
						t.Fatalf("publication after source purge: %v", secondErr)
					}
				} else if secondErr != nil {
					t.Fatal(secondErr)
				}
				retained := firstAction == "publication" && outcome == "commit" || firstAction == "purge" && outcome == "rollback"
				service := NewService(storage.Repository(pool))
				target, _ := service.Project(project.ProjectID)
				ref := ArtifactRef{Namespace: "outputs", Name: "result"}
				if retained {
					ref = published.TargetRef
				}
				got, err := target.Read(ctx, ref)
				if retained {
					if err != nil || string(got.Payload.Data) != "accepted" || published.SourceRef.Revision == nil || *published.SourceRef.Revision != *source.Revision {
						t.Fatalf("retained exact publication: %+v, %v", got, err)
					}
				} else if !errors.Is(err, ErrArtifactNotFound) {
					t.Fatalf("uncommitted publication survived: %v", err)
				}
				sourceSurvives := firstAction == "purge" && outcome == "rollback"
				run, _ := service.Run("review-run")
				_, err = run.Read(ctx, source)
				if sourceSurvives && err != nil || !sourceSurvives && !errors.Is(err, ErrArtifactNotFound) {
					t.Fatalf("source survives=%t: %v", sourceSurvives, err)
				}
				var lineage, versions, blobs int
				if err := pool.QueryRow(ctx, `SELECT (SELECT count(*) FROM artifact_lineage),
(SELECT count(*) FROM artifact_versions), (SELECT count(*) FROM artifact_blobs)`).Scan(&lineage, &versions, &blobs); err != nil {
					t.Fatal(err)
				}
				wantLineage, wantContent := 0, 0
				if sourceSurvives {
					wantLineage = 2 // output_bind and project_output_publish
				}
				if retained {
					wantContent = 1
				}
				if lineage != wantLineage || versions != wantContent || blobs != wantContent {
					t.Fatalf("lineage/versions/blobs=%d/%d/%d, want %d/%d/%d", lineage, versions, blobs, wantLineage, wantContent, wantContent)
				}
			})
		}
	}
}

func TestPostgresPublicationPilotProjectFence(t *testing.T) {
	for _, firstAction := range []string{"publication", "deletion"} {
		for _, outcome := range []string{"commit", "rollback"} {
			t.Run(firstAction+"/"+outcome, func(t *testing.T) {
				ctx, cancel := context.WithTimeout(t.Context(), 30*time.Second)
				defer cancel()
				pool := isolatedArtifactPool(t, ctx)
				storage := artifactpgtest.NewStorage()
				project, source := storageReviewProject(t, ctx, pool, storage)
				publish := func(tx pgx.Tx) error {
					_, err := publishOutput(ctx, NewService(storage.Repository(tx)), "review-run", project.ProjectID, "result", source)
					return err
				}
				remove := func(tx pgx.Tx) error {
					_, _, err := projectstore.NewPostgresStore(tx).BeginDeletion(ctx, projectstore.BeginDeletionParams{
						OwnerID: project.OwnerID, ProjectID: project.ProjectID, ExpectedRevision: project.Revision,
					})
					return err
				}
				first, second := publish, remove
				if firstAction == "deletion" {
					first, second = remove, publish
				}
				err := publicationPilotRace(t, ctx, pool, first, second, outcome == "rollback")
				if firstAction == "deletion" && outcome == "commit" {
					if !errors.Is(err, ErrScopeDeleting) {
						t.Fatalf("publication after Project fence: %v", err)
					}
				} else if err != nil {
					t.Fatal(err)
				}
				retained := firstAction == "publication" && outcome == "commit" || firstAction == "deletion" && outcome == "rollback"
				target, _ := NewService(storage.Repository(pool)).Project(project.ProjectID)
				got, err := target.Read(ctx, ArtifactRef{Namespace: "outputs", Name: "result"})
				if retained {
					if err != nil || string(got.Payload.Data) != "accepted" {
						t.Fatalf("committed publication: %+v, %v", got, err)
					}
				} else if !errors.Is(err, ErrArtifactNotFound) {
					t.Fatalf("rejected/rolled-back publication left content: %v", err)
				}
				actual, err := projectstore.NewPostgresStore(pool).Get(ctx, project.OwnerID, project.ProjectID)
				want := projectstore.LifecycleDeleting
				if firstAction == "deletion" && outcome == "rollback" {
					want = projectstore.LifecycleActive
				}
				if err != nil || actual.Lifecycle != want {
					t.Fatalf("Project lifecycle=%s, want %s: %v", actual.Lifecycle, want, err)
				}
				var revisions, lineage int
				if err := pool.QueryRow(ctx, `SELECT
(SELECT count(*) FROM artifact_binding_revisions WHERE scope_kind='project'),
(SELECT count(*) FROM artifact_lineage WHERE lineage_kind='project_output_publish')`).Scan(&revisions, &lineage); err != nil {
					t.Fatal(err)
				}
				count := 0
				if retained {
					count = 1
				}
				if revisions != count || lineage != count {
					t.Fatalf("Project revisions/lineage=%d/%d, want %d", revisions, lineage, count)
				}
			})
		}
	}
}

func TestPostgresPublicationPilotComposedDeletion(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 30*time.Second)
	defer cancel()
	pool := isolatedArtifactPool(t, ctx)
	storage := artifactpgtest.NewStorage()
	project, source := storageReviewProject(t, ctx, pool, storage)
	service := NewService(storage.Repository(pool))
	publication, err := publishOutput(ctx, service, "review-run", project.ProjectID, "result", source)
	if err != nil {
		t.Fatal(err)
	}
	// A second, independently owned version shares only the physical blob.
	user, _ := service.User("user-1")
	kept, err := user.Write(ctx, ArtifactRef{Namespace: "keep", Name: "shared"}, Payload{MediaType: "text/plain", Data: []byte("accepted")}, nil)
	if err != nil {
		t.Fatal(err)
	}
	runs := runstore.NewPostgresStore(pool, storage)
	for _, transition := range [][2]runstore.WorkflowRunState{
		{runstore.RunInitializing, runstore.RunPending},
		{runstore.RunPending, runstore.RunRunning},
		{runstore.RunRunning, runstore.RunSucceeded},
	} {
		if _, err := runs.TransitionRun(ctx, "review-run", transition[0], transition[1], runstore.Reason{Code: "pilot"}); err != nil {
			t.Fatal(err)
		}
	}
	// Rollback the composed operation, then retry through its ordinary owner.
	tx, err := pool.Begin(ctx)
	if err != nil {
		t.Fatal(err)
	}
	defer tx.Rollback(context.Background())
	if err := rundeletion.New(tx, storage).DeleteReleasedTerminalRun(ctx, "user-1", "review-run"); err != nil {
		t.Fatal(err)
	}
	if err := tx.Rollback(ctx); err != nil {
		t.Fatal(err)
	}
	run, _ := service.Run("review-run")
	if _, err := run.Read(ctx, source); err != nil {
		t.Fatalf("rollback lost the source: %v", err)
	}
	if err := rundeletion.New(pool, storage).DeleteReleasedTerminalRun(ctx, "user-1", "review-run"); err != nil {
		t.Fatal(err)
	}
	target, _ := service.Project(project.ProjectID)
	if got, err := target.Read(ctx, publication.TargetRef); err != nil || string(got.Payload.Data) != "accepted" {
		t.Fatalf("publication after composed Run deletion: %+v, %v", got, err)
	}
	if _, _, err := projectstore.NewPostgresStore(pool).BeginDeletion(ctx, projectstore.BeginDeletionParams{
		OwnerID: project.OwnerID, ProjectID: project.ProjectID, ExpectedRevision: project.Revision,
	}); err != nil {
		t.Fatal(err)
	}
	// One batch removes the sole Project revision; the next removes its scope.
	for pass := range 2 {
		tx, err := pool.Begin(ctx)
		if err != nil {
			t.Fatal(err)
		}
		defer tx.Rollback(context.Background())
		purger, err := storage.Purger(tx)
		if err != nil {
			t.Fatal(err)
		}
		done, err := purger.PurgeProject(ctx, project.ProjectID)
		if err != nil || done != (pass == 1) {
			t.Fatalf("Project purge pass %d: done=%t, err=%v", pass, done, err)
		}
		if err := tx.Commit(ctx); err != nil {
			t.Fatal(err)
		}
	}
	if _, err := target.Read(ctx, publication.TargetRef); !errors.Is(err, ErrArtifactNotFound) {
		t.Fatalf("purged Project publication still readable: %v", err)
	}
	if got, err := user.Read(ctx, kept.Ref); err != nil || string(got.Payload.Data) != "accepted" {
		t.Fatalf("other scope's shared blob was collected: %+v, %v", got, err)
	}
	var runsLeft, revisions, versions, blobs, lineage int
	if err := pool.QueryRow(ctx, `SELECT (SELECT count(*) FROM workflow_runs),
(SELECT count(*) FROM artifact_binding_revisions), (SELECT count(*) FROM artifact_versions),
(SELECT count(*) FROM artifact_blobs), (SELECT count(*) FROM artifact_lineage)`).Scan(&runsLeft, &revisions, &versions, &blobs, &lineage); err != nil {
		t.Fatal(err)
	}
	if runsLeft != 0 || revisions != 1 || versions != 1 || blobs != 1 || lineage != 0 {
		t.Fatalf("after both purges runs/revisions/versions/blobs/lineage=%d/%d/%d/%d/%d", runsLeft, revisions, versions, blobs, lineage)
	}
}

func publicationPilotRace(t *testing.T, ctx context.Context, pool *pgxpool.Pool, firstAction, secondAction func(pgx.Tx) error, rollback bool) error {
	t.Helper()
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
	if err := firstAction(first); err != nil {
		t.Fatal(err)
	}
	pid := second.Conn().PgConn().PID()
	result := make(chan error, 1)
	go func() { result <- secondAction(second) }()
	// On a failed assertion release the blocker and join before touching the
	// second connection or letting fixture cleanup drop the schema.
	joined := false
	defer func() {
		if !joined {
			_ = first.Rollback(context.Background())
			<-result
		}
	}()
	waitArtifactLock(t, ctx, pool, pid)
	if rollback {
		err = first.Rollback(ctx)
	} else {
		err = first.Commit(ctx)
	}
	if err != nil {
		t.Fatal(err)
	}
	err = <-result
	joined = true
	if err != nil {
		return err
	}
	return second.Commit(ctx)
}
