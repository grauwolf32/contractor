package projectlifecycle

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

var deletionPilotPhases = []projectstore.DeletionPhase{
	projectstore.DeletionCancelling, projectstore.DeletionDraining,
	projectstore.DeletionPurgingRuns, projectstore.DeletionPurgingArtifacts,
}

// V498-006: expiry permits replacement but does not revoke the unchanged ID.
// Once replaced, old phase/claim mutations must leave the new owner untouched.
func TestProjectDeletionPilotClaimTakeover(t *testing.T) {
	for _, phase := range deletionPilotPhases {
		t.Run(string(phase), func(t *testing.T) {
			ctx, cancel := context.WithTimeout(t.Context(), 45*time.Second)
			defer cancel()
			pool, runs, id := deletionPilotFixture(t, ctx, phase)
			first := newTestController(t, pool, runs, &recordingNotifier{}, "first")
			old, err := first.claim(ctx, "pilot-old")
			if err != nil {
				t.Fatal(err)
			}
			second := newTestController(t, pool, runs, &recordingNotifier{}, "second")
			if _, err := second.claim(ctx, "pilot-too-soon"); !errors.Is(err, errNoDeletion) {
				t.Fatalf("live claim was replaced: %v", err)
			}
			expireDeletionClaim(t, ctx, pool, id)
			if err := pgx.BeginFunc(ctx, pool, func(tx pgx.Tx) error {
				return projectstore.NewPostgresTxStore(tx).LockDeletionClaim(ctx, old)
			}); err != nil {
				t.Fatalf("expired but unreplaced ID was revoked: %v", err)
			}
			current, err := second.claim(ctx, "pilot-new")
			if err != nil || current.Phase != phase {
				t.Fatalf("claim handover: %+v, %v", current, err)
			}
			before := deletionPilotSnapshot(t, ctx, pool, id)
			for name, operation := range map[string]func() error{
				"advance": func() error { _, _, err := first.advancePhase(ctx, old, projectstore.DeletionDraining); return err },
				"release": func() error { return first.releaseClaim(ctx, old) },
				"defer":   func() error { return first.deferClaim(ctx, old) },
				"lock": func() error {
					return pgx.BeginFunc(ctx, pool, func(tx pgx.Tx) error {
						return projectstore.NewPostgresTxStore(tx).LockDeletionClaim(ctx, old)
					})
				},
				"project-purge": func() error { _, err := first.purgeProject(ctx, old); return err },
			} {
				if err := operation(); !errors.Is(err, projectstore.ErrDeletionClaimLost) {
					t.Fatalf("stale %s: %v", name, err)
				}
				if after := deletionPilotSnapshot(t, ctx, pool, id); after != before {
					t.Fatalf("stale %s changed durable state: before=%+v after=%+v", name, before, after)
				}
			}
			if err := second.releaseClaim(ctx, current); err != nil {
				t.Fatal(err)
			}
			deletionPilotResume(t, ctx, pool, runs, id)
		})
	}
}

// Reconstruct a controller after a crash before the step, or after durable
// work succeeded but its result/bookkeeping were lost. This is not a process
// kill or network proxy; the ordinary PostgreSQL commit is real.
func TestProjectDeletionPilotRestart(t *testing.T) {
	for _, phase := range deletionPilotPhases {
		for _, point := range []string{"before-step", "after-commit"} {
			t.Run(string(phase)+"/"+point, func(t *testing.T) {
				ctx, cancel := context.WithTimeout(t.Context(), 45*time.Second)
				defer cancel()
				pool, runs, id := deletionPilotFixture(t, ctx, phase)
				first := newTestController(t, pool, runs, &recordingNotifier{}, "crashed")
				claim, err := first.claim(ctx, "crashed-claim")
				if err != nil {
					t.Fatal(err)
				}
				beforeStep := deletionPilotSnapshot(t, ctx, pool, id)
				if point == "after-commit" {
					worked, completed, err := first.advance(ctx, claim)
					if err != nil || !worked || completed {
						t.Fatalf("step before crash: worked=%t completed=%t err=%v", worked, completed, err)
					}
				}
				committed := deletionPilotSnapshot(t, ctx, pool, id)
				if point == "after-commit" && committed == beforeStep {
					t.Fatal("successful step recorded no durable progress")
				}
				restarted := newTestController(t, pool, runs, &recordingNotifier{}, "restart")
				if worked, err := restarted.RunOnce(ctx); err != nil || worked {
					t.Fatalf("restart acquired the still-live crash claim: %t, %v", worked, err)
				}
				if got := deletionPilotSnapshot(t, ctx, pool, id); got != committed {
					t.Fatalf("restart changed committed progress while claim held: %+v -> %+v", committed, got)
				}
				expireDeletionClaim(t, ctx, pool, id)
				deletionPilotResume(t, ctx, pool, runs, id)
			})
		}
	}
}

func TestProjectDeletionPilotLockedBatch(t *testing.T) {
	for _, outcome := range []string{"commit", "rollback"} {
		t.Run(outcome, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(t.Context(), 45*time.Second)
			defer cancel()
			pool, runs, id := deletionPilotFixture(t, ctx, projectstore.DeletionPurgingArtifacts)
			storage := artifactpgtest.NewStorage()
			first := newTestController(t, pool, runs, &recordingNotifier{}, "locked")
			old, err := first.claim(ctx, "locked-claim")
			if err != nil {
				t.Fatal(err)
			}
			expireDeletionClaim(t, ctx, pool, id)
			before := deletionPilotSnapshot(t, ctx, pool, id)
			tx, err := pool.Begin(ctx)
			if err != nil {
				t.Fatal(err)
			}
			defer tx.Rollback(context.Background())
			if err := projectstore.NewPostgresTxStore(tx).LockDeletionClaim(ctx, old); err != nil {
				t.Fatal(err)
			}
			// The fixture is drained and has no Eval/Finding/receipt rows. This
			// is the real locked Artifact tail of Controller.purgeProject.
			purger, err := storage.Purger(tx)
			if err != nil {
				t.Fatal(err)
			}
			if gone, err := purger.PurgeProject(ctx, id); err != nil || gone {
				t.Fatalf("stage one Artifact batch: gone=%t err=%v", gone, err)
			}
			second := newTestController(t, pool, runs, &recordingNotifier{}, "competitor")
			if _, err := second.claim(ctx, "skipped-lock"); !errors.Is(err, errNoDeletion) {
				t.Fatalf("SKIP LOCKED admitted a competing claim: %v", err)
			}
			if visible := deletionPilotSnapshot(t, ctx, pool, id); visible != before {
				t.Fatalf("uncommitted purge became visible: %+v -> %+v", before, visible)
			}
			if outcome == "commit" {
				err = tx.Commit(ctx)
			} else {
				err = tx.Rollback(ctx)
			}
			if err != nil {
				t.Fatal(err)
			}
			after := deletionPilotSnapshot(t, ctx, pool, id)
			want := before.Revisions
			if outcome == "commit" {
				want = 0 // All three fixture revisions fit the production batch.
			}
			if after.Revisions != want {
				t.Fatalf("%s batch: %d revisions, want %d", outcome, after.Revisions, want)
			}
			current, err := second.claim(ctx, "after-lock")
			if err != nil {
				t.Fatal(err)
			}
			if _, err := first.purgeProject(ctx, old); !errors.Is(err, projectstore.ErrDeletionClaimLost) {
				t.Fatalf("old batch owner after takeover: %v", err)
			}
			if err := second.releaseClaim(ctx, current); err != nil {
				t.Fatal(err)
			}
			deletionPilotResume(t, ctx, pool, runs, id)
		})
	}
}

func deletionPilotFixture(t *testing.T, ctx context.Context, phase projectstore.DeletionPhase) (*pgxpool.Pool, *runstore.PostgresStore, string) {
	t.Helper()
	pool := isolatedPool(t, ctx)
	storage := artifactpgtest.NewStorage()
	projects := projectstore.NewPostgresStore(pool)
	project, _, err := projects.Create(ctx, projectstore.CreateParams{
		ProjectID: "pilot-project", OwnerID: "user-1", Kind: projectstore.KindProject,
		Name: "Deletion pilot", IdempotencyKey: "pilot-create", RequestDigest: "sha256:" + strings.Repeat("a", 64),
	})
	if err != nil {
		t.Fatal(err)
	}
	runs := runstore.NewPostgresStore(pool, storage)
	run := createProjectRun(t, ctx, runs, "pilot-run", project.ProjectID)
	if _, err := runs.TransitionRun(ctx, run.RunID, runstore.RunInitializing, runstore.RunFailed, runstore.Reason{Code: "fixture"}); err != nil {
		t.Fatal(err)
	}
	store, _ := artifacts.NewService(storage.Repository(pool)).Project(project.ProjectID)
	for i := range 3 {
		if _, err := store.Write(ctx, artifacts.ArtifactRef{Namespace: "notes", Name: fmt.Sprintf("note-%d", i)}, artifacts.Payload{MediaType: "text/plain", Data: []byte("retained until its purge")}, nil); err != nil {
			t.Fatal(err)
		}
	}
	if _, _, err := projects.BeginDeletion(ctx, projectstore.BeginDeletionParams{OwnerID: project.OwnerID, ProjectID: project.ProjectID, ExpectedRevision: project.Revision}); err != nil {
		t.Fatal(err)
	}
	setup := newTestController(t, pool, runs, &recordingNotifier{}, "setup")
	for step := 0; step < 8; step++ {
		if deletionPilotSnapshot(t, ctx, pool, project.ProjectID).Phase == string(phase) {
			return pool, runs, project.ProjectID
		}
		if worked, err := setup.RunOnce(ctx); err != nil || !worked {
			t.Fatalf("prepare phase %s: %t, %v", phase, worked, err)
		}
	}
	t.Fatal("fixture failed to reach phase")
	return nil, nil, ""
}

type deletionPilotState struct {
	Phase, Claim string
	Expires      time.Time
	Revision     int64
	Runs         int
	Revisions    int
}

func deletionPilotSnapshot(t *testing.T, ctx context.Context, pool *pgxpool.Pool, id string) deletionPilotState {
	t.Helper()
	var state deletionPilotState
	if err := pool.QueryRow(ctx, `SELECT deletion_phase, COALESCE(deletion_claim_id,''),
COALESCE(deletion_claim_expires_at, 'epoch'::timestamptz), revision,
(SELECT count(*) FROM workflow_runs WHERE project_id=$1),
(SELECT count(*) FROM artifact_binding_revisions WHERE scope_kind='project' AND scope_id=$1)
FROM projects WHERE project_id=$1`, id).Scan(&state.Phase, &state.Claim, &state.Expires, &state.Revision, &state.Runs, &state.Revisions); err != nil {
		t.Fatal(err)
	}
	return state
}

func deletionPilotResume(t *testing.T, ctx context.Context, pool *pgxpool.Pool, runs *runstore.PostgresStore, id string) {
	t.Helper()
	previous := deletionPilotSnapshot(t, ctx, pool, id)
	for step := range 16 {
		// Each object uses fresh claim IDs; no in-memory phase/progress survives.
		controller := newTestController(t, pool, runs, &recordingNotifier{}, fmt.Sprintf("fresh-%d", step))
		if worked, err := controller.RunOnce(ctx); err != nil || !worked {
			t.Fatalf("resume step %d: %t, %v", step, worked, err)
		}
		if _, err := projectstore.NewPostgresStore(pool).Get(ctx, "user-1", id); errors.Is(err, projectstore.ErrNotFound) {
			var leftovers int
			if err := pool.QueryRow(ctx, `SELECT (SELECT count(*) FROM workflow_runs WHERE project_id=$1)
+ (SELECT count(*) FROM artifact_scopes WHERE scope_kind='project' AND scope_id=$1)
+ (SELECT count(*) FROM artifact_versions) + (SELECT count(*) FROM artifact_blobs)`, id).Scan(&leftovers); err != nil || leftovers != 0 {
				t.Fatalf("final deletion left %d resources: %v", leftovers, err)
			}
			return
		} else if err != nil {
			t.Fatal(err)
		}
		current := deletionPilotSnapshot(t, ctx, pool, id)
		if slices.Index(deletionPilotPhases, projectstore.DeletionPhase(current.Phase)) < slices.Index(deletionPilotPhases, projectstore.DeletionPhase(previous.Phase)) {
			t.Fatalf("restart regressed the phase: %+v -> %+v", previous, current)
		}
		if current.Runs > previous.Runs || current.Revisions > previous.Revisions {
			t.Fatalf("restart resurrected removed resources: %+v -> %+v", previous, current)
		}
		previous = current
	}
	t.Fatal("restarted controllers did not finish within the fixture's bounded steps")
}
