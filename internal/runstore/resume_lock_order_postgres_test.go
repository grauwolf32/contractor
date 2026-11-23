package runstore

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"

	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/projectstore"
)

func TestPostgresResumeReplaysReceiptAfterProjectDeletionStarts(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 45*time.Second)
	defer cancel()
	pool := isolatedRunStorePool(t, ctx)
	projects := projectstore.NewPostgresStore(pool)
	project, _, err := projects.Create(ctx, projectstore.CreateParams{
		ProjectID: "project-resume-replay", OwnerID: "user-1", Kind: projectstore.KindProject,
		Name: "Resume replay", IdempotencyKey: "resume-replay", RequestDigest: "sha256:" + strings.Repeat("a", 64),
	})
	if err != nil {
		t.Fatal(err)
	}
	params := testRunParams("run-resume-replay")
	params.ProjectID = &project.ProjectID
	store, source := failedResumeFixtureOf(t, ctx, pool, params)
	first, err := store.ResumeFailedRun(ctx, "user-1", params.RunID, source.StageExecutionID, "replay-target")
	if err != nil {
		t.Fatal(err)
	}
	failResumeStage(t, ctx, store, params.RunID, first.StageExecutionID)
	if _, _, err := projects.BeginDeletion(ctx, projectstore.BeginDeletionParams{
		ProjectID: project.ProjectID, OwnerID: "user-1", ExpectedRevision: project.Revision,
	}); err != nil {
		t.Fatal(err)
	}
	replay, err := store.ResumeFailedRun(ctx, "user-1", params.RunID, source.StageExecutionID, "unused-replay-target")
	if err != nil || replay != first {
		t.Fatalf("committed resume replay = (%+v, %v), want %+v", replay, err, first)
	}
	if _, err := store.ResumeFailedRun(ctx, "other-owner", params.RunID, source.StageExecutionID, "foreign-target"); !errors.Is(err, ErrNotFound) {
		t.Fatalf("foreign receipt replay = %v, want ErrNotFound", err)
	}
	if _, err := store.ResumeFailedRun(ctx, "user-1", params.RunID, first.StageExecutionID, "new-target"); !errors.Is(err, ErrConflict) {
		t.Fatalf("new resume in deleting Project = %v, want ErrConflict", err)
	}
	var stages, receipts int
	if err := pool.QueryRow(ctx, `SELECT count(*) FROM stage_executions WHERE run_id=$1`, params.RunID).Scan(&stages); err != nil {
		t.Fatal(err)
	}
	if err := pool.QueryRow(ctx, `SELECT count(*) FROM stage_executions WHERE run_id=$1 AND resume_source_execution_id IS NOT NULL`, params.RunID).Scan(&receipts); err != nil {
		t.Fatal(err)
	}
	if stages != 3 || receipts != 1 {
		t.Fatalf("replay/new resume left %d Stages and %d receipts, want 3 and 1", stages, receipts)
	}
}

// Resuming a Project's Run locks the Project before the Run, as Project
// deletion does: while deletion holds the Project, the resume waits for it
// without holding the Run row, and it continues once the Project is free.
func TestPostgresResumeLocksTheProjectBeforeTheRun(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	pool := isolatedRunStorePool(t, ctx)
	const projectID, runID = "project-resume-order", "run-resume-order"
	if _, _, err := projectstore.NewPostgresStore(pool).Create(ctx, projectstore.CreateParams{
		ProjectID: projectID, OwnerID: "user-1", Kind: projectstore.KindProject,
		Name: projectID, IdempotencyKey: projectID, RequestDigest: "sha256:" + strings.Repeat("c", 64),
	}); err != nil {
		t.Fatal(err)
	}
	params := testRunParams(runID)
	params.ProjectID = new(projectID)
	store, source := failedResumeFixtureOf(t, ctx, pool, params)
	holder, err := pool.Begin(ctx)
	if err != nil {
		t.Fatal(err)
	}
	defer holder.Rollback(context.Background())
	if _, err := holder.Exec(ctx, `SELECT 1 FROM projects WHERE project_id = $1 FOR UPDATE`, projectID); err != nil {
		t.Fatal(err)
	}
	holderPID := int(holder.Conn().PgConn().PID())
	done := make(chan error, 1)
	go func() {
		_, err := store.ResumeFailedRun(ctx, "user-1", runID, source.StageExecutionID, "resume-order-target")
		done <- err
	}()
	deadline := time.NewTimer(5 * time.Second)
	defer deadline.Stop()
	for waiting := 0; waiting == 0; {
		if err := pool.QueryRow(ctx, `SELECT count(*) FROM pg_stat_activity WHERE $1::int = ANY(pg_blocking_pids(pid))`,
			holderPID).Scan(&waiting); err != nil {
			t.Fatal(err)
		}
		select {
		case <-deadline.C:
			t.Fatal("resume never waited for the locked Project")
		case err := <-done:
			t.Fatalf("resume ended while the Project was locked: %v", err)
		default:
		}
	}
	probe, err := pool.Begin(ctx)
	if err != nil {
		t.Fatal(err)
	}
	defer probe.Rollback(context.Background())
	_, err = probe.Exec(ctx, `SELECT 1 FROM workflow_runs WHERE run_id = $1 FOR UPDATE NOWAIT`, runID)
	if persistencepostgres.SQLState(err) == persistencepostgres.SQLStateLockNotAvailable {
		t.Fatal("resume locked the Run before the Project")
	}
	if err != nil {
		t.Fatal(err)
	}
	if err := probe.Rollback(ctx); err != nil {
		t.Fatal(err)
	}
	if err := holder.Rollback(ctx); err != nil {
		t.Fatal(err)
	}
	if err := <-done; err != nil {
		t.Fatalf("resume after the Project lock ended = %v", err)
	}
}
