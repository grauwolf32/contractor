package runstore

import (
	"context"
	"strings"
	"testing"
	"time"

	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/projectstore"
)

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
