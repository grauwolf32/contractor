package rundeletion

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/pagelimit"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

func TestPostgresRunDeletionGatesMatchDeletion(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	pool := isolatedDeletionPool(t, ctx)
	runs := runstore.NewPostgresStore(pool)
	createDeletionRun(t, ctx, pool, "user-1", "run-active", runstore.RunRunning)
	createDeletionRun(t, ctx, pool, "user-1", "run-release-pending", runstore.RunFailed)
	createDeletionRun(t, ctx, pool, "user-1", "run-released", runstore.RunSucceeded)
	createDeletionRun(t, ctx, pool, "user-2", "run-foreign", runstore.RunSucceeded)
	execution, err := runs.CreateStageExecution(ctx, runstore.CreateStageExecutionParams{
		StageExecutionID: "stage-release-pending", RunID: "run-release-pending", StageName: "copy", Attempt: 1,
		StageSpecSchemaVersion: contracts.APIVersion, StageSpecSnapshot: json.RawMessage(`{}`),
		StageContextSchemaVersion: contracts.APIVersion, StageContext: runstore.StageContextSnapshot{},
	})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `
INSERT INTO stage_allocations (
    allocation_id, stage_execution_id, logical_agent_name, namespace,
    agent_template_ref, worker_runtime_ref, runtime_agent_instance_id
) VALUES ('allocation-release-pending', $1, 'builder', 'builder', '{}'::jsonb, '{}'::jsonb, 'runtime-1')`,
		execution.StageExecutionID,
	); err != nil {
		t.Fatal(err)
	}

	deletions := New(pool)
	want := map[string]*Reason{
		"run-active":          reasonOf(RunNotTerminal),
		"run-release-pending": reasonOf(AllocationReleasePending),
		"run-released":        nil,
	}
	for runID, reason := range want {
		got, err := deletions.RunDeletionBlocker(ctx, "user-1", runID)
		if err != nil || (got == nil) != (reason == nil) || (got != nil && *got != *reason) {
			t.Fatalf("RunDeletionBlocker(%s) = (%v, %v), want %v", runID, got, err, reason)
		}
	}
	for _, runID := range []string{"run-foreign", "run-missing"} {
		if _, err := deletions.RunDeletionBlocker(ctx, "user-1", runID); !errors.Is(err, runstore.ErrNotFound) {
			t.Fatalf("RunDeletionBlocker(%s) error = %v, want not found", runID, err)
		}
	}
	deletable, err := deletions.DeletableRuns(ctx, "user-1", []string{
		"run-active", "run-release-pending", "run-released", "run-foreign", "run-missing",
	})
	wantDeletable := map[string]bool{
		"run-active": false, "run-release-pending": false, "run-released": true,
		"run-foreign": false, "run-missing": false,
	}
	if err != nil || fmt.Sprint(deletable) != fmt.Sprint(wantDeletable) {
		t.Fatalf("DeletableRuns = (%v, %v), want %v", deletable, err, wantDeletable)
	}
	oversized := make([]string, pagelimit.MaxRows+1)
	for index := range oversized {
		oversized[index] = fmt.Sprintf("run-%d", index)
	}
	if _, err := deletions.DeletableRuns(ctx, "user-1", oversized); !errors.Is(err, runstore.ErrInvalid) {
		t.Fatalf("oversized DeletableRuns error = %v, want invalid", err)
	}

	for runID, reason := range want {
		if reason == nil {
			continue
		}
		err := deletions.DeleteReleasedTerminalRun(ctx, "user-1", runID)
		var blocked *NotDeletableError
		if !errors.As(err, &blocked) || blocked.Reason != *reason || !errors.Is(err, runstore.ErrConflict) {
			t.Fatalf("delete %s error = %v, want %s conflict", runID, err, *reason)
		}
	}
	if err := deletions.DeleteReleasedTerminalRun(ctx, "user-1", "run-foreign"); !errors.Is(err, runstore.ErrNotFound) {
		t.Fatalf("foreign deletion error = %v, want not found", err)
	}
	if err := deletions.DeleteReleasedTerminalRun(ctx, "user-1", "run-released"); err != nil {
		t.Fatalf("delete released terminal Run: %v", err)
	}
	if _, err := runs.GetRun(ctx, "run-released"); !errors.Is(err, runstore.ErrNotFound) {
		t.Fatalf("deleted Run lookup error = %v", err)
	}
	if _, err := deletions.RunDeletionBlocker(ctx, "user-1", "run-released"); !errors.Is(err, runstore.ErrNotFound) {
		t.Fatalf("deleted Run blocker error = %v, want not found", err)
	}
}

func TestPostgresDeletionLocksRequireTransaction(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	pool := isolatedDeletionPool(t, ctx)
	createDeletionRun(t, ctx, pool, "user-1", "run-done", runstore.RunSucceeded)
	runs := runstore.NewPostgresStore(pool)
	if _, err := runs.LockRunForDeletion(ctx, "user-1", "run-done"); err == nil ||
		!strings.Contains(err.Error(), "transaction-backed store is required") {
		t.Fatalf("pool-backed Run lock error = %v", err)
	}
	if err := runs.DeleteLockedRun(ctx, "user-1", "run-done"); err == nil ||
		!strings.Contains(err.Error(), "transaction-backed store is required") {
		t.Fatalf("pool-backed Run delete error = %v", err)
	}
	if _, err := auditstore.NewPostgresStore(pool).LockRunDeletionAudits(ctx, "user-1", "run-done"); err == nil ||
		!strings.Contains(err.Error(), "transaction-backed store is required") {
		t.Fatalf("pool-backed Audit lock error = %v", err)
	}
	if _, err := runs.GetRun(ctx, "run-done"); err != nil {
		t.Fatalf("refused steps changed the Run: %v", err)
	}
}

func createDeletionRun(
	t *testing.T, ctx context.Context, pool *pgxpool.Pool,
	ownerID, runID string, state runstore.WorkflowRunState,
) {
	t.Helper()
	projectID := "project-" + ownerID
	if _, _, err := projectstore.NewPostgresStore(pool).Create(ctx, projectstore.CreateParams{
		ProjectID: projectID, OwnerID: ownerID, Kind: projectstore.KindProject,
		Name: "Run deletion " + ownerID, IdempotencyKey: "create-" + projectID,
		RequestDigest: "sha256:" + strings.Repeat("a", 64),
	}); err != nil {
		t.Fatal(err)
	}
	runs := runstore.NewPostgresStore(pool)
	if _, err := runs.CreateRun(ctx, runstore.CreateRunParams{
		RunID: runID, OwnerID: ownerID, ProjectID: &projectID,
		WorkflowName: "artifact-copy", WorkflowVersion: "1",
		WorkflowSchemaVersion: contracts.APIVersion,
		WorkflowSnapshot:      json.RawMessage(`{"ref":{"name":"artifact-copy","version":"1"}}`),
		Parameters:            map[string]string{},
		RuntimeConfig:         runtimeconfig.BuiltInRunSnapshot(),
	}); err != nil {
		t.Fatal(err)
	}
	from := runstore.RunInitializing
	if state != runstore.RunFailed {
		// Initialization commits pending; only admission sets running (RUN-R26, RUN-R28).
		for _, next := range []runstore.WorkflowRunState{runstore.RunPending, runstore.RunRunning} {
			if _, err := runs.TransitionRun(ctx, runID, from, next, runstore.Reason{Code: "started"}); err != nil {
				t.Fatal(err)
			}
			from = next
		}
	}
	if state != from {
		if _, err := runs.TransitionRun(ctx, runID, from, state, runstore.Reason{Code: "finished"}); err != nil {
			t.Fatal(err)
		}
	}
}

func isolatedDeletionPool(t *testing.T, ctx context.Context) *pgxpool.Pool {
	t.Helper()
	databaseURL := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if databaseURL == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	admin, err := pgxpool.New(ctx, databaseURL)
	if err != nil {
		t.Fatal(err)
	}
	if err := admin.Ping(ctx); err != nil {
		admin.Close()
		t.Fatal(err)
	}
	random := make([]byte, 8)
	if _, err := rand.Read(random); err != nil {
		admin.Close()
		t.Fatal(err)
	}
	schema := "contractor_run_deletion_test_" + hex.EncodeToString(random)
	identifier := pgx.Identifier{schema}.Sanitize()
	if _, err := admin.Exec(ctx, `CREATE SCHEMA `+identifier); err != nil {
		admin.Close()
		t.Fatal(err)
	}
	config, err := pgxpool.ParseConfig(databaseURL)
	if err != nil {
		admin.Close()
		t.Fatal(err)
	}
	config.ConnConfig.RuntimeParams["search_path"] = schema
	pool, err := pgxpool.NewWithConfig(ctx, config)
	if err != nil {
		admin.Close()
		t.Fatal(err)
	}
	t.Cleanup(func() {
		pool.Close()
		cleanupContext, cleanupCancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cleanupCancel()
		_, _ = admin.Exec(cleanupContext, `DROP SCHEMA `+identifier+` CASCADE`)
		admin.Close()
	})
	if _, err := persistencepostgres.ApplyMigrations(ctx, pool); err != nil {
		t.Fatal(err)
	}
	return pool
}
