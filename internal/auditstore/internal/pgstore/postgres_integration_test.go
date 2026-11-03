package pgstore

import (
	"context"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/persistence/explaintest"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/persistence/postgrestest"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

const auditTestRuntimeSnapshot = `{"default":{"label":"default","explicit":false,"bindingRevision":1,"config":{"name":"contractor-empty","version":"1","digest":"sha256:80a1754c01f8443c29fdc8f650a2254b2461694819918b204a55a7ad3425dc5f"}},"labels":[],"llmCredentialIds":[],"runtimeCredentialIds":[]}`

func insertAndBindTestRun(
	t *testing.T,
	ctx context.Context,
	pool *pgxpool.Pool,
	claim auditstore.ControllerClaim,
	execution auditstore.Execution,
	runID string,
	ownerID string,
	projectID string,
) (auditstore.Execution, error) {
	t.Helper()
	var bound auditstore.Execution
	err := persistencepostgres.InTx(ctx, pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		if err := insertTestRun(ctx, tx, runID, ownerID, projectID, &execution); err != nil {
			return err
		}
		var err error
		bound, err = newTestStore(tx).BindRun(ctx, auditstore.BindRunParams{
			Claim: claim, ExecutionID: execution.ExecutionID, RunID: runID,
		})
		return err
	})
	return bound, err
}

func insertTestRun(
	ctx context.Context,
	db persistencepostgres.DBTX,
	runID string,
	ownerID string,
	projectID string,
	execution *auditstore.Execution,
) error {
	publicationMode := "ordinary"
	var executionID, submissionKey *string
	if execution != nil {
		publicationMode = "audit-managed"
		executionID = &execution.ExecutionID
		submissionKey = &execution.SubmissionKey
	}
	_, err := db.Exec(ctx, `
INSERT INTO workflow_runs (
    run_id, owner_id, project_id, workflow_name, workflow_version,
    workflow_schema_version, workflow_snapshot, parameters,
    runtime_labels, runtime_config_snapshot, publication_mode,
    audit_execution_id, audit_submission_key, scheduler_lane_class, state, state_reason_code
) VALUES ($1, $2, $3, 'audit-check', '1', 'contractor/v1alpha1',
          '{}'::jsonb, '{}'::jsonb, ARRAY[]::text[], $4::jsonb,
          $5, $6, $7, 'model', 'initializing', 'created')`,
		runID, ownerID, projectID, auditTestRuntimeSnapshot,
		publicationMode, executionID, submissionKey,
	)
	return err
}

func terminateTestRun(
	t *testing.T, ctx context.Context, pool *pgxpool.Pool, runID, outcome string,
) (string, uint64) {
	t.Helper()
	_, err := pool.Exec(ctx, `
UPDATE workflow_runs
   SET state = $2, state_reason_code = 'test-terminal',
       started_at = COALESCE(started_at, clock_timestamp()),
       finished_at = clock_timestamp(), updated_at = clock_timestamp()
 WHERE run_id = $1`, runID, outcome)
	if err != nil {
		t.Fatalf("terminate test WorkflowRun: %v", err)
	}
	var generation string
	var sequence int64
	if err := pool.QueryRow(ctx, `
SELECT run_event_generation, next_run_event_sequence - 1
  FROM workflow_runs WHERE run_id = $1`, runID).Scan(&generation, &sequence); err != nil {
		t.Fatal(err)
	}
	return generation, uint64(sequence)
}

func isolatedAuditPool(t *testing.T, ctx context.Context, databaseURL string) *pgxpool.Pool {
	t.Helper()
	pool := postgrestest.NewPool(t, ctx, databaseURL, postgrestest.Options{})
	if err := applyAuditMigrations(ctx, pool); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		cleanup, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		requireDenseAuditEvents(t, cleanup, pool)
	})
	return pool
}

// budgetAuditPool is isolatedAuditPool on a database of its own
// (explaintest.Database), for a fixture whose reads a test budgets: no other
// package's snapshot keeps the superseded versions its VACUUM must remove.
func budgetAuditPool(t *testing.T, ctx context.Context, databaseURL string) *pgxpool.Pool {
	t.Helper()
	pool := explaintest.Database(t, ctx, databaseURL, applyAuditMigrations)
	t.Cleanup(func() {
		cleanupCtx, cleanupCancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cleanupCancel()
		requireDenseAuditEvents(t, cleanupCtx, pool)
	})
	return pool
}

func applyAuditMigrations(ctx context.Context, pool *pgxpool.Pool) error {
	_, err := persistencepostgres.ApplyMigrations(ctx, pool, persistencepostgres.MigrationBudgets{})
	return err
}

func waitForPostgresLock(
	t *testing.T,
	ctx context.Context,
	pool *pgxpool.Pool,
	backendPID int,
	done <-chan error,
) {
	t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		select {
		case err := <-done:
			t.Fatalf("mutation completed before the claim fence moved: %v", err)
		default:
		}
		var waitEventType *string
		if err := pool.QueryRow(ctx, `
SELECT wait_event_type FROM pg_stat_activity WHERE pid = $1`, backendPID).Scan(&waitEventType); err != nil {
			t.Fatal(err)
		}
		if waitEventType != nil && *waitEventType == "Lock" {
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatal("mutation did not reach the PostgreSQL claim fence")
}

func stringPointer(value string) *string { return &value }
