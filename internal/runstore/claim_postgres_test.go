package runstore

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/reporting"
	"github.com/grauwolf32/contractor/internal/runtimeconfig/runtimeconfigtest"
	"github.com/grauwolf32/contractor/internal/settingsstore"
	"github.com/jackc/pgx/v5/pgxpool"
)

// claimQueueSize is the queue the claim plan test reads. Every Run in it is
// claimable, so a claim that sorted its candidates would sort all of them.
const claimQueueSize = 10000

// populateClaimQueueSQL loads $2 claimable Runs over $3 owners, oldest first:
// a tenth running, a tenth waiting, a tenth with Skill initialization pending
// and the rest pending. Every 13th was deferred just now and every 97th holds
// a live claim. $1 is a valid RuntimeConfig snapshot.
const populateClaimQueueSQL = `
INSERT INTO workflow_runs (
    run_id, owner_id, workflow_name, workflow_version, workflow_schema_version,
    workflow_snapshot, parameters, runtime_labels, runtime_config_snapshot,
    state, state_reason_code, created_at, updated_at, started_at, scheduler_deferred,
    scheduler_claim_id, scheduler_claimed_at, scheduler_claim_expires_at
)
SELECT 'run-claim-' || lpad(i::text, 5, '0'), 'owner-' || (i % $3),
       'claim', '1', 'contractor/v1alpha1', '{}'::jsonb, '{}'::jsonb, '{}'::text[], $1::jsonb,
       run.state,
       CASE WHEN run.state = 'initializing' THEN 'skill_initialization_pending' ELSE 'queued' END,
       run.created_at,
       CASE WHEN i % 13 = 0 THEN clock_timestamp() ELSE run.created_at END,
       CASE WHEN run.state IN ('running', 'waiting') THEN run.created_at END,
       i % 13 = 0,
       CASE WHEN i % 97 = 0 THEN 'claim-held-' || i END,
       CASE WHEN i % 97 = 0 THEN clock_timestamp() END,
       CASE WHEN i % 97 = 0 THEN clock_timestamp() + interval '1 hour' END
FROM generate_series(1, $2) AS i
CROSS JOIN LATERAL (
    SELECT CASE i % 10
               WHEN 0 THEN 'running'
               WHEN 1 THEN 'waiting'
               WHEN 2 THEN 'initializing'
               ELSE 'pending'
           END AS state,
           timestamptz '2026-09-01 00:00:00+00' + i * interval '1 second' AS created_at
) AS run`

// populateClaimQueue loads the queue with the lifecycle-event trigger off,
// as populateRunHistory does, and analyzes it.
func populateClaimQueue(t *testing.T, ctx context.Context, pool *pgxpool.Pool) {
	t.Helper()
	snapshot, err := json.Marshal(runtimeconfigtest.BuiltInRunSnapshot())
	if err != nil {
		t.Fatal(err)
	}
	loadWithoutLifecycleEvents(t, ctx, pool, "workflow_runs", "workflow_runs_capture_lifecycle_event",
		populateClaimQueueSQL, snapshot, claimQueueSize, 7)
}

// loadWithoutLifecycleEvents runs one bulk insert into table with its
// per-row lifecycle-event trigger off, which would otherwise dominate the
// test's run time, then analyzes the table so the planner sees the load.
func loadWithoutLifecycleEvents(
	t *testing.T, ctx context.Context, pool *pgxpool.Pool, table, trigger, sql string, args ...any,
) {
	t.Helper()
	tx, err := pool.Begin(ctx)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = tx.Rollback(context.Background()) }()
	for _, statement := range []struct {
		sql  string
		args []any
	}{
		// Deferred constraint triggers would otherwise block re-enabling.
		{sql: `SET CONSTRAINTS ALL IMMEDIATE`},
		{sql: `ALTER TABLE ` + table + ` DISABLE TRIGGER ` + trigger},
		{sql: sql, args: args},
		{sql: `ALTER TABLE ` + table + ` ENABLE TRIGGER ` + trigger},
	} {
		if _, err := tx.Exec(ctx, statement.sql, statement.args...); err != nil {
			t.Fatalf("load %s: %v", table, err)
		}
	}
	if err := tx.Commit(ctx); err != nil {
		t.Fatalf("commit %s load: %v", table, err)
	}
	if _, err := pool.Exec(ctx, `VACUUM ANALYZE `+table); err != nil {
		t.Fatal(err)
	}
}

// TestPostgresClaimReadsTenThousandRunsInClaimOrder explains the claim over
// ten thousand claimable Runs, one owner's Queue paused: both the custom and
// the generic plan read the Runs in claim order from the claim indexes and
// stop at the first one they can lock, with no sort of the candidates and no
// scan of workflow_runs. The claims then follow that order: cancelling first,
// then the oldest admitted Run, skipping deferred and claimed ones.
func TestPostgresClaimReadsTenThousandRunsInClaimOrder(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
	defer cancel()
	pool := isolatedRunStorePool(t, ctx)
	populateClaimQueue(t, ctx, pool)
	store := NewPostgresStore(pool)
	if _, err := store.UpdateOwnerQueueControl(ctx, UpdateOwnerQueueControlParams{OwnerID: "owner-3", Paused: true}); err != nil {
		t.Fatal(err)
	}

	excluded := make([]string, settingsstore.MaximumConcurrentRuns)
	for index := range excluded {
		excluded[index] = fmt.Sprintf("run-claim-%05d", index+1)
	}
	claim := captureStatement(t, func(store *PostgresStore) error {
		_, err := store.ClaimRunnableRun(ctx, "claim-plan", time.Minute, excluded)
		return err
	})
	assertPlanUsesIndexes(t, ctx, pool, "Scheduler claim over the queue", claim,
		"workflow_runs_claim_idx", "workflow_runs_deferred_claim_idx")
	for _, generic := range []bool{false, true} {
		if plan := explainPlan(t, ctx, pool, claim, generic); strings.Contains(plan, "Sort") {
			t.Errorf("claim (generic plan %v) sorts its candidates:\n%s", generic, plan)
		}
	}

	cancelling := createTestRun(t, ctx, store, "run-claim-cancelling")
	if _, err := store.RequestRunCancellation(ctx, cancelling.RunID, WorkflowRunCancellation{
		Code: CancellationUserRequested, RequestedAt: time.Now(),
	}); err != nil {
		t.Fatal(err)
	}
	// run-claim-00001 is the oldest admitted Run (waiting) and neither
	// deferred nor claimed.
	for _, want := range []string{cancelling.RunID, "run-claim-00001"} {
		claimed, err := store.ClaimRunnableRun(ctx, "claim-"+want, time.Minute, nil)
		if err != nil || claimed.RunID != want {
			t.Fatalf("claim = (%s, %v), want %s", claimed.RunID, err, want)
		}
	}
}

func TestPostgresClaimExcludesLocallyActiveRuns(t *testing.T) {
	for _, deferred := range []bool{false, true} {
		t.Run(fmt.Sprintf("deferred=%v", deferred), func(t *testing.T) {
			ctx, cancel := context.WithTimeout(t.Context(), 15*time.Second)
			defer cancel()
			pool := isolatedRunStorePool(t, ctx)
			store := NewPostgresStore(pool)
			for _, id := range []string{"run-live", "run-peer"} {
				createTestRun(t, ctx, store, id)
				if _, err := admitTestRun(ctx, store, id); err != nil {
					t.Fatal(err)
				}
			}
			if _, err := pool.Exec(ctx, `UPDATE workflow_runs
SET scheduler_deferred = $1, updated_at = clock_timestamp(),
    scheduler_claim_id = CASE WHEN run_id = 'run-live' THEN 'original-lane' END,
    scheduler_claimed_at = CASE WHEN run_id = 'run-live' THEN clock_timestamp() - interval '3 minutes' END,
    scheduler_claim_expires_at = CASE WHEN run_id = 'run-live' THEN clock_timestamp() - interval '1 minute' END`, deferred); err != nil {
				t.Fatal(err)
			}
			claimed, err := store.ClaimRunnableRun(ctx, "peer-lane", time.Minute, []string{"run-live"})
			if err != nil || claimed.RunID != "run-peer" {
				t.Fatalf("claim excluding live owner = %s, %v", claimed.RunID, err)
			}
			if _, err := store.ClaimRunnableRun(ctx, "no-work", time.Minute, []string{"run-live"}); !errors.Is(err, ErrNoWork) {
				t.Fatalf("remaining locally active Run was claimed: %v", err)
			}
			if _, err := store.RenewRunClaim(ctx, "run-live", "original-lane", time.Minute); err != nil {
				t.Fatal(err)
			}
		})
	}
}

// TestPostgresClaimSkipsPendingRunsOfPausedOwners pauses an owner's Queue: the
// claim no longer takes its pending Runs, deferred or not, whose Stage
// admission could not start (SCH-R163), but still takes its cancelling and
// admitted Runs, its pending Runs that hold release work (an unreleased
// allocation or an aborting Stage) and other owners' Runs, and takes the
// skipped ones again once the Queue resumes.
func TestPostgresClaimSkipsPendingRunsOfPausedOwners(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	pool := isolatedRunStorePool(t, ctx)
	store := NewPostgresStore(pool)
	create := func(runID, ownerID string) WorkflowRun {
		t.Helper()
		params := testRunParams(runID)
		params.OwnerID = ownerID
		run, err := createRun(ctx, store, params)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := store.TransitionRun(ctx, runID, RunInitializing, RunPending, Reason{Code: "initialized"}); err != nil {
			t.Fatal(err)
		}
		return run
	}
	createStage := func(run WorkflowRun) StageExecution {
		t.Helper()
		stage, err := store.CreateStageExecution(ctx, testActiveStageParams(run.RunID, run.RunID+"-stage", "copy"))
		if err != nil {
			t.Fatal(err)
		}
		return stage
	}
	allocate := func(stage StageExecution) string {
		t.Helper()
		allocation := testStageAllocation(stage.StageExecutionID, "allocation-"+stage.StageExecutionID)
		if err := store.RecordStageAllocation(ctx, allocation); err != nil {
			t.Fatal(err)
		}
		return allocation.AllocationID
	}
	claim := func(want string) {
		t.Helper()
		claimed, err := store.ClaimRunnableRun(ctx, "claim-"+want, time.Minute, nil)
		if err != nil || claimed.RunID != want {
			t.Fatalf("claim = (%s, %v), want %s", claimed.RunID, err, want)
		}
	}

	deferred := create("run-paused-deferred", "user-paused")
	claim(deferred.RunID)
	if err := store.DeferRunClaim(ctx, deferred.RunID, "claim-"+deferred.RunID); err != nil {
		t.Fatal(err)
	}
	pending := create("run-paused-pending", "user-paused")
	// A placement whose discard failed is still held (SCH-R152).
	placed := create("run-paused-placed", "user-paused")
	allocate(createStage(placed))
	released := create("run-paused-released", "user-paused")
	if err := store.MarkStageAllocationReleased(ctx, allocate(createStage(released))); err != nil {
		t.Fatal(err)
	}
	aborting := create("run-paused-aborting", "user-paused")
	if err := store.EnterAborting(ctx, EnterAbortingParams{
		StageExecutionID: createStage(aborting).StageExecutionID, ExpectedState: StagePreparing,
		TerminationSchemaVersion: contracts.APIVersion,
		Termination: StageTermination{
			Outcome: TerminationInterrupted, Code: "context_artifact_missing", Message: "source is missing",
			Phase: TerminationPreparing, OccurredAt: time.Now(),
		},
		AbortID: "abort-paused", Deadline: time.Now().Add(time.Minute), Reason: Reason{Code: "stage_interrupted"},
	}); err != nil {
		t.Fatal(err)
	}
	admitted := create("run-paused-admitted", "user-paused")
	if _, err := store.TransitionRun(ctx, admitted.RunID, RunPending, RunRunning, Reason{Code: "admitted"}); err != nil {
		t.Fatal(err)
	}
	cancelling := create("run-paused-cancelling", "user-paused")
	if _, err := store.RequestRunCancellation(ctx, cancelling.RunID, WorkflowRunCancellation{
		Code: CancellationUserRequested, RequestedAt: time.Now(),
	}); err != nil {
		t.Fatal(err)
	}
	other := create("run-other-pending", "user-other")
	control, err := store.UpdateOwnerQueueControl(ctx, UpdateOwnerQueueControlParams{OwnerID: "user-paused", Paused: true})
	if err != nil {
		t.Fatal(err)
	}

	// Claims stay held, so each claim takes the next Run in claim order.
	claim(cancelling.RunID)
	claim(admitted.RunID)
	claim(placed.RunID)
	claim(aborting.RunID)
	claim(other.RunID)
	if run, err := store.ClaimRunnableRun(ctx, "claim-paused", time.Minute, nil); !errors.Is(err, ErrNoWork) {
		t.Fatalf("claim while the owner's Queue is paused = (%s, %v), want no work", run.RunID, err)
	}

	if _, err := store.UpdateOwnerQueueControl(ctx, UpdateOwnerQueueControlParams{
		OwnerID: "user-paused", ExpectedRevision: control.Revision, Paused: false,
	}); err != nil {
		t.Fatal(err)
	}
	// Resume makes them claimable without rewriting them (SCH-R154); the
	// deferred Run still ranks behind pending work within its window.
	claim(pending.RunID)
	claim(released.RunID)
	claim(deferred.RunID)
}

// populateStageHistorySQL gives each of the $2 Runs named $1 || n a history
// of $3 interrupted StageExecutions, one per Stage name, and one preparing
// StageExecution, as long retry chains would. $4 is a valid
// StageTermination and $5 a valid StageContext snapshot.
const populateStageHistorySQL = `
INSERT INTO stage_executions (
    stage_execution_id, run_id, stage_name, attempt,
    stage_spec_schema_version, stage_spec_snapshot,
    stage_context_schema_version, stage_context_snapshot,
    state, state_reason_code, termination_schema_version, stage_termination,
    abort_id, abort_deadline, terminal_at, created_at, updated_at
)
SELECT history.run_id || '-stage-' || lpad(attempt::text, 4, '0'), history.run_id, 'stage-' || attempt, 1,
       'contractor/v1alpha1', '{}'::jsonb, 'contractor/v1alpha1', $5::jsonb,
       CASE WHEN attempt <= $3 THEN 'interrupted' ELSE 'preparing' END,
       CASE WHEN attempt <= $3 THEN 'termination_committed' ELSE 'created' END,
       CASE WHEN attempt <= $3 THEN 'contractor/v1alpha1' END,
       CASE WHEN attempt <= $3 THEN $4::jsonb END,
       CASE WHEN attempt <= $3 THEN 'abort-' || attempt END,
       CASE WHEN attempt <= $3 THEN started END,
       CASE WHEN attempt <= $3 THEN started END,
       started, started
FROM generate_series(1, $2) AS run_number
CROSS JOIN LATERAL (SELECT $1::text || run_number AS run_id) AS history
CROSS JOIN LATERAL generate_series(1, $3 + 1) AS attempt
CROSS JOIN LATERAL (
    SELECT timestamptz '2026-09-01 00:00:00+00' + attempt * interval '1 second' AS started
) AS timing`

// TestPostgresActiveStageReadUsesActiveStageIndex gives Runs the longest
// Stage history LIM-RUN-2 allows: the tick's read returns the one active
// StageExecution through stage_executions_run_active_idx in both the custom
// and the generic plan, without reading the history, and reports a Run
// without an active Stage as not found and one with two as a broken
// SCH-R114 invariant. GetRunState reads only the state.
func TestPostgresActiveStageReadUsesActiveStageIndex(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
	defer cancel()
	pool := isolatedRunStorePool(t, ctx)
	store := NewPostgresStore(pool)
	const runs, history = 10, 1023
	for index := 1; index <= runs; index++ {
		if _, err := admitTestRun(ctx, store, createTestRun(t, ctx, store, fmt.Sprintf("run-history-%d", index)).RunID); err != nil {
			t.Fatal(err)
		}
	}
	termination, err := json.Marshal(StageTermination{
		Outcome: TerminationInterrupted, Code: "control_lease_expired", Message: "allocation lost",
		Retryable: true, Phase: TerminationPreparing, OccurredAt: time.Date(2026, 9, 1, 0, 0, 0, 0, time.UTC),
	})
	if err != nil {
		t.Fatal(err)
	}
	stageContext, err := json.Marshal(StageContextSnapshot{
		Parameters: map[string]string{}, Artifacts: map[string]PinnedContextArtifact{},
	})
	if err != nil {
		t.Fatal(err)
	}
	loadWithoutLifecycleEvents(t, ctx, pool, "stage_executions", "stage_executions_capture_lifecycle_event",
		populateStageHistorySQL, "run-history-", runs, history, termination, stageContext)

	active, err := store.GetActiveStageExecution(ctx, "run-history-3")
	if err != nil || active.StageExecutionID != "run-history-3-stage-1024" || active.State != StagePreparing {
		t.Fatalf("active StageExecution = (%s %s, %v), want the preparing attempt 1024",
			active.StageExecutionID, active.State, err)
	}
	read := captureStatement(t, func(store *PostgresStore) error {
		_, err := store.GetActiveStageExecution(ctx, "run-history-3")
		return err
	})
	for _, generic := range []bool{false, true} {
		plan := explainPlan(t, ctx, pool, read, generic)
		if !strings.Contains(plan, "stage_executions_run_active_idx") || strings.Contains(plan, "Seq Scan") {
			t.Errorf("active StageExecution read (generic plan %v) reads the Stage history:\n%s", generic, plan)
		}
	}
	if state, err := store.GetRunState(ctx, "run-history-3"); err != nil || state != RunRunning {
		t.Fatalf("Run state = (%s, %v), want running", state, err)
	}
	if _, err := store.GetRunState(ctx, "run-missing"); !errors.Is(err, ErrNotFound) {
		t.Fatalf("missing Run state error = %v, want not found", err)
	}

	fresh := createTestRun(t, ctx, store, "run-without-active-stage")
	if _, err := store.GetActiveStageExecution(ctx, fresh.RunID); !errors.Is(err, ErrNotFound) {
		t.Fatalf("Run without a StageExecution: %v, want not found", err)
	}
	if _, err := store.CreateStageExecution(ctx, testActiveStageParams("run-history-5", "run-history-5-second", "second")); err != nil {
		t.Fatal(err)
	}
	if _, err := store.GetActiveStageExecution(ctx, "run-history-5"); !errors.Is(err, ErrMultipleActiveStages) {
		t.Fatalf("Run with two active StageExecutions: %v, want the SCH-R114 invariant error", err)
	}
}

func testActiveStageParams(runID, stageExecutionID, stageName string) CreateStageExecutionParams {
	return CreateStageExecutionParams{
		StageExecutionID: stageExecutionID, RunID: runID, StageName: stageName, Attempt: 1,
		StageSpecSchemaVersion: contracts.APIVersion, StageSpecSnapshot: json.RawMessage(`{}`),
		StageContextSchemaVersion: contracts.APIVersion, StageContext: StageContextSnapshot{},
	}
}

func testStageAllocation(stageExecutionID, allocationID string) StageAllocation {
	return StageAllocation{
		AllocationID: allocationID, StageExecutionID: stageExecutionID,
		LogicalAgentName: "builder", Namespace: "builder",
		AgentTemplateRef: contracts.AgentTemplateRef{
			TemplateID: "builder", Version: "1", Digest: "sha256:" + strings.Repeat("a", 64),
		},
		WorkerRuntimeRef:                  contracts.WorkerRuntimeRef{RuntimeID: "adk", Version: "1"},
		RuntimeAgentID:                    strings.Repeat("3", 64),
		RuntimeAgentInstanceID:            "runtime-paused",
		RuntimeAgentLabelRevision:         1,
		RuntimeConfigurationSchemaVersion: AllocationRuntimeConfigurationSchemaVersion,
		RuntimeConfiguration:              testAllocationRuntimeConfiguration(),
		PerformanceCollectionPolicy:       reporting.PerformanceCollectionDisabled,
	}
}
