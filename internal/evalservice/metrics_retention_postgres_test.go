package evalservice

import (
	"encoding/json"
	"reflect"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/auditstore/auditstoretest"
	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/evaldomain"
	"github.com/grauwolf32/contractor/internal/evalstore"
	pg "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/planner/stageresult"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/runstore/runstoretest"
	"github.com/grauwolf32/contractor/internal/runtimeconfig/runtimeconfigtest"
	"github.com/grauwolf32/contractor/internal/telemetry"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

// Telemetry retention keeps the StageMetrics an Eval member reads (OBS-R43,
// EVM-R239). Cleanup after they expire renews them and deletes only rows no
// member reads, so a finished experiment saves no new result, publishes no new
// view and still observes the usage it was collected with, which is what a
// later external result is checked against. Renewed rows leave the expired
// range, and once the experiment is purged they expire like any other.
func TestSharedPostgresEvalKeepsMemberStageMetricsPastTelemetryRetention(t *testing.T) {
	h := newHarness(t)
	e := h.prepared(t, "workflow")
	h.command(t, e, "start")
	c := h.coordinator(t, "metrics-retention")
	store := evalstore.NewPostgresStore(h.pool)
	stages := map[string]string{}
	for n := 0; n < 80; n++ {
		tick(t, c)
		finishRunsWithStageMetrics(t, h, stages)
		e = h.get(t, e.ID)
		view, err := store.LatestView(t.Context(), e.OwnerID, e.ID)
		if err != nil {
			t.Fatal(err)
		}
		if e.State == evaldomain.StateFinished && view != nil && view.Freshness == "current" {
			break
		}
	}
	if e.State != evaldomain.StateFinished || len(stages) == 0 {
		t.Fatalf("experiment did not finish with member metrics: state=%s stages=%d", e.State, len(stages))
	}
	memberStages := make([]string, 0, len(stages))
	for _, stage := range stages {
		memberStages = append(memberStages, stage)
	}
	records, projection := evalRecordCount(t, h.pool, e.ID), evalProjection(t, h.pool, e.ID)
	if records == 0 || projection[0] != projection[1] {
		t.Fatalf("fixture has %d records and projection %v", records, projection)
	}
	collected := map[string]evaldomain.Usage{}
	for _, member := range evalExecutedMembers(t, h.pool, e.ID) {
		record, err := store.LatestNativeRecord(t.Context(), e.OwnerID, e.ID, member, evaldomain.RecordKindResult)
		if err != nil || record == nil {
			t.Fatalf("member %s result = (%v, %v)", member, record, err)
		}
		var result evaldomain.ResultInput
		if err := json.Unmarshal(record.Document.Bytes(), &result); err != nil {
			t.Fatal(err)
		}
		if result.Usage.TotalTokens.Value == nil {
			t.Fatalf("member %s was collected without token usage: %+v", member, result.Usage.TotalTokens)
		}
		collected[member] = result.Usage
	}

	// A Run outside the experiment whose metrics expired after the members'.
	runs := runstore.NewPostgresStore(h.pool)
	if _, err := runstoretest.CreateRun(t.Context(), runs, runstore.CreateRunParams{
		LaneClass: resolved.LaneClassModel, RunID: "run-outside-eval", OwnerID: e.OwnerID,
		WorkflowName: "fixture", WorkflowVersion: "1", WorkflowSchemaVersion: contracts.APIVersion,
		WorkflowSnapshot: json.RawMessage(`{"name":"fixture"}`), Parameters: map[string]string{},
		RuntimeConfig: runtimeconfigtest.BuiltInRunSnapshot(),
	}); err != nil {
		t.Fatal(err)
	}
	for _, target := range []runstore.WorkflowRunState{runstore.RunPending, runstore.RunRunning} {
		run, err := runs.GetRun(t.Context(), "run-outside-eval")
		if err != nil {
			t.Fatal(err)
		}
		if _, err := runs.TransitionRun(t.Context(), run.RunID, run.State, target, runstore.Reason{Code: "fixture"}); err != nil {
			t.Fatal(err)
		}
	}
	finishedStageWithMetrics(t, h.pool, runs, "run-outside-eval", "stage-outside-eval")
	expire := func(stages []string, age string) {
		t.Helper()
		if _, err := h.pool.Exec(t.Context(), `UPDATE stage_metrics SET expires_at = clock_timestamp() - $2::interval
WHERE stage_execution_id = ANY($1)`, stages, age); err != nil {
			t.Fatal(err)
		}
	}
	expire(memberStages, "2 days")
	expire([]string{"stage-outside-eval"}, "1 day")

	cleanup := telemetry.NewRepository(h.pool)
	// The members' rows come first in expiry order and fill the first pass.
	if settled, err := cleanup.CleanupExpired(t.Context(), time.Now().UTC(), len(memberStages)); err != nil || settled != int64(len(memberStages)) {
		t.Fatalf("first cleanup settled (%d, %v), want %d member rows", settled, err, len(memberStages))
	}
	var kept, renewed int
	if err := h.pool.QueryRow(t.Context(), `SELECT count(*), count(*) FILTER (WHERE expires_at > clock_timestamp() + interval '29 days')
FROM stage_metrics WHERE stage_execution_id = ANY($1)`, memberStages).Scan(&kept, &renewed); err != nil {
		t.Fatal(err)
	}
	if kept != len(memberStages) || renewed != kept {
		t.Fatalf("cleanup kept %d and renewed %d of %d member metric rows", kept, renewed, len(memberStages))
	}
	// The renewed rows left the expired range, so a budget of one reaches the
	// other Run's row.
	if settled, err := cleanup.CleanupExpired(t.Context(), time.Now().UTC(), 1); err != nil || settled != 1 {
		t.Fatalf("second cleanup settled (%d, %v), want the row outside the experiment", settled, err)
	}
	if n := stageMetricCount(t, h.pool, []string{"stage-outside-eval"}); n != 0 {
		t.Fatalf("metrics outside the experiment were kept: %d", n)
	}

	for range 5 {
		tick(t, c)
	}
	if got := evalRecordCount(t, h.pool, e.ID); got != records {
		t.Fatalf("retention cleanup made the coordinator save %d records, had %d", got, records)
	}
	if got := evalProjection(t, h.pool, e.ID); got != projection {
		t.Fatalf("retention cleanup changed the projection revision from %v to %v", projection, got)
	}
	for member, usage := range collected {
		row, err := store.ExecutionObservation(t.Context(), e.OwnerID, e.ID, member)
		if err != nil {
			t.Fatal(err)
		}
		inventory, err := store.Inventory(t.Context(), e.OwnerID, e.ID, member)
		if err != nil {
			t.Fatal(err)
		}
		current, err := observedUsage(t.Context(), h.pool, e.OwnerID, member, executionView(row), inventory)
		if err != nil || !reflect.DeepEqual(current, usage) {
			t.Fatalf("member %s usage after cleanup = (%+v, %v), collected %+v", member, current, err, usage)
		}
	}

	// A change of the metrics content still marks the members for recollection.
	if _, err := h.pool.Exec(t.Context(), `UPDATE stage_metrics SET summary = summary WHERE stage_execution_id = $1`, memberStages[0]); err != nil {
		t.Fatal(err)
	}
	if got := evalProjection(t, h.pool, e.ID); got[0] == got[1] {
		t.Fatalf("a metrics content change did not mark the experiment: %v", got)
	}

	// Once the experiment is purged, its Runs' rows expire like any other.
	if err := pg.InTx(t.Context(), h.pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		for _, statement := range []string{
			`SELECT set_config('contractor.eval_purge', 'on', true)`,
			`DELETE FROM eval_selections WHERE experiment_id = '` + e.ID + `'`,
			`DELETE FROM eval_experiments WHERE experiment_id = '` + e.ID + `'`,
		} {
			if _, err := tx.Exec(t.Context(), statement); err != nil {
				return err
			}
		}
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	expire(memberStages, "1 day")
	if settled, err := cleanup.CleanupExpired(t.Context(), time.Now().UTC(), 500); err != nil || settled != int64(len(memberStages)) {
		t.Fatalf("cleanup after purge settled (%d, %v), want %d", settled, err, len(memberStages))
	}
	if n := stageMetricCount(t, h.pool, memberStages); n != 0 {
		t.Fatalf("cleanup kept %d metric rows of a purged experiment", n)
	}
}

// The StageMetrics of a child Run of the Audit an Eval member executed are
// kept the same way (EVM-R239).
func TestSharedPostgresEvalKeepsAuditChildStageMetricsPastTelemetryRetention(t *testing.T) {
	h, _, _, claim, execution := evalAuditExecution(t)
	audit, err := h.audit.Get(t.Context(), h.scope.OwnerID, claim.AuditID)
	if err != nil {
		t.Fatal(err)
	}
	runID := "run-" + execution.ExecutionID
	if err := pg.InTx(t.Context(), h.pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		if _, err := runstore.NewPostgresStore(tx).CreateAuditRun(t.Context(), runstore.CreateAuditRunParams{
			CreateRunParams: runstore.CreateRunParams{
				LaneClass: resolved.LaneClassModel, RunID: runID, OwnerID: audit.OwnerID, ProjectID: &audit.ProjectID,
				WorkflowName: "fixture", WorkflowVersion: "1", WorkflowSchemaVersion: contracts.APIVersion,
				WorkflowSnapshot: json.RawMessage(`{"name":"fixture"}`), Parameters: map[string]string{},
				RuntimeConfig: runtimeconfigtest.BuiltInRunSnapshot(),
			},
			AuditExecutionID: execution.ExecutionID, AuditSubmissionKey: execution.SubmissionKey,
		}); err != nil {
			return err
		}
		_, err := auditstoretest.NewStore(tx).BindRun(t.Context(), auditstore.BindRunParams{
			Claim: claim, ExecutionID: execution.ExecutionID, RunID: runID,
		})
		return err
	}); err != nil {
		t.Fatal(err)
	}
	runs := runstore.NewPostgresStore(h.pool)
	for _, transition := range [][2]runstore.WorkflowRunState{
		{runstore.RunInitializing, runstore.RunPending}, {runstore.RunPending, runstore.RunRunning},
	} {
		if _, err := runs.TransitionRun(t.Context(), runID, transition[0], transition[1], runstore.Reason{Code: "fixture"}); err != nil {
			t.Fatal(err)
		}
	}
	finishedStageWithMetrics(t, h.pool, runs, runID, "stage-audit-child")
	if _, err := h.pool.Exec(t.Context(), `UPDATE stage_metrics SET expires_at = clock_timestamp() - interval '1 day'
WHERE stage_execution_id = 'stage-audit-child'`); err != nil {
		t.Fatal(err)
	}
	if settled, err := telemetry.NewRepository(h.pool).CleanupExpired(t.Context(), time.Now().UTC(), 500); err != nil || settled != 1 {
		t.Fatalf("cleanup settled (%d, %v), want the Audit child row", settled, err)
	}
	var renewed bool
	if err := h.pool.QueryRow(t.Context(), `SELECT expires_at > clock_timestamp() + interval '29 days'
FROM stage_metrics WHERE stage_execution_id = 'stage-audit-child'`).Scan(&renewed); err != nil || !renewed {
		t.Fatalf("Audit child metrics renewed = (%v, %v)", renewed, err)
	}
}

// finishRunsWithStageMetrics gives each pending or running fixture Run one
// succeeded StageExecution with complete Planner metrics, records it in
// stages, and then finishes the Runs.
func finishRunsWithStageMetrics(t *testing.T, h *serviceHarness, stages map[string]string) {
	t.Helper()
	rows, err := h.pool.Query(t.Context(), `SELECT run_id, state FROM workflow_runs WHERE state IN ('pending', 'running') ORDER BY run_id`)
	if err != nil {
		t.Fatal(err)
	}
	states, err := pgx.CollectRows(rows, func(row pgx.CollectableRow) ([2]string, error) {
		var value [2]string
		return value, row.Scan(&value[0], &value[1])
	})
	if err != nil {
		t.Fatal(err)
	}
	runs := runstore.NewPostgresStore(h.pool)
	for _, run := range states {
		if runstore.WorkflowRunState(run[1]) == runstore.RunPending {
			if _, err := runs.TransitionRun(t.Context(), run[0], runstore.RunPending, runstore.RunRunning, runstore.Reason{Code: "fixture_admitted"}); err != nil {
				t.Fatal(err)
			}
		}
		if _, ok := stages[run[0]]; !ok {
			stages[run[0]] = "stage-" + run[0]
			finishedStageWithMetrics(t, h.pool, runs, run[0], stages[run[0]])
		}
	}
	h.finishRuns(t)
}

// finishedStageWithMetrics stores a succeeded StageExecution of running Run
// runID whose StageMetrics report complete Planner usage of 42 tokens.
func finishedStageWithMetrics(t *testing.T, pool *pgxpool.Pool, runs *runstore.PostgresStore, runID, stageID string) {
	t.Helper()
	if err := runstoretest.CreateRunningStage(t.Context(), runs, runID, stageID, contracts.APIVersion); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(t.Context(), `INSERT INTO stage_metrics(stage_execution_id, metrics_schema_version, metrics, summary)
VALUES($1, $2, '{"planner":{"complete":true,"truncated":false,"metrics":{"modelCalls":2,"inputTokens":30,"outputTokens":12,"totalTokens":42,"tools":{}}},"workers":{},"runtime":{}}',
       '{"totalTokens":42}')`, stageID, contracts.APIVersion); err != nil {
		t.Fatal(err)
	}
	result := stageresult.StageContentResult{
		APIVersion: contracts.APIVersion, Outcome: stageresult.StageSucceeded,
		Summary: "done", Artifacts: map[string]contracts.ArtifactRef{},
	}
	if err := runs.EnterFinalizing(t.Context(), runstore.EnterFinalizingParams{
		StageExecutionID: stageID, ResultSchemaVersion: contracts.APIVersion,
		Candidate: result, FinalizationID: "finalize-" + stageID,
		Deadline: time.Now().Add(time.Minute), Reason: runstore.Reason{Code: "planner_completed"},
	}); err != nil {
		t.Fatal(err)
	}
	if err := runs.CompleteStageResult(t.Context(), stageID, contracts.APIVersion, result); err != nil {
		t.Fatal(err)
	}
}

func evalRecordCount(t *testing.T, pool *pgxpool.Pool, experiment string) int {
	t.Helper()
	var n int
	if err := pool.QueryRow(t.Context(), `SELECT count(*) FROM eval_records WHERE experiment_id = $1`, experiment).Scan(&n); err != nil {
		t.Fatal(err)
	}
	return n
}

// evalProjection returns the projection queue's revision and published
// revision of experiment.
func evalProjection(t *testing.T, pool *pgxpool.Pool, experiment string) [2]int64 {
	t.Helper()
	var revision [2]int64
	if err := pool.QueryRow(t.Context(), `SELECT revision, published_revision FROM eval_projection_queue WHERE experiment_id = $1`,
		experiment).Scan(&revision[0], &revision[1]); err != nil {
		t.Fatal(err)
	}
	return revision
}

func evalExecutedMembers(t *testing.T, pool *pgxpool.Pool, experiment string) []string {
	t.Helper()
	rows, err := pool.Query(t.Context(), `SELECT member_id FROM eval_submissions
WHERE experiment_id = $1 AND execution_id IS NOT NULL ORDER BY member_id`, experiment)
	if err != nil {
		t.Fatal(err)
	}
	members, err := pgx.CollectRows(rows, pgx.RowTo[string])
	if err != nil || len(members) == 0 {
		t.Fatalf("executed members = (%v, %v)", members, err)
	}
	return members
}

func stageMetricCount(t *testing.T, pool *pgxpool.Pool, stages []string) int {
	t.Helper()
	var n int
	if err := pool.QueryRow(t.Context(), `SELECT count(*) FROM stage_metrics WHERE stage_execution_id = ANY($1)`, stages).Scan(&n); err != nil {
		t.Fatal(err)
	}
	return n
}
