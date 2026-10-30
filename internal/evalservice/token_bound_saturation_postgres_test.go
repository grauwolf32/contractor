package evalservice

import (
	"encoding/json"
	"fmt"
	"math"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/reporting"
	"github.com/grauwolf32/contractor/internal/evaldomain"
	"github.com/grauwolf32/contractor/internal/evalstore"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/telemetry"
	"github.com/jackc/pgx/v5"
)

// TestSharedPostgresEvalTokenLowerBoundSaturatesPastInt64 drives two accepted
// members through the coordinator while each owned Run holds two stages whose
// summaries sum past 2^63-1. Each stage is the largest a valid Stage can
// summarize: its Planner and 1,000 Workers, every report at the 2^53-1 counter
// bound, 9,016,206,453,995,731,991 tokens. The known lower bound saturates at
// the int64 maximum instead of failing to scan, the experiment sum stays at
// that maximum instead of overflowing bigint, the maximum reaches even the
// largest token limit, and the members settle once their Runs drain
// (EVM-R248).
func TestSharedPostgresEvalTokenLowerBoundSaturatesPastInt64(t *testing.T) {
	h := newHarness(t)
	e := h.create(t, "workflow")
	var draft evaldomain.Draft
	if err := json.Unmarshal(e.Draft.Bytes(), &draft); err != nil {
		t.Fatal(err)
	}
	limit := int64(math.MaxInt64)
	draft.Budgets.MaxInFlight = 2
	draft.Budgets.MaxObservedTotalTokens = &limit
	update := frozen(t, "DraftUpdate", evaldomain.DraftUpdate{Name: e.Name, Draft: draft})
	if err := h.service.tx(t.Context(), func(store *evalstore.Store) error {
		_, err := store.UpdateDraft(t.Context(), h.scope, e.ID, update, identity(t, "saturation", e.Revision, update))
		return err
	}); err != nil {
		t.Fatal(err)
	}
	h.command(t, h.get(t, e.ID), "prepare")
	tick(t, h.coordinator(t, "prepare"))
	h.command(t, h.get(t, e.ID), "start")
	c := h.coordinator(t, "saturation-controller")
	accepted := func() int {
		t.Helper()
		var n int
		if err := h.pool.QueryRow(t.Context(), `SELECT count(*) FROM eval_submissions
WHERE experiment_id = $1 AND state = 'accepted' AND execution_id IS NOT NULL`, e.ID).Scan(&n); err != nil {
			t.Fatal(err)
		}
		return n
	}
	for attempt := 0; attempt < 30 && accepted() < 2; attempt++ {
		tick(t, c)
	}
	if n := accepted(); n != 2 {
		t.Fatalf("fixture accepted %d members, want 2", n)
	}

	metrics, summary := largestStageMetrics(t)
	rows, err := h.pool.Query(t.Context(), `SELECT run_id, state FROM workflow_runs ORDER BY run_id`)
	if err != nil {
		t.Fatal(err)
	}
	runStates, err := pgx.CollectRows(rows, func(row pgx.CollectableRow) ([2]string, error) {
		var value [2]string
		return value, row.Scan(&value[0], &value[1])
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(runStates) != 2 {
		t.Fatalf("fixture has %d Runs, want one per accepted member", len(runStates))
	}
	runs := runstore.NewPostgresStore(h.pool)
	for _, run := range runStates {
		if runstore.WorkflowRunState(run[1]) == runstore.RunPending {
			if _, err := runs.TransitionRun(t.Context(), run[0], runstore.RunPending, runstore.RunRunning, runstore.Reason{Code: "fixture_admitted"}); err != nil {
				t.Fatal(err)
			}
		}
		for stage := range 2 {
			finishedStageWithMetricsDocument(t, h.pool, runs, run[0],
				fmt.Sprintf("stage-%s-%d", run[0], stage), fmt.Sprintf("stage-%d", stage), metrics, summary)
		}
	}

	tick(t, c)
	e = h.get(t, e.ID)
	if e.ObservedTokens != math.MaxInt64 {
		t.Fatalf("experiment observed tokens = %d, want the saturated %d", e.ObservedTokens, int64(math.MaxInt64))
	}
	var saturatedMembers int
	if err := h.pool.QueryRow(t.Context(), `SELECT count(*) FROM eval_submissions
WHERE experiment_id = $1 AND observed_tokens = 9223372036854775807`, e.ID).Scan(&saturatedMembers); err != nil {
		t.Fatal(err)
	}
	if saturatedMembers != 2 {
		t.Fatalf("%d members hold the saturated lower bound, want 2", saturatedMembers)
	}
	if e.State != evaldomain.StateCancelling {
		t.Fatalf("saturated lower bound did not reach the token limit: state %s", e.State)
	}
	for attempt := 0; attempt < 6 && h.get(t, e.ID).State != evaldomain.StateCancelled; attempt++ {
		tick(t, c)
		h.finishRuns(t)
	}
	e = h.get(t, e.ID)
	if e.State != evaldomain.StateCancelled || e.Outstanding != 0 || accepted() != 0 {
		t.Fatalf("members did not settle: state=%s outstanding=%d accepted=%d", e.State, e.Outstanding, accepted())
	}
	if e.ObservedTokens != math.MaxInt64 {
		t.Fatalf("settled experiment observed tokens = %d", e.ObservedTokens)
	}
}

// largestStageMetrics returns the StageMetrics document and summary of a
// Stage whose Planner and 1,000 Workers each report every token and call
// counter at the report bound, 2^53-1.
func largestStageMetrics(t *testing.T) (string, string) {
	t.Helper()
	bound := int64(contracts.MaxExactInteger)
	report := func(id string) reporting.ExecutionReport {
		return reporting.ExecutionReport{
			ReportID: id, Complete: true,
			Metrics: reporting.ExecutionMetrics{
				ModelCalls: &bound, InputTokens: &bound, OutputTokens: &bound, TotalTokens: &bound,
				Tools: map[string]reporting.ToolMetrics{},
			},
			ToolCalls: []reporting.ToolCallRecord{}, Errors: []reporting.ExecutionError{},
		}
	}
	planner := report("planner")
	workers := make(map[string]reporting.AllocationFinalReport, 1000)
	for index := range 1000 {
		name := fmt.Sprintf("worker-%04d", index)
		workers[name] = reporting.AllocationFinalReport{
			Worker: report(name), Runtime: reporting.RuntimeReport{Complete: true},
		}
	}
	stage := telemetry.BuildStageMetrics(&planner, workers)
	if err := stage.Validate(); err != nil {
		t.Fatal(err)
	}
	summary := telemetry.Summarize(stage)
	if summary.TotalTokens != 1001*bound {
		t.Fatalf("largest stage summary = %d tokens, want %d", summary.TotalTokens, 1001*bound)
	}
	metrics, err := json.Marshal(stage)
	if err != nil {
		t.Fatal(err)
	}
	encodedSummary, err := json.Marshal(summary)
	if err != nil {
		t.Fatal(err)
	}
	return string(metrics), string(encodedSummary)
}
