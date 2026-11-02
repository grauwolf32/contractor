package evalservice

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"slices"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"

	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
	"github.com/grauwolf32/contractor/internal/contracts/reporting"
	"github.com/grauwolf32/contractor/internal/evaldomain"
	"github.com/grauwolf32/contractor/internal/evalstore"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/runstore/runstoretest"
	"github.com/grauwolf32/contractor/internal/runtimeconfig/runtimeconfigtest"
	"github.com/grauwolf32/contractor/internal/telemetry"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

// createRunningUsageRun stores Run runID of owner "owner" and admits it, so
// its Stages can store metrics that observedRunUsage collects.
func createRunningUsageRun(t *testing.T, runs *runstore.PostgresStore, runID string) {
	t.Helper()
	if _, err := runstoretest.CreateRun(t.Context(), runs, runstore.CreateRunParams{
		LaneClass: resolved.LaneClassModel,
		RunID:     runID, OwnerID: "owner", WorkflowName: "fixture", WorkflowVersion: "1",
		WorkflowSchemaVersion: contracts.APIVersion, WorkflowSnapshot: json.RawMessage(`{"name":"fixture"}`),
		Parameters: map[string]string{}, RuntimeConfig: runtimeconfigtest.BuiltInRunSnapshot(),
	}); err != nil {
		t.Fatal(err)
	}
	if _, err := runs.TransitionRun(
		t.Context(), runID, runstore.RunInitializing, runstore.RunPending, runstore.Reason{Code: "fixture"},
	); err != nil {
		t.Fatal(err)
	}
	if _, err := runs.TransitionRun(
		t.Context(), runID, runstore.RunPending, runstore.RunRunning, runstore.Reason{Code: "admitted"},
	); err != nil {
		t.Fatal(err)
	}
}

// observedRunUsage collects the usage of a member that executed Run runID.
func observedRunUsage(t *testing.T, pool *pgxpool.Pool, runID string) (evaldomain.Usage, error) {
	t.Helper()
	ref := evaldomain.ExecutionRef{Kind: "run", ID: runID}
	inventory := evalstore.Inventory{Complete: true, Gaps: []string{}, Entries: []evalstore.InventoryEntry{{Execution: &ref, Available: true}}}
	start, finish := time.Unix(0, 0).UTC(), time.Unix(1, 0).UTC()
	return observedUsage(t.Context(), pool, "owner", evaldomain.Digest([]byte(runID))[7:], ExecutionView{
		Ref: &ref, State: "succeeded", StartedAt: &start, FinishedAt: &finish,
	}, inventory, artifactpgtest.NewStorage())
}

// Collection reads the Planner's counters as stored, whatever the Stage's
// Planner: a report without model counters leaves them unavailable even for
// the model-free passthrough@1, whose own reports state their zeros
// (EVM-R81). No reader derives a counter from the Planner identity.
func TestEvalObservedUsageReadsPlannerCountersAsReported(t *testing.T) {
	pool := serviceTestPool(t)
	runs := runstore.NewPostgresStore(pool, artifactpgtest.NewStorage())
	for _, planner := range []string{"passthrough", "streamline"} {
		createRunningUsageRun(t, runs, planner)
		if _, err := runs.CreateStageExecution(t.Context(), runstore.CreateStageExecutionParams{
			LaneClass:        resolved.LaneClassModel,
			StageExecutionID: planner, RunID: planner, StageName: "fixture", Attempt: 1,
			StageSpecSchemaVersion:    contracts.APIVersion,
			StageSpecSnapshot:         json.RawMessage(`{"planner":{"plannerId":"` + planner + `","version":"1"}}`),
			StageContextSchemaVersion: contracts.APIVersion, StageContext: runstore.StageContextSnapshot{},
		}); err != nil {
			t.Fatal(err)
		}
		if _, err := pool.Exec(t.Context(), `INSERT INTO stage_metrics(stage_execution_id,metrics_schema_version,metrics,summary)
VALUES($1,$2,'{"planner":{"complete":true,"truncated":false,"metrics":{"tools":{}}},"workers":{},"runtime":{}}','{}')`, planner, contracts.APIVersion); err != nil {
			t.Fatal(err)
		}
		usage, err := observedRunUsage(t, pool, planner)
		if err != nil {
			t.Fatal(err)
		}
		for name, measure := range map[string]evaldomain.Measure{"totalTokens": usage.TotalTokens, "modelCalls": usage.ModelCalls} {
			if measure.Value != nil || measure.Completeness != "unavailable" {
				t.Fatalf("%s Stage %s = %+v, want unavailable without the reported counter", planner, name, measure)
			}
		}
		if usage.ToolCalls.Value == nil || *usage.ToolCalls.Value != 0 || usage.ToolCalls.Completeness != "complete" {
			t.Fatalf("%s Stage tool calls = %+v, want the reported zero", planner, usage.ToolCalls)
		}
	}
}

// A passthrough@1 Planner report as the Planner builds it, recorded and
// rebuilt into StageMetrics by the telemetry repository, passes the strict
// reader and yields complete usage: the report states its zero model
// counters itself (EVM-R81).
func TestEvalObservedUsageOfRecordedPassthroughReportIsComplete(t *testing.T) {
	pool := serviceTestPool(t)
	runs := runstore.NewPostgresStore(pool, artifactpgtest.NewStorage())
	createRunningUsageRun(t, runs, "passthrough-run")
	const stage = "passthrough-stage"
	if err := runstoretest.CreateRunningNamedStage(t.Context(), runs, "passthrough-run", stage, "fixture", contracts.APIVersion); err != nil {
		t.Fatal(err)
	}
	value := func(v int64) *int64 { return &v }
	session := stage + "-session"
	started := time.Now().UTC().Truncate(time.Millisecond)
	repository := telemetry.NewRepository(pool)
	if err := repository.RecordPlannerReport(t.Context(), telemetry.PlannerReportEnvelope{
		StageExecutionID: stage, SessionID: session, InvocationID: stage + "-invocation",
		StartedAt: started, FinishedAt: started.Add(time.Second), ReportSchemaVersion: contracts.APIVersion,
		Report: reporting.ExecutionReport{
			ReportID: "planner-" + session, Complete: true,
			Metrics: reporting.ExecutionMetrics{
				DurationMS: value(1000), ModelCalls: value(0), InputTokens: value(0),
				OutputTokens: value(0), TotalTokens: value(0),
				Tools: map[string]reporting.ToolMetrics{"a2a.invoke": {Calls: value(1), Succeeded: value(1), Failed: value(0)}},
			},
			ToolCalls: []reporting.ToolCallRecord{{
				CallID: "a2a-planner-" + session, Tool: "a2a.invoke",
				Arguments: reporting.ToolArguments{"binding": "builder"}, Outcome: reporting.ToolCallSucceeded,
				DurationMS: value(900),
			}},
			Errors: []reporting.ExecutionError{},
		},
	}); err != nil {
		t.Fatal(err)
	}
	if _, err := repository.RebuildStageMetrics(t.Context(), stage, contracts.APIVersion); err != nil {
		t.Fatal(err)
	}
	usage, err := observedRunUsage(t, pool, "passthrough-run")
	if err != nil {
		t.Fatal(err)
	}
	for name, want := range map[string]struct {
		measure evaldomain.Measure
		value   float64
	}{
		"modelCalls": {usage.ModelCalls, 0}, "inputTokens": {usage.InputTokens, 0},
		"outputTokens": {usage.OutputTokens, 0}, "totalTokens": {usage.TotalTokens, 0},
		"toolCalls": {usage.ToolCalls, 1}, "toolFailures": {usage.ToolFailures, 0},
	} {
		if want.measure.Value == nil || *want.measure.Value != want.value || want.measure.Completeness != "complete" {
			t.Fatalf("passthrough %s = %+v, want complete %v", name, want.measure, want.value)
		}
	}
}

// A stored StageMetrics document whose projection breaks the contract its
// reports were validated against on write fails collection as an integrity
// error instead of yielding usage. jsonb keeps one member of a duplicated
// name, so duplicate members are covered on the decoded bytes
// (TestEvalAttemptMetricsRejectStoredMetricsThatBreakTheirContract).
func TestEvalObservedUsageRejectsStoredMetricsThatBreakTheirContract(t *testing.T) {
	pool := serviceTestPool(t)
	runs := runstore.NewPostgresStore(pool, artifactpgtest.NewStorage())
	for index, document := range []string{
		`{"planner":{"complete":true,"truncated":false,"metrics":{"modelCalls":-1,"tools":{}}},"workers":{},"runtime":{}}`,
		`{"planner":{"complete":true,"truncated":false,"metrics":{"totalTokens":9007199254740992,"tools":{}}},"workers":{},"runtime":{}}`,
		`{"planner":{"complete":true,"truncated":false,"metrics":{"tools":{"read":{"calls":-1}}}},"workers":{},"runtime":{}}`,
		`{"planner":{"complete":true,"truncated":false,"metrics":{"cachedTokens":1,"tools":{}}},"workers":{},"runtime":{}}`,
		`{"planner":{"complete":true,"truncated":false,"metrics":{"tools":{"read":{"calls":1,"retries":0}}}},"workers":{},"runtime":{}}`,
		`{"planner":{"complete":true,"metrics":{"tools":{}}},"workers":{},"runtime":{}}`,
		`{"workers":{"builder":{"complete":true,"truncated":false,"metrics":{}}},"runtime":{"builder":{"complete":true}}}`,
		`{"workers":{},"runtime":{"builder":{}}}`,
	} {
		run := fmt.Sprintf("corrupt-%d", index)
		createRunningUsageRun(t, runs, run)
		if err := runstoretest.CreateRunningNamedStage(t.Context(), runs, run, run+"-stage", "fixture", contracts.APIVersion); err != nil {
			t.Fatal(err)
		}
		if _, err := pool.Exec(t.Context(), `INSERT INTO stage_metrics(stage_execution_id,metrics_schema_version,metrics,summary)
VALUES($1,$2,$3::jsonb,'{}')`, run+"-stage", contracts.APIVersion, document); err != nil {
			t.Fatal(err)
		}
		if usage, err := observedRunUsage(t, pool, run); !errors.Is(err, errStoredMetrics) {
			t.Fatalf("stored %s = (%+v, %v), want an integrity error", document, usage, err)
		}
	}
}

// The inventory authority/pagination journey is covered at the public boundary.
// This test exercises real metric SQL and normalization over its all-role fixture.
func TestSharedEvalObservedAuditUsageCountsStagesOnceAndKeepsParentDuration(t *testing.T) {
	pool := serviceTestPool(t)
	raw := contractstest.ReadFile(t, "api", "testdata", "evals", "audit-accounting.json")
	var fixture struct {
		Observation evaldomain.UsageObservation `json:"observation"`
	}
	err := json.Unmarshal(raw, &fixture)
	if err != nil {
		t.Fatal(err)
	}
	input := fixture.Observation
	inventory := evalstore.Inventory{Complete: true, Gaps: []string{}}
	runs := runstore.NewPostgresStore(pool, artifactpgtest.NewStorage())
	for _, ref := range input.Executions {
		inventory.Entries = append(inventory.Entries, evalstore.InventoryEntry{Execution: &ref, Available: true})
		if ref.Kind != "run" {
			continue
		}
		_, err = runstoretest.CreateRun(t.Context(), runs, runstore.CreateRunParams{
			LaneClass: resolved.LaneClassModel,
			RunID:     ref.ID, OwnerID: "owner", WorkflowName: "fixture", WorkflowVersion: "1",
			WorkflowSchemaVersion: contracts.APIVersion, WorkflowSnapshot: json.RawMessage(`{"name":"fixture"}`),
			Parameters: map[string]string{}, RuntimeConfig: runtimeconfigtest.BuiltInRunSnapshot(),
		})

		if err != nil {
			t.Fatal(err)
		}
		if _, err = runs.TransitionRun(
			t.Context(), ref.ID, runstore.RunInitializing, runstore.RunPending, runstore.Reason{Code: "fixture"},
		); err != nil {
			t.Fatal(err)
		}
		if _, err = runs.TransitionRun(
			t.Context(), ref.ID, runstore.RunPending, runstore.RunRunning, runstore.Reason{Code: "admitted"},
		); err != nil {
			t.Fatal(err)
		}
	}
	seen := map[string]bool{}
	for _, attempt := range input.Attempts {
		if !seen[attempt.StageExecutionID] {
			_, err = runs.CreateStageExecution(t.Context(), runstore.CreateStageExecutionParams{
				LaneClass:        resolved.LaneClassModel,
				StageExecutionID: attempt.StageExecutionID, RunID: attempt.RunID, StageName: attempt.StageExecutionID,
				Attempt: 1, StageSpecSchemaVersion: contracts.APIVersion, StageSpecSnapshot: json.RawMessage(`{"objective":"fixture"}`),
				StageContextSchemaVersion: contracts.APIVersion, StageContext: runstore.StageContextSnapshot{},
			})
			if err != nil {
				t.Fatal(err)
			}
			seen[attempt.StageExecutionID] = true
		}
		metrics := map[string]any{
			"inputTokens": attempt.Metrics["inputTokens"], "outputTokens": attempt.Metrics["outputTokens"],
			"totalTokens": attempt.Metrics["totalTokens"], "modelCalls": attempt.Metrics["modelCalls"],
			"tools": map[string]any{},
		}
		document, err := json.Marshal(map[string]any{
			"planner": map[string]any{"complete": true, "truncated": false, "metrics": metrics, "prompt": "PRIVATE_PROMPT"},
			"workers": map[string]any{}, "runtime": map[string]any{},
		})
		if err != nil {
			t.Fatal(err)
		}
		// The repeated snapshot is an overwrite, never a token delta. Summary
		// totals and private report fields must not become collection authority.
		_, err = pool.Exec(t.Context(), `
INSERT INTO stage_metrics(stage_execution_id, metrics_schema_version, metrics, summary)
VALUES($1,$2,$3,'{"totalTokens":99999}'::jsonb)
ON CONFLICT(stage_execution_id) DO UPDATE SET metrics=EXCLUDED.metrics`, attempt.StageExecutionID, contracts.APIVersion, document)
		if err != nil {
			t.Fatal(err)
		}
	}
	start, err := time.Parse(time.RFC3339Nano, *input.StartedAt)
	if err != nil {
		t.Fatal(err)
	}
	finish, err := time.Parse(time.RFC3339Nano, *input.FinishedAt)
	if err != nil {
		t.Fatal(err)
	}
	execution := evaldomain.ExecutionView{Ref: &input.Parent, State: "succeeded", StartedAt: &start, FinishedAt: &finish}
	usage, err := observedUsage(t.Context(), pool, "owner", input.MemberID, execution, inventory, artifactpgtest.NewStorage())
	if err != nil || usage.TotalTokens.Value == nil || *usage.TotalTokens.Value != 100 || usage.TotalTokens.Completeness != "complete" || *usage.ModelCalls.Value != 4 || *usage.WallMS.Value != 2000 {
		t.Fatal("wrong SQL stage/parent accounting", usage, err)
	}
	runsRead := []string{"check-1", "discovery-1", "assessment-1"}
	snapshots, err := evalstore.NewPostgresStore(pool, artifactpgtest.NewStorage()).MetricSnapshots(t.Context(), "owner", runsRead)
	if err != nil || len(snapshots) != 4 {
		t.Fatal("metric snapshot cohort", snapshots, err)
	}
	for _, snapshot := range snapshots {
		if bytes.Contains(snapshot.Document, []byte("PRIVATE_")) || bytes.Contains(snapshot.Document, []byte("99999")) {
			t.Fatal("private/aggregate fields leaked into metric projection")
		}
	}
	inventory.Entries = append(inventory.Entries, inventory.Entries[1])
	replay, err := observedUsage(t.Context(), pool, "owner", input.MemberID, execution, inventory, artifactpgtest.NewStorage())
	if err != nil || !reflect.DeepEqual(usage, replay) {
		t.Fatal("repeated inventory or snapshots doubled usage", err)
	}
	if _, err = pool.Exec(t.Context(), `DELETE FROM stage_metrics WHERE stage_execution_id='assessment'`); err != nil {
		t.Fatal(err)
	}
	partial, err := observedUsage(t.Context(), pool, "owner", input.MemberID, execution, inventory, artifactpgtest.NewStorage())
	if err != nil || partial.TotalTokens.Value == nil || *partial.TotalTokens.Value != 60 || partial.TotalTokens.Completeness != "partial" || partial.WallMS.Completeness != "complete" || *partial.WallMS.Value != 2000 {
		t.Fatal("missing child metrics lost partial scope or parent duration", partial, err)
	}
	snapshots, err = evalstore.NewPostgresStore(pool, artifactpgtest.NewStorage()).MetricSnapshots(t.Context(), "foreign", runsRead)
	if err != nil || len(snapshots) != 0 {
		t.Fatal("foreign child metrics disclosed", err)
	}
}

// Identical overflowing StageMetrics always produce the same usage: a counter
// whose exact sum exceeds the int64 range, within one snapshot or across a
// member's snapshots, is unavailable in every collection, so recollecting the
// unchanged observations appends no native record after the accepted one
// (EVM-R240, EVM-R82).
func TestSharedPostgresEvalOverflowingUsageRecollectsWithoutNewRecords(t *testing.T) {
	h := newHarness(t)
	e := h.prepared(t, "workflow")
	h.command(t, e, "start")
	c := h.coordinator(t, "usage-overflow")
	store := evalstore.NewPostgresStore(h.pool, artifactpgtest.NewStorage())
	// Every counter lies within 2^53-1 (OBS-R111), so only many contributions
	// leave the int64 range. The first stage's 1,025 tools at the bound
	// overflow its tool calls inside its snapshot; 1,024 of them failed, which
	// fits the range at 2^63-1,024 and overflows only together with the
	// second stage's 1,024 failures. Small summaries keep the budget's known
	// lower bound out of this test.
	bound := int64(contracts.MaxExactInteger)
	tools := map[string]reporting.ToolMetrics{}
	for index := range 1025 {
		calls, failed := bound, bound
		if index == 1024 {
			failed = 0
		}
		tools[fmt.Sprintf("tool-%04d", index)] = reporting.ToolMetrics{Calls: &calls, Failed: &failed}
	}
	encodedTools, err := json.Marshal(tools)
	if err != nil {
		t.Fatal(err)
	}
	documents := []string{
		`{"planner":{"complete":true,"truncated":false,"metrics":{"modelCalls":1,"inputTokens":100,` +
			`"outputTokens":20,"totalTokens":120,"tools":` + string(encodedTools) + `}},"workers":{},"runtime":{}}`,
		`{"planner":{"complete":true,"truncated":false,"metrics":{"modelCalls":1,"inputTokens":200,` +
			`"outputTokens":0,"totalTokens":200,"tools":{"read":{"calls":1024,"failed":1024}}}},"workers":{},"runtime":{}}`,
	}
	runs := runstore.NewPostgresStore(h.pool, artifactpgtest.NewStorage())
	stages := []string{}
	finished := map[string]bool{}
	for n := 0; n < 80; n++ {
		tick(t, c)
		rows, err := h.pool.Query(t.Context(), `SELECT run_id FROM workflow_runs WHERE state IN ('pending','running') ORDER BY run_id`)
		if err != nil {
			t.Fatal(err)
		}
		active, err := pgx.CollectRows(rows, pgx.RowTo[string])
		if err != nil {
			t.Fatal(err)
		}
		for _, run := range active {
			if finished[run] {
				continue
			}
			current, err := runs.GetRun(t.Context(), run)
			if err != nil {
				t.Fatal(err)
			}
			if current.State == runstore.RunPending {
				if _, err := runs.TransitionRun(t.Context(), run, runstore.RunPending, runstore.RunRunning, runstore.Reason{Code: "fixture_admitted"}); err != nil {
					t.Fatal(err)
				}
			}
			for i, document := range documents {
				stage := fmt.Sprintf("stage-%s-%d", run, i)
				finishedStageWithMetricsDocument(t, h.pool, runs, run, stage, stage, document, `{"totalTokens":1}`)
				stages = append(stages, stage)
			}
			finished[run] = true
		}
		h.finishRuns(t)
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
	collected := func() map[string]evaldomain.Usage {
		t.Helper()
		usage := map[string]evaldomain.Usage{}
		for _, member := range evalExecutedMembers(t, h.pool, e.ID) {
			record, err := store.LatestNativeRecord(t.Context(), e.OwnerID, e.ID, member, evaldomain.RecordKindResult)
			if err != nil || record == nil {
				t.Fatalf("member %s result = (%v, %v)", member, record, err)
			}
			var result evaldomain.ResultInput
			if err := json.Unmarshal(record.Document.Bytes(), &result); err != nil {
				t.Fatal(err)
			}
			usage[member] = result.Usage
		}
		return usage
	}
	accepted := collected()
	for member, usage := range accepted {
		for name, m := range map[string]evaldomain.Measure{"toolCalls": usage.ToolCalls, "toolFailures": usage.ToolFailures} {
			if m.Value != nil || m.Completeness != "unavailable" || !slices.Equal(m.Scope.Missing, []string{"counter overflow"}) || len(m.SourceRefs) != 2 {
				t.Fatalf("member %s %s = %+v, want unavailable with a counter overflow gap", member, name, m)
			}
		}
		for name, want := range map[string]struct {
			m     evaldomain.Measure
			value float64
		}{
			"modelCalls": {usage.ModelCalls, 2}, "inputTokens": {usage.InputTokens, 300},
			"outputTokens": {usage.OutputTokens, 20}, "totalTokens": {usage.TotalTokens, 320},
		} {
			if want.m.Value == nil || *want.m.Value != want.value || want.m.Completeness != "complete" || len(want.m.SourceRefs) != 2 {
				t.Fatalf("member %s %s = %+v, want complete %v", member, name, want.m, want.value)
			}
		}
	}
	records := evalRecordCount(t, h.pool, e.ID)
	for round := range 4 {
		// Rewriting the same documents marks every member for recollection.
		if _, err := h.pool.Exec(t.Context(), `UPDATE stage_metrics SET metrics = metrics WHERE stage_execution_id = ANY($1)`, stages); err != nil {
			t.Fatal(err)
		}
		if got := evalProjection(t, h.pool, e.ID); got[0] == got[1] {
			t.Fatalf("round %d: unchanged metrics did not mark the experiment: %v", round, got)
		}
		for n := 0; n < 20; n++ {
			if got := evalProjection(t, h.pool, e.ID); got[0] == got[1] {
				break
			}
			tick(t, c)
		}
		if got := evalProjection(t, h.pool, e.ID); got[0] != got[1] {
			t.Fatalf("round %d: recollection did not publish: %v", round, got)
		}
		if got := evalRecordCount(t, h.pool, e.ID); got != records {
			t.Fatalf("round %d: recollecting identical observations saved %d records, had %d", round, got, records)
		}
		if again := collected(); !reflect.DeepEqual(again, accepted) {
			t.Fatalf("round %d: recollected usage changed", round)
		}
	}
}
