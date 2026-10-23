package postgres

import (
	"encoding/json"
	"os"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contracts"
)

// Migration 000026 classifies every stored Run by the Stage it progresses
// next, its active StageExecution or else its entry Stage, with the SQL copy
// of resolved.StageLaneClass; the test checks the copy against the Go rule
// on snapshots serialized as the Server writes them (SCH-R175). The settings
// singleton gains a tool pool of two, and both claim indexes lead with the
// class.
func TestPostgresModelFreeLanePoolMigrationClassifiesExistingRuns(t *testing.T) {
	url := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if url == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	ctx := t.Context()
	pool, _ := isolatedPools(t, ctx, url)
	installMigrationPrefix(t, ctx, pool, 25)
	stage := func(plannerID string, runtimes ...string) resolved.ResolvedStage {
		agents := map[string]resolved.ResolvedAgentBinding{}
		for index, runtimeID := range runtimes {
			agents[string(rune('a'+index))] = resolved.ResolvedAgentBinding{Template: contracts.ResolvedAgentTemplate{
				Runtime: contracts.WorkerRuntimeRef{RuntimeID: runtimeID, Version: "1"},
			}}
		}
		return resolved.ResolvedStage{Planner: resolved.PlannerRef{PlannerID: plannerID, Version: "1"}, Agents: agents}
	}
	cases := []struct {
		runID  string
		entry  resolved.ResolvedStage
		active *resolved.ResolvedStage
	}{
		{runID: "run-entry-scan", entry: stage("scan-plan", "tool", "tool")},
		{runID: "run-entry-passthrough-tool", entry: stage("passthrough", "tool")},
		{runID: "run-entry-passthrough-adk", entry: stage("passthrough", "adk")},
		{runID: "run-entry-mixed", entry: stage("scan-plan", "tool", "adk")},
		{runID: "run-entry-router-tool", entry: stage("router", "tool", "tool")},
		{runID: "run-active-scan", entry: stage("streamline", "adk"), active: new(stage("passthrough", "tool"))},
		{runID: "run-active-model", entry: stage("scan-plan", "tool"), active: new(stage("streamline", "adk"))},
	}
	for _, test := range cases {
		workflow := resolved.ResolvedWorkflow{EntryStage: "entry", Stages: map[string]resolved.ResolvedStage{"entry": test.entry}}
		state := "pending"
		if test.active != nil {
			workflow.Stages["active"] = *test.active
			state = "running"
		}
		snapshot, err := json.Marshal(workflow)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := pool.Exec(ctx, `INSERT INTO workflow_runs (
    run_id, owner_id, workflow_name, workflow_version, workflow_schema_version, workflow_snapshot,
    parameters, runtime_labels, runtime_config_snapshot, state, state_reason_code, started_at
) VALUES ($1, 'user', 'workflow', '1', 'v1', $2::jsonb, '{}', '{}', $3::jsonb, $4, 'test',
          CASE WHEN $4 = 'running' THEN clock_timestamp() END)`,
			test.runID, snapshot, builtInRunRuntimeSnapshot, state); err != nil {
			t.Fatal(err)
		}
		if test.active == nil {
			continue
		}
		spec, err := json.Marshal(*test.active)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := pool.Exec(ctx, `INSERT INTO stage_executions (
    stage_execution_id, run_id, stage_name, attempt, stage_spec_schema_version, stage_spec_snapshot,
    stage_context_schema_version, stage_context_snapshot, state, state_reason_code
) VALUES ($1 || '-active', $1, 'active', 1, 'v1', $2::jsonb, 'v1', '{}', 'preparing', 'test')`,
			test.runID, spec); err != nil {
			t.Fatal(err)
		}
	}

	if _, err := ApplyMigrations(ctx, pool, MigrationBudgets{}); err != nil {
		t.Fatal(err)
	}
	for _, test := range cases {
		want := resolved.StageLaneClass(test.entry)
		if test.active != nil {
			want = resolved.StageLaneClass(*test.active)
		}
		var class string
		if err := pool.QueryRow(ctx, `SELECT scheduler_lane_class FROM workflow_runs WHERE run_id = $1`, test.runID).Scan(&class); err != nil || class != string(want) {
			t.Errorf("%s migrated to class %q (%v), want %q", test.runID, class, err, want)
		}
	}
	var toolRuns, revision int
	var toolDefault *string
	if err := pool.QueryRow(ctx, `SELECT settings.max_concurrent_tool_runs, settings.revision, columns.column_default
FROM scheduler_settings AS settings, information_schema.columns AS columns
WHERE columns.table_schema = current_schema() AND columns.table_name = 'scheduler_settings'
  AND columns.column_name = 'max_concurrent_tool_runs'`).Scan(&toolRuns, &revision, &toolDefault); err != nil ||
		toolRuns != 2 || revision != 1 || toolDefault != nil {
		t.Fatalf("migrated settings = tool %d revision %d default %v (%v)", toolRuns, revision, toolDefault, err)
	}
	for _, index := range []string{"workflow_runs_claim_idx", "workflow_runs_deferred_claim_idx"} {
		var definition string
		if err := pool.QueryRow(ctx, `SELECT indexdef FROM pg_indexes WHERE schemaname = current_schema() AND indexname = $1`, index).Scan(&definition); err != nil ||
			!strings.Contains(definition, "(scheduler_lane_class, ") {
			t.Fatalf("%s = %q (%v), want the lane class first", index, definition, err)
		}
	}
	if _, err := pool.Exec(ctx, `UPDATE workflow_runs SET scheduler_lane_class = 'scan' WHERE run_id = 'run-entry-scan'`); Classify(err) != ClassInvalid {
		t.Fatalf("unknown lane class reached storage: %v", err)
	}
}
