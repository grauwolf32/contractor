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

	// An Audit-managed Run queues the deferred binding check on every UPDATE,
	// including the lane backfill. Ordinary Runs above do not exercise it.
	auditWorkflow := resolved.ResolvedWorkflow{EntryStage: "entry", Stages: map[string]resolved.ResolvedStage{"entry": cases[0].entry}}
	workflowJSON, err := json.Marshal(auditWorkflow)
	if err != nil {
		t.Fatal(err)
	}
	profile, err := json.Marshal(map[string]any{"workflows": map[string]any{"scan": map[string]any{"workflow": auditWorkflow}}})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `INSERT INTO projects(project_id,owner_id,kind,name,request_idempotency_key,request_digest)
 VALUES('lanes','user','project','Lanes','create','sha256:'||repeat('a',64))`); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `INSERT INTO audits(audit_id,owner_id,project_id,profile_name,profile_version,profile_digest,
 profile_snapshot,input_selection,baseline_snapshot,state,phase,started_at,max_rounds,batch_size,max_items_per_round,max_items_total,
 max_submitted_runs,max_item_run_attempts,max_evidence_bytes,reserved_run_count,outstanding_run_count)
 VALUES('audit-lanes','user','lanes','profile','1','sha256:'||repeat('a',64),$1::jsonb,'{}',
 '{"schema":"contractor.audit.baseline.v1","inputs":{},"scope":{}}',
 'active','rounds',clock_timestamp(),1,1,10,10,20,1,1024,1,1)`, profile); err != nil {
		t.Fatal(err)
	}
	tx, err := pool.Begin(ctx)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = tx.Rollback(ctx) }()
	if _, err := tx.Exec(ctx, `INSERT INTO audit_executions(
 execution_id,audit_id,role,workflow_role,role_attempt,manifest_ref,manifest_digest,submission_key,request_digest)
 VALUES('execution-scan','audit-lanes','discovery','scan',1,'{}','sha256:'||repeat('b',64),
 'submit-scan','sha256:'||repeat('b',64))`); err != nil {
		t.Fatal(err)
	}
	if _, err := tx.Exec(ctx, `INSERT INTO workflow_runs (
 run_id, owner_id, project_id, workflow_name, workflow_version, workflow_schema_version, workflow_snapshot,
 parameters, runtime_labels, runtime_config_snapshot, state, state_reason_code,
 publication_mode, audit_execution_id, audit_submission_key
) VALUES ('run-audit-scan','user','lanes','workflow','1','v1',$1::jsonb,
 '{}','{}',$2::jsonb,'pending','test','audit-managed','execution-scan','submit-scan')`, workflowJSON, builtInRunRuntimeSnapshot); err != nil {
		t.Fatal(err)
	}
	if _, err := tx.Exec(ctx, `UPDATE audit_executions SET state='submitted', run_id='run-audit-scan',
 run_provenance='{"schema":"contractor.audit.run-provenance.v1","runId":"run-audit-scan"}'
 WHERE execution_id='execution-scan'`); err != nil {
		t.Fatal(err)
	}

	if err := tx.Commit(ctx); err != nil {
		t.Fatal(err)
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
	var runClass, executionClass string
	if err := pool.QueryRow(ctx, `SELECT r.scheduler_lane_class, e.lane_class
 FROM workflow_runs r JOIN audit_executions e ON e.execution_id=r.audit_execution_id
 WHERE r.run_id='run-audit-scan' AND e.run_id=r.run_id AND e.submission_key=r.audit_submission_key`).Scan(&runClass, &executionClass); err != nil || runClass != "tool" || executionClass != "tool" {
		t.Fatalf("migrated Audit binding classes = %q, %q (%v)", runClass, executionClass, err)
	}
	var deferred bool
	if err := pool.QueryRow(ctx, `SELECT tgdeferrable AND tginitdeferred FROM pg_trigger
 WHERE tgrelid='workflow_runs'::regclass AND tgname='workflow_runs_require_complete_audit_binding'`).Scan(&deferred); err != nil || !deferred {
		t.Fatalf("Audit binding trigger lost its deferred contract: %v (%v)", deferred, err)
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
