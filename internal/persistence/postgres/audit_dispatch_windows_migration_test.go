package postgres

import (
	"encoding/json"
	"fmt"
	"os"
	"testing"

	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contracts"
)

// Migration 000027 classifies every stored Audit execution by its role's
// pinned Workflow, with the SQL copy of resolved.WorkflowLaneClass that the
// test checks against the Go rule on Workflows serialized as the Server
// writes them, and counts each Audit's outstanding `tool` executions (states
// intent and submitted) into outstanding_tool_run_count (AUX-R3).
func TestPostgresAuditDispatchWindowsMigrationSplitsOutstandingRuns(t *testing.T) {
	url := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if url == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	ctx := t.Context()
	pool, _ := isolatedPools(t, ctx, url)
	installMigrationPrefix(t, ctx, pool, 26)
	stage := func(plannerID string, runtimes ...string) resolved.ResolvedStage {
		agents := map[string]resolved.ResolvedAgentBinding{}
		for index, runtimeID := range runtimes {
			agents[string(rune('a'+index))] = resolved.ResolvedAgentBinding{Template: contracts.ResolvedAgentTemplate{
				Runtime: contracts.WorkerRuntimeRef{RuntimeID: runtimeID, Version: "1"},
			}}
		}
		return resolved.ResolvedStage{Planner: resolved.PlannerRef{PlannerID: plannerID, Version: "1"}, Agents: agents}
	}
	workflows := map[string]resolved.ResolvedWorkflow{
		"scan": {EntryStage: "scan", Stages: map[string]resolved.ResolvedStage{
			"scan": stage("scan-plan", "tool"), "fetch": stage("passthrough", "tool"),
		}},
		"scan-then-triage": {EntryStage: "scan", Stages: map[string]resolved.ResolvedStage{
			"scan": stage("scan-plan", "tool"), "triage": stage("streamline", "adk"),
		}},
		"check": {EntryStage: "check", Stages: map[string]resolved.ResolvedStage{"check": stage("passthrough", "adk")}},
	}
	bindings := map[string]map[string]resolved.ResolvedWorkflow{}
	for role, workflow := range workflows {
		bindings[role] = map[string]resolved.ResolvedWorkflow{"workflow": workflow}
	}
	profile, err := json.Marshal(map[string]any{"workflows": bindings})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `INSERT INTO projects(project_id,owner_id,kind,name,request_idempotency_key,request_digest)
 VALUES('lanes','owner','project','Lanes','create','sha256:'||repeat('a',64))`); err != nil {
		t.Fatal(err)
	}
	// Each role has two outstanding executions and one observed one.
	outstanding := map[string]int{"scan": 2, "scan-then-triage": 2, "check": 2}
	for _, auditID := range []string{"audit-busy", "audit-idle"} {
		running := 0
		if auditID == "audit-busy" {
			running = 6
		}
		if _, err := pool.Exec(ctx, `INSERT INTO audits(audit_id,owner_id,project_id,profile_name,profile_version,profile_digest,
 profile_snapshot,input_selection,baseline_snapshot,state,phase,started_at,max_rounds,batch_size,max_items_per_round,max_items_total,
 max_submitted_runs,max_item_run_attempts,max_evidence_bytes,reserved_run_count,outstanding_run_count)
 VALUES($1,'owner','lanes','profile','1','sha256:'||repeat('a',64),$2::jsonb,'{}',
 '{"schema":"contractor.audit.baseline.v1","inputs":{},"scope":{}}',
 'active','rounds',clock_timestamp(),1,1,10,10,20,1,1024,9,$3)`, auditID, profile, running); err != nil {
			t.Fatal(err)
		}
	}
	for role := range workflows {
		for attempt := 1; attempt <= 3; attempt++ {
			observed := attempt > outstanding[role]
			executionID := fmt.Sprintf("%s-%d", role, attempt)
			if _, err := pool.Exec(ctx, `INSERT INTO audit_executions(
    execution_id,audit_id,role,workflow_role,role_attempt,manifest_ref,manifest_digest,submission_key,request_digest,
    state,terminal_outcome,terminal_observed_at)
 VALUES($1,'audit-busy','discovery',$2,$3,'{}'::jsonb,'sha256:'||repeat('b',64),$1,'sha256:'||repeat('b',64),
        CASE WHEN $4 THEN 'collecting' ELSE 'intent' END,
        CASE WHEN $4 THEN 'submission-failed' END,
        CASE WHEN $4 THEN clock_timestamp() END)`, executionID, role, attempt, observed); err != nil {
				t.Fatal(err)
			}
		}
	}

	if _, err := ApplyMigrations(ctx, pool, MigrationBudgets{}); err != nil {
		t.Fatal(err)
	}
	for role, workflow := range workflows {
		want := resolved.WorkflowLaneClass(workflow)
		var classes []string
		rows, err := pool.Query(ctx, `SELECT DISTINCT lane_class FROM audit_executions WHERE workflow_role = $1`, role)
		if err != nil {
			t.Fatal(err)
		}
		for rows.Next() {
			var class string
			if err := rows.Scan(&class); err != nil {
				t.Fatal(err)
			}
			classes = append(classes, class)
		}
		if rows.Close(); rows.Err() != nil || len(classes) != 1 || classes[0] != string(want) {
			t.Errorf("role %s migrated to classes %v (%v), want %q", role, classes, rows.Err(), want)
		}
	}
	for auditID, want := range map[string][2]int{"audit-busy": {6, 2}, "audit-idle": {0, 0}} {
		var total, tool int
		if err := pool.QueryRow(ctx, `SELECT outstanding_run_count, outstanding_tool_run_count FROM audits WHERE audit_id = $1`, auditID).
			Scan(&total, &tool); err != nil || total != want[0] || tool != want[1] {
			t.Fatalf("%s outstanding = (%d, tool %d) (%v), want (%d, tool %d)", auditID, total, tool, err, want[0], want[1])
		}
	}
	if _, err := pool.Exec(ctx, `UPDATE audit_executions SET lane_class = 'scan' WHERE execution_id = 'scan-1'`); Classify(err) != ClassInvalid {
		t.Fatalf("unknown lane class reached storage: %v", err)
	}
	if _, err := pool.Exec(ctx, `UPDATE audits SET outstanding_tool_run_count = 7 WHERE audit_id = 'audit-busy'`); Classify(err) != ClassInvalid {
		t.Fatalf("a tool count above the total reached storage: %v", err)
	}
}
