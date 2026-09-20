package evalservice

import (
	"bytes"
	"encoding/json"
	"os"
	"reflect"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/evaldomain"
	"github.com/grauwolf32/contractor/internal/evalstore"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
)

func TestEvalObservedLegacyPlannerCountersUseImmutableStageIdentity(t *testing.T) {
	pool := serviceTestPool(t)
	runs := runstore.NewPostgresStore(pool)
	for _, planner := range []string{"passthrough", "streamline"} {
		_, err := runs.CreateRun(t.Context(), runstore.CreateRunParams{
			RunID: planner, OwnerID: "owner", WorkflowName: "fixture", WorkflowVersion: "1",
			WorkflowSchemaVersion: contracts.APIVersion, WorkflowSnapshot: json.RawMessage(`{"name":"fixture"}`),
			Parameters: map[string]string{}, RuntimeConfig: runtimeconfig.BuiltInRunSnapshot(),
		})
		if err != nil {
			t.Fatal(err)
		}
		if _, err = runs.TransitionRun(t.Context(), planner, runstore.RunInitializing, runstore.RunRunning, runstore.Reason{Code: "fixture"}); err != nil {
			t.Fatal(err)
		}
		_, err = runs.CreateStageExecution(t.Context(), runstore.CreateStageExecutionParams{
			StageExecutionID: planner, RunID: planner, StageName: "fixture", Attempt: 1,
			StageSpecSchemaVersion:    contracts.APIVersion,
			StageSpecSnapshot:         json.RawMessage(`{"planner":{"plannerId":"` + planner + `","version":"1"}}`),
			StageContextSchemaVersion: contracts.APIVersion, StageContext: runstore.StageContextSnapshot{},
		})
		if err != nil {
			t.Fatal(err)
		}
		// The same retained report has no model counters. Only the immutable
		// passthrough identity can establish that its own model work is zero.
		_, err = pool.Exec(t.Context(), `INSERT INTO stage_metrics(stage_execution_id,metrics_schema_version,metrics,summary)
VALUES($1,$2,'{"planner":{"complete":true,"metrics":{"tools":{}}},"workers":{},"runtime":{}}','{}')`, planner, contracts.APIVersion)
		if err != nil {
			t.Fatal(err)
		}
		ref := evaldomain.ExecutionRef{Kind: "run", ID: planner}
		inventory := evalstore.Inventory{Complete: true, Gaps: []string{}, Entries: []evalstore.InventoryEntry{{Execution: &ref, Available: true}}}
		start, finish := time.Unix(0, 0).UTC(), time.Unix(1, 0).UTC()
		usage, err := observedUsage(t.Context(), pool, "owner", evaldomain.Digest([]byte(planner))[7:], ExecutionView{
			Ref: &ref, State: "succeeded", StartedAt: &start, FinishedAt: &finish,
		}, inventory)
		if err != nil {
			t.Fatal(err)
		}
		if planner == "passthrough" {
			if usage.TotalTokens.Value == nil || *usage.TotalTokens.Value != 0 || usage.TotalTokens.Completeness != "complete" {
				t.Fatalf("legacy passthrough metrics = %+v", usage.TotalTokens)
			}
		} else if usage.TotalTokens.Value != nil || usage.TotalTokens.Completeness != "unavailable" {
			t.Fatalf("missing model usage became zero: %+v", usage.TotalTokens)
		}
	}
}

// The inventory authority/pagination journey is covered at the public boundary.
// This test exercises real metric SQL and normalization over its all-role fixture.
func TestEvalObservedAuditUsageCountsStagesOnceAndKeepsParentDuration(t *testing.T) {
	pool := serviceTestPool(t)
	raw, err := os.ReadFile("../../api/testdata/evals/audit-accounting.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		Observation evaldomain.UsageObservation `json:"observation"`
	}
	if err = json.Unmarshal(raw, &fixture); err != nil {
		t.Fatal(err)
	}
	input := fixture.Observation
	inventory := evalstore.Inventory{Complete: true, Gaps: []string{}}
	runs := runstore.NewPostgresStore(pool)
	for _, ref := range input.Executions {
		inventory.Entries = append(inventory.Entries, evalstore.InventoryEntry{Execution: &ref, Available: true})
		if ref.Kind != "run" {
			continue
		}
		_, err = runs.CreateRun(t.Context(), runstore.CreateRunParams{
			RunID: ref.ID, OwnerID: "owner", WorkflowName: "fixture", WorkflowVersion: "1",
			WorkflowSchemaVersion: contracts.APIVersion, WorkflowSnapshot: json.RawMessage(`{"name":"fixture"}`),
			Parameters: map[string]string{}, RuntimeConfig: runtimeconfig.BuiltInRunSnapshot(),
		})
		if err != nil {
			t.Fatal(err)
		}
		if _, err = runs.TransitionRun(t.Context(), ref.ID, runstore.RunInitializing, runstore.RunRunning, runstore.Reason{Code: "fixture"}); err != nil {
			t.Fatal(err)
		}
	}
	seen := map[string]bool{}
	for _, attempt := range input.Attempts {
		if !seen[attempt.StageExecutionID] {
			_, err = runs.CreateStageExecution(t.Context(), runstore.CreateStageExecutionParams{
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
	usage, err := observedUsage(t.Context(), pool, "owner", input.MemberID, execution, inventory)
	if err != nil || usage.TotalTokens.Value == nil || *usage.TotalTokens.Value != 100 || usage.TotalTokens.Completeness != "complete" || *usage.ModelCalls.Value != 4 || *usage.WallMS.Value != 2000 {
		t.Fatal("wrong SQL stage/parent accounting", usage, err)
	}
	runsRead := []string{"check-1", "discovery-1", "assessment-1"}
	snapshots, err := evalstore.NewPostgresStore(pool).MetricSnapshots(t.Context(), "owner", runsRead)
	if err != nil || len(snapshots) != 4 {
		t.Fatal("metric snapshot cohort", snapshots, err)
	}
	for _, snapshot := range snapshots {
		if bytes.Contains(snapshot.Document, []byte("PRIVATE_")) || bytes.Contains(snapshot.Document, []byte("99999")) {
			t.Fatal("private/aggregate fields leaked into metric projection")
		}
	}
	inventory.Entries = append(inventory.Entries, inventory.Entries[1])
	replay, err := observedUsage(t.Context(), pool, "owner", input.MemberID, execution, inventory)
	if err != nil || !reflect.DeepEqual(usage, replay) {
		t.Fatal("repeated inventory or snapshots doubled usage", err)
	}
	if _, err = pool.Exec(t.Context(), `DELETE FROM stage_metrics WHERE stage_execution_id='assessment'`); err != nil {
		t.Fatal(err)
	}
	partial, err := observedUsage(t.Context(), pool, "owner", input.MemberID, execution, inventory)
	if err != nil || partial.TotalTokens.Value == nil || *partial.TotalTokens.Value != 60 || partial.TotalTokens.Completeness != "partial" || partial.WallMS.Completeness != "complete" || *partial.WallMS.Value != 2000 {
		t.Fatal("missing child metrics lost partial scope or parent duration", partial, err)
	}
	snapshots, err = evalstore.NewPostgresStore(pool).MetricSnapshots(t.Context(), "foreign", runsRead)
	if err != nil || len(snapshots) != 0 {
		t.Fatal("foreign child metrics disclosed", err)
	}
}
