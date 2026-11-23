package runstore

import (
	"context"
	"maps"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"

	"github.com/grauwolf32/contractor/internal/persistence/explaintest"
)

// TestPostgresStageAllocationUpdatesFireOnlyTheTriggersTheyConcern counts
// the triggers of stage_allocations that each allocation write calls
// (PER-R22). Recording an allocation runs the completion check with its
// lookup of the Run's authority; a release attempt, a completed release and
// the report-received mark change no guarded column and call none of them.
// Assigning a guarded column calls its guard, and changing any one guarded
// column alone is still rejected by the guard that protects it, so a column
// missing from a guard's UPDATE OF list fails here.
func TestPostgresStageAllocationUpdatesFireOnlyTheTriggersTheyConcern(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	pool := isolatedRunStorePool(t, ctx)
	run := createTestRun(t, ctx, NewPostgresStore(pool, artifactpgtest.NewStorage()), "run-allocation-triggers")
	if _, err := admitTestRun(ctx, NewPostgresStore(pool, artifactpgtest.NewStorage()), run.RunID); err != nil {
		t.Fatal(err)
	}
	stage, err := NewPostgresStore(pool, artifactpgtest.NewStorage()).CreateStageExecution(ctx,
		testActiveStageParams(run.RunID, "stage-allocation-triggers", "copy"))
	if err != nil {
		t.Fatal(err)
	}
	tx, err := pool.Begin(ctx)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = tx.Rollback(context.WithoutCancel(ctx)) }()
	if _, err := tx.Exec(ctx, `SET CONSTRAINTS ALL IMMEDIATE`); err != nil {
		t.Fatal(err)
	}
	probe := explaintest.NewProbe(t, tx)
	store := NewPostgresStore(probe, artifactpgtest.NewStorage())
	const (
		identity   = "stage_allocations_protect_identity"
		policy     = "stage_allocations_protect_performance_policy"
		completion = "stage_allocations_require_completion"
	)
	expectFired := func(step string, want map[string]int64, action func() error) {
		t.Helper()
		probe.Reset()
		if err := action(); err != nil {
			t.Fatalf("%s: %v", step, err)
		}
		if fired := probe.Fired("stage_allocations"); !maps.Equal(fired, want) {
			t.Errorf("%s fired the stage_allocations triggers %v, want %v", step, fired, want)
		}
	}
	const allocationID = "allocation-triggers"
	expectFired("allocation", map[string]int64{completion: 1}, func() error {
		return store.RecordStageAllocation(ctx, testStageAllocation(stage.StageExecutionID, allocationID))
	})
	expectFired("release attempt", map[string]int64{}, func() error {
		return store.MarkStageAllocationReleaseAttempt(ctx, allocationID)
	})
	expectFired("release", map[string]int64{}, func() error {
		return store.MarkStageAllocationReleased(ctx, allocationID)
	})
	// The statement contractor_mark_allocation_report_received runs for each
	// stored execution report; EXPLAIN does not see inside that trigger.
	expectFired("report received", map[string]int64{}, func() error {
		_, err := probe.Exec(ctx, `UPDATE stage_allocations SET execution_report_received = true
WHERE allocation_id = $1 AND NOT execution_report_received`, allocationID)
		return err
	})
	// Assigning a guarded column calls its guard even when the value stays.
	for column, guard := range map[string]string{
		"namespace": identity, "performance_collection_policy": policy, "completion_contract": completion,
	} {
		expectFired("unchanged "+column, map[string]int64{guard: 1}, func() error {
			_, err := probe.Exec(ctx, `UPDATE stage_allocations SET `+column+` = `+column+` WHERE allocation_id = $1`, allocationID)
			return err
		})
	}

	// Each guarded column changed alone is rejected by its own guard, before
	// any check constraint sees the value. Explain rolls each attempt back.
	const (
		identityMessage   = "Stage allocation identity and Runtime configuration are immutable"
		policyMessage     = "Stage allocation performance collection policy is immutable"
		completionMessage = "Allocation completion is immutable"
		labelsMessage     = "Runtime placement labels are immutable"
	)
	for column, change := range map[string]struct{ value, message string }{
		"allocation_id":                        {`'allocation-other'`, identityMessage},
		"stage_execution_id":                   {`'stage-other'`, identityMessage},
		"logical_agent_name":                   {`'other'`, identityMessage},
		"namespace":                            {`'other'`, identityMessage},
		"agent_template_ref":                   {`'{"templateId":"other"}'`, identityMessage},
		"worker_runtime_ref":                   {`'{"runtimeId":"other"}'`, identityMessage},
		"runtime_agent_instance_id":            {`'other'`, identityMessage},
		"runtime_agent_id":                     {`repeat('4', 64)`, identityMessage},
		"runtime_agent_label_revision":         {`runtime_agent_label_revision + 1`, identityMessage},
		"runtime_configuration_schema_version": {`'other'`, identityMessage},
		"runtime_configuration":                {`'{}'`, identityMessage},
		"created_at":                           {`created_at + interval '1 second'`, identityMessage},
		"performance_collection_policy":        {`'requested'`, policyMessage},
		"completion_contract":                  {`'{"kind":"audit-check-results@1"}'`, completionMessage},
		"runtime_placement_labels":             {`'{gpu}'`, labelsMessage},
	} {
		statement := `UPDATE stage_allocations SET ` + column + ` = ` + change.value + ` WHERE allocation_id = $1`
		if _, err := explaintest.Explain(ctx, tx, statement, allocationID); err == nil || !strings.Contains(err.Error(), change.message) {
			t.Errorf("%s = %v, want %q", statement, err, change.message)
		}
	}
	// An INSERT still looks up the Run's completion authority: an ordinary
	// Run has none, so an allocation may not claim a completion contract.
	if _, err := explaintest.Explain(ctx, tx, `
INSERT INTO stage_allocations (
    allocation_id, stage_execution_id, logical_agent_name, namespace, agent_template_ref,
    worker_runtime_ref, runtime_agent_instance_id, completion_contract
)
SELECT 'allocation-claimed', stage_execution_id, 'claimant', namespace, agent_template_ref,
       worker_runtime_ref, runtime_agent_instance_id, '{"kind":"audit-check-results@1"}'
FROM stage_allocations WHERE allocation_id = $1`, allocationID); err == nil ||
		!strings.Contains(err.Error(), "Allocation has no completion authority") {
		t.Errorf("allocation claiming a completion on an ordinary Run = %v, want rejected", err)
	}
}
