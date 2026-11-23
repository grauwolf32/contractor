package runstore

import (
	"context"
	"encoding/json"
	"os"
	"testing"

	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"

	"github.com/grauwolf32/contractor/internal/persistence/explaintest"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/jackc/pgx/v5/pgxpool"
)

func TestPostgresTerminalReleaseCandidatesDoNotScanTheBacklog(t *testing.T) {
	ctx := t.Context()
	databaseURL := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if databaseURL == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is unset")
	}
	pool := explaintest.Database(t, ctx, databaseURL, func(ctx context.Context, pool *pgxpool.Pool) error {
		_, err := persistencepostgres.ApplyMigrations(ctx, pool, persistencepostgres.MigrationBudgets{})
		return err
	})
	populateConsolidationRuns(t, ctx, pool)
	allocation := testStageAllocation("unused", "unused")
	template, _ := json.Marshal(allocation.AgentTemplateRef)
	runtime, _ := json.Marshal(allocation.WorkerRuntimeRef)
	configuration, _ := json.Marshal(allocation.RuntimeConfiguration)
	// More than fourteen thousand pending allocations, with several per
	// Stage and both active and terminal executions. No released history is
	// needed to expose the old whole-backlog aggregation.
	if _, err := pool.Exec(ctx, `INSERT INTO stage_allocations (
allocation_id,stage_execution_id,logical_agent_name,namespace,agent_template_ref,worker_runtime_ref,
runtime_agent_id,runtime_agent_instance_id,runtime_agent_label_revision,
runtime_configuration_schema_version,runtime_configuration,performance_collection_policy)
SELECT 'a-'||stage_execution_id||'-'||n,stage_execution_id,'worker-'||n,'worker-'||n,
$1::jsonb,$2::jsonb,$3,'release-runtime',1,$4,$5::jsonb,'disabled'
FROM stage_executions CROSS JOIN LATERAL generate_series(1,
CASE WHEN stage_execution_id='run-00001-stage-1' THEN 128 ELSE 4 END) n`,
		template, runtime, allocation.RuntimeAgentID, allocation.RuntimeConfigurationSchemaVersion, configuration); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `VACUUM ANALYZE stage_allocations`); err != nil {
		t.Fatal(err)
	}
	statement := listTerminalStageExecutionsWithAllocationsSQL + prefixedStageExecutionColumns("execution") + `
FROM pending JOIN stage_executions execution USING(stage_execution_id)
ORDER BY pending.next_attempt_at, execution.stage_execution_id`
	const busyStage = "run-00001-stage-1"
	store := NewPostgresStore(pool, artifactpgtest.NewStorage())
	for _, attempted := range []bool{false, true} {
		if attempted {
			if _, err := pool.Exec(ctx, `UPDATE stage_allocations SET release_attempted_at=timestamptz '2026-01-01 00:00:00Z'`); err != nil {
				t.Fatal(err)
			}
			if _, err := pool.Exec(ctx, `VACUUM ANALYZE stage_allocations`); err != nil {
				t.Fatal(err)
			}
		}
		for _, mode := range explaintest.Modes {
			analysis, err := explaintest.ExplainPrepared(ctx, pool, mode, statement,
				terminalReleaseCandidateBatchLimit, []string{busyStage}, terminalReleaseRecoveryBatchLimit)
			if err != nil {
				t.Fatal(err)
			}
			allocations, stages := analysis.Reads.Of("stage_allocations"), analysis.Reads.Of("stage_executions")
			// Includes the 128-allocation excluded Stage and active allocations
			// the index walk skips, but never the fourteen-thousand-row backlog.
			if allocations > 512 || stages > 1024 {
				t.Fatalf("%s attempted=%t read %d allocations and %d Stages:\n%s",
					mode, attempted, allocations, stages, analysis.Plan)
			}
			t.Logf("%s attempted=%t: %d allocations, %d Stages", mode, attempted, allocations, stages)
		}
		batch, err := store.ListTerminalStageExecutionsWithAllocations(ctx, []string{busyStage})
		if err != nil || len(batch) != terminalReleaseRecoveryBatchLimit {
			t.Fatalf("release batch = (%d, %v), want %d Stages", len(batch), err, terminalReleaseRecoveryBatchLimit)
		}
		for _, stage := range batch {
			if stage.StageExecutionID == busyStage || !stage.State.Terminal() {
				t.Fatalf("release batch contains busy or active Stage %s", stage.StageExecutionID)
			}
		}
	}
	// An older attempted Stage follows every never-attempted Stage, and a
	// completed release disappears even though its attempt was oldest.
	if _, err := pool.Exec(ctx, `UPDATE stage_allocations SET release_attempted_at=NULL WHERE stage_execution_id='run-00001-stage-2'`); err != nil {
		t.Fatal(err)
	}
	batch, err := store.ListTerminalStageExecutionsWithAllocations(ctx, nil)
	if err != nil || len(batch) == 0 || batch[0].StageExecutionID != "run-00001-stage-2" {
		t.Fatalf("never-attempted Stage did not lead the batch: %v", err)
	}
	if _, err := pool.Exec(ctx, `UPDATE stage_allocations SET release_attempted_at=timestamptz '2025-01-01 00:00:00Z', release_completed_at=timestamptz '2025-01-01 00:00:00Z' WHERE stage_execution_id=$1`, busyStage); err != nil {
		t.Fatal(err)
	}
	batch, err = store.ListTerminalStageExecutionsWithAllocations(ctx, nil)
	if err != nil {
		t.Fatal(err)
	}
	for _, stage := range batch {
		if stage.StageExecutionID == busyStage {
			t.Fatal("completed release remained in the batch")
		}
	}
}
