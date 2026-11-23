package projectstore

import (
	"os"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"
	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/persistence/explaintest"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/runstore/runstoretest"
	"github.com/grauwolf32/contractor/internal/runtimeconfig/runtimeconfigtest"
)

func TestPostgresLLMCostReadDoesNotScanCompletedRunHistory(t *testing.T) {
	url := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if url == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	ctx := t.Context()
	pool := isolatedProjectPool(t, ctx, url)
	projects := NewPostgresStore(pool)
	for _, id := range []string{"cost-history", "cost-template"} {
		if _, _, err := projects.Create(ctx, CreateParams{ProjectID: id, OwnerID: "owner", Kind: KindProject, Name: id, IdempotencyKey: id, RequestDigest: "sha256:" + strings.Repeat("a", 64)}); err != nil {
			t.Fatal(err)
		}
	}
	runs := runstore.NewPostgresStore(pool, artifactpgtest.NewStorage())
	create := func(id, project string) {
		t.Helper()
		if _, err := runstoretest.CreateRun(ctx, runs, runstore.CreateRunParams{RunID: id, OwnerID: "owner", ProjectID: &project, LaneClass: resolved.LaneClassModel, WorkflowName: "workflow", WorkflowVersion: "1", WorkflowSchemaVersion: contracts.APIVersion, WorkflowSnapshot: []byte(`{"name":"workflow"}`), Parameters: map[string]string{}, RuntimeConfig: runtimeconfigtest.BuiltInRunSnapshot()}); err != nil {
			t.Fatal(err)
		}
	}
	// Only insert rows: no unrelated test snapshot can hold dead fixture
	// versions away from VACUUM. The live template belongs to another Project.
	create("cost-seed", "cost-template")
	if _, err := pool.Exec(ctx, `
INSERT INTO workflow_runs(run_id,owner_id,project_id,workflow_name,workflow_version,
 workflow_schema_version,workflow_snapshot,state,state_reason_code,created_at,updated_at,
 runtime_labels,runtime_config_snapshot,scheduler_lane_class,finished_at)
SELECT 'cost-history-'||n,seed.owner_id,'cost-history',seed.workflow_name,seed.workflow_version,
 seed.workflow_schema_version,seed.workflow_snapshot,'succeeded','fixture',seed.created_at,seed.updated_at,
 seed.runtime_labels,seed.runtime_config_snapshot,seed.scheduler_lane_class,seed.created_at
FROM workflow_runs seed CROSS JOIN generate_series(1,12000) n WHERE seed.run_id='cost-seed'`); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `VACUUM ANALYZE workflow_runs`); err != nil {
		t.Fatal(err)
	}
	measure := func() {
		t.Helper()
		for _, mode := range explaintest.Modes {
			analysis, err := explaintest.ExplainPrepared(ctx, pool, mode, readProjectCostSQL, "owner", "cost-history")
			if err != nil {
				t.Fatal(err)
			}
			if reads := analysis.Reads.Of("workflow_runs"); reads > 4 {
				t.Fatalf("%s cost read visited %d Runs, budget 4:\n%s", mode, reads, analysis.Plan)
			}
		}
	}
	measure()
	if _, pending, work, err := projects.ReadLLMCost(ctx, "owner", "cost-history"); err != nil || pending || !work {
		t.Fatalf("settled history: pending=%t work=%t %v", pending, work, err)
	}
	create("cost-active", "cost-history")
	measure()
	if _, pending, work, err := projects.ReadLLMCost(ctx, "owner", "cost-history"); err != nil || !pending || !work {
		t.Fatalf("new active Run: pending=%t work=%t %v", pending, work, err)
	}
}
