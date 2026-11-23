package postgres

import (
	"context"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/persistence/explaintest"
)

// TestWorkflowRunGuardsFireOnTheColumnsTheyCompare checks the column-scoped
// guards of workflow_runs (PER-R22): each reads through NEW and OLD only the
// columns it lists in UPDATE OF and the declared qualifiers, and no BEFORE
// UPDATE trigger assigns a column of NEW, which would change a value without
// listing it.
func TestWorkflowRunGuardsFireOnTheColumnsTheyCompare(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 45*time.Second)
	defer cancel()
	pool, _ := isolatedPools(t, ctx, storageReviewDatabaseURL(t))
	if _, err := ApplyMigrations(ctx, pool, MigrationBudgets{}); err != nil {
		t.Fatal(err)
	}
	// Each guard below acts only when its listed column changes and reads
	// the Run's state to decide whether that change is still allowed.
	qualifiers := explaintest.Qualifiers{
		"workflow_runs_protect_audit_input_materialization": {"state", "publication_mode"},
		"workflow_runs_protect_audit_completion":            {"state"},
		"workflow_runs_protect_skill_snapshot":              {"state"},
	}
	if guards := explaintest.RequireColumnScopedGuards(t, ctx, pool, "workflow_runs", qualifiers); guards < 4 {
		t.Fatalf("found %d column-scoped guards of workflow_runs, want the four including placement requirements (PER-R22, RUN-R108)", guards)
	}
}

// TestStageExecutionGuardsFireOnTheColumnsTheyCompare checks the same for
// stage_executions: the execution, automatic transition and manual receipt
// guards list exactly the columns they protect. The manual receipt guard also
// validates every INSERT (PER-R22).
func TestStageExecutionGuardsFireOnTheColumnsTheyCompare(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 45*time.Second)
	defer cancel()
	pool, _ := isolatedPools(t, ctx, storageReviewDatabaseURL(t))
	if _, err := ApplyMigrations(ctx, pool, MigrationBudgets{}); err != nil {
		t.Fatal(err)
	}
	if guards := explaintest.RequireColumnScopedGuards(t, ctx, pool, "stage_executions", nil); guards != 3 {
		t.Fatalf("found %d column-scoped guards of stage_executions, want the execution, transition and manual receipt guards (PER-R22)", guards)
	}
}
