package postgres

import (
	"context"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/persistence/explaintest"
)

// TestWorkflowRunGuardsFireOnTheColumnsTheyCompare checks the column-scoped
// guards of workflow_runs (PER-R22): each lists exactly the columns its
// function or WHEN compares between NEW and OLD, and no BEFORE UPDATE trigger
// assigns a column of NEW, which would change a value without listing it.
func TestWorkflowRunGuardsFireOnTheColumnsTheyCompare(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 45*time.Second)
	defer cancel()
	pool, _ := isolatedPools(t, ctx, storageReviewDatabaseURL(t))
	if _, err := ApplyMigrations(ctx, pool, MigrationBudgets{}); err != nil {
		t.Fatal(err)
	}
	if guards := explaintest.RequireColumnScopedGuards(t, ctx, pool, "workflow_runs"); guards < 4 {
		t.Fatalf("found %d column-scoped guards of workflow_runs, want the four including placement requirements (PER-R22, RUN-R108)", guards)
	}
}

// TestStageExecutionGuardsFireOnTheColumnsTheyCompare checks the same for
// stage_executions: the immutability guard lists the columns it protects and
// the manual continuation guard, which also validates every INSERT, fires on
// an UPDATE only for resume_source_execution_id (PER-R22).
func TestStageExecutionGuardsFireOnTheColumnsTheyCompare(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 45*time.Second)
	defer cancel()
	pool, _ := isolatedPools(t, ctx, storageReviewDatabaseURL(t))
	if _, err := ApplyMigrations(ctx, pool, MigrationBudgets{}); err != nil {
		t.Fatal(err)
	}
	if guards := explaintest.RequireColumnScopedGuards(t, ctx, pool, "stage_executions"); guards != 2 {
		t.Fatalf("found %d column-scoped guards of stage_executions, want the immutability and the manual continuation guards (PER-R22)", guards)
	}
}
