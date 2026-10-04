package postgres

import (
	"context"
	"slices"
	"strings"
	"testing"
	"time"
)

func TestPostgresEvalProjectDependenciesFoldMigration(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 45*time.Second)
	defer cancel()
	pool, _ := isolatedPools(t, ctx, storageReviewDatabaseURL(t))
	installStorageReviewPrefix(t, ctx, pool, 90)

	result, err := ApplyMigrations(ctx, pool)
	if err != nil || !slices.Contains(result.AppliedVersions, 91) {
		t.Fatalf("fold eval Project dependencies = %+v, %v", result, err)
	}
	var tableGone bool
	var fence, retain string
	if err := pool.QueryRow(ctx, `
SELECT to_regclass('eval_project_dependencies') IS NULL,
       pg_get_functiondef('contractor_eval_fence_project'::regproc),
       pg_get_functiondef('contractor_eval_retain_deleted_execution'::regproc)`,
	).Scan(&tableGone, &fence, &retain); err != nil {
		t.Fatal(err)
	}
	if !tableGone || !strings.Contains(fence, "execution_project_id") || !strings.Contains(retain, "execution_project_id") {
		t.Fatalf("dependency table dropped %t; triggers read submissions: fence=%t retain=%t",
			tableGone, strings.Contains(fence, "execution_project_id"), strings.Contains(retain, "execution_project_id"))
	}
}
