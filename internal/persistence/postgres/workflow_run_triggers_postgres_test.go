package postgres

import (
	"context"
	"regexp"
	"slices"
	"testing"
	"time"

	"github.com/jackc/pgx/v5"
)

// workflowRunBeforeUpdateTriggersSQL lists each BEFORE UPDATE trigger of
// workflow_runs with its function source and its UPDATE OF columns, if any.
const workflowRunBeforeUpdateTriggersSQL = `
SELECT trigger.tgname::text, function.prosrc,
       ARRAY(SELECT attribute.attname::text
               FROM unnest(trigger.tgattr::int2[]) AS column_number
               JOIN pg_attribute AS attribute
                 ON attribute.attrelid = trigger.tgrelid AND attribute.attnum = column_number)
  FROM pg_trigger AS trigger
  JOIN pg_proc AS function ON function.oid = trigger.tgfoid
 WHERE trigger.tgrelid = 'workflow_runs'::regclass AND NOT trigger.tgisinternal
   -- BEFORE (2) and UPDATE (16) of pg_trigger.tgtype
   AND trigger.tgtype & 18 = 18
 ORDER BY 1`

var (
	comparedTriggerColumn = regexp.MustCompile(`NEW\.([a-z_]+) IS DISTINCT FROM OLD\.([a-z_]+)`)
	assignedTriggerColumn = regexp.MustCompile(`NEW\.([a-z_]+)\s*:=`)
)

// TestWorkflowRunGuardsFireOnTheColumnsTheyCompare checks the column-scoped
// guards of workflow_runs (PER-R22). An UPDATE OF trigger runs only when an
// UPDATE lists one of its columns, so each guard must list exactly the
// columns its function compares between NEW and OLD, and no BEFORE UPDATE
// trigger may assign a column of NEW, which would change a value without
// listing it.
func TestWorkflowRunGuardsFireOnTheColumnsTheyCompare(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 45*time.Second)
	defer cancel()
	pool, _ := isolatedPools(t, ctx, storageReviewDatabaseURL(t))
	if _, err := ApplyMigrations(ctx, pool, MigrationBudgets{}); err != nil {
		t.Fatal(err)
	}
	rows, err := pool.Query(ctx, workflowRunBeforeUpdateTriggersSQL)
	if err != nil {
		t.Fatal(err)
	}
	type beforeUpdateTrigger struct {
		name, source string
		updateOf     []string
	}
	triggers, err := pgx.CollectRows(rows, func(row pgx.CollectableRow) (beforeUpdateTrigger, error) {
		var trigger beforeUpdateTrigger
		err := row.Scan(&trigger.name, &trigger.source, &trigger.updateOf)
		return trigger, err
	})
	if err != nil {
		t.Fatal(err)
	}
	guards := 0
	for _, trigger := range triggers {
		for _, assignment := range assignedTriggerColumn.FindAllStringSubmatch(trigger.source, -1) {
			t.Errorf("%s assigns NEW.%s, which no UPDATE OF guard sees", trigger.name, assignment[1])
		}
		if len(trigger.updateOf) == 0 {
			continue
		}
		guards++
		var compared []string
		for _, comparison := range comparedTriggerColumn.FindAllStringSubmatch(trigger.source, -1) {
			if comparison[1] != comparison[2] {
				t.Errorf("%s compares NEW.%s with OLD.%s", trigger.name, comparison[1], comparison[2])
			}
			compared = append(compared, comparison[1])
		}
		slices.Sort(compared)
		slices.Sort(trigger.updateOf)
		if compared = slices.Compact(compared); !slices.Equal(compared, trigger.updateOf) {
			t.Errorf("%s fires on UPDATE OF %v but compares %v", trigger.name, trigger.updateOf, compared)
		}
	}
	if guards < 3 {
		t.Fatalf("found %d column-scoped guards of workflow_runs, want the three of PER-R22", guards)
	}
}
