package explaintest

import (
	"context"
	"regexp"
	"slices"
	"testing"

	"github.com/jackc/pgx/v5"
)

// Querier is the part of a pgx pool or transaction that
// RequireColumnScopedGuards uses.
type Querier interface {
	Query(ctx context.Context, sql string, args ...any) (pgx.Rows, error)
}

var (
	comparedTriggerColumn = regexp.MustCompile(`(?i)NEW\.([a-z_]+) IS DISTINCT FROM OLD\.([a-z_]+)`)
	assignedTriggerColumn = regexp.MustCompile(`(?i)NEW\.([a-z_]+)\s*:=`)
)

// RequireColumnScopedGuards checks the BEFORE UPDATE row triggers of table,
// resolved through the connection's search_path, and returns how many of
// them are column-scoped (PER-R22). An UPDATE OF trigger runs only when an
// UPDATE lists one of its columns, so each such guard must list exactly the
// columns its function or WHEN condition compares between NEW and OLD
// (written NEW.c IS DISTINCT FROM OLD.c), and no BEFORE UPDATE trigger of the
// table may assign a column of NEW, which would change a value without
// listing it. The caller states how many guards it expects, so a guard that
// loses its column list fails too.
func RequireColumnScopedGuards(t testing.TB, ctx context.Context, q Querier, table string) int {
	t.Helper()
	rows, err := q.Query(ctx, beforeUpdateTriggersSQL, table)
	if err != nil {
		t.Fatalf("list the BEFORE UPDATE triggers of %s: %v", table, err)
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
		t.Fatalf("read the BEFORE UPDATE triggers of %s: %v", table, err)
	}
	guards := 0
	for _, trigger := range triggers {
		for _, assignment := range assignedTriggerColumn.FindAllStringSubmatch(trigger.source, -1) {
			t.Errorf("%s.%s assigns NEW.%s, which no UPDATE OF guard sees", table, trigger.name, assignment[1])
		}
		if len(trigger.updateOf) == 0 {
			continue
		}
		guards++
		var compared []string
		for _, comparison := range comparedTriggerColumn.FindAllStringSubmatch(trigger.source, -1) {
			if comparison[1] != comparison[2] {
				t.Errorf("%s.%s compares NEW.%s with OLD.%s", table, trigger.name, comparison[1], comparison[2])
			}
			compared = append(compared, comparison[1])
		}
		slices.Sort(compared)
		slices.Sort(trigger.updateOf)
		if compared = slices.Compact(compared); !slices.Equal(compared, trigger.updateOf) {
			t.Errorf("%s.%s fires on UPDATE OF %v but compares %v", table, trigger.name, trigger.updateOf, compared)
		}
	}
	return guards
}
