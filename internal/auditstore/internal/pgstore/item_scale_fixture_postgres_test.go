package pgstore

import (
	"context"
	"testing"

	"github.com/grauwolf32/contractor/internal/persistence/explaintest"
	"github.com/jackc/pgx/v5/pgxpool"
)

const (

	// planningEntries bounds the index entries the planner reads to find a
	// column's current extremes, in custom and generic plans alike, when a
	// comparison value, such as the other side's extreme in a merge join's
	// estimate, falls in a histogram end bin (see explaintest). ANALYZE
	// samples the large tables, so whether it does varies between runs (the
	// reconcile window's planning read 0 or 2 items); in a scale fixture's
	// own vacuumed database each probe reads one live entry. Row budgets add
	// it.
	planningEntries = 8
)

// explainPrepared analyzes sql with args prepared and planned in mode on a
// connection of pool, and fails t when the analysis fails.
func explainPrepared(
	t *testing.T, ctx context.Context, pool *pgxpool.Pool, mode explaintest.Mode, sql string, args ...any,
) explaintest.Analysis {
	t.Helper()
	analysis, err := explaintest.ExplainPrepared(ctx, pool, mode, sql, args...)
	if err != nil {
		t.Fatalf("%s: explain: %v\n%s", mode, err, sql)
	}
	return analysis
}
