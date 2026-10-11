package memory

import (
	"context"
	"testing"
	"time"

	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/jackc/pgx/v5"
)

// TestPostgresPlannerMemoryTransactionKeepsTheOperationBudget checks that a
// Memory transaction, which classifies its own commit, still runs under the
// operation budget selected on its context, as every InTx transaction does.
func TestPostgresPlannerMemoryTransactionKeepsTheOperationBudget(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	pool := isolatedMemoryPool(t, ctx)
	runID, stageID := "run-memory-budget", "stage-memory-budget"
	createRunningMemoryStage(t, ctx, pool, runID, stageID)
	budgets := persistencepostgres.Budgets{
		AcquireTimeout: time.Second, QueryTimeout: 9 * time.Second, StatementTimeout: 7 * time.Second,
		LockTimeout: 1500 * time.Millisecond, IdleTransactionTimeout: 11 * time.Second,
	}
	budgeted, err := persistencepostgres.WithOperationBudgets(ctx, budgets)
	if err != nil {
		t.Fatal(err)
	}
	binding := Binding{RunID: runID, StageExecutionID: stageID, Namespace: "builder"}
	for _, mutation := range []bool{false, true} {
		settings, err := withActiveStage(budgeted, pool, mutation, binding, nil, func(tx pgx.Tx) ([2]int64, error) {
			var settings [2]int64
			err := tx.QueryRow(budgeted, `SELECT (extract(epoch FROM current_setting('statement_timeout')::interval)*1000)::bigint,
(extract(epoch FROM current_setting('lock_timeout')::interval)*1000)::bigint`).Scan(&settings[0], &settings[1])
			return settings, err
		})
		if err != nil || settings != [2]int64{budgets.StatementTimeout.Milliseconds(), budgets.LockTimeout.Milliseconds()} {
			t.Fatalf("mutation=%t Memory transaction settings = (%v, %v), want %+v", mutation, settings, err, budgets)
		}
	}
}
