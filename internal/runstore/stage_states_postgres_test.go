package runstore

import (
	"context"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/persistence/statesettest"
)

// TestPostgresStageExecutionStateSetsMatchSchema holds the migrated
// stage_executions table to the StageExecutionState declaration (SCH-R42):
// its state CHECK lists exactly the declared states, and every partial index
// admits a named subset. The active and terminal subsets are the production
// sets and must split the declared states, so a new state fails until it is
// classified; the terminal
// history indexes read terminal_at, which stage_execution_state_shape sets in
// exactly the terminal states.
func TestPostgresStageExecutionStateSetsMatchSchema(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	pool := isolatedRunStorePool(t, ctx)
	statesettest.Require(t, ctx, pool, statesettest.Table{
		Name: "stage_executions",
		Columns: []statesettest.Column{{
			Name: "state", Check: "stage_executions_state_check",
			Values: statesettest.DeclaredValues(t, "internal/runstore", "StageExecutionState"),
			Subsets: map[string][]string{
				"active":   stageStateStrings(activeStageStates[:]),
				"terminal": stageStateStrings(terminalStageStates[:]),
			},
			Partitions: [][]string{{"active", "terminal"}},
			Markers: []statesettest.Marker{
				{Column: "terminal_at", Check: "stage_execution_state_shape", Subset: "terminal"},
			},
		}},
		Indexes: map[string]map[string]string{
			// Escalation ordinals are unique per configuration variant in
			// every state.
			"stage_execution_escalation_ordinal_unique": nil,
			"stage_execution_transition_run_idx":        nil,
			"stage_executions_run_active_idx":           {"state": "active"},
			"stage_executions_run_terminal_history_idx": {"state": "terminal"},
			"stage_executions_terminal_history_idx":     {"state": "terminal"},
		},
	})
}
