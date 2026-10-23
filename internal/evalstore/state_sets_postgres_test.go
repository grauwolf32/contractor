package evalstore

import (
	"context"
	"slices"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/evaldomain"
	"github.com/grauwolf32/contractor/internal/persistence/statesettest"
)

// TestPostgresEvalStateSetsMatchSchema holds the state columns of the
// migrated eval_experiments and eval_submissions tables to the
// evaldomain.State and SubmissionState declarations: each state CHECK lists
// exactly the declared constants, and every partial index admits a named
// subset. The handwritten subsets must split the declared states, so a state
// added to either type fails until it is classified.
func TestPostgresEvalStateSetsMatchSchema(t *testing.T) {
	pool := testPool(t)
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()

	experiments := statesettest.DeclaredValues(t, "internal/evaldomain", "State")
	terminal := slices.DeleteFunc(slices.Clone(experiments), func(value string) bool {
		return !evaldomain.State(value).Terminal()
	})
	for _, table := range []statesettest.Table{
		{
			Name: "eval_experiments",
			Columns: []statesettest.Column{{
				Name: "state", Check: "eval_experiments_state_check", Values: experiments,
				Subsets: map[string][]string{
					// The coordinator claims an experiment in an active state
					// at once and a paused one when its deadline passes
					// (claimStatement); idle and terminal ones have no work.
					"active": {
						string(evaldomain.StatePreparing), string(evaldomain.StateRunning),
						string(evaldomain.StateSettling), string(evaldomain.StatePausing),
						string(evaldomain.StateCancelling),
					},
					"paused":   {string(evaldomain.StatePaused)},
					"idle":     {string(evaldomain.StateDraft), string(evaldomain.StateReady)},
					"terminal": terminal,
				},
				Partitions: [][]string{{"active", "paused", "idle", "terminal"}},
			}},
			Indexes: map[string]map[string]string{
				"eval_experiments_claim_active_idx":          {"state": "active"},
				"eval_experiments_claim_paused_deadline_idx": {"state": "paused"},
				"eval_experiments_dataset_revision_idx":      nil,
			},
		},
		{
			Name: "eval_submissions",
			Columns: []statesettest.Column{{
				Name: "state", Check: "eval_submissions_state_check",
				Values: statesettest.DeclaredValues(t, "internal/evalstore", "SubmissionState"),
				Subsets: map[string][]string{
					// An outstanding submission still holds the experiment's
					// outstanding count; Settle moves it to a settled state.
					"outstanding": {string(SubmissionIntent), string(SubmissionAccepted)},
					"settled":     {string(SubmissionTerminal), string(SubmissionRejected)},
				},
				Partitions: [][]string{{"outstanding", "settled"}},
			}},
			Indexes: map[string]map[string]string{
				"eval_submissions_execution_idx":   nil,
				"eval_submissions_outstanding_idx": {"state": "outstanding"},
				"eval_submissions_reconcile_idx":   {"state": "outstanding"},
				"eval_submissions_workspace_idx":   nil,
			},
		},
	} {
		t.Run(table.Name, func(t *testing.T) {
			statesettest.Require(t, ctx, pool, table)
		})
	}
}
