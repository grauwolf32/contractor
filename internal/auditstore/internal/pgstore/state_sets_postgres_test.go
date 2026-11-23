package pgstore

import (
	"context"
	"os"
	"slices"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/persistence/statesettest"
)

// TestPostgresAuditStateSetsMatchSchema holds the state columns of the Audit
// and finding tables to their auditstore declarations: each state CHECK lists
// exactly the declared constants, each constant passes its type's Valid, and
// every partial index admits a named subset derived from the declaration, so
// a state added in Go or in the schema alone fails here.
func TestPostgresAuditStateSetsMatchSchema(t *testing.T) {
	databaseURL := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if databaseURL == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	pool := isolatedAuditPool(t, ctx, databaseURL)

	audits := declaredAuditValues(t, "AuditState", func(value string) bool { return auditstore.AuditState(value).Valid() })
	tasks := statesettest.DeclaredValues(t, "internal/auditstore", "TaskState")
	executions := declaredAuditValues(t, "ExecutionState", func(value string) bool { return auditstore.ExecutionState(value).Valid() })
	findings := declaredAuditValues(t, "FindingState", func(value string) bool { return auditstore.FindingState(value).Valid() })
	without := func(values []string, excluded ...string) []string {
		return slices.DeleteFunc(slices.Clone(values), func(value string) bool { return slices.Contains(excluded, value) })
	}

	for _, table := range []statesettest.Table{
		{
			Name: "audits",
			Columns: []statesettest.Column{{
				Name: "state", Check: "audits_state_check", Values: audits,
				Subsets: map[string][]string{
					// The controller reconciles every started Audit until it
					// is terminal, deleting ones included (claimAuditSQL).
					"reconcilable": slices.DeleteFunc(slices.Clone(audits), func(value string) bool {
						state := auditstore.AuditState(value)
						return state.Terminal() || state == auditstore.AuditDraft
					}),
				},
			}, {
				Name: "hold_state", Check: "audits_hold_state_check",
				Values: statesettest.DeclaredValues(t, "internal/auditstore", "HoldState"),
				Subsets: map[string][]string{
					// Credential deletion checks the Audits that hold their
					// baseline credentials (listHeldAuditIDsByCredential).
					"held": {string(auditstore.HoldHeld)},
				},
			}},
			Indexes: map[string]map[string]string{
				"audits_reconcile_idx":                {"state": "reconcilable"},
				"audits_held_llm_credentials_idx":     {"hold_state": "held"},
				"audits_held_runtime_credentials_idx": {"hold_state": "held"},
			},
		},
		{
			Name: "audit_tasks",
			Columns: []statesettest.Column{{Name: "state", Check: "audit_tasks_state_check", Values: tasks,
				Subsets: map[string][]string{"ready": {string(auditstore.TaskReady)}, "settled": {string(auditstore.TaskSettled)}}}},
			Indexes: map[string]map[string]string{"audit_tasks_ready_workflow_idx": {"state": "ready"}, "audit_tasks_last_execution_idx": nil, "audit_tasks_continuation_cursor": nil, "audit_tasks_hypothesis_idx": nil, "audit_tasks_workflow_external_ordinal": {"state": "settled"}},
		},
		{
			Name: "audit_executions",
			Columns: []statesettest.Column{{
				Name: "state", Check: "audit_executions_state_check", Values: executions,
				Subsets: map[string][]string{
					"uncollected": without(executions, string(auditstore.ExecutionCollected)),
				},
			}},
			Indexes: map[string]map[string]string{
				"audit_executions_audit_run_cleanup_idx": nil,
				"audit_executions_reconcile_idx":         {"state": "uncollected"},
				"audit_executions_run_unique":            nil,
			},
		},

		{
			// The pointer indexes serve foreign-key probes and purge: they
			// select the rows that hold a pointer, whatever their state.
			Name:    "project_findings",
			Columns: []statesettest.Column{{Name: "state", Check: "project_findings_state_check", Values: findings}},
			Indexes: map[string]map[string]string{
				"project_findings_duplicate_idx": nil,

				"project_findings_qualified_key_idx": nil,
			},
		},
	} {
		t.Run(table.Name, func(t *testing.T) {
			statesettest.Require(t, ctx, pool, table)
		})
	}
}

// declaredAuditValues returns the constants of the auditstore type typeName
// and fails unless its Valid accepts each of them.
func declaredAuditValues(t *testing.T, typeName string, valid func(string) bool) []string {
	t.Helper()
	values := statesettest.DeclaredValues(t, "internal/auditstore", typeName)
	for _, value := range values {
		if !valid(value) {
			t.Errorf("auditstore.%s(%q) is declared but not Valid", typeName, value)
		}
	}
	if valid("not-a-declared-value") {
		t.Errorf("auditstore.%s accepts an undeclared value", typeName)
	}
	return values
}
