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
	items := declaredAuditValues(t, "ItemState", func(value string) bool { return auditstore.ItemState(value).Valid() })
	executions := declaredAuditValues(t, "ExecutionState", func(value string) bool { return auditstore.ExecutionState(value).Valid() })
	reviews := declaredAuditValues(t, "ReviewState", func(value string) bool { return auditstore.ReviewState(value).Valid() })
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
			}},
			Indexes: map[string]map[string]string{
				"audits_reconcile_idx": {"state": "reconcilable"},
			},
		},
		{
			Name: "audit_items",
			Columns: []statesettest.Column{{
				Name: "state", Check: "audit_items_state_check", Values: items,
				Subsets: map[string][]string{
					"unsettled":       without(items, string(auditstore.ItemSettled)),
					"awaiting_review": {string(auditstore.ItemAwaitingReview)},
					"ready":           {string(auditstore.ItemReady)},
				},
			}},
			Indexes: map[string]map[string]string{
				"audit_items_dispatch_idx":         {"state": "unsettled"},
				"audit_items_pending_approval_idx": {"state": "awaiting_review"},
				"audit_items_purge_pointer_idx":    nil,
				"audit_items_ready_role_idx":       {"state": "ready"},
			},
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
				"audit_executions_audit_run_cleanup_idx":   nil,
				"audit_executions_prepare_accepted_unique": nil,
				"audit_executions_prepare_attempt_unique":  nil,
				"audit_executions_reconcile_idx":           {"state": "uncollected"},
				"audit_executions_role_attempt_unique":     nil,
				"audit_executions_run_unique":              nil,
			},
		},
		{
			Name: "audit_review_requests",
			Columns: []statesettest.Column{{
				Name: "state", Check: "audit_review_requests_state_check", Values: reviews,
				Subsets: map[string][]string{"pending": {string(auditstore.ReviewPending)}},
			}},
			Indexes: map[string]map[string]string{
				"audit_review_requests_finding_idx":             nil,
				"audit_review_requests_pending_subject_unique":  {"state": "pending"},
				"audit_review_requests_report_candidate_unique": nil,
				"audit_review_requests_report_round_idx":        nil,
			},
		},
		{
			// The pointer indexes serve foreign-key probes and purge: they
			// select the rows that hold a pointer, whatever their state.
			Name: "audit_findings",
			Columns: []statesettest.Column{{
				Name: "state", Check: "audit_findings_state_check", Values: findings,
			}},
			Indexes: map[string]map[string]string{
				"audit_findings_correlation_audit_group_order_idx":   nil,
				"audit_findings_correlation_group_idx":               nil,
				"audit_findings_correlation_project_group_order_idx": nil,
				"audit_findings_current_assessment_idx":              nil,
				"audit_findings_current_decision_idx":                nil,
				"audit_findings_duplicate_target_idx":                nil,
				"audit_findings_purge_pointers_idx":                  nil,
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
