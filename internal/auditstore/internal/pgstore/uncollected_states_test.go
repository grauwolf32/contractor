package pgstore

import (
	"slices"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/auditstore"
)

// TestUncollectedExecutionStatesSQLMatchesDeclaration keeps the state list
// of the uncollected-execution reads equal to every declared execution state
// but collected, the subset audit_executions_reconcile_idx holds
// (TestPostgresAuditStateSetsMatchSchema compares that index with it).
func TestUncollectedExecutionStatesSQLMatchesDeclaration(t *testing.T) {
	var listed []string
	for _, value := range strings.Split(strings.Trim(uncollectedExecutionStatesSQL, "()"), ",") {
		listed = append(listed, strings.Trim(strings.TrimSpace(value), "'"))
	}
	want := slices.DeleteFunc(statesOfExecutions(t), func(state string) bool {
		return state == string(auditstore.ExecutionCollected)
	})
	slices.Sort(listed)
	slices.Sort(want)
	if !slices.Equal(listed, want) {
		t.Fatalf("uncollectedExecutionStatesSQL lists %v, want every execution state but collected: %v", listed, want)
	}
}

// statesOfExecutions is every declared auditstore.ExecutionState.
func statesOfExecutions(t *testing.T) []string {
	t.Helper()
	return declaredAuditValues(t, "ExecutionState", func(value string) bool { return auditstore.ExecutionState(value).Valid() })
}
