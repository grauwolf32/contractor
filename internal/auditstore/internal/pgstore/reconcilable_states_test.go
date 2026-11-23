package pgstore

import (
	"slices"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/auditstore"
)

// TestReconcilableAuditStatesSQLMatchesDeclaration keeps the state list of
// the Audit claim equal to every declared Audit state but draft and the
// terminal ones, the subset audits_reconcile_idx holds
// (TestPostgresAuditStateSetsMatchSchema compares that index with it).
func TestReconcilableAuditStatesSQLMatchesDeclaration(t *testing.T) {
	var listed []string
	for _, value := range strings.Split(strings.Trim(reconcilableAuditStatesSQL, "()"), ",") {
		listed = append(listed, strings.Trim(strings.TrimSpace(value), "'"))
	}
	want := slices.DeleteFunc(
		declaredAuditValues(t, "AuditState", func(value string) bool { return auditstore.AuditState(value).Valid() }),
		func(value string) bool {
			state := auditstore.AuditState(value)
			return state.Terminal() || state == auditstore.AuditDraft
		})
	slices.Sort(listed)
	slices.Sort(want)
	if !slices.Equal(listed, want) {
		t.Fatalf("reconcilableAuditStatesSQL lists %v, want every Audit state but draft and the terminal ones: %v", listed, want)
	}
}
