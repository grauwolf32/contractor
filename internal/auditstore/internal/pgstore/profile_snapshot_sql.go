package pgstore

import (
	"github.com/grauwolf32/contractor/internal/config"
)

// SQL fragments that read the ResolvedAuditProfile an Audit retains in
// audits.profile_snapshot. The JSON members are the constants beside the
// encoded type; internal/persistence pins them against an encoded profile.

// profileWorkflowsSQL is the role-keyed Workflow bindings of the profile.
const profileWorkflowsSQL = "audit.profile_snapshot->'" + config.AuditProfileWorkflowsMember + "'"

// prepareRoleSQL holds when the jsonb_each row role binds a prepare Workflow.
const prepareRoleSQL = "role.value->>'" + config.AuditWorkflowKindMember + "' = '" +
	string(config.AuditWorkflowPrepare) + "'"

// roleMaxRunAttemptsSQL is the attempt budget of the jsonb_each row role.
const roleMaxRunAttemptsSQL = "(role.value->>'" + config.AuditWorkflowMaxRunAttemptsMember + "')::integer"

// profileRoleSQL applies operator (#> or #>>) to the binding of the role named
// by the SQL expression role, or to members of that binding.
func profileRoleSQL(operator, role string, members ...string) string {
	path := "ARRAY['" + config.AuditProfileWorkflowsMember + "', " + role
	for _, member := range members {
		path += ", '" + member + "'"
	}
	return "audit.profile_snapshot " + operator + " " + path + "]"
}
