package pgstore

import "github.com/grauwolf32/contractor/internal/config/auditprofile"

// SQL fragments that read the ResolvedAuditProfile an Audit retains in
// audits.profile_snapshot. The JSON members are the constants beside the
// encoded type; internal/persistence pins them against an encoded profile.

// profileRoleSQL applies operator (#> or #>>) to the binding of the role named
// by the SQL expression role, or to members of that binding.
func profileRoleSQL(operator, role string, members ...string) string {
	path := "ARRAY['" + auditprofile.AuditProfileWorkflowsMember + "', " + role
	for _, member := range members {
		path += ", '" + member + "'"
	}
	return "audit.profile_snapshot " + operator + " " + path + "]"
}
