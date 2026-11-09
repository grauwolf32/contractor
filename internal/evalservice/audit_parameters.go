package evalservice

import (
	"github.com/grauwolf32/contractor/internal/auditservice"
	"github.com/grauwolf32/contractor/internal/contracts/auditcontract"
	"github.com/grauwolf32/contractor/internal/evaldomain"
)

// Audit scope names retain their existing meaning. When a Program also declares
// one of those names, its value is explicitly bound to both destinations.
func mapAuditParameters(c evaldomain.Case, v evaldomain.Variant, program auditcontract.Program) (auditservice.Scope, map[string]string, error) {
	mapped := MapParameters(c, v)
	scope := auditservice.Scope{Objective: c.Task.Objective, Target: mapped["target"], AuthorizationScope: mapped["authorizationScope"]}
	if value, ok := mapped["objective"]; ok {
		scope.Objective = value
	}
	parameters := map[string]string{}
	for name, value := range mapped {
		_, declared := program.Parameters[name]
		if declared || (name != "objective" && name != "target" && name != "authorizationScope") {
			parameters[name] = value
		}
	}
	_, err := program.BindParameters(parameters)
	return scope, parameters, err
}
