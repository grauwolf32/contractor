package evalservice

import (
	"maps"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/config/auditprofile"
	"github.com/grauwolf32/contractor/internal/contracts/auditcontract"
	"github.com/grauwolf32/contractor/internal/evaldomain"
)

func TestAuditParameterEligibilityUsesProgramContract(t *testing.T) {
	program := auditcontract.Program{Parameters: map[string]auditcontract.ProgramParameter{
		"server": {Required: true}, "operations": {Required: true}, "test_parameters": {Required: true},
		"credential_id": {}, "tags": {}, "target": {},
	}}
	snapshot := BindingSnapshot{Audit: &auditprofile.PinnedProgram{Program: program}}
	for _, test := range []struct {
		name, key, value, missing, state string
	}{
		{name: "required-values", state: "eligible"},
		{name: "credential", key: "credential_id", value: "test-credential", state: "eligible"},
		{name: "web-filter", key: "tags", value: "web", state: "eligible"},
		{name: "scope-overlap", key: "target", value: "https://target.test", state: "eligible"},
		{name: "scope", key: "authorizationScope", value: "local fixture", state: "eligible"},
		{name: "no-launch-values", missing: "*", state: "unsupported"},
		{name: "missing-server", missing: "server", state: "unsupported"},
		{name: "missing-operations", missing: "operations", state: "unsupported"},
		{name: "missing-test-parameters", missing: "test_parameters", state: "unsupported"},
		{name: "unknown", key: "undeclared", value: "value", state: "unsupported"},
		{name: "over-limit", key: "server", value: strings.Repeat("x", 4097), state: "unsupported"},
		{name: "nul", key: "server", value: "a\x00b", state: "unsupported"},
	} {
		t.Run(test.name, func(t *testing.T) {
			c := evaldomain.Case{}
			c.Task.Parameters = map[string]string{"server": "https://case.test", "operations": "getPet", "test_parameters": "id"}
			delete(c.Task.Parameters, test.missing)
			if test.missing == "*" {
				c.Task.Parameters = map[string]string{}
			}
			v := evaldomain.Variant{Kind: "audit", Parameters: map[string]string{}}
			if test.key != "" {
				v.Parameters[test.key] = test.value
			}
			if got := caseEligibility(c, v, snapshot, nil); got.State != test.state {
				t.Fatalf("eligibility = %+v, want %s", got, test.state)
			}
		})
	}
}

func TestAuditScopeNamesAlsoBindWhenDeclaredByProgram(t *testing.T) {
	c := evaldomain.Case{}
	c.Task.Objective = "task objective"
	c.Task.Parameters = map[string]string{"objective": "case objective", "target": "case target", "authorizationScope": "case scope"}
	v := evaldomain.Variant{Parameters: map[string]string{"objective": "$task.objective", "target": "variant target"}}
	p := auditcontract.Program{Parameters: map[string]auditcontract.ProgramParameter{
		"objective": {Required: true}, "target": {Required: true}, "authorizationScope": {Required: true},
	}}
	scope, values, err := mapAuditParameters(c, v, p)
	want := map[string]string{"objective": "task objective", "target": "variant target", "authorizationScope": "case scope"}
	if err != nil || !maps.Equal(values, want) || scope.Objective != want["objective"] || scope.Target != want["target"] || scope.AuthorizationScope != want["authorizationScope"] {
		t.Fatalf("mapping = %+v %+v, %v", scope, values, err)
	}
	if c.Task.Parameters["target"] != "case target" || v.Parameters["objective"] != "$task.objective" {
		t.Fatal("mapping mutated the source recipe")
	}
}
