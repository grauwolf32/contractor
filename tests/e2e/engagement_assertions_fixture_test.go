//go:build e2e

package e2e

import (
	"encoding/json"
	"testing"

	"github.com/grauwolf32/contractor/internal/auditdomain"
)

func engagementAssertionSource(t *testing.T, statement string) []byte {
	t.Helper()
	raw, err := json.Marshal(map[string]any{
		"schema": "contractor.audit.assertions.v1",
		"items": []any{map[string]any{
			"key": "check-preflight", "version": "1", "statement": statement,
			"scope": map[string]string{}, "grounds": "An authored Engagement conformance fixture requirement",
			"applicability": "always", "allowed_methods": []string{"static"}, "required_evidence": []string{},
		}},
	})
	if err != nil {
		t.Fatal(err)
	}
	return raw
}

func TestEngagementAssertionSourcesUseCurrentDomainContract(t *testing.T) {
	for _, statement := range []string{"Record bounded preflight conformance.", "Record private identity conformance."} {
		plan, err := auditdomain.ChecklistAssertions(engagementAssertionSource(t, statement), "application/json", "check")
		if err != nil {
			t.Fatal(err)
		}
		seed := plan.Hypotheses["check-preflight"]
		if len(plan.Hypotheses) != 1 || len(plan.Work) != 1 || seed.Statement != statement || seed.Grounds == "" || seed.Scope["allowed_methods"] != `["static"]` || seed.Scope["required_evidence"] != `[]` {
			t.Fatalf("Engagement assertion lost its authored contract: %+v", plan)
		}
	}
}
