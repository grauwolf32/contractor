package findingintake

import (
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts"
)

func TestFindingIntakeAuthorizesOnlySelectedProposalOperations(t *testing.T) {
	for _, test := range []struct {
		version string
		tools   []string
		allowed bool
	}{
		{"1", []string{"finding"}, true},
		{"1", []string{"list_findings"}, false},
		{"2", []string{"finding"}, false},
		{"2", []string{"list_findings"}, false},
		{"1", []string{"finding", "list_findings"}, true},
		{"2", nil, false},
		{"3", []string{"finding"}, false},
		{"4", []string{"finding"}, false},
	} {
		selection := []contracts.ToolsetSelection{{
			Ref: contracts.ToolsetRef{ToolsetID: "security-findings", Version: test.version}, Tools: test.tools,
		}}
		if got := selectsFindingTool(selection); got != test.allowed {
			t.Fatalf("%+v authorized = %v", selection, got)
		}
	}
	for _, toolset := range []string{"security-findings-code", "security-findings-http"} {
		selection := []contracts.ToolsetSelection{{
			Ref: contracts.ToolsetRef{ToolsetID: toolset, Version: "1"}, Tools: []string{"finding"},
		}}
		if !selectsFindingTool(selection) {
			t.Fatalf("specialist publisher %s was not authorized", toolset)
		}
	}
	if selectsFindingTool(nil) || selectsFindingTool([]contracts.ToolsetSelection{{
		Ref: contracts.ToolsetRef{ToolsetID: "unknown", Version: "2"}, Tools: []string{"finding"},
	}}) {
		t.Fatal("unregistered proposal operation authorized")
	}
}
