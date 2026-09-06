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
		{"2", []string{"finding"}, true},
		{"2", []string{"list_findings"}, false},
		{"2", []string{"finding", "list_findings"}, true},
		{"2", nil, false},
		{"3", []string{"finding"}, false},
	} {
		selection := []contracts.ToolsetSelection{{
			Ref: contracts.ToolsetRef{ToolsetID: "security-findings", Version: test.version}, Tools: test.tools,
		}}
		if got := selectsFindingTool(selection); got != test.allowed {
			t.Fatalf("%+v authorized = %v", selection, got)
		}
	}
	if selectsFindingTool(nil) || selectsFindingTool([]contracts.ToolsetSelection{{
		Ref: contracts.ToolsetRef{ToolsetID: "unknown", Version: "2"}, Tools: []string{"finding"},
	}}) {
		t.Fatal("unregistered proposal operation authorized")
	}
}
