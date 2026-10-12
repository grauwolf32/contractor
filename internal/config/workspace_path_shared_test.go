package config

import (
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
)

// TestSharedWorkspaceTargetCases checks Workflow authoring against the
// workspace target rule of api/testdata/v1alpha1/workspace-path-cases.json,
// which allocation validation and the Runtime apply as well, so a target that
// publishes always prepares.
func TestSharedWorkspaceTargetCases(t *testing.T) {
	t.Parallel()
	_, _, cases := contractstest.WorkspacePathCases(t)
	for _, test := range cases {
		if got := validateWorkspaceTarget(test.Value) == nil; got != test.WorkspaceTarget {
			t.Errorf("%s: Workflow workspace target accepted = %t, want %t", test.Name, got, test.WorkspaceTarget)
		}
	}
}
