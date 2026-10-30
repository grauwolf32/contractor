package streamline

import (
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts"
)

// A Worker is workspace-eligible when one of its selections names a tool its
// Toolset descriptor classifies as observing the workspace (TLS-R38).
func TestSelectedWorkspaceObservationReadsTheToolsetDescriptor(t *testing.T) {
	t.Parallel()
	selection := func(toolset string, tools ...string) contracts.ToolsetSelection {
		return contracts.ToolsetSelection{Ref: contracts.ToolsetRef{ToolsetID: toolset, Version: "1"}, Tools: tools}
	}
	for name, test := range map[string]struct {
		toolsets []contracts.ToolsetSelection
		want     bool
	}{
		"no Toolsets":           {nil, false},
		"code analysis only":    {[]contracts.ToolsetSelection{selection("code-analysis", "find_symbol")}, false},
		"workspace diff only":   {[]contracts.ToolsetSelection{selection("workspace-changes", "diff", "rollback_changes")}, false},
		"changed paths":         {[]contracts.ToolsetSelection{selection("workspace-changes", "changed_paths")}, true},
		"read beside artifacts": {[]contracts.ToolsetSelection{selection("run-artifacts", "read_artifact"), selection("filesystem", "read_file")}, true},
		"mkdir":                 {[]contracts.ToolsetSelection{selection("edit-files", "mkdir")}, true},
	} {
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			if got := selectedWorkspaceObservation(contracts.ResolvedAgentTemplate{Toolsets: test.toolsets}); got != test.want {
				t.Fatalf("workspace-eligible = %v, want %v", got, test.want)
			}
		})
	}
}
