package e2e

import (
	"path/filepath"
	"testing"

	"github.com/grauwolf32/contractor/internal/configload"
)

func TestAuditProgramsResolveAsRunnableMVPProfiles(t *testing.T) {
	t.Parallel()

	snapshot, _, err := configload.LoadUnionReadOnly("../../configs", filepath.Join(t.TempDir(), "managed"))
	if err != nil {
		t.Fatal(err)
	}
	for _, selector := range commonAuditProfileSelectors {
		profile, err := snapshot.AuditProfile(selector)
		if err != nil {
			t.Fatal(err)
		}
		if len(profile.Program.Hypotheses) != 0 || len(profile.Program.InitialWork) == 0 || len(profile.Workflows) != len(profile.Program.Workflows) {
			t.Fatalf("%s has no common exploratory program", selector)
		}
		for alias, binding := range profile.Program.Workflows {
			workflow := profile.Workflows[alias]
			if workflow.Ref.Name+"@"+workflow.Ref.Version != binding.Ref {
				t.Fatalf("%s lost exact Workflow closure %s", selector, alias)
			}
		}
	}
}

var commonAuditProfileSelectors = []string{"openapi-operation-observe@1", "openapi-operation-trace@1", "openapi-nuclei-scan@1", "openapi-sqlmap-scan@1", "source-openapi-sqlmap-scan@1", "web-scan@1"}
