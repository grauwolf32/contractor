//go:build e2e

package e2e

import (
	"path/filepath"
	"testing"

	"github.com/grauwolf32/contractor/internal/configload"
)

func TestAuditIterativeFixtureConfigurationLoads(t *testing.T) {
	h := &scanProcessHarness{repositoryRoot: repoRoot(t), temporaryRoot: t.TempDir()}
	root := stageE2EConfiguration(t, filepath.Join(h.repositoryRoot, "configs"), filepath.Join(h.temporaryRoot, "configs"), "http://127.0.0.1:1/v1")
	installTypedCheckFixture(t, root)
	installIterativeFixture(t, h, root)
	snapshot, _, err := configload.LoadUnionReadOnly(root, filepath.Join(t.TempDir(), "managed"))
	if err != nil {
		t.Fatal(err)
	}
	for _, profile := range []string{"iterative-fixture@1", "iterative-default-fixture@1"} {
		if _, err := snapshot.AuditProfile(profile); err != nil {
			t.Fatal(err)
		}
	}
	for _, workflow := range []string{"iterative-check-fixture@1", "iterative-default-check-fixture@1", "iterative-assessment-fixture@1", "iterative-model-free-fixture@1"} {
		if _, err := snapshot.Workflow(workflow); err != nil {
			t.Fatal(err)
		}
	}
}
