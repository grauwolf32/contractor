//go:build e2e

package e2e

import (
	"path/filepath"
	"testing"

	"github.com/grauwolf32/contractor/internal/configload"
)

func TestAuditCommonFixtureConfigurationLoads(t *testing.T) {
	h := &scanProcessHarness{repositoryRoot: repoRoot(t), temporaryRoot: t.TempDir()}
	root := stageE2EConfiguration(t, filepath.Join(h.repositoryRoot, "configs"), filepath.Join(h.temporaryRoot, "configs"), "http://127.0.0.1:1/v1")
	installBasicAuditProgram(t, root)
	installCommonPrepared256Program(t, root)
	snapshot, _, err := configload.LoadUnionReadOnly(root, filepath.Join(t.TempDir(), "managed"))
	if err != nil {
		t.Fatal(err)
	}
	for _, profile := range []string{"basic-process@1", "prepared-256@1"} {
		if _, err := snapshot.AuditProfile(profile); err != nil {
			t.Fatal(err)
		}
	}
}
