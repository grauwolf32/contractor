//go:build e2e

package e2e

import (
	"path/filepath"
	"testing"

	"github.com/grauwolf32/contractor/internal/configload"
)

func TestAuditProgramsE2EConfigurationLoads(t *testing.T) {
	root := stageE2EConfiguration(t, filepath.Join(repoRoot(t), "configs"), filepath.Join(t.TempDir(), "configs"), "http://127.0.0.1:1/v1")
	snapshot, _, err := configload.LoadUnionReadOnly(root, filepath.Join(t.TempDir(), "managed"))
	if err != nil {
		t.Fatal(err)
	}
	for _, selector := range commonAuditProfileSelectors {
		if _, err := snapshot.AuditProfile(selector); err != nil {
			t.Fatal(err)
		}
	}
}
