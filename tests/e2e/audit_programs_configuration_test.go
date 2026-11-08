//go:build e2e

package e2e

import (
	"path/filepath"
	"reflect"
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

func TestAuditProgramCatalogRemovalConfigurationClosure(t *testing.T) {
	root := stageE2EConfiguration(t, filepath.Join(repoRoot(t), "configs"), filepath.Join(t.TempDir(), "configs"), "http://127.0.0.1:1/v1")
	before, _, err := configload.LoadUnionReadOnly(root, filepath.Join(t.TempDir(), "before-managed"))
	if err != nil {
		t.Fatal(err)
	}
	removeAuditProgramAuthoringEntries(t, root)
	after, _, err := configload.LoadUnionReadOnly(root, filepath.Join(t.TempDir(), "after-managed"))
	if err != nil {
		t.Fatalf("catalog replacement must retain a complete authoring closure: %v", err)
	}
	removed := 0
	for _, profile := range before.AuditProfiles() {
		selector := profile.Ref.Name + "@" + profile.Ref.Version
		current, err := after.AuditProfile(selector)
		if removedAuditProgramProfiles[selector] {
			removed++
			if err == nil {
				t.Fatalf("removed Audit profile still resolves: %s", selector)
			}
		} else {
			if err != nil || !reflect.DeepEqual(current, profile) {
				t.Fatalf("unrelated Audit profile must remain unchanged: %s: %v", selector, err)
			}
		}
	}
	if removed != 4 || len(after.AuditProfiles()) != len(before.AuditProfiles())-removed {
		t.Fatalf("unexpected removed profile closure: removed=%d before=%d after=%d", removed, len(before.AuditProfiles()), len(after.AuditProfiles()))
	}
}
