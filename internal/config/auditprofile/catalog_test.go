package auditprofile_test

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/config/auditprofile"
	"github.com/grauwolf32/contractor/internal/configload"
	"github.com/grauwolf32/contractor/internal/configtest"
	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
)

// The profiles these tests decode again are resolved from the configuration
// loader's frozen fixtures (../testdata/README.md): its core catalog plus the
// named slices.
const loaderFixtures = "../testdata"

func copyCatalog(t *testing.T, slices ...string) string {
	t.Helper()
	root := configtest.CopyWithPolicies(t, filepath.Join(loaderFixtures, "valid"))
	for _, slice := range slices {
		if err := os.CopyFS(root, os.DirFS(filepath.Join(loaderFixtures, slice))); err != nil {
			t.Fatal(err)
		}
	}
	return root
}

func loadCatalog(t *testing.T, root string) *config.Snapshot {
	t.Helper()
	snapshot, err := configload.Load(root)
	if err != nil {
		t.Fatalf("load %s: %v", root, err)
	}
	return snapshot
}

func loadProfile(t *testing.T, root, ref string) auditprofile.ResolvedAuditProfile {
	t.Helper()
	profile, err := loadCatalog(t, root).AuditProfile(ref)
	if err != nil {
		t.Fatal(err)
	}
	return profile
}

// writeAuditProfile installs one AuditProfile manifest into root.
func writeAuditProfile(t *testing.T, root, name, manifest string) {
	t.Helper()
	path := filepath.Join(root, "audit-profiles", name+".yaml")
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(manifest), 0o644); err != nil {
		t.Fatal(err)
	}
}

// templateAuditProfileYAML is the shared custom-checklist profile fixture
// under name.
func templateAuditProfileYAML(t *testing.T, name string) string {
	t.Helper()
	data, err := os.ReadFile(filepath.Join(loaderFixtures, "audit-profile-template.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	return strings.Replace(string(data), "PROFILE_NAME", name, 1)
}

// sharedAuditProfileYAML is a profile of api/testdata/audit-composition, so
// only a TestShared* test may call it.
func sharedAuditProfileYAML(t *testing.T, name string) string {
	t.Helper()
	return string(contractstest.ReadFile(t, "api", "testdata", "audit-composition", name+".yaml"))
}

// redigest pins profile's current content under a valid digest.
func redigest(t *testing.T, profile *auditprofile.ResolvedAuditProfile) {
	t.Helper()
	digest, err := auditprofile.Digest(*profile)
	if err != nil {
		t.Fatal(err)
	}
	profile.Ref.Digest = digest
}
