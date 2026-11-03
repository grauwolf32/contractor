package auditprofile_test

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/configload"
	"github.com/grauwolf32/contractor/internal/configtest"
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
	snapshot, _, err := configload.LoadUnionReadOnly(root, filepath.Join(t.TempDir(), "managed"))
	if err != nil {
		t.Fatalf("load %s: %v", root, err)
	}
	return snapshot
}
