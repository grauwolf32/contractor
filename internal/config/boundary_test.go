package config

import (
	"go/build"
	"slices"
	"testing"
)

// The catalog loader sits low in the dependency graph. Audit vocabulary and
// bundle formats reach it as data (Toolset descriptors, contracts) or as
// checks composed by internal/configload, never as imports.
func TestConfigDoesNotImportProductContexts(t *testing.T) {
	t.Parallel()

	pkg, err := build.ImportDir(".", 0)
	if err != nil {
		t.Fatal(err)
	}
	for _, forbidden := range []string{"auditdomain", "auditstandards", "agentskills"} {
		if slices.Contains(pkg.Imports, "github.com/grauwolf32/contractor/internal/"+forbidden) {
			t.Errorf("internal/config imports internal/%s", forbidden)
		}
	}
}
