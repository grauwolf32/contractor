package config

import (
	"go/build"
	"go/parser"
	"go/token"
	"io/fs"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
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

// catalogOwners are the only packages whose production code imports the
// loader: they load the catalog, hold its Manager or publish and discover its
// resources. Every other package reads the resolved model from
// internal/config/resolved and AuditProfiles from internal/config/auditprofile,
// so a change to loading, publication or discovery never reaches them.
var catalogOwners = map[string]string{
	"internal/app":            "composition root that starts the configuration Manager and validates roots offline",
	"internal/configload":     "the one load path, which adds the bundle checks",
	"internal/httpapi/public": "public catalog discovery, publication and AgentTemplate instructions",
}

func TestOnlyCatalogOwnersImportTheLoader(t *testing.T) {
	t.Parallel()

	moduleRoot, err := filepath.Abs(filepath.Join("..", ".."))
	if err != nil {
		t.Fatal(err)
	}
	const loader = "github.com/grauwolf32/contractor/internal/config"
	importers := map[string]string{}
	fset := token.NewFileSet()
	for _, sourceRoot := range []string{"api", "cmd", "internal", "tools"} {
		err := filepath.WalkDir(filepath.Join(moduleRoot, sourceRoot), func(path string, entry fs.DirEntry, walkErr error) error {
			if walkErr != nil {
				return walkErr
			}
			if entry.IsDir() {
				if name := entry.Name(); name == "testdata" || strings.HasPrefix(name, ".") {
					return filepath.SkipDir
				}
				return nil
			}
			if filepath.Ext(path) != ".go" || strings.HasSuffix(path, "_test.go") {
				return nil
			}
			source, err := os.ReadFile(path)
			if err != nil {
				return err
			}
			file, err := parser.ParseFile(fset, path, source, parser.ImportsOnly)
			if err != nil {
				return err
			}
			for _, spec := range file.Imports {
				if imported, _ := strconv.Unquote(spec.Path.Value); imported == loader {
					relative, _ := filepath.Rel(moduleRoot, path)
					importers[filepath.ToSlash(filepath.Dir(relative))] = filepath.ToSlash(relative)
				}
			}
			return nil
		})
		if err != nil {
			t.Fatal(err)
		}
	}
	for pkg, file := range importers {
		if _, owner := catalogOwners[pkg]; !owner {
			t.Errorf("%s imports internal/config; read the model from internal/config/resolved or internal/config/auditprofile, or load the catalog through internal/configload", file)
		}
	}
	for pkg := range catalogOwners {
		if _, imports := importers[pkg]; !imports {
			t.Errorf("catalog owner %s no longer imports internal/config; delete it from catalogOwners", pkg)
		}
	}
}
