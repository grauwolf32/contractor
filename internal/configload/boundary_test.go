package configload

import (
	"bytes"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"io/fs"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
)

const configImportPath = "github.com/grauwolf32/contractor/internal/config"

// catalogLoaders are the internal/config entry points that read configuration
// roots. Outside internal/config itself, only this package may name them, so
// no load can silently skip the bundle checks.
var catalogLoaders = map[string]bool{
	"Load": true, "LoadUnion": true, "LoadUnionReadOnly": true, "NewManager": true,
}

// referenceRule restricts the named exports of one package to the listed
// internal/ packages (including their tests).
type referenceRule struct {
	importPath string
	names      map[string]bool
	allowed    []string
}

func TestCatalogLoadersAreReachedOnlyThroughConfigload(t *testing.T) {
	t.Parallel()

	for _, violation := range moduleReferences(t, referenceRule{
		importPath: configImportPath, names: catalogLoaders, allowed: []string{"config", "configload"},
	}) {
		t.Errorf("%s: use configload instead so the bundle checks apply", violation)
	}
}

// The checks in this package discover and validate the bundles on every load
// and return them, so the Server walks its operator root once at startup and
// seeds that plan; no other package discovers the bundles again.
func TestBundleDiscoveryIsReachedOnlyThroughConfigload(t *testing.T) {
	t.Parallel()

	discover := map[string]bool{"DiscoverBundled": true}
	for _, violation := range moduleReferences(t,
		referenceRule{
			importPath: "github.com/grauwolf32/contractor/internal/agentskills",
			names:      discover, allowed: []string{"agentskills", "configload"},
		},
		referenceRule{
			importPath: "github.com/grauwolf32/contractor/internal/auditstandards",
			names:      discover, allowed: []string{"auditstandards", "configload"},
		},
	) {
		t.Errorf("%s: use the Bundles a configload load returns instead of discovering the root again", violation)
	}
}

// moduleReferences returns every reference a rule forbids in the Go files of
// the module, walked once for all rules.
func moduleReferences(t *testing.T, rules ...referenceRule) []string {
	t.Helper()
	moduleRoot, err := filepath.Abs(filepath.Join("..", ".."))
	if err != nil {
		t.Fatal(err)
	}
	var violations []string
	fset := token.NewFileSet()
	err = filepath.WalkDir(moduleRoot, func(path string, entry fs.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		if entry.IsDir() {
			name := entry.Name()
			if path != moduleRoot && (strings.HasPrefix(name, ".") || name == "node_modules" ||
				name == "testdata" || name == "runtime" || name == "ui") {
				return filepath.SkipDir
			}
			return nil
		}
		if filepath.Ext(path) != ".go" {
			return nil
		}
		source, err := os.ReadFile(path)
		if err != nil {
			return err
		}
		var file *ast.File
		for _, rule := range rules {
			if rule.allows(moduleRoot, filepath.Dir(path)) ||
				!bytes.Contains(source, []byte(strconv.Quote(rule.importPath))) {
				continue
			}
			if file == nil {
				if file, err = parser.ParseFile(fset, path, source, parser.SkipObjectResolution); err != nil {
					return err
				}
			}
			violations = append(violations, importReferences(fset, moduleRoot, file, rule)...)
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	return violations
}

func (r referenceRule) allows(moduleRoot, directory string) bool {
	for _, name := range r.allowed {
		if directory == filepath.Join(moduleRoot, "internal", name) {
			return true
		}
	}
	return false
}

// importReferences finds every selector naming one of the rule's exports
// through the file's import of the rule's package, under any alias.
func importReferences(fset *token.FileSet, moduleRoot string, file *ast.File, rule referenceRule) []string {
	var found []string
	report := func(position token.Pos, what string) {
		at := fset.Position(position)
		relative, _ := filepath.Rel(moduleRoot, at.Filename)
		found = append(found, fmt.Sprintf("%s:%d: %s", relative, at.Line, what))
	}
	alias := ""
	for _, spec := range file.Imports {
		if path, _ := strconv.Unquote(spec.Path.Value); path != rule.importPath {
			continue
		}
		alias = rule.importPath[strings.LastIndex(rule.importPath, "/")+1:]
		if spec.Name != nil {
			alias = spec.Name.Name
		}
		if alias == "." {
			report(spec.Pos(), "dot import of "+rule.importPath+" hides restricted calls")
		}
	}
	if alias == "" || alias == "_" || alias == "." {
		return found
	}
	ast.Inspect(file, func(node ast.Node) bool {
		selector, ok := node.(*ast.SelectorExpr)
		if !ok {
			return true
		}
		if ident, ok := selector.X.(*ast.Ident); ok && ident.Name == alias && rule.names[selector.Sel.Name] {
			report(selector.Pos(), alias+"."+selector.Sel.Name)
		}
		return true
	})
	return found
}

func TestLoaderReferencesFindAliasedCallsAndValues(t *testing.T) {
	t.Parallel()

	source := `package sample

import (
	workflowconfig "github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/agentskills"
)

var load = workflowconfig.Load

func f() {
	_, _ = workflowconfig.NewManager(workflowconfig.ManagerOptions{})
	_, _ = workflowconfig.ParseConfigurationKind("model-policies")
	_, _ = agentskills.DiscoverBundled("configs")
}
`
	fset := token.NewFileSet()
	file, err := parser.ParseFile(fset, "/module/sample.go", source, parser.SkipObjectResolution)
	if err != nil {
		t.Fatal(err)
	}
	got := strings.Join(importReferences(fset, "/module", file, referenceRule{
		importPath: configImportPath, names: catalogLoaders,
	}), "\n")
	want := "sample.go:8: workflowconfig.Load\nsample.go:11: workflowconfig.NewManager"
	if got != want {
		t.Fatalf("references =\n%s\nwant\n%s", got, want)
	}
	got = strings.Join(importReferences(fset, "/module", file, referenceRule{
		importPath: "github.com/grauwolf32/contractor/internal/agentskills",
		names:      map[string]bool{"DiscoverBundled": true},
	}), "\n")
	if want := "sample.go:13: agentskills.DiscoverBundled"; got != want {
		t.Fatalf("discovery references = %q, want %q", got, want)
	}
}
