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

func TestCatalogLoadersAreReachedOnlyThroughConfigload(t *testing.T) {
	t.Parallel()

	moduleRoot, err := filepath.Abs(filepath.Join("..", ".."))
	if err != nil {
		t.Fatal(err)
	}
	allowed := map[string]bool{
		filepath.Join(moduleRoot, "internal", "config"):     true,
		filepath.Join(moduleRoot, "internal", "configload"): true,
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
		if filepath.Ext(path) != ".go" || allowed[filepath.Dir(path)] {
			return nil
		}
		source, err := os.ReadFile(path)
		if err != nil {
			return err
		}
		if !bytes.Contains(source, []byte(strconv.Quote(configImportPath))) {
			return nil
		}
		file, err := parser.ParseFile(fset, path, source, parser.SkipObjectResolution)
		if err != nil {
			return err
		}
		violations = append(violations, loaderReferences(fset, moduleRoot, file)...)
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	for _, violation := range violations {
		t.Errorf("%s: use configload instead so the bundle checks apply", violation)
	}
}

func loaderReferences(fset *token.FileSet, moduleRoot string, file *ast.File) []string {
	var found []string
	report := func(position token.Pos, what string) {
		at := fset.Position(position)
		relative, _ := filepath.Rel(moduleRoot, at.Filename)
		found = append(found, fmt.Sprintf("%s:%d: %s", relative, at.Line, what))
	}
	alias := ""
	for _, spec := range file.Imports {
		if path, _ := strconv.Unquote(spec.Path.Value); path != configImportPath {
			continue
		}
		alias = "config"
		if spec.Name != nil {
			alias = spec.Name.Name
		}
		if alias == "." {
			report(spec.Pos(), "dot import of internal/config hides catalog loader calls")
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
		if ident, ok := selector.X.(*ast.Ident); ok && ident.Name == alias && catalogLoaders[selector.Sel.Name] {
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
)

var load = workflowconfig.Load

func f() {
	_, _ = workflowconfig.NewManager(workflowconfig.ManagerOptions{})
	_ = workflowconfig.MVPDescriptors()
}
`
	fset := token.NewFileSet()
	file, err := parser.ParseFile(fset, "/module/sample.go", source, parser.SkipObjectResolution)
	if err != nil {
		t.Fatal(err)
	}
	got := strings.Join(loaderReferences(fset, "/module", file), "\n")
	want := "sample.go:7: workflowconfig.Load\nsample.go:10: workflowconfig.NewManager"
	if got != want {
		t.Fatalf("references =\n%s\nwant\n%s", got, want)
	}
}
