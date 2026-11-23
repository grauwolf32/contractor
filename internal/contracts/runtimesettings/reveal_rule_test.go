package runtimesettings

import (
	"go/ast"
	"go/parser"
	"go/token"
	"io/fs"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"testing"
)

// revealJSONCallers are the production files allowed to write RuntimeSettings
// secrets in plaintext: the prepare request encoder and the Control Plane
// client that sends it to the Runtime Agent over mTLS (SEC-R29, SEC-R88).
// Every other encoding of RuntimeSettings must use default JSON, which
// redacts SecretString.
var revealJSONCallers = []string{
	"internal/contracts/control/allocation_reveal.go",
	"internal/controlplane/runtime_request.go",
}

func TestRevealJSONIsCalledOnlyOnTheAllocationWire(t *testing.T) {
	root, err := filepath.Abs(filepath.Join("..", "..", ".."))
	if err != nil {
		t.Fatal(err)
	}
	files := 0
	var violations []string
	for _, dir := range []string{"cmd", "internal", "tools"} {
		err := filepath.WalkDir(filepath.Join(root, dir), func(path string, entry fs.DirEntry, err error) error {
			if err != nil {
				return err
			}
			if entry.IsDir() {
				if entry.Name() == "testdata" {
					return filepath.SkipDir
				}
				return nil
			}
			if !strings.HasSuffix(path, ".go") || strings.HasSuffix(path, "_test.go") {
				return nil
			}
			fset := token.NewFileSet()
			file, err := parser.ParseFile(fset, path, nil, parser.SkipObjectResolution)
			if err != nil {
				return err
			}
			relative, err := filepath.Rel(root, path)
			if err != nil {
				return err
			}
			files++
			violations = append(violations, revealJSONViolations(fset, file, filepath.ToSlash(relative))...)
			return nil
		})
		if err != nil {
			t.Fatal(err)
		}
	}
	if files == 0 {
		t.Fatal("found no Go source files")
	}
	for _, violation := range violations {
		t.Errorf("%s: RevealJSON writes RuntimeSettings secrets in plaintext and is reserved for the allocation prepare request", violation)
	}
}

func TestRevealJSONRuleReportsOtherCallers(t *testing.T) {
	source := `package example

func store(request preparer) ([]byte, error) {
	return request.RevealJSON()
}

func log(settings any) {
	encode := settings.(revealer).RevealJSON
	_, _ = encode()
}

func (s settings) RevealJSON() ([]byte, error) { return nil, nil }
`
	fset := token.NewFileSet()
	file, err := parser.ParseFile(fset, "example.go", source, parser.SkipObjectResolution)
	if err != nil {
		t.Fatal(err)
	}
	got := revealJSONViolations(fset, file, "internal/example/example.go")
	want := []string{"internal/example/example.go:4", "internal/example/example.go:8"}
	if !slices.Equal(got, want) {
		t.Fatalf("violations = %v, want %v", got, want)
	}
	if got := revealJSONViolations(fset, file, revealJSONCallers[0]); len(got) != 0 {
		t.Fatalf("allowed caller reported: %v", got)
	}
}

// revealJSONViolations reports every call of, or method value naming,
// RevealJSON outside the allowed files. Declarations are not uses.
func revealJSONViolations(fset *token.FileSet, file *ast.File, path string) []string {
	if slices.Contains(revealJSONCallers, path) {
		return nil
	}
	var violations []string
	ast.Inspect(file, func(node ast.Node) bool {
		if selector, ok := node.(*ast.SelectorExpr); ok && selector.Sel.Name == "RevealJSON" {
			violations = append(violations, path+":"+strconv.Itoa(fset.Position(selector.Pos()).Line))
		}
		return true
	})
	return violations
}
