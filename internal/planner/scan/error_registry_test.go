package scan

import (
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"testing"
)

func TestScanPlannerErrorRegistry(t *testing.T) {
	paths, err := filepath.Glob("*.go")
	if err != nil {
		t.Fatal(err)
	}
	codes := map[string]bool{}
	for _, path := range paths {
		if strings.HasSuffix(path, "_test.go") {
			continue
		}
		source, err := parser.ParseFile(token.NewFileSet(), path, nil, 0)
		if err != nil {
			t.Fatal(err)
		}
		literalCode := func(expression ast.Expr) {
			literal, ok := expression.(*ast.BasicLit)
			if !ok || literal.Kind != token.STRING {
				t.Fatalf("%s: scan failure code must be a literal checked against the registry", path)
			}
			code, err := strconv.Unquote(literal.Value)
			if err != nil {
				t.Fatal(err)
			}
			codes[code] = true
		}
		ast.Inspect(source, func(node ast.Node) bool {
			if call, ok := node.(*ast.CallExpr); ok {
				name := ""
				switch function := call.Fun.(type) {
				case *ast.Ident:
					name = function.Name
				case *ast.SelectorExpr:
					name = function.Sel.Name
				}
				if name == "scanError" {
					literalCode(call.Args[0])
				}
			}
			// Completed scan failures carry the code on the retained result.
			if value, ok := node.(*ast.CompositeLit); ok {
				if typ, ok := value.Type.(*ast.SelectorExpr); ok && typ.Sel.Name == "TerminationError" {
					for _, element := range value.Elts {
						field, ok := element.(*ast.KeyValueExpr)
						if !ok {
							continue
						}
						if key, ok := field.Key.(*ast.Ident); ok && key.Name == "Code" {
							literalCode(field.Value)
						}
					}
				}
			}
			return true
		})
	}
	if len(codes) == 0 {
		t.Fatal("registry gate found no scan Planner codes")
	}
	for _, path := range []string{"../../../docs/spec/tools/scanners.md", "../../../docs/spec/platform/errors.md"} {
		registry, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		if missing := missingScanCodes(codes, string(registry)); len(missing) != 0 {
			t.Errorf("%s is missing scan Planner codes: %v", path, missing)
		}
		if missing := missingScanCodes(map[string]bool{"scan_unregistered_test_code": true}, string(registry)); len(missing) != 1 {
			t.Fatalf("%s: registry gate accepted an unregistered code", path)
		}
	}
}

func missingScanCodes(codes map[string]bool, registry string) []string {
	rows := regexp.MustCompile("(?m)^\\| `([a-z][a-z0-9_]*)` \\|").FindAllStringSubmatch(registry, -1)
	registered := make(map[string]bool, len(rows))
	for _, row := range rows {
		registered[row[1]] = true
	}
	var missing []string
	for code := range codes {
		if !registered[code] {
			missing = append(missing, code)
		}
	}
	sort.Strings(missing)
	return missing
}
