package contracts

import (
	"go/ast"
	"go/parser"
	"go/token"
	"io/fs"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"testing"
)

const contractsImportPath = "github.com/grauwolf32/contractor/internal/contracts"

// stageResultImportPath holds the Stage result bounds, which count the same
// compact encoding as the wire bounds.
const stageResultImportPath = "github.com/grauwolf32/contractor/internal/planner/stageresult"

var contractByteLimit = regexp.MustCompile(`^Max[A-Za-z0-9]*Bytes$`)

// json.Marshal escapes <, > and & as six bytes each and U+2028/U+2029 as
// \u escapes, so its length overcounts the compact UTF-8 encoding that the
// Runtime measures (WRK-R37). A contract byte limit compared with that length
// rejects values the other side accepts; measure with ResultJSONSize or call
// the contract's Validate instead.
func TestContractByteLimitsAreNotComparedWithJSONMarshal(t *testing.T) {
	root, err := filepath.Abs(filepath.Join("..", ".."))
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
			violations = append(violations, marshalLimitViolations(fset, file, filepath.ToSlash(relative))...)
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
		t.Errorf("%s: a contract byte limit is compared with json.Marshal output; use contracts.ResultJSONSize or the contract's Validate", violation)
	}
}

func TestContractByteLimitRuleReportsMarshalComparisons(t *testing.T) {
	source := `package example

import (
	"encoding/json"
	wire "github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/reporting"
	"github.com/grauwolf32/contractor/internal/planner/stageresult"
)

func rejected(value any) bool {
	encoded, err := json.Marshal(value)
	return err != nil || len(encoded) > stageresult.MaxStageResultBytes
}

func nested(value any) bool {
	check := func() bool {
		var data, _ = json.Marshal(value)
		return reporting.MaxAgentStateSnapshotBytes-1 < len(data)
	}
	return check()
}

func accepted(value any) bool {
	size, err := wire.ResultJSONSize(value)
	encoded, _ := json.Marshal(value)
	return err != nil || size > wire.MaxWorkerCompletionBytes || len(encoded) > 1024
}

func wireBound(value any) bool {
	encoded, _ := json.Marshal(value)
	return len(encoded) > wire.MaxStageRequestBytes
}
`
	fset := token.NewFileSet()
	file, err := parser.ParseFile(fset, "example.go", source, parser.SkipObjectResolution)
	if err != nil {
		t.Fatal(err)
	}
	got := marshalLimitViolations(fset, file, "internal/example/example.go")
	want := []string{"internal/example/example.go:12", "internal/example/example.go:18", "internal/example/example.go:31"}
	if strings.Join(got, ",") != strings.Join(want, ",") {
		t.Fatalf("violations = %v, want %v", got, want)
	}
	for _, inPackage := range []struct{ source, path string }{
		{"package contracts\n\nimport \"encoding/json\"\n\nfunc bounded(value any) bool {\n\tencoded, _ := json.Marshal(value)\n\treturn len(encoded) >= MaxWorkerCompletionBytes\n}\n", "internal/contracts/bounded.go"},
		{"package stageresult\n\nimport \"encoding/json\"\n\nfunc bounded(value any) bool {\n\tencoded, _ := json.Marshal(value)\n\treturn len(encoded) >= MaxStageResultBytes\n}\n", "internal/planner/stageresult/bounded.go"},
	} {
		file, err = parser.ParseFile(fset, "bounded.go", inPackage.source, parser.SkipObjectResolution)
		if err != nil {
			t.Fatal(err)
		}
		if got := marshalLimitViolations(fset, file, inPackage.path); len(got) != 1 {
			t.Fatalf("%s: in-package violations = %v, want one", inPackage.path, got)
		}
		if got := marshalLimitViolations(fset, file, "internal/other/bounded.go"); len(got) != 0 {
			t.Fatalf("unqualified limit outside a limit package reported: %v", got)
		}
	}
}

// marshalLimitViolations reports comparisons of len(x), where x was assigned
// from json.Marshal in the same function, with a Max*Bytes constant of a
// package under internal/contracts or of internal/planner/stageresult, which
// holds the Stage result bounds.
func marshalLimitViolations(fset *token.FileSet, file *ast.File, relative string) []string {
	jsonName := ""
	contractPackages := map[string]bool{}
	for _, spec := range file.Imports {
		path, err := strconv.Unquote(spec.Path.Value)
		if err != nil {
			continue
		}
		name := path[strings.LastIndex(path, "/")+1:]
		if spec.Name != nil {
			name = spec.Name.Name
		}
		switch {
		case path == "encoding/json":
			jsonName = name
		case path == contractsImportPath || strings.HasPrefix(path, contractsImportPath+"/") || path == stageResultImportPath:
			contractPackages[name] = true
		}
	}
	if jsonName == "" {
		return nil
	}
	inContracts := strings.HasPrefix(relative, "internal/contracts/") || strings.HasPrefix(relative, "internal/planner/stageresult/")
	isLimit := func(expression ast.Expr) bool {
		found := false
		ast.Inspect(expression, func(node ast.Node) bool {
			switch value := node.(type) {
			case *ast.SelectorExpr:
				if pkg, ok := value.X.(*ast.Ident); ok && contractPackages[pkg.Name] && contractByteLimit.MatchString(value.Sel.Name) {
					found = true
				}
				return false
			case *ast.Ident:
				if inContracts && contractByteLimit.MatchString(value.Name) {
					found = true
				}
			}
			return !found
		})
		return found
	}
	isMarshal := func(expression ast.Expr) bool {
		call, ok := expression.(*ast.CallExpr)
		if !ok {
			return false
		}
		selector, ok := call.Fun.(*ast.SelectorExpr)
		if !ok {
			return false
		}
		pkg, ok := selector.X.(*ast.Ident)
		return ok && pkg.Name == jsonName && (selector.Sel.Name == "Marshal" || selector.Sel.Name == "MarshalIndent")
	}
	var violations []string
	for _, declaration := range file.Decls {
		function, ok := declaration.(*ast.FuncDecl)
		if !ok || function.Body == nil {
			continue
		}
		marshaled := map[string]bool{}
		ast.Inspect(function.Body, func(node ast.Node) bool {
			switch value := node.(type) {
			case *ast.AssignStmt:
				if len(value.Rhs) == 1 && isMarshal(value.Rhs[0]) {
					if name, ok := value.Lhs[0].(*ast.Ident); ok {
						marshaled[name.Name] = true
					}
				}
			case *ast.ValueSpec:
				if len(value.Values) == 1 && isMarshal(value.Values[0]) && len(value.Names) > 0 {
					marshaled[value.Names[0].Name] = true
				}
			}
			return true
		})
		if len(marshaled) == 0 {
			continue
		}
		isMarshaledLength := func(expression ast.Expr) bool {
			call, ok := expression.(*ast.CallExpr)
			if !ok || len(call.Args) != 1 {
				return false
			}
			if fun, ok := call.Fun.(*ast.Ident); !ok || fun.Name != "len" {
				return false
			}
			argument, ok := call.Args[0].(*ast.Ident)
			return ok && marshaled[argument.Name]
		}
		ast.Inspect(function.Body, func(node ast.Node) bool {
			comparison, ok := node.(*ast.BinaryExpr)
			if !ok {
				return true
			}
			switch comparison.Op {
			case token.GTR, token.GEQ, token.LSS, token.LEQ, token.EQL, token.NEQ:
			default:
				return true
			}
			if (isMarshaledLength(comparison.X) && isLimit(comparison.Y)) ||
				(isMarshaledLength(comparison.Y) && isLimit(comparison.X)) {
				violations = append(violations, relative+":"+strconv.Itoa(fset.Position(comparison.Pos()).Line))
			}
			return true
		})
	}
	return violations
}
