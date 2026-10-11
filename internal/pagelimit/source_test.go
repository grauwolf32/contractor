package pagelimit

import (
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"io/fs"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"testing"
)

// boundName matches identifiers that name a page size or a list limit.
var boundName = regexp.MustCompile(`(?i)(page|limit)`)

// TestNoOtherPublicPageBound fails when production code outside this package
// spells the public page bound itself: a limit compared with, assigned or
// declared as the literal MaxRows or MaxFetchRows. Each such copy is how a
// store came to reject the documented limit=200 page.
func TestNoOtherPublicPageBound(t *testing.T) {
	root, err := filepath.Abs(filepath.Join("..", ".."))
	if err != nil {
		t.Fatal(err)
	}
	var problems []string
	files := 0
	for _, dir := range []string{"cmd", "internal"} {
		err := filepath.WalkDir(filepath.Join(root, dir), func(path string, entry fs.DirEntry, walkErr error) error {
			if walkErr != nil {
				return walkErr
			}
			if entry.IsDir() {
				switch entry.Name() {
				case "testdata", "generated", "pagelimit":
					return filepath.SkipDir
				}
				return nil
			}
			if !strings.HasSuffix(path, ".go") || strings.HasSuffix(path, "_test.go") {
				return nil
			}
			files++
			found, err := literalPageBounds(path)
			if err != nil {
				return err
			}
			relative, _ := filepath.Rel(root, path)
			for _, problem := range found {
				problems = append(problems, relative+":"+problem)
			}
			return nil
		})
		if err != nil {
			t.Fatal(err)
		}
	}
	if files == 0 {
		t.Fatal("found no Go source files")
	}
	if len(problems) > 0 {
		t.Fatalf("use pagelimit.MaxRows, MaxFetchRows, ValidPage or ValidFetch instead of a literal page bound:\n%s",
			strings.Join(problems, "\n"))
	}
}

// The scanner is tested on synthetic source so each pattern is known to fail.
func TestLiteralPageBoundsReportsEachPattern(t *testing.T) {
	path := filepath.Join(t.TempDir(), "store.go")
	source := `package store

const maximumPageSize = 200

func check(limit int, query struct{ Limit int }) bool {
	query.Limit = 201
	_ = struct{ Limit int }{Limit: 200}
	return limit > 200 || 201 < query.Limit || limit <= 199
}
`
	if err := os.WriteFile(path, []byte(source), 0o600); err != nil {
		t.Fatal(err)
	}
	found, err := literalPageBounds(path)
	if err != nil {
		t.Fatal(err)
	}
	want := []string{
		"3:7: declares maximumPageSize = 200",
		"6:2: assigns query.Limit = 201",
		"7:26: sets Limit: 200",
		"8:9: compares limit with 200",
		"8:24: compares query.Limit with 201",
	}
	if strings.Join(found, "\n") != strings.Join(want, "\n") {
		t.Fatalf("literalPageBounds =\n%s\nwant\n%s", strings.Join(found, "\n"), strings.Join(want, "\n"))
	}
}

func literalPageBounds(path string) ([]string, error) {
	files := token.NewFileSet()
	file, err := parser.ParseFile(files, path, nil, parser.SkipObjectResolution)
	if err != nil {
		return nil, err
	}
	var found []string
	report := func(node ast.Node, format string, args ...any) {
		position := files.Position(node.Pos())
		found = append(found, fmt.Sprintf("%d:%d: ", position.Line, position.Column)+fmt.Sprintf(format, args...))
	}
	ast.Inspect(file, func(node ast.Node) bool {
		switch node := node.(type) {
		case *ast.ValueSpec:
			for index, name := range node.Names {
				if index < len(node.Values) && boundName.MatchString(name.Name) {
					if value, ok := pageBoundLiteral(node.Values[index]); ok {
						report(name, "declares %s = %d", name.Name, value)
					}
				}
			}
		case *ast.AssignStmt:
			for index, target := range node.Lhs {
				if index < len(node.Rhs) && boundName.MatchString(expressionName(target)) {
					if value, ok := pageBoundLiteral(node.Rhs[index]); ok {
						report(target, "assigns %s = %d", expressionName(target), value)
					}
				}
			}
		case *ast.KeyValueExpr:
			if key := expressionName(node.Key); boundName.MatchString(key) {
				if value, ok := pageBoundLiteral(node.Value); ok {
					report(node, "sets %s: %d", key, value)
				}
			}
		case *ast.BinaryExpr:
			switch node.Op {
			case token.LSS, token.LEQ, token.GTR, token.GEQ, token.EQL, token.NEQ:
			default:
				return true
			}
			for _, pair := range [][2]ast.Expr{{node.X, node.Y}, {node.Y, node.X}} {
				if name := expressionName(pair[0]); boundName.MatchString(name) {
					if value, ok := pageBoundLiteral(pair[1]); ok {
						report(node, "compares %s with %d", name, value)
					}
				}
			}
		}
		return true
	})
	return found, nil
}

// pageBoundLiteral reports whether expression is the integer literal MaxRows
// or MaxFetchRows.
func pageBoundLiteral(expression ast.Expr) (int, bool) {
	literal, ok := expression.(*ast.BasicLit)
	if !ok || literal.Kind != token.INT {
		return 0, false
	}
	value, err := strconv.Atoi(literal.Value)
	if err != nil || value != MaxRows && value != MaxFetchRows {
		return 0, false
	}
	return value, true
}

// expressionName renders an identifier or selector chain such as params.Limit.
func expressionName(expression ast.Expr) string {
	switch expression := expression.(type) {
	case *ast.Ident:
		return expression.Name
	case *ast.SelectorExpr:
		if prefix := expressionName(expression.X); prefix != "" {
			return prefix + "." + expression.Sel.Name
		}
		return expression.Sel.Name
	}
	return ""
}
