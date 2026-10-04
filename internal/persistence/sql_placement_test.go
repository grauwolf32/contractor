package persistence

import (
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"io/fs"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
)

// maxInlineSQLLines is the longest raw-string SQL statement that production
// code may keep inline. Longer statements belong in a *_sql.go file next to
// their caller, where each one is named and commented.
const maxInlineSQLLines = 9

var sqlStatementStart = regexp.MustCompile(`(?is)^\s*(--[^\n]*\n\s*)*(WITH|SELECT|INSERT|UPDATE|DELETE|CREATE|ALTER|DROP|LOCK|TRUNCATE|VALUES)\b`)

func TestSQLPlacementKeepsLongStatementsInSQLFiles(t *testing.T) {
	root, err := filepath.Abs(filepath.Join("..", ".."))
	if err != nil {
		t.Fatal(err)
	}
	var violations []string
	for _, dir := range []string{"cmd", "internal"} {
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
			if !strings.HasSuffix(path, ".go") || strings.HasSuffix(path, "_test.go") || strings.HasSuffix(path, "_sql.go") {
				return nil
			}
			fset := token.NewFileSet()
			file, err := parser.ParseFile(fset, path, nil, 0)
			if err != nil {
				return err
			}
			relative, _ := filepath.Rel(root, path)
			violations = append(violations, inlineSQLViolations(fset, file, relative)...)
			return nil
		})
		if err != nil {
			t.Fatal(err)
		}
	}
	for _, violation := range violations {
		t.Errorf("%s: move this SQL statement into a commented *_sql.go file", violation)
	}
}

func TestSQLPlacementReportsLongInlineStatements(t *testing.T) {
	long := "SELECT a\n" + strings.Repeat("     , b\n", maxInlineSQLLines) + "  FROM t"
	short := "SELECT a\n  FROM t"
	source := "package sample\n\nfunc f() {\n\t_ = `" + long + "`\n\t_ = `" + short + "`\n\t_ = `" + strings.Repeat("text\n", 20) + "`\n}\n"
	fset := token.NewFileSet()
	file, err := parser.ParseFile(fset, "sample.go", source, 0)
	if err != nil {
		t.Fatal(err)
	}
	got := inlineSQLViolations(fset, file, "sample.go")
	if len(got) != 1 || !strings.HasPrefix(got[0], "sample.go:4:") {
		t.Fatalf("violations = %q, want one at sample.go:4", got)
	}
}

func inlineSQLViolations(fset *token.FileSet, file *ast.File, path string) []string {
	var violations []string
	ast.Inspect(file, func(node ast.Node) bool {
		literal, ok := node.(*ast.BasicLit)
		if !ok || literal.Kind != token.STRING || !strings.HasPrefix(literal.Value, "`") {
			return true
		}
		body := literal.Value[1 : len(literal.Value)-1]
		if !sqlStatementStart.MatchString(body) {
			return true
		}
		if lines := strings.Count(strings.Trim(body, "\n"), "\n") + 1; lines > maxInlineSQLLines {
			position := fset.Position(literal.Pos())
			violations = append(violations, fmt.Sprintf("%s:%d: %d lines", path, position.Line, lines))
		}
		return true
	})
	return violations
}
