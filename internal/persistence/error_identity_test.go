package persistence

import (
	"go/ast"
	"go/parser"
	"go/token"
	"io/fs"
	"path/filepath"
	"strings"
	"testing"
)

// A failed database call must retain its cause even when its public message
// must be static. WrapError keeps SQL and credentials out of Error() (PER-R13).
func TestDatabaseFailuresKeepTheirCause(t *testing.T) {
	root := filepath.Join("..", "..", "internal")
	files := 0
	err := filepath.WalkDir(root, func(path string, entry fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if entry.IsDir() || !strings.HasSuffix(path, ".go") || strings.HasSuffix(path, "_test.go") {
			return nil
		}
		set := token.NewFileSet()
		file, err := parser.ParseFile(set, path, nil, 0)
		if err != nil {
			return err
		}
		files++
		for _, position := range discardedDatabaseFailures(set, file) {
			t.Errorf("%s: preserve the database cause with postgres.WrapError", position)
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if files < 100 {
		t.Fatalf("scanned only %d production files", files)
	}
}

func TestDatabaseFailureScan(t *testing.T) {
	for _, sample := range []struct {
		body string
		want int
	}{
		{`_, err := db.Exec(ctx, sql); if err != nil { return errors.New("safe") }; return nil`, 1},
		{`if err := tx.QueryRow(ctx, sql).Scan(&v); err != nil { return errors.New("safe") }; return nil`, 1},
		{`if err := rows.Err(); err != nil { return errors.New("safe") }; return nil`, 1},
		{`_, err := db.Exec(ctx, sql); if err != nil { return postgres.WrapError("safe", err) }; return nil`, 0},
		{`_, err := db.Exec(ctx, sql); if err != nil { return err }; return nil`, 0},
		{`if !valid { return errors.New("invalid input") }; return nil`, 0},
	} {
		set := token.NewFileSet()
		file, err := parser.ParseFile(set, "probe.go", "package probe; func f() error { "+sample.body+" }", 0)
		if err != nil {
			t.Fatal(err)
		}
		if found := discardedDatabaseFailures(set, file); len(found) != sample.want {
			t.Errorf("%s: got %d violations, want %d", sample.body, len(found), sample.want)
		}
	}
}

func discardedDatabaseFailures(set *token.FileSet, file *ast.File) []token.Position {
	var result []token.Position
	ast.Inspect(file, func(node ast.Node) bool {
		block, ok := node.(*ast.BlockStmt)
		if !ok {
			return true
		}
		for index, statement := range block.List {
			branch, ok := statement.(*ast.IfStmt)
			if !ok {
				continue
			}
			origin := branch.Init
			if origin == nil && index > 0 {
				if assignment, ok := block.List[index-1].(*ast.AssignStmt); ok {
					origin = assignment
				}
			}
			if (!hasDatabaseCall(origin) && !hasDatabaseCall(branch.Cond)) || !checksFailure(branch.Cond) {
				continue
			}
			// Inspect returns only, never error constructors in diagnostics or a
			// closure with its own error handling.
			ast.Inspect(branch, func(n ast.Node) bool {
				if _, nested := n.(*ast.FuncLit); nested {
					return false
				}
				returned, ok := n.(*ast.ReturnStmt)
				if !ok {
					return true
				}
				for _, value := range returned.Results {
					call, ok := value.(*ast.CallExpr)
					if !ok {
						continue
					}
					selector, ok := call.Fun.(*ast.SelectorExpr)
					if !ok || selector.Sel.Name != "New" {
						continue
					}
					if name, ok := selector.X.(*ast.Ident); ok && name.Name == "errors" {
						result = append(result, set.Position(value.Pos()))
					}
				}
				return false
			})
		}
		return true
	})
	return result
}

func checksFailure(condition ast.Expr) bool {
	found := false
	ast.Inspect(condition, func(node ast.Node) bool {
		if comparison, ok := node.(*ast.BinaryExpr); ok && comparison.Op == token.NEQ {
			if name, ok := comparison.Y.(*ast.Ident); ok && name.Name == "nil" {
				found = true
			}
		}
		if call, ok := node.(*ast.CallExpr); ok {
			if selector, ok := call.Fun.(*ast.SelectorExpr); ok && selector.Sel.Name == "Is" {
				if name, ok := selector.X.(*ast.Ident); ok && name.Name == "errors" {
					found = true
				}
			}
		}
		return true
	})
	return found
}

func hasDatabaseCall(node ast.Node) bool {
	found := false
	if node == nil {
		return false
	}
	ast.Inspect(node, func(n ast.Node) bool {
		call, ok := n.(*ast.CallExpr)
		if !ok {
			return true
		}
		selector, ok := call.Fun.(*ast.SelectorExpr)
		if !ok {
			return true
		}
		switch selector.Sel.Name {
		case "Exec", "Query", "QueryRow", "Scan", "Begin", "BeginTx", "Commit", "Rollback":
			found = true
		case "Err":
			if name, ok := selector.X.(*ast.Ident); ok && strings.HasSuffix(strings.ToLower(name.Name), "rows") {
				found = true
			}
		}
		return true
	})
	return found
}
