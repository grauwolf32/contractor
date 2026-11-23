package persistence

import (
	"go/ast"
	"go/parser"
	"go/token"
	"path/filepath"
	"strconv"
	"testing"
)

// A pgx safe-to-retry hint is not proof that COMMIT was not sent (PER-R13).
// Production code must use the central class and the callee's replay contract.
func TestProductionDoesNotUsePGXSafeToRetry(t *testing.T) {
	files, err := eachSourcePackage(filepath.Join("..", ".."), func(_ string, sources map[string][]byte) error {
		for path, source := range sources {
			set := token.NewFileSet()
			file, err := parser.ParseFile(set, path, source, parser.SkipObjectResolution)
			if err != nil {
				return err
			}
			for _, position := range driverRetryHints(set, file) {
				t.Errorf("%s: pgconn.SafeToRetry does not prove an uncommitted transaction; use the central class and existing replay contract", position)
			}
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

func TestDriverRetryHintScan(t *testing.T) {
	for _, test := range []struct {
		name, source string
		want         int
	}{
		{"direct", `import "github.com/jackc/pgx/v5/pgconn"; var retry = pgconn.SafeToRetry`, 1},
		{"alias", `import driver "github.com/jackc/pgx/v5/pgconn"; func retry(err error) bool { return driver.SafeToRetry(err) }`, 1},
		{"dot", `import . "github.com/jackc/pgx/v5/pgconn"; var retry = SafeToRetry`, 1},
		{"central class", `import "github.com/jackc/pgx/v5/pgconn"; var closed = pgconn.ErrConnClosed`, 0},
		{"unrelated method", `type driver struct{}; func (driver) SafeToRetry(error) bool { return false }`, 0},
	} {
		t.Run(test.name, func(t *testing.T) {
			set := token.NewFileSet()
			file, err := parser.ParseFile(set, "probe.go", "package probe; "+test.source, parser.SkipObjectResolution)
			if err != nil {
				t.Fatal(err)
			}
			if found := driverRetryHints(set, file); len(found) != test.want {
				t.Fatalf("violations = %v, want %d", found, test.want)
			}
		})
	}
}

func driverRetryHints(set *token.FileSet, file *ast.File) []token.Position {
	alias := ""
	for _, imported := range file.Imports {
		path, err := strconv.Unquote(imported.Path.Value)
		if err == nil && path == "github.com/jackc/pgx/v5/pgconn" {
			alias = "pgconn"
			if imported.Name != nil {
				alias = imported.Name.Name
			}
		}
	}
	var found []token.Position
	ast.Inspect(file, func(node ast.Node) bool {
		if selector, ok := node.(*ast.SelectorExpr); ok && selector.Sel.Name == "SafeToRetry" {
			if qualifier, ok := selector.X.(*ast.Ident); ok && qualifier.Name == alias {
				found = append(found, set.Position(selector.Pos()))
			}
			return false
		}
		if name, ok := node.(*ast.Ident); ok && alias == "." && name.Name == "SafeToRetry" {
			found = append(found, set.Position(name.Pos()))
		}
		return true
	})
	return found
}
