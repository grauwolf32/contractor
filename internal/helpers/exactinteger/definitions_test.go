package exactinteger_test

import (
	"go/ast"
	"go/constant"
	"go/parser"
	"go/token"
	"io/fs"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
)

func integerLiteral(expr ast.Expr) constant.Value {
	switch v := expr.(type) {
	case *ast.BasicLit:
		if v.Kind == token.INT || v.Kind == token.FLOAT {
			return constant.MakeFromLiteral(v.Value, v.Kind, 0)
		}
	case *ast.ParenExpr:
		return integerLiteral(v.X)
	case *ast.UnaryExpr:
		if x := integerLiteral(v.X); x != nil && (v.Op == token.ADD || v.Op == token.SUB) {
			return constant.UnaryOp(v.Op, x, 0)
		}
	case *ast.BinaryExpr:
		x, y := integerLiteral(v.X), integerLiteral(v.Y)
		if x == nil || y == nil {
			return nil
		}
		switch v.Op {
		case token.ADD, token.SUB:
			return constant.BinaryOp(x, v.Op, y)
		case token.SHL:
			if shift, ok := constant.Uint64Val(y); ok && shift <= 64 && x.Kind() == constant.Int {
				return constant.Shift(x, v.Op, uint(shift))
			}
		}
	}
	return nil
}

func copiedBoundPositions(name string, source []byte) ([]token.Position, error) {
	set := token.NewFileSet()
	file, err := parser.ParseFile(set, name, source, parser.SkipObjectResolution)
	if err != nil {
		return nil, err
	}
	var canonical ast.Expr
	if name == "internal/helpers/exactinteger/integer.go" {
		ast.Inspect(file, func(n ast.Node) bool {
			if spec, ok := n.(*ast.ValueSpec); ok && len(spec.Names) == 1 && spec.Names[0].Name == "Max" && len(spec.Values) == 1 {
				canonical = spec.Values[0]
			}
			return true
		})
	}
	var positions []token.Position
	ast.Inspect(file, func(n ast.Node) bool {
		expr, ok := n.(ast.Expr)
		if !ok {
			return true
		}
		if expr == canonical {
			return false
		}
		value := integerLiteral(expr)
		if literal, ok := expr.(*ast.BasicLit); ok && literal.Kind == token.STRING {
			text, _ := strconv.Unquote(literal.Value)
			value = constant.MakeFromLiteral(text, token.INT, 0)
		}
		if value == nil || value.Kind() == constant.Unknown {
			return true
		}
		// Pin both inclusive and exclusive spellings. Fixture/test literals
		// are outside this production-source scan.
		if constant.Sign(value) < 0 {
			value = constant.UnaryOp(token.SUB, value, 0)
		}
		for _, target := range []string{"9007199254740991", "9007199254740992"} {
			if constant.Compare(value, token.EQL, constant.MakeFromLiteral(target, token.INT, 0)) {
				positions = append(positions, set.Position(expr.Pos()))
				return false
			}
		}
		return true
	})
	return positions, nil
}

func TestExactIntegerDefinitionsHaveOneSource(t *testing.T) {
	root := filepath.Join("..", "..", "..")
	files := 0
	for _, dir := range []string{"internal", "cmd", "api", "tools"} {
		err := filepath.WalkDir(filepath.Join(root, dir), func(path string, entry fs.DirEntry, err error) error {
			if err != nil {
				return err
			}
			if entry.IsDir() {
				if entry.Name() == "testdata" || entry.Name() == "generated" || strings.HasPrefix(entry.Name(), ".") {
					return filepath.SkipDir
				}
				return nil
			}
			if !strings.HasSuffix(path, ".go") || strings.HasSuffix(path, "_test.go") {
				return nil
			}
			name, err := filepath.Rel(root, path)
			if err != nil {
				return err
			}
			source, err := os.ReadFile(path)
			if err != nil {
				return err
			}
			positions, err := copiedBoundPositions(filepath.ToSlash(name), source)
			if err != nil {
				return err
			}
			files++
			for _, position := range positions {
				t.Errorf("%s: derive the fixed bound from exactinteger.Max or contracts.MaxExactInteger", position)
			}
			return nil
		})
		if err != nil {
			t.Fatal(err)
		}
	}
	if files < 100 {
		t.Fatalf("source scan found only %d files", files)
	}
}

func TestExactIntegerCopyGuardFindsProbes(t *testing.T) {
	for _, expr := range []string{"9007199254740991", "1<<53 - 1", "(1 << 53) - 1", "0x1fffffffffffff", `"9007199254740991"`, "1<<53"} {
		positions, err := copiedBoundPositions("probe.go", []byte("package probe\nconst copy = "+expr))
		if err != nil || len(positions) != 1 {
			t.Errorf("%s: positions=%v err=%v", expr, positions, err)
		}
	}
	for name, source := range map[string]string{
		"probe.go": "package probe\nconst copy = contracts.MaxExactInteger",
		"internal/helpers/exactinteger/integer.go": "package exactinteger\nconst Max = 1<<53 - 1",
	} {
		positions, err := copiedBoundPositions(name, []byte(source))
		if err != nil || len(positions) != 0 {
			t.Errorf("%s: positions=%v err=%v", name, positions, err)
		}
	}
}
