package persistence

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

var directRoutineSource = regexp.MustCompile(`(?i)\bprosrc\b`)

// TestRoutineInspectionsUseTheSharedBodyReader keeps new routine checks from
// overlooking SQL-standard bodies. The helper is the sole ordinary reader;
// the trigger-only query is exempt because SQL functions cannot be triggers.
func TestRoutineInspectionsUseTheSharedBodyReader(t *testing.T) {
	root := moduleRootDir(t)
	files := 0
	for _, directory := range []string{"internal", "tests", "scripts"} {
		err := filepath.WalkDir(filepath.Join(root, directory), func(path string, entry fs.DirEntry, err error) error {
			if err != nil {
				return err
			}
			if entry.IsDir() {
				switch entry.Name() {
				case "testdata", "node_modules", "__pycache__":
					return filepath.SkipDir
				}
				return nil
			}
			name, err := filepath.Rel(root, path)
			if err != nil {
				return err
			}
			switch filepath.ToSlash(name) {
			case "internal/persistence/explaintest/routine.go", "internal/persistence/explaintest/guards_sql.go":
				return nil
			case "internal/persistence/routine_source_test.go":
				return nil // This guard's negative probes are data, never database reads.
			}
			ext := filepath.Ext(path)
			if ext != ".go" && ext != ".py" && ext != ".sql" {
				return nil
			}
			source, err := os.ReadFile(path)
			if err != nil {
				return err
			}
			files++
			violations, err := routineSourceReferences(name, source)
			if err != nil {
				return err
			}
			for _, position := range violations {
				t.Errorf("%s: inspect routine bodies through explaintest.RoutineBodySQL", position)
			}
			return nil
		})
		if err != nil {
			t.Fatal(err)
		}
	}
	if files < 100 {
		t.Fatalf("scanned only %d files; the source walk is broken", files)
	}
}

func routineSourceReferences(name string, source []byte) ([]string, error) {
	var positions []string
	if filepath.Ext(name) == ".go" {
		set := token.NewFileSet()
		file, err := parser.ParseFile(set, name, source, 0)
		if err != nil {
			return nil, err
		}
		ast.Inspect(file, func(node ast.Node) bool {
			if literal, ok := node.(*ast.BasicLit); ok && literal.Kind == token.STRING {
				value, err := strconv.Unquote(literal.Value)
				if err == nil && directRoutineSource.MatchString(value) {
					positions = append(positions, set.Position(literal.Pos()).String())
				}
			}
			return true
		})
	} else {
		for index, line := range strings.Split(string(source), "\n") {
			trimmed := strings.TrimSpace(line)
			if strings.HasPrefix(trimmed, "#") || strings.HasPrefix(trimmed, "--") {
				continue
			}
			if directRoutineSource.MatchString(line) {
				positions = append(positions, fmt.Sprintf("%s:%d", name, index+1))
			}
		}
	}
	return positions, nil
}

func TestRoutineSourceGuardRejectsDirectReaders(t *testing.T) {
	for name, source := range map[string]string{
		"check.go":  "package check\nconst query = `SELECT p.prosrc FROM pg_proc p`",
		"check.py":  `query = "SELECT PROSRC FROM pg_proc"`,
		"check.sql": `SELECT prosrc FROM pg_proc;`,
	} {
		positions, err := routineSourceReferences(name, []byte(source))
		if err != nil || len(positions) != 1 {
			t.Errorf("%s: direct read was missed: %v, %v", name, positions, err)
		}
	}
	positions, err := routineSourceReferences("check.go", []byte("package check\n// prosrc is insufficient.\nconst query = `SELECT something_else FROM pg_proc`"))
	if err != nil || len(positions) != 0 {
		t.Fatalf("non-reader = %v, %v", positions, err)
	}
}
