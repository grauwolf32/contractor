package rundeletion

import (
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"io/fs"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"testing"
)

// deletionSteps are the store methods that lock or change state for a Run
// deletion. Called outside this package, they would delete a Run without the
// rest of the cascade, so only the stores that define them and this package
// may name them.
var deletionSteps = map[string]bool{
	"LockRunForDeletion": true, "DeleteLockedRun": true,
	"LockRunDeletionAudits": true, "MarkExecutionRunDeleted": true,
	"InvalidateDeletedRunProjections": true,
}

func TestDeletionStepsAreReachedOnlyThroughRundeletion(t *testing.T) {
	t.Parallel()

	moduleRoot := moduleRoot(t)
	allowed := map[string]bool{}
	for _, dir := range []string{"internal/rundeletion", "internal/runstore", "internal/auditstore/internal/pgstore"} {
		allowed[filepath.Join(moduleRoot, filepath.FromSlash(dir))] = true
	}
	fset := token.NewFileSet()
	var violations []string
	walkGoFiles(t, moduleRoot, func(path string, source []byte) {
		if allowed[filepath.Dir(path)] || !mentionsDeletionStep(source) {
			return
		}
		file, err := parser.ParseFile(fset, path, source, parser.SkipObjectResolution)
		if err != nil {
			t.Fatal(err)
		}
		violations = append(violations, deletionStepReferences(fset, moduleRoot, file)...)
	})
	for _, violation := range violations {
		t.Errorf("%s: delete Runs through internal/rundeletion so the whole cascade applies", violation)
	}
}

// TestRunstoreNamesNoAuditOrFindingTables keeps Run deletion knowledge about
// Audit and finding tables out of runstore, where the import check cannot see
// it because SQL text needs no Go import.
func TestRunstoreNamesNoAuditOrFindingTables(t *testing.T) {
	t.Parallel()

	moduleRoot := moduleRoot(t)
	tables := auditAndFindingTables(t, moduleRoot)
	if len(tables) == 0 {
		t.Fatal("no Audit or finding tables found in migrations")
	}
	pattern := regexp.MustCompile(`\b(` + strings.Join(tables, "|") + `)\b`)
	runstoreDir := filepath.Join(moduleRoot, "internal", "runstore")
	entries, err := os.ReadDir(runstoreDir)
	if err != nil {
		t.Fatal(err)
	}
	for _, entry := range entries {
		name := entry.Name()
		if entry.IsDir() || filepath.Ext(name) != ".go" || strings.HasSuffix(name, "_test.go") {
			continue
		}
		source, err := os.ReadFile(filepath.Join(runstoreDir, name))
		if err != nil {
			t.Fatal(err)
		}
		for index, line := range strings.Split(string(source), "\n") {
			if match := pattern.FindString(line); match != "" {
				t.Errorf("internal/runstore/%s:%d names table %s; compose it in internal/rundeletion or the owning store", name, index+1, match)
			}
		}
	}
}

func TestDeletionStepReferencesFindMethodValuesAndCalls(t *testing.T) {
	t.Parallel()

	source := `package sample

func f(store, audits any) {
	_ = store.DeleteLockedRun(ctx, "owner", "run")
	lock := audits.LockRunDeletionAudits
	_ = store.GetRun(ctx, "run")
}
`
	fset := token.NewFileSet()
	file, err := parser.ParseFile(fset, "/module/sample.go", source, parser.SkipObjectResolution)
	if err != nil {
		t.Fatal(err)
	}
	got := strings.Join(deletionStepReferences(fset, "/module", file), "\n")
	want := "sample.go:4: DeleteLockedRun\nsample.go:5: LockRunDeletionAudits"
	if got != want {
		t.Fatalf("references =\n%s\nwant\n%s", got, want)
	}
}

func moduleRoot(t *testing.T) string {
	t.Helper()
	root, err := filepath.Abs(filepath.Join("..", ".."))
	if err != nil {
		t.Fatal(err)
	}
	return root
}

func walkGoFiles(t *testing.T, root string, visit func(string, []byte)) {
	t.Helper()
	err := filepath.WalkDir(root, func(path string, entry fs.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		if entry.IsDir() {
			name := entry.Name()
			if path != root && (strings.HasPrefix(name, ".") || name == "node_modules" ||
				name == "testdata" || name == "runtime" || name == "ui") {
				return filepath.SkipDir
			}
			return nil
		}
		if filepath.Ext(path) != ".go" {
			return nil
		}
		source, err := os.ReadFile(path)
		if err != nil {
			return err
		}
		visit(path, source)
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
}

func mentionsDeletionStep(source []byte) bool {
	for name := range deletionSteps {
		if strings.Contains(string(source), name) {
			return true
		}
	}
	return false
}

func deletionStepReferences(fset *token.FileSet, moduleRoot string, file *ast.File) []string {
	var found []string
	ast.Inspect(file, func(node ast.Node) bool {
		selector, ok := node.(*ast.SelectorExpr)
		if !ok || !deletionSteps[selector.Sel.Name] {
			return true
		}
		at := fset.Position(selector.Sel.Pos())
		relative, _ := filepath.Rel(moduleRoot, at.Filename)
		found = append(found, fmt.Sprintf("%s:%d: %s", relative, at.Line, selector.Sel.Name))
		return true
	})
	return found
}

var createTablePattern = regexp.MustCompile(`(?i)CREATE TABLE (?:IF NOT EXISTS )?((?:audit|finding)[a-z0-9_]*)`)

// auditAndFindingTables lists every Audit and finding table any migration
// creates, so a renamed table stays covered.
func auditAndFindingTables(t *testing.T, moduleRoot string) []string {
	t.Helper()
	paths, err := filepath.Glob(filepath.Join(moduleRoot, "internal", "persistence", "migrations", "*.sql"))
	if err != nil {
		t.Fatal(err)
	}
	seen := map[string]bool{}
	for _, path := range paths {
		source, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		for _, match := range createTablePattern.FindAllStringSubmatch(string(source), -1) {
			seen[strings.ToLower(match[1])] = true
		}
	}
	tables := make([]string, 0, len(seen))
	for table := range seen {
		tables = append(tables, table)
	}
	sort.Strings(tables)
	return tables
}
