package rundeletion

import (
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// deletionSteps are the store methods and functions that lock or change
// state for a Run deletion. Called outside this package, they would delete a
// Run without the rest of the cascade, so only the stores that define them
// and this package may name them.
var deletionSteps = map[string]bool{
	"LockRunForDeletion": true, "DeleteLockedRunOutputPublications": true, "DeleteLockedRun": true,
	"LockRunDeletionAudits": true, "MarkExecutionRunDeleted": true,
	"InvalidateDeletedRunProjections": true, "MarkSourceRunDeleted": true,
}

func TestDeletionStepsAreReachedOnlyThroughRundeletion(t *testing.T) {
	t.Parallel()

	moduleRoot := moduleRoot(t)
	allowed := map[string]bool{}
	for _, dir := range []string{
		"internal/rundeletion", "internal/runstore", "internal/auditstore", "internal/findingintake",
	} {
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
