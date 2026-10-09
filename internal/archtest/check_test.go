package archtest

import (
	"bufio"
	"fmt"
	"go/build/constraint"
	"go/parser"
	"go/token"
	"io/fs"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"testing"
)

// tablePath is where developers edit layers and knownViolations.
const tablePath = "internal/archtest/layers_test.go"

// sourceRoots are the module directories whose non-test packages are
// checked. tests/ holds test harnesses, which may import any layer.
var sourceRoots = []string{"api", "cmd", "internal", "tools"}

type layerSpec struct {
	name     string
	role     string
	packages []string
}

type knownViolation struct {
	from, to string
	reason   string
	fix      string
}

// importGraph maps each package to the module packages it imports, with the
// position of the first such import. Paths are relative to the module root.
type importGraph map[string]map[string]string

func TestLayerDependencies(t *testing.T) {
	root, err := filepath.Abs(filepath.Join("..", ".."))
	if err != nil {
		t.Fatal(err)
	}
	module, err := modulePath(filepath.Join(root, "go.mod"))
	if err != nil {
		t.Fatal(err)
	}
	graph, err := loadImportGraph(root, module)
	if err != nil {
		t.Fatal(err)
	}
	if len(graph) == 0 {
		t.Fatal("found no Go packages under " + strings.Join(sourceRoots, ", "))
	}
	if problems := checkLayers(layers, knownViolations, graph); len(problems) > 0 {
		t.Fatal(report(problems))
	}
}

// The check itself is tested on a synthetic graph so each rule is known to
// fail when it should, independently of the current code base.
func TestLayerCheckReportsEachRule(t *testing.T) {
	table := []layerSpec{
		{name: "low", role: "bottom", packages: []string{"internal/a", "internal/removed"}},
		{name: "high", role: "top", packages: []string{"internal/b", "cmd/..."}},
	}
	known := []knownViolation{
		{from: "internal/a", to: "internal/b", reason: "r", fix: "f"},
		{from: "internal/a", to: "internal/gone", reason: "r", fix: "f"},
		{from: "internal/b", to: "internal/a", reason: "r", fix: "f"},
		{from: "cmd/x", to: "internal/b"},
	}
	graph := importGraph{
		"internal/a":   {"internal/b": "internal/a/a.go:3"},
		"internal/b":   {"internal/a": "internal/b/b.go:3"},
		"internal/new": {},
		"cmd/x":        {"internal/a": "cmd/x/main.go:3", "internal/b": "cmd/x/main.go:4"},
	}
	problems := checkLayers(table, known, graph)
	for _, want := range []string{
		"internal/new is not assigned to a layer",
		"internal/removed matches no package",
		"internal/a -> internal/gone: the import no longer exists",
		"internal/b -> internal/a: no longer an upward import",
		"cmd/x -> internal/b needs both a reason and a fix",
	} {
		if !slices.ContainsFunc(problems, func(problem string) bool { return strings.Contains(problem, want) }) {
			t.Errorf("no problem reports %q; got:\n%s", want, strings.Join(problems, "\n"))
		}
	}
	// The allowlisted upward import is not reported as a new violation.
	for _, problem := range problems {
		if strings.Contains(problem, "a higher layer") {
			t.Errorf("allowlisted import reported as a violation: %s", problem)
		}
	}

	problems = checkLayers(table, nil, importGraph{
		"internal/a": {"internal/b": "internal/a/a.go:3"},
		"internal/b": {},
	})
	want := "internal/a (low) imports internal/b (high), a higher layer, at internal/a/a.go:3"
	if !slices.Contains(problems, want) {
		t.Errorf("upward import not reported as %q; got:\n%s", want, strings.Join(problems, "\n"))
	}
}

// checkLayers returns one message per broken rule, in a stable order.
func checkLayers(table []layerSpec, known []knownViolation, graph importGraph) []string {
	var problems []string
	layerOf := map[string]int{}
	patterns := map[string]int{}
	used := map[string]bool{}
	listed := map[string]bool{}
	for index, spec := range table {
		for _, entry := range spec.packages {
			if listed[entry] {
				problems = append(problems, fmt.Sprintf("layer entry %s is listed twice in %s", entry, tablePath))
			}
			listed[entry] = true
			if prefix, ok := strings.CutSuffix(entry, "/..."); ok {
				patterns[prefix] = index
			} else {
				layerOf[entry] = index
			}
		}
	}
	lookup := func(pkg string) (int, bool) {
		if index, ok := layerOf[pkg]; ok {
			used[pkg] = true
			return index, true
		}
		best, found := "", false
		for prefix := range patterns {
			if (pkg == prefix || strings.HasPrefix(pkg, prefix+"/")) && len(prefix) >= len(best) {
				best, found = prefix, true
			}
		}
		if !found {
			return 0, false
		}
		used[best+"/..."] = true
		return patterns[best], true
	}
	allowed := map[[2]string]bool{}
	for _, violation := range known {
		allowed[[2]string{violation.from, violation.to}] = true
	}
	matched := map[[2]string]bool{}
	reported := map[string]bool{}
	unclassified := func(pkg, context string) {
		if !reported[pkg] {
			reported[pkg] = true
			problems = append(problems, fmt.Sprintf("%s is not assigned to a layer%s; add it to one layer in %s: %s",
				pkg, context, tablePath, layerSummary(table)))
		}
	}
	for _, pkg := range sortedKeys(graph) {
		fromLayer, ok := lookup(pkg)
		if !ok {
			unclassified(pkg, "")
			continue
		}
		for _, imported := range sortedKeys(graph[pkg]) {
			toLayer, ok := lookup(imported)
			if !ok {
				unclassified(imported, " (imported by "+pkg+")")
				continue
			}
			if toLayer <= fromLayer {
				continue
			}
			edge := [2]string{pkg, imported}
			if allowed[edge] {
				matched[edge] = true
				continue
			}
			problems = append(problems, fmt.Sprintf("%s (%s) imports %s (%s), a higher layer, at %s",
				pkg, table[fromLayer].name, imported, table[toLayer].name, graph[pkg][imported]))
		}
	}

	for _, violation := range known {
		edge := [2]string{violation.from, violation.to}
		name := violation.from + " -> " + violation.to
		switch {
		case violation.reason == "" || violation.fix == "":
			problems = append(problems, fmt.Sprintf("knownViolations entry %s needs both a reason and a fix", name))
		case matched[edge]:
		case graph[violation.from][violation.to] != "":
			// The import exists but no longer points upward.
			problems = append(problems, fmt.Sprintf("knownViolations entry %s: no longer an upward import under the current layers; delete the entry", name))
		default:
			problems = append(problems, fmt.Sprintf("knownViolations entry %s: the import no longer exists; delete the entry", name))
		}
	}

	for _, spec := range table {
		for _, entry := range spec.packages {
			if !used[entry] {
				problems = append(problems, fmt.Sprintf("layer entry %s matches no package; delete it from %s", entry, tablePath))
			}
		}
	}
	return problems
}

func layerSummary(table []layerSpec) string {
	names := make([]string, 0, len(table))
	for _, spec := range table {
		names = append(names, spec.name+" ("+spec.role+")")
	}
	return strings.Join(names, ", ")
}

func report(problems []string) string {
	return "Go layer check failed; the layer table is " + tablePath + ".\n\n" +
		strings.Join(problems, "\n") + `

A package may import only its own layer or a lower one. To fix an upward import:
  - invert the dependency: declare an interface or callback in the lower package
    and let the higher package, or the composition root internal/app, supply it;
  - move the shared type or function down into the lower layer, or below it;
  - if the dependency is genuinely intended, move a package to another layer,
    or add the import to knownViolations with a reason and the intended fix.
A stale knownViolations entry means the violation is gone: delete the entry.`
}

// loadImportGraph parses the imports of every non-test Go file under the
// source roots. Files of every build configuration count, except those
// excluded with //go:build ignore.
func loadImportGraph(root, module string) (importGraph, error) {
	graph := importGraph{}
	fset := token.NewFileSet()
	for _, sourceRoot := range sourceRoots {
		start := filepath.Join(root, sourceRoot)
		err := filepath.WalkDir(start, func(path string, entry fs.DirEntry, err error) error {
			if err != nil {
				return err
			}
			if entry.IsDir() {
				name := entry.Name()
				if path != start && (name == "testdata" || name == "vendor" ||
					strings.HasPrefix(name, ".") || strings.HasPrefix(name, "_") || isModuleRoot(path)) {
					return filepath.SkipDir
				}
				return nil
			}
			if !strings.HasSuffix(path, ".go") || strings.HasSuffix(path, "_test.go") {
				return nil
			}
			file, err := parser.ParseFile(fset, path, nil, parser.ImportsOnly|parser.ParseComments)
			if err != nil {
				return err
			}
			for _, group := range file.Comments {
				if group.Pos() > file.Package {
					break
				}
				for _, comment := range group.List {
					if expression, err := constraint.Parse(comment.Text); err == nil {
						if tag, ok := expression.(*constraint.TagExpr); ok && tag.Tag == "ignore" {
							return nil
						}
					}
				}
			}
			relative, err := filepath.Rel(root, path)
			if err != nil {
				return err
			}
			pkg := filepath.ToSlash(filepath.Dir(relative))
			if graph[pkg] == nil {
				graph[pkg] = map[string]string{}
			}
			for _, spec := range file.Imports {
				imported, err := strconv.Unquote(spec.Path.Value)
				if err != nil {
					return err
				}
				target, ok := strings.CutPrefix(imported, module+"/")
				if !ok {
					continue
				}
				if _, seen := graph[pkg][target]; !seen {
					graph[pkg][target] = fmt.Sprintf("%s:%d", filepath.ToSlash(relative), fset.Position(spec.Pos()).Line)
				}
			}
			return nil
		})
		if err != nil {
			return nil, err
		}
	}
	return graph, nil
}

func isModuleRoot(dir string) bool {
	_, err := os.Stat(filepath.Join(dir, "go.mod"))
	return err == nil
}

func modulePath(goMod string) (string, error) {
	file, err := os.Open(goMod)
	if err != nil {
		return "", err
	}
	defer file.Close()
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		if module, ok := strings.CutPrefix(strings.TrimSpace(scanner.Text()), "module "); ok {
			return strings.Trim(strings.TrimSpace(module), `"`), nil
		}
	}
	if err := scanner.Err(); err != nil {
		return "", err
	}
	return "", fmt.Errorf("%s declares no module path", goMod)
}

func sortedKeys[V any](values map[string]V) []string {
	keys := make([]string, 0, len(values))
	for key := range values {
		keys = append(keys, key)
	}
	slices.Sort(keys)
	return keys
}
