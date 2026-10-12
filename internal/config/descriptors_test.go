package config

import (
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestRegisteredDescriptorsNormalize reports a defect in the built-in table
// as a test failure instead of the panic of the registry's first reader.
func TestRegisteredDescriptorsNormalize(t *testing.T) {
	t.Parallel()
	if _, err := normalizeDescriptors(builtInDescriptors()); err != nil {
		t.Fatalf("built-in descriptor table: %v", err)
	}
}

// Every check of a selection against registered code reads one normalized
// registry, so no caller can be handed, or build, a table of its own: the
// table and its types are unexported, the loader and the decoders take no
// descriptor argument, and inside this package only registeredDescriptors
// reads the built-in table.
func TestBuiltInDescriptorTableHasOneReader(t *testing.T) {
	t.Parallel()
	paths, err := filepath.Glob("*.go")
	if err != nil {
		t.Fatal(err)
	}
	fset := token.NewFileSet()
	var readers []string
	for _, path := range paths {
		if strings.HasSuffix(path, "_test.go") {
			continue
		}
		source, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		file, err := parser.ParseFile(fset, path, source, parser.SkipObjectResolution)
		if err != nil {
			t.Fatal(err)
		}
		for _, declaration := range file.Decls {
			ast.Inspect(declaration, func(node ast.Node) bool {
				call, ok := node.(*ast.CallExpr)
				if !ok {
					return true
				}
				if ident, ok := call.Fun.(*ast.Ident); ok && ident.Name == "builtInDescriptors" {
					readers = append(readers, enclosingName(declaration)+" at "+fset.Position(call.Pos()).String())
				}
				return true
			})
		}
	}
	if len(readers) != 1 || !strings.HasPrefix(readers[0], "registeredDescriptors ") {
		t.Fatalf("builtInDescriptors is read by %v; read the registry through registeredDescriptors instead", readers)
	}
}

func enclosingName(declaration ast.Decl) string {
	switch value := declaration.(type) {
	case *ast.FuncDecl:
		return value.Name.Name
	case *ast.GenDecl:
		for _, spec := range value.Specs {
			if variable, ok := spec.(*ast.ValueSpec); ok && len(variable.Names) > 0 {
				return variable.Names[0].Name
			}
		}
	}
	return "unknown declaration"
}

// The Planner facts live on the registered PlannerFactory descriptor; the
// catalog load, every snapshot decoder and therefore the Scheduler read them
// from there (LIM-PLN-1, LIM-PLN-2).
func TestPlannerDescriptorsDecideAgentCountAndModelUse(t *testing.T) {
	t.Parallel()
	for _, test := range []struct {
		planner   PlannerRef
		modelFree bool
		accepts   map[int]bool
	}{
		{PlannerRef{PlannerID: "passthrough", Version: "1"}, true, map[int]bool{0: false, 1: true, 2: false}},
		{PlannerRef{PlannerID: "streamline", Version: "1"}, false, map[int]bool{0: false, 1: true, 2: false}},
		{PlannerRef{PlannerID: "router", Version: "1"}, false, map[int]bool{0: false, 1: true, 2: true}},
		{PlannerRef{PlannerID: "scan-plan", Version: "1"}, true, map[int]bool{0: false, 1: true, 2: true}},
	} {
		if !KnownPlanner(test.planner) || IsModelFreePlanner(test.planner) != test.modelFree {
			t.Errorf("%+v: known = %v, model-free = %v", test.planner, KnownPlanner(test.planner), IsModelFreePlanner(test.planner))
		}
		for agents, accepted := range test.accepts {
			if err := ValidatePlannerAgents(test.planner, agents); (err == nil) != accepted {
				t.Errorf("%+v with %d logical Agents: %v", test.planner, agents, err)
			}
		}
	}
	retired := PlannerRef{PlannerID: "passthrough", Version: "2"}
	if KnownPlanner(retired) || IsModelFreePlanner(retired) {
		t.Fatal("an unregistered Planner is known")
	}
	if err := ValidatePlannerAgents(retired, 1); err == nil || !strings.Contains(err.Error(), `unknown PlannerFactory "passthrough@2"`) {
		t.Fatalf("unregistered Planner error = %v", err)
	}
}
