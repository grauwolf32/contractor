package archtest

import (
	"go/ast"
	"go/parser"
	"go/token"
	"io/fs"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/config/resolved"
)

// plannerIdentityOwners define and implement the registered Planners
// (CAT-R79): internal/config/resolved names each one and holds its facts, and
// internal/planner implements them.
var plannerIdentityOwners = []string{"internal/config/resolved", "internal/planner"}

// Everywhere else a Planner identity compared with a string literal is a
// second definition of the registry that a renamed or new version misses
// silently; such code compares PlannerRef.Selector() with a resolved Planner
// constant or reads a resolved Planner fact. Test files are not scanned: they
// pin the expected Planner of an authored fixture with a literal, which is an
// expectation rather than a rule.
func TestPlannerIdentitiesAreNotComparedWithLiterals(t *testing.T) {
	identities := registeredPlannerIdentities(t)
	files := 0
	for _, root := range sourceRoots {
		err := filepath.WalkDir(filepath.Join("..", "..", root), func(path string, entry fs.DirEntry, err error) error {
			if err != nil {
				return err
			}
			relative, err := filepath.Rel(filepath.Join("..", ".."), path)
			if err != nil {
				return err
			}
			relative = filepath.ToSlash(relative)
			if entry.IsDir() {
				if entry.Name() == "testdata" || entry.Name() == "node_modules" || isPlannerIdentityOwner(relative) {
					return filepath.SkipDir
				}
				return nil
			}
			if !strings.HasSuffix(path, ".go") || strings.HasSuffix(path, "_test.go") {
				return nil
			}
			positions := token.NewFileSet()
			file, err := parser.ParseFile(positions, path, nil, 0)
			if err != nil {
				return err
			}
			files++
			for _, finding := range plannerIdentityComparisons(positions, file, identities) {
				t.Errorf("%s: Planner identity compared with a string literal; use PlannerRef.Selector() with a resolved Planner constant or a resolved Planner fact", finding)
			}
			return nil
		})
		if err != nil {
			t.Fatal(err)
		}
	}
	if files == 0 {
		t.Fatal("scanned no Go files")
	}
}

func TestPlannerIdentityScanReportsLiteralComparisons(t *testing.T) {
	identities := registeredPlannerIdentities(t)
	for name, tc := range map[string]struct {
		body     string
		findings int
	}{
		"selector literal":         {`return ref == "streamline@1"`, 1},
		"reversed selector":        {`return "router@1" != ref`, 1},
		"PlannerID literal":        {`return stage.Planner.PlannerID != "passthrough" || stage.Planner.Version != "1"`, 2},
		"PlannerRef literal":       {`return stage.Planner != (resolved.PlannerRef{PlannerID: "scan-plan", Version: "1"})`, 1},
		"switch on a selector":     {`switch ref { case "passthrough@1": return true }; return false`, 1},
		"switch on a PlannerID":    {`switch stage.Planner.PlannerID { case "next": return true }; return false`, 1},
		"resolved constant":        {`return stage.Planner.Selector() != resolved.PassthroughPlanner`, 0},
		"resolved fact":            {`return resolved.IsSingleAgentPlanner(stage.Planner)`, 0},
		"unrelated Version":        {`return manifest.Version == "1"`, 0},
		"unregistered id@version":  {`return ref == "sitemap-openapi-join@1"`, 0},
		"selector used as a value": {`stage.Planner = resolved.PlannerRef{PlannerID: "router", Version: "1"}; return true`, 0},
	} {
		t.Run(name, func(t *testing.T) {
			positions := token.NewFileSet()
			file, err := parser.ParseFile(positions, "probe.go", "package probe\n\nfunc probe() bool {\n\t"+tc.body+"\n}\n", 0)
			if err != nil {
				t.Fatal(err)
			}
			if got := plannerIdentityComparisons(positions, file, identities); len(got) != tc.findings {
				t.Fatalf("findings = %q, want %d", got, tc.findings)
			}
		})
	}
}

func isPlannerIdentityOwner(relative string) bool {
	for _, owner := range plannerIdentityOwners {
		if relative == owner || strings.HasPrefix(relative, owner+"/") {
			return true
		}
	}
	return false
}

// registeredPlannerIdentities returns the selectors of the registered
// PlannerFactories.
func registeredPlannerIdentities(t *testing.T) map[string]bool {
	t.Helper()
	selectors := resolved.PlannerSelectors()
	if len(selectors) == 0 {
		t.Fatal("the descriptor registry has no PlannerFactory")
	}
	identities := make(map[string]bool, len(selectors))
	for _, selector := range selectors {
		identities[selector] = true
	}
	return identities
}

// plannerIdentityComparisons returns the positions in file where an equality
// comparison or a switch case compares a Planner identity with a string
// literal: a registered id@version selector, a PlannerID or a Planner Version
// field, or a PlannerRef composite literal with a literal PlannerID.
func plannerIdentityComparisons(positions *token.FileSet, file *ast.File, selectors map[string]bool) []string {
	var findings []string
	report := func(node ast.Node) {
		findings = append(findings, positions.Position(node.Pos()).String())
	}
	ast.Inspect(file, func(node ast.Node) bool {
		switch node := node.(type) {
		case *ast.BinaryExpr:
			if node.Op != token.EQL && node.Op != token.NEQ {
				return true
			}
			if comparesPlannerIdentity(node.X, node.Y, selectors) || comparesPlannerIdentity(node.Y, node.X, selectors) {
				report(node)
			}
		case *ast.SwitchStmt:
			for _, statement := range node.Body.List {
				clause, ok := statement.(*ast.CaseClause)
				if !ok {
					continue
				}
				for _, value := range clause.List {
					if (node.Tag != nil && comparesPlannerIdentity(value, node.Tag, selectors)) ||
						isRegisteredSelectorLiteral(value, selectors) {
						report(value)
					}
				}
			}
		}
		return true
	})
	return findings
}

// comparesPlannerIdentity reports whether comparing literal with other
// compares a Planner identity with a string literal.
func comparesPlannerIdentity(literal, other ast.Expr, selectors map[string]bool) bool {
	literal = ast.Unparen(literal)
	if composite, ok := literal.(*ast.CompositeLit); ok {
		for _, element := range composite.Elts {
			pair, ok := element.(*ast.KeyValueExpr)
			if !ok {
				continue
			}
			if key, ok := pair.Key.(*ast.Ident); ok && key.Name == "PlannerID" && isStringLiteral(pair.Value) {
				return true
			}
		}
		return false
	}
	if !isStringLiteral(literal) {
		return false
	}
	if isRegisteredSelectorLiteral(literal, selectors) {
		return true
	}
	field, ok := ast.Unparen(other).(*ast.SelectorExpr)
	if !ok {
		return false
	}
	if field.Sel.Name == "PlannerID" {
		return true
	}
	owner, ok := ast.Unparen(field.X).(*ast.SelectorExpr)
	return field.Sel.Name == "Version" && ok && owner.Sel.Name == "Planner"
}

func isRegisteredSelectorLiteral(expression ast.Expr, selectors map[string]bool) bool {
	literal, ok := ast.Unparen(expression).(*ast.BasicLit)
	if !ok || literal.Kind != token.STRING {
		return false
	}
	value, err := strconv.Unquote(literal.Value)
	return err == nil && selectors[value]
}

func isStringLiteral(expression ast.Expr) bool {
	literal, ok := ast.Unparen(expression).(*ast.BasicLit)
	return ok && literal.Kind == token.STRING
}
