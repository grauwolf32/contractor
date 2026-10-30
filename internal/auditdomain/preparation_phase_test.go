package auditdomain

import (
	"go/ast"
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

func TestPrecedesRoundsSelectsPreparingAndInventory(t *testing.T) {
	for phase, want := range map[AuditPhase]bool{
		AuditPhaseNotStarted: false,
		AuditPhasePreparing:  true,
		AuditPhaseInventory:  true,
		AuditPhaseRounds:     false,
		"":                   false,
		"planning":           false,
	} {
		if got := phase.PrecedesRounds(); got != want {
			t.Errorf("AuditPhase(%q).PrecedesRounds() = %t, want %t", phase, got, want)
		}
	}
}

// precedesRoundsHome holds the one Go definition of the phases before Rounds.
const precedesRoundsHome = "internal/auditdomain/preparation.go"

// TestPrecedesRoundsHasOneDefinition fails on production Go code that writes
// the phase set of AuditPhase.PrecedesRounds out again: a condition, a switch
// case or a list that names exactly AuditPhasePreparing and
// AuditPhaseInventory among the phases.
func TestPrecedesRoundsHasOneDefinition(t *testing.T) {
	probe := filepath.Join(t.TempDir(), "probe.go")
	if err := os.WriteFile(probe, []byte(`package probe

func copies(p, q auditdomain.AuditPhase, round *string) bool {
	if (p == auditdomain.AuditPhasePreparing || p == auditdomain.AuditPhaseInventory) && round == nil {
		return true
	}
	if p != AuditPhasePreparing && p != AuditPhaseInventory {
		return false
	}
	switch q {
	case auditdomain.AuditPhaseInventory, auditdomain.AuditPhasePreparing:
		return true
	}
	return len([]auditdomain.AuditPhase{auditdomain.AuditPhasePreparing, auditdomain.AuditPhaseInventory}) > 0
}

func otherSets(p auditdomain.AuditPhase) bool {
	if p == auditdomain.AuditPhaseInventory || p == auditdomain.AuditPhaseRounds {
		return true
	}
	if p == auditdomain.AuditPhasePreparing && p.Valid() {
		return true
	}
	switch p {
	case auditdomain.AuditPhasePreparing:
		return p == auditdomain.AuditPhasePreparing
	case auditdomain.AuditPhaseInventory:
		return true
	}
	return p == auditdomain.AuditPhaseNotStarted || p == auditdomain.AuditPhasePreparing || p == auditdomain.AuditPhaseInventory
}
`), 0o600); err != nil {
		t.Fatal(err)
	}
	found := phaseSetCopies(t, probe)
	if len(found) != 4 {
		t.Fatalf("the scan found %d copies in its probe, want 4: %v", len(found), found)
	}

	root, err := filepath.Abs("../..")
	if err != nil {
		t.Fatal(err)
	}
	var copies []string
	for _, directory := range []string{"internal", "cmd"} {
		err := filepath.WalkDir(filepath.Join(root, directory), func(path string, entry fs.DirEntry, walkErr error) error {
			if walkErr != nil {
				return walkErr
			}
			relative, err := filepath.Rel(root, path)
			if err != nil {
				return err
			}
			relative = filepath.ToSlash(relative)
			if entry.IsDir() || !strings.HasSuffix(relative, ".go") || strings.HasSuffix(relative, "_test.go") || relative == precedesRoundsHome {
				return nil
			}
			for _, line := range phaseSetCopies(t, path) {
				copies = append(copies, relative+":"+line)
			}
			return nil
		})
		if err != nil {
			t.Fatal(err)
		}
	}
	if len(copies) != 0 {
		t.Errorf("the phases before Rounds are written out outside %s; use AuditPhase.PrecedesRounds:\n%s",
			precedesRoundsHome, strings.Join(copies, "\n"))
	}
}

// phaseSetCopies returns the line of each condition, switch case and list
// in one Go file whose AuditPhase constants are exactly AuditPhasePreparing
// and AuditPhaseInventory. A condition is a whole chain of && and ||.
func phaseSetCopies(t *testing.T, path string) []string {
	t.Helper()
	set := token.NewFileSet()
	file, err := parser.ParseFile(set, path, nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	var found []string
	check := func(node ast.Node, parts []ast.Node) {
		phases := map[string]bool{}
		for _, part := range parts {
			ast.Inspect(part, func(inner ast.Node) bool {
				var name string
				switch value := inner.(type) {
				case *ast.Ident:
					name = value.Name
				case *ast.SelectorExpr:
					name = value.Sel.Name
				}
				switch name {
				case "AuditPhaseNotStarted", "AuditPhasePreparing", "AuditPhaseInventory", "AuditPhaseRounds":
					phases[name] = true
				}
				return true
			})
		}
		if len(phases) == 2 && phases["AuditPhasePreparing"] && phases["AuditPhaseInventory"] {
			found = append(found, strconv.Itoa(set.Position(node.Pos()).Line))
		}
	}
	var visit func(ast.Node) bool
	visit = func(node ast.Node) bool {
		switch value := node.(type) {
		case *ast.BinaryExpr:
			if value.Op != token.LOR && value.Op != token.LAND {
				return true
			}
			leaves := chainLeaves(value)
			check(value, leaves)
			for _, leaf := range leaves {
				ast.Inspect(leaf, visit)
			}
			return false
		case *ast.CaseClause:
			check(value, expressionNodes(value.List))
		case *ast.CompositeLit:
			check(value, expressionNodes(value.Elts))
		}
		return true
	}
	ast.Inspect(file, visit)
	slices.Sort(found)
	return slices.Compact(found)
}

// chainLeaves flattens a chain of && and || through parentheses.
func chainLeaves(expression ast.Expr) []ast.Node {
	switch value := expression.(type) {
	case *ast.ParenExpr:
		return chainLeaves(value.X)
	case *ast.BinaryExpr:
		if value.Op == token.LOR || value.Op == token.LAND {
			return append(chainLeaves(value.X), chainLeaves(value.Y)...)
		}
	}
	return []ast.Node{expression}
}

func expressionNodes(expressions []ast.Expr) []ast.Node {
	nodes := make([]ast.Node, len(expressions))
	for index, expression := range expressions {
		nodes[index] = expression
	}
	return nodes
}
