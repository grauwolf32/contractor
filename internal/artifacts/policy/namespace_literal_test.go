package policy

import (
	"go/ast"
	"go/parser"
	"go/token"
	"io/fs"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"testing"
)

// reservedNamespaceOwners maps each distinctive reserved Namespace to the
// constant that names it: the purpose-reserved Run Namespaces (ART-R114) and
// the UserScope Audit standard catalog (ART-R25). Go code outside this package
// names them only through the constant, so a rename cannot miss a copy
// (AGENTS.md 3.1). inputs, outputs and skills are ordinary words of other rules
// and are not scanned. Test files are excluded: their fixtures pin the
// persisted and wire value independently of the constant. The Runtime
// (runtime/tests/test_security_findings_toolset.py) and the UI
// (ui/src/api/artifacts.test.ts) scan their sources for the same names.
var reservedNamespaceOwners = map[string]string{
	HTTPExchangeNamespace:         "artifactpolicy.HTTPExchangeNamespace",
	AuditStandardCatalogNamespace: "artifactpolicy.AuditStandardCatalogNamespace",
	FindingProposalNamespace:      "artifactpolicy.FindingProposalNamespace",
	FindingMaterialsNamespace:     "artifactpolicy.FindingMaterialsNamespace",
	HistoricalAuditsNamespace:     "artifactpolicy.HistoricalAuditsNamespace",
	FindingHistoryNamespace:       "artifactpolicy.FindingHistoryNamespace",
	AuditContextNamespace:         "artifactpolicy.AuditContextNamespace",
}

// reservedNamespaceCopy is one string literal that spells a reserved Namespace.
type reservedNamespaceCopy struct {
	position token.Position
	owner    string
}

// reservedNamespaceLiterals returns the string literals of file that spell a
// reserved Namespace: the whole literal, as a Go comparison, ref or directory
// name would, or a quoted SQL string inside it. A route such as
// .../{runId}/finding-proposals or /v1/audit-standards names the resource,
// another rule, and is not reported.
func reservedNamespaceLiterals(fset *token.FileSet, file *ast.File) []reservedNamespaceCopy {
	var copies []reservedNamespaceCopy
	flagNames := registeredFlagNames(file)
	ast.Inspect(file, func(node ast.Node) bool {
		literal, ok := node.(*ast.BasicLit)
		if !ok || literal.Kind != token.STRING || flagNames[literal] {
			return true
		}
		value, err := strconv.Unquote(literal.Value)
		if err != nil {
			return true
		}
		for namespace, owner := range reservedNamespaceOwners {
			if value == namespace || strings.Contains(value, "'"+namespace+"'") {
				copies = append(copies, reservedNamespaceCopy{fset.Position(literal.Pos()), owner})
			}
		}
		return true
	})
	return copies
}

// CLI flag names have their own owner. Only the name argument is exempt;
// a default value or another literal can still duplicate a Namespace.
func registeredFlagNames(file *ast.File) map[ast.Expr]bool {
	packages, sets := map[string]bool{}, map[*ast.Object]bool{}
	for _, imported := range file.Imports {
		path, _ := strconv.Unquote(imported.Path.Value)
		if path == "flag" {
			name := "flag"
			if imported.Name != nil {
				name = imported.Name.Name
			}
			packages[name] = true
		}
	}
	bind := func(name ast.Expr, value ast.Expr) {
		identifier, ok := name.(*ast.Ident)
		call, isCall := value.(*ast.CallExpr)
		if !ok || !isCall {
			return
		}
		selector, ok := call.Fun.(*ast.SelectorExpr)
		if !ok || selector.Sel.Name != "NewFlagSet" {
			return
		}
		if pkg, ok := selector.X.(*ast.Ident); ok && pkg.Obj == nil && packages[pkg.Name] {
			sets[identifier.Obj] = true
		}
	}
	ast.Inspect(file, func(node ast.Node) bool {
		switch node := node.(type) {
		case *ast.AssignStmt:
			for index, value := range node.Rhs {
				if index < len(node.Lhs) {
					bind(node.Lhs[index], value)
				}
			}
		case *ast.ValueSpec:
			for index, value := range node.Values {
				if index < len(node.Names) {
					bind(node.Names[index], value)
				}
			}
		}
		return true
	})
	names := map[ast.Expr]bool{}
	ast.Inspect(file, func(node ast.Node) bool {
		call, ok := node.(*ast.CallExpr)
		if !ok {
			return true
		}
		selector, ok := call.Fun.(*ast.SelectorExpr)
		if !ok {
			return true
		}
		known := false
		switch receiver := selector.X.(type) {
		case *ast.Ident:
			known = receiver.Obj == nil && packages[receiver.Name] || receiver.Obj != nil && sets[receiver.Obj]
		case *ast.SelectorExpr:
			pkg, ok := receiver.X.(*ast.Ident)
			known = ok && pkg.Obj == nil && packages[pkg.Name] && receiver.Sel.Name == "CommandLine"
		}
		if !known {
			return true
		}
		var index int
		switch selector.Sel.Name {
		case "Bool", "BoolFunc", "Duration", "Float64", "Func", "Int", "Int64", "String", "Uint", "Uint64":
		case "BoolVar", "DurationVar", "Float64Var", "IntVar", "Int64Var", "StringVar", "TextVar", "UintVar", "Uint64Var", "Var":
			index = 1
		default:
			return true
		}
		if index < len(call.Args) {
			names[call.Args[index]] = true
		}
		return true
	})
	return names
}

func TestReservedNamespacesAreNamedOnlyByArtifactPolicy(t *testing.T) {
	t.Parallel()
	root, err := filepath.Abs(filepath.Join("..", "..", ".."))
	if err != nil {
		t.Fatal(err)
	}
	owner := filepath.Join(root, "internal", "artifacts", "policy")
	fset := token.NewFileSet()
	var copies []string
	scanned := 0
	for _, top := range []string{"api", "cmd", "internal", "tests", "tools"} {
		err := filepath.WalkDir(filepath.Join(root, top), func(path string, entry fs.DirEntry, err error) error {
			if err != nil {
				return err
			}
			if entry.IsDir() {
				name := entry.Name()
				if path == owner || name == "testdata" || name == "node_modules" || strings.HasPrefix(name, ".") {
					return filepath.SkipDir
				}
				return nil
			}
			if !strings.HasSuffix(path, ".go") || strings.HasSuffix(path, "_test.go") {
				return nil
			}
			file, err := parser.ParseFile(fset, path, nil, 0)
			if err != nil {
				return err
			}
			scanned++
			for _, found := range reservedNamespaceLiterals(fset, file) {
				relative, _ := filepath.Rel(root, found.position.Filename)
				copies = append(copies, filepath.ToSlash(relative)+":"+strconv.Itoa(found.position.Line)+": use "+found.owner)
			}
			return nil
		})
		if err != nil {
			t.Fatal(err)
		}
	}
	if scanned < 500 {
		t.Fatalf("scanned only %d Go files; the repository walk is broken", scanned)
	}
	if len(copies) > 0 {
		t.Fatalf("Go copies of a reserved Namespace outside internal/artifacts/policy:\n%s", strings.Join(copies, "\n"))
	}
}

func TestReservedNamespaceScanReportsCopiesNotRoutes(t *testing.T) {
	t.Parallel()
	source := "package sample\n\n" +
		"func reserved(namespace string) bool {\n" +
		"\treturn namespace == \"finding-proposals\" || namespace == `audit-context`\n" +
		"}\n\n" +
		"const route = \"POST /private/v1/allocations/{allocationID}/finding-proposals\"\n" +
		"const query = `SELECT 1 FROM artifact_bindings WHERE namespace = 'finding-proposals'`\n" +
		"const derived = \"audit-finding-proposals\"\n" +
		"const CatalogNamespace = \"audit-standards\"\n" +
		"const catalogRoute = \"/v1/audit-standards/{scheme}/versions/{version}\"\n" +
		"const notes = \"audit-standards-notes\"\n"
	fset := token.NewFileSet()
	file, err := parser.ParseFile(fset, "sample.go", source, parser.SkipObjectResolution)
	if err != nil {
		t.Fatal(err)
	}
	var lines []int
	for _, found := range reservedNamespaceLiterals(fset, file) {
		lines = append(lines, found.position.Line)
	}
	slices.Sort(lines)
	if want := []int{4, 4, 8, 10}; !slices.Equal(lines, want) {
		t.Fatalf("reported lines %v, want %v", lines, want)
	}
}

func TestReservedNamespaceScanDistinguishesFlagNamesFromValues(t *testing.T) {
	source := `package sample
import flags "flag"
var value string
var global = flags.NewFlagSet("command", flags.ContinueOnError)
func configure() {
	local := flags.NewFlagSet("command", flags.ContinueOnError)
	local.StringVar(&value, "audit-context", "", "trusted context file")
	global.String("finding-proposals", "", "proposal file")
	flags.String("audit-context", "", "system file")
	flags.CommandLine.Bool("audit-standards", false, "catalog option")
	local.String("ordinary", "audit-context", "Namespace default")
	_ = "audit-context"
	unrelated.StringVar(&value, "audit-context", "", "not a standard flag")
}
func otherScope() {
	local := unrelated
	local.StringVar(&value, "audit-context", "", "same identifier, different binding")
}
func shadowImport() {
	flags := unrelated
	flags.String("audit-context", "", "shadowed package name")
}
`
	fset := token.NewFileSet()
	file, err := parser.ParseFile(fset, "sample.go", source, 0)
	if err != nil {
		t.Fatal(err)
	}
	var lines []int
	for _, found := range reservedNamespaceLiterals(fset, file) {
		lines = append(lines, found.position.Line)
	}
	if want := []int{11, 12, 13, 17, 21}; !slices.Equal(lines, want) {
		t.Fatalf("reported lines %v, want Namespace values %v", lines, want)
	}
}
