package archtest

import (
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

// contractorModule is the import path prefix of the packages in this
// repository.
const contractorModule = "github.com/grauwolf32/contractor"

// shadowRoots are the directories whose Go files, production and test, the
// import shadowing check reads.
var shadowRoots = []string{"api", "cmd", "internal", "tests", "tools"}

// A local named after a package its file imports hides that package for the
// rest of its scope: a later edit that needs the package there fails to
// compile or, worse, reads the local.
func TestLocalsDoNotShadowImportedPackages(t *testing.T) {
	names := packageNames{}
	files := 0
	for _, root := range shadowRoots {
		err := filepath.WalkDir(filepath.Join("..", "..", root), func(path string, entry fs.DirEntry, err error) error {
			if err != nil {
				return err
			}
			if entry.IsDir() {
				if entry.Name() == "testdata" || entry.Name() == "node_modules" {
					return filepath.SkipDir
				}
				return nil
			}
			if !strings.HasSuffix(path, ".go") {
				return nil
			}
			positions := token.NewFileSet()
			file, err := parser.ParseFile(positions, path, nil, parser.ParseComments)
			if err != nil {
				return err
			}
			if ast.IsGenerated(file) {
				return nil
			}
			files++
			for _, finding := range importShadows(positions, file, names.lookup) {
				t.Error(finding)
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

func TestImportShadowCheckReportsEachDeclarationForm(t *testing.T) {
	const header = "package probe\n\nimport (\n\t\"context\"\n\n\t\"" + contractorModule + "/internal/config/resolved\"\n\tyaml \"gopkg.in/yaml.v3\"\n)\n\n"
	for name, tc := range map[string]struct {
		source string
		want   []string
	}{
		"short variable": {
			"func f() { resolved := 1; _ = resolved }",
			[]string{"resolved"},
		},
		"var": {
			"func f() { var context int; _ = context }",
			[]string{"context"},
		},
		"range variables": {
			"func f(items map[string]int) { for resolved, yaml := range items { _, _ = resolved, yaml } }",
			[]string{"resolved", "yaml"},
		},
		"parameter and named result": {
			"func f(context int) (resolved error) { return nil }",
			[]string{"context", "resolved"},
		},
		"function literal and receiver": {
			"type t struct{}\nfunc (yaml t) f() { _ = func(resolved int) {} }",
			[]string{"yaml", "resolved"},
		},
		"type switch and select": {
			"func f(v any, ch chan int) { switch context := v.(type) { default: _ = context }; select { case yaml := <-ch: _ = yaml } }",
			[]string{"context", "yaml"},
		},
		"distinct names": {
			"func f(ctx context.Context, items map[string]int) (snapshot resolved.ResolvedWorkflow) { workflow := 1; for name := range items { _ = name }; _, _ = ctx, workflow; _ = yaml.Marshal; return }",
			nil,
		},
	} {
		t.Run(name, func(t *testing.T) {
			positions := token.NewFileSet()
			file, err := parser.ParseFile(positions, "probe.go", header+tc.source+"\n", 0)
			if err != nil {
				t.Fatal(err)
			}
			var got []string
			for _, finding := range importShadows(positions, file, packageNames{}.lookup) {
				got = append(got, finding[strings.LastIndex(finding, " ")+1:])
			}
			if strings.Join(got, ",") != strings.Join(tc.want, ",") {
				t.Fatalf("findings = %q, want %q", got, tc.want)
			}
		})
	}
}

// importShadows returns one finding for each local declared in file with the
// name of a package that file imports.
func importShadows(positions *token.FileSet, file *ast.File, packageName func(string) string) []string {
	imported := map[string]bool{}
	for _, spec := range file.Imports {
		importPath, err := strconv.Unquote(spec.Path.Value)
		if err != nil {
			continue
		}
		name := packageName(importPath)
		if spec.Name != nil {
			name = spec.Name.Name
		}
		if name != "_" && name != "." {
			imported[name] = true
		}
	}
	var findings []string
	declare := func(identifier *ast.Ident) {
		if identifier != nil && imported[identifier.Name] {
			findings = append(findings, positions.Position(identifier.Pos()).String()+": local shadows the imported package "+identifier.Name)
		}
	}
	fields := func(list *ast.FieldList) {
		if list == nil {
			return
		}
		for _, field := range list.List {
			for _, name := range field.Names {
				declare(name)
			}
		}
	}
	for _, declaration := range file.Decls {
		function, ok := declaration.(*ast.FuncDecl)
		if !ok {
			continue
		}
		fields(function.Recv)
		fields(function.Type.Params)
		fields(function.Type.Results)
		if function.Body == nil {
			continue
		}
		ast.Inspect(function.Body, func(node ast.Node) bool {
			switch node := node.(type) {
			case *ast.AssignStmt:
				if node.Tok == token.DEFINE {
					for _, target := range node.Lhs {
						identifier, _ := target.(*ast.Ident)
						declare(identifier)
					}
				}
			case *ast.ValueSpec:
				for _, name := range node.Names {
					declare(name)
				}
			case *ast.RangeStmt:
				if node.Tok == token.DEFINE {
					key, _ := node.Key.(*ast.Ident)
					value, _ := node.Value.(*ast.Ident)
					declare(key)
					declare(value)
				}
			case *ast.FuncLit:
				fields(node.Type.Params)
				fields(node.Type.Results)
			}
			return true
		})
	}
	return findings
}

// packageNames caches the declared package name of each package of this
// module; other packages are named by the last element of their import path
// without a major version suffix.
type packageNames map[string]string

var majorVersion = regexp.MustCompile(`^v[0-9]+$`)

func (names packageNames) lookup(importPath string) string {
	if name, ok := names[importPath]; ok {
		return name
	}
	name := names.declared(importPath)
	if name == "" {
		elements := strings.Split(importPath, "/")
		name = elements[len(elements)-1]
		if majorVersion.MatchString(name) && len(elements) > 1 {
			name = elements[len(elements)-2]
		}
		name, _, _ = strings.Cut(name, ".")
		name = strings.ReplaceAll(strings.TrimPrefix(name, "go-"), "-", "")
	}
	if names != nil {
		names[importPath] = name
	}
	return name
}

// declared reads the package clause of a package of this module.
func (packageNames) declared(importPath string) string {
	relative, ok := strings.CutPrefix(importPath, contractorModule+"/")
	if !ok {
		return ""
	}
	directory := filepath.Join("..", "..", filepath.FromSlash(relative))
	entries, err := os.ReadDir(directory)
	if err != nil {
		return ""
	}
	for _, entry := range entries {
		if entry.IsDir() || filepath.Ext(entry.Name()) != ".go" || strings.HasSuffix(entry.Name(), "_test.go") {
			continue
		}
		file, err := parser.ParseFile(token.NewFileSet(), filepath.Join(directory, entry.Name()), nil, parser.PackageClauseOnly)
		if err == nil {
			return file.Name.Name
		}
	}
	return ""
}
