package persistence

import (
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"maps"
	"regexp"
	"slices"
	"strconv"
	"strings"
	"testing"
)

// evalJSONBCast is a production declaration whose SQL reads an Eval table
// and casts a stored value to json or jsonb, with the number of casts.
type evalJSONBCast struct {
	casts  int
	reason string
}

// evalJSONBCasts lists every such declaration by file and name. PostgreSQL
// rejects the JSON escape \u0000 in a jsonb cast and in every json operator
// (SQLSTATE 22P05), so a cast of a stored Eval document is safe only because
// every Eval codec rejects that escape before a document is stored
// (EVF-R236). A new cast is a decision: list it here with what it reads, or
// read the value without a JSON cast.
var evalJSONBCasts = map[string]evalJSONBCast{
	"internal/evalstore/public_read_sql.go summaryPageSQL": {2,
		"The experiment list reads the frozen setup, or the draft before Prepare, and the published view summary of every experiment on its page."},
	"internal/evalstore/report_sources_sql.go viewSourcesSQL": {3,
		"The comparison report reads both sides of each pair document of its view generation and the source of each selected result and assessment record."},
}

// TestEvalSQLCastsStoredValuesToJSONBOnlyWhereListed names every production
// statement that casts a stored value of an Eval table to json or jsonb
// (V414-017). Eval tables are those internal/evalstore owns (PER-R23); the
// scan reads the SQL of every package, so a foreign read is seen as well.
// Migration backfills are not production reads and are not scanned: the
// two that cast stored Eval documents rewrite the escape first (PER-R32).
func TestEvalSQLCastsStoredValuesToJSONBOnlyWhereListed(t *testing.T) {
	root := moduleRootDir(t)
	registry := loadOwnershipRegistry(t, root)
	relations := map[string]bool{}
	for name, relation := range registry.relations {
		if relation.owner == "internal/evalstore" {
			relations[name] = true
		}
	}
	if !relations["eval_experiments"] || !relations["eval_records"] {
		t.Fatalf("the ownership registry assigns no Eval tables to internal/evalstore: %v", slices.Sorted(maps.Keys(relations)))
	}
	found := map[string]int{}
	files, err := eachSourcePackage(root, func(_ string, sources map[string][]byte) error {
		casts, err := scanPackageJSONBCasts(sources, relations)
		maps.Copy(found, casts)
		return err
	})
	if err != nil {
		t.Fatal(err)
	}
	if files < 100 {
		t.Fatalf("scanned only %d Go files; the repository walk is broken", files)
	}
	for _, name := range slices.Sorted(maps.Keys(found)) {
		if listed, ok := evalJSONBCasts[name]; !ok || listed.casts != found[name] {
			t.Errorf("%s casts a stored Eval value to JSON %d times; evalJSONBCasts lists %d. A stored document is "+
				"castable only because EVF-R236 keeps the NUL escape out of it: list the cast with what it reads, "+
				"or read the value without a JSON cast", name, found[name], listed.casts)
		}
	}
	for _, name := range slices.Sorted(maps.Keys(evalJSONBCasts)) {
		if evalJSONBCasts[name].reason == "" {
			t.Errorf("evalJSONBCasts entry %s has no reason", name)
		}
		if _, ok := found[name]; !ok {
			t.Errorf("evalJSONBCasts lists %s, which no longer casts a stored Eval value; remove the entry", name)
		}
	}
}

// TestEvalJSONBCastScanCountsCastsOfStoredValues pins what the scan counts:
// a cast of a column or expression, in either spelling, in a statement that
// reads an Eval table, including one assembled from constants; not a cast of
// a string constant or a bind parameter, and not a statement over other
// tables.
func TestEvalJSONBCastScanCountsCastsOfStoredValues(t *testing.T) {
	sources := map[string][]byte{"internal/evalstore/sample_sql.go": []byte("package evalstore\n\n" +
		"var castSQL = `SELECT convert_from(r.document, 'UTF8')::jsonb -> 'source', CAST(r.setup AS json)\n" +
		"FROM eval_records r`\n\n" +
		"var constantSQL = `SELECT '[]'::jsonb, $1::jsonb, $12 :: json, r.document FROM eval_records r`\n\n" +
		"var foreignSQL = `SELECT convert_from(document, 'UTF8')::jsonb FROM workflow_runs`\n\n" +
		"const pairs = `eval_view_pairs`\n\n" +
		"type Store struct{}\n\n" +
		"func (s *Store) Page() string { return `SELECT p.document::JSONB FROM ` + pairs }\n")}
	relations := map[string]bool{"eval_records": true, "eval_view_pairs": true}
	got, err := scanPackageJSONBCasts(sources, relations)
	if err != nil {
		t.Fatal(err)
	}
	want := map[string]int{
		"internal/evalstore/sample_sql.go castSQL":    2,
		"internal/evalstore/sample_sql.go Store.Page": 1,
	}
	if !maps.Equal(got, want) {
		t.Fatalf("scan found %v, want %v", got, want)
	}
}

// jsonCast matches a cast to json or jsonb: the :: operator, or the end of
// CAST(... AS json).
var jsonCast = regexp.MustCompile(`(?i)::\s*jsonb?\b|\bAS\s+jsonb?\s*\)`)

// castOfConstant matches the text before a :: cast whose operand is a string
// constant or a bind parameter.
var castOfConstant = regexp.MustCompile(`('|\$[0-9]+)\s*$`)

// scanPackageJSONBCasts returns, by "file declaration", how many casts to
// json or jsonb each declaration of one package makes in SQL that reads one
// of relations. A declaration is a function or method, or a package-level
// constant or variable; its SQL is every string literal in it and every
// concatenation of literals and constants, as the ownership scan reads it.
func scanPackageJSONBCasts(sources map[string][]byte, relations map[string]bool) (map[string]int, error) {
	fset := token.NewFileSet()
	scanner := &packageScanner{fset: fset, relations: relations, constants: map[string]ast.Expr{}, resolving: map[string]bool{}}
	files := map[string]*ast.File{}
	for _, name := range slices.Sorted(maps.Keys(sources)) {
		file, err := parser.ParseFile(fset, name, sources[name], parser.SkipObjectResolution)
		if err != nil {
			return nil, err
		}
		files[name] = file
		for _, decl := range file.Decls {
			general, ok := decl.(*ast.GenDecl)
			if !ok || (general.Tok != token.CONST && general.Tok != token.VAR) {
				continue
			}
			for _, spec := range general.Specs {
				value := spec.(*ast.ValueSpec)
				if len(value.Names) == len(value.Values) {
					for index, ident := range value.Names {
						scanner.constants[ident.Name] = value.Values[index]
					}
				}
			}
		}
	}
	found := map[string]int{}
	record := func(file, name string, node ast.Node) {
		readsEval, casts := false, 0
		for _, text := range declarationSQL(scanner, node) {
			for _, access := range sqlRelationAccesses(text, relations) {
				readsEval = readsEval || access.relation != ""
			}
			for _, match := range jsonCast.FindAllStringIndex(text, -1) {
				if !strings.HasPrefix(text[match[0]:], "::") || !castOfConstant.MatchString(text[:match[0]]) {
					casts++
				}
			}
		}
		if readsEval && casts > 0 {
			found[file+" "+name] += casts
		}
	}
	for _, name := range slices.Sorted(maps.Keys(files)) {
		for _, decl := range files[name].Decls {
			switch decl := decl.(type) {
			case *ast.FuncDecl:
				record(name, functionName(decl), decl)
			case *ast.GenDecl:
				for _, spec := range decl.Specs {
					value, ok := spec.(*ast.ValueSpec)
					if !ok {
						continue
					}
					for index, ident := range value.Names {
						if index < len(value.Values) {
							record(name, ident.Name, value.Values[index])
						}
					}
				}
			}
		}
	}
	return found, nil
}

// declarationSQL returns the SQL text of a declaration: each string literal
// that is not part of a concatenation, and the text of each concatenation
// with the constants it names resolved.
func declarationSQL(scanner *packageScanner, node ast.Node) []string {
	var texts []string
	var visit func(ast.Node)
	visit = func(node ast.Node) {
		ast.Inspect(node, func(node ast.Node) bool {
			switch node := node.(type) {
			case *ast.ImportSpec:
				return false
			case *ast.Field:
				// A struct tag is not SQL.
				if node.Type != nil {
					visit(node.Type)
				}
				return false
			case *ast.BinaryExpr:
				if node.Op != token.ADD || !scanner.isStringConcatenation(node) {
					return true
				}
				var text strings.Builder
				var opaque []ast.Expr
				scanner.flatten(node, &text, &opaque)
				texts = append(texts, text.String())
				for _, operand := range opaque {
					visit(operand)
				}
				return false
			case *ast.BasicLit:
				if value, err := strconv.Unquote(node.Value); err == nil && node.Kind == token.STRING {
					texts = append(texts, value)
				}
			}
			return true
		})
	}
	visit(node)
	return texts
}

// functionName is a function's name, or Type.Method for a method.
func functionName(decl *ast.FuncDecl) string {
	if decl.Recv == nil || len(decl.Recv.List) == 0 {
		return decl.Name.Name
	}
	receiver := decl.Recv.List[0].Type
	for {
		switch expr := receiver.(type) {
		case *ast.StarExpr:
			receiver = expr.X
			continue
		case *ast.IndexExpr:
			receiver = expr.X
			continue
		case *ast.IndexListExpr:
			receiver = expr.X
			continue
		case *ast.Ident:
			return expr.Name + "." + decl.Name.Name
		}
		return fmt.Sprintf("%T.%s", receiver, decl.Name.Name)
	}
}
