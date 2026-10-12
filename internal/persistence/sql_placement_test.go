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

// This file holds the SQL placement check: production Go code keeps an SQL
// statement inline only while it has at most maxInlineSQLLines lines. A
// longer statement belongs in a *_sql.go file next to its caller, where it is
// named and commented.
//
// Like the table ownership scanner in sql_access_test.go, the check folds a
// string literal, or a chain of operands joined with + (through
// parentheses), into one text before it measures it:
//
//   - a string literal contributes its value;
//   - an identifier that names a constant or variable declared once at
//     package level in the same package, with a value, contributes the fold
//     of that value, so `SELECT ` + columns + ` FROM t` reads whole;
//   - any other operand, such as a call (profileRoleSQL(...)), a method call
//     (executionIntentEvents.advance()), a local variable, a parameter or a
//     constant of another package, is unknown text (unknownSQL): it stands
//     for part of one line and does not break the chain;
//   - where two adjacent operands sit on different source lines, the fold
//     breaks the line there unless the text already breaks, so a statement
//     written as one quoted string per line counts the lines a reader sees.
//
// A chain is a statement when its fold, after any leading unknown text,
// starts with an SQL keyword; comment lines may come first. Its length is the
// number of lines of the fold, ignoring line breaks at its ends, in which
// every constant or variable declared in a *_sql.go file is unknown text too:
// that text is already named and commented. A named statement completed at
// its call site, as in
//
//	transitionSQL + prefixedAuditColumns("changed") + ` FROM changed`
//
// is therefore one line long, while a statement whose keyword comes from a
// *_sql.go constant and whose long body is spelled out inline still fails.
//
// Limits: identifiers resolve by name, so a local name that shadows a
// package-level one reads as the package-level one. A statement assembled
// with +=, fmt.Sprintf, strings.Join or a strings.Builder is measured one
// literal at a time. Production code uses those only to add a short clause or
// to fill in a statement named in a *_sql.go file; a change that builds a
// longer statement that way must extend this fold first.

// maxInlineSQLLines is the longest SQL statement that production code may
// keep inline.
const maxInlineSQLLines = 9

var sqlStatementStart = regexp.MustCompile(`(?is)^\s*(--[^\n]*\n\s*)*(WITH|SELECT|INSERT|UPDATE|DELETE|CREATE|ALTER|DROP|LOCK|TRUNCATE|VALUES)\b`)

func TestSQLPlacementKeepsLongStatementsInSQLFiles(t *testing.T) {
	var placement sqlPlacement
	files, err := eachSourcePackage(moduleRootDir(t), func(_ string, sources map[string][]byte) error {
		return placement.scanPackage(sources)
	})
	if err != nil {
		t.Fatal(err)
	}
	if files < 100 || placement.statements == 0 {
		t.Fatalf("scanned %d Go files and %d SQL statements; the repository walk is broken", files, placement.statements)
	}
	for _, violation := range placement.violations {
		t.Errorf("%s: move this SQL statement into a commented *_sql.go file", violation)
	}
}

func TestSQLPlacementReportsLongInlineStatements(t *testing.T) {
	long := "SELECT a\n" + strings.Repeat("     , b\n", maxInlineSQLLines) + "  FROM t"
	short := "SELECT a\n  FROM t"
	source := "package sample\n\nfunc f() {\n\t_ = `" + long + "`\n\t_ = `" + short + "`\n\t_ = `" + strings.Repeat("text\n", 20) + "`\n}\n"
	var placement sqlPlacement
	if err := placement.scanPackage(map[string][]byte{"sample.go": []byte(source)}); err != nil {
		t.Fatal(err)
	}
	if got := placement.violations; len(got) != 1 || !strings.HasPrefix(got[0], "sample.go:4:") {
		t.Fatalf("violations = %q, want one at sample.go:4", got)
	}
}

// TestSQLPlacementFoldsConcatenatedStatements checks the fold on the ways
// repository code joins a statement: around calls, from package constants,
// after an unknown first operand, one quoted string per line, and from
// statements already named in a *_sql.go file.
func TestSQLPlacementFoldsConcatenatedStatements(t *testing.T) {
	header := []string{
		"package sample",
		"",
		`func project(alias string) string { return alias + ".item_id" }`,
		"",
		`func cte() string { return "WITH ranked AS (SELECT 1)" }`,
		"",
		"func wrap(statement string) string { return statement }",
		"",
	}
	columns := strings.Join([]string{
		"package sample",
		"",
		"const itemColumns = `",
		"item.item_id, item.round_id,",
		"item.state, item.revision,",
		"item.created_at, item.updated_at`",
	}, "\n") + "\n"
	named := strings.Join([]string{
		"package sample",
		"",
		"// changedSQL moves item $1 from state $3 at revision $4 to state $2 and",
		"// selects from the changed row; the caller appends the columns.",
		"var changedSQL = `",
		"WITH changed AS (",
		"    UPDATE items AS item",
		"       SET state = $2, revision = item.revision + 1",
		"     WHERE item.item_id = $1",
		"       AND item.state = $3",
		"       AND item.revision = $4",
		"       AND item.deleted_at IS NULL",
		"    RETURNING item.*",
		")",
		"SELECT `",
		"",
		"// selectItemsSQL is the head of the item reads.",
		"const selectItemsSQL = `SELECT ` + itemColumns",
	}, "\n") + "\n"
	cases := []struct {
		name  string
		chain []string // source lines of the expression assigned to probe
		lines int      // the length reported, or 0 when the chain passes
	}{{
		name: "a 12-line statement concatenated around a call",
		chain: []string{
			"`",
			"SELECT item.item_id,",
			"       item.state, ` + project(\"item\") + `",
			"  FROM items AS item",
			"  JOIN rounds AS round USING (round_id)",
			" WHERE item.audit_id = $1",
			"   AND item.round_id = $2",
			"   AND item.state = $3",
			"   AND round.state = $4",
			"   AND item.revision > $5",
			" ORDER BY item.ordinal",
			" LIMIT $6",
			"   FOR UPDATE OF item`",
		},
		lines: 12,
	}, {
		name: "a statement after an unknown first operand",
		chain: []string{
			"cte() + `",
			"SELECT item_id",
			"  FROM ranked",
			"  JOIN items USING (item_id)",
			" WHERE audit_id = $1",
			"   AND round_id = $2",
			"   AND state = $3",
			"   AND revision > $4",
			" ORDER BY ordinal",
			" LIMIT $5`",
		},
		lines: 10,
	}, {
		name: "a statement with a constant of another file in parentheses",
		chain: []string{
			"`SELECT ` + (itemColumns + `",
			"  FROM items AS item",
			" WHERE item.audit_id = $1",
			"   AND item.round_id = $2",
			"   AND item.state = $3",
			" ORDER BY item.ordinal",
			" LIMIT $4`)",
		},
		lines: 10,
	}, {
		name: "a statement written one quoted string per line",
		chain: []string{
			`"SELECT item_id, state " +`,
			`	"FROM items " +`,
			`	"WHERE audit_id = $1 " +`,
			`	"AND round_id = $2 " +`,
			`	"AND state = $3 " +`,
			`	"AND revision > $4 " +`,
			`	"AND subject_key = $5 " +`,
			`	"AND kind = $6 " +`,
			`	"ORDER BY ordinal " +`,
			`	"LIMIT $7"`,
		},
		lines: 10,
	}, {
		name: "a statement whose keyword comes from a named head",
		chain: []string{
			"selectItemsSQL + `",
			"  FROM items AS item",
			"  JOIN rounds AS round USING (round_id)",
			" WHERE item.audit_id = $1",
			"   AND item.round_id = $2",
			"   AND item.state = $3",
			"   AND item.revision > $4",
			"   AND round.state = $5",
			" ORDER BY item.ordinal",
			" LIMIT $6`",
		},
		lines: 10,
	}, {
		name: "a statement inside an unknown operand",
		chain: []string{
			"wrap(`",
			"SELECT item_id",
			"  FROM items",
			" WHERE audit_id = $1",
			"   AND round_id = $2",
			"   AND state = $3",
			"   AND revision > $4",
			"   AND subject_key = $5",
			"   AND kind = $6",
			" ORDER BY ordinal",
			" LIMIT $7`) + ` FOR UPDATE`",
		},
		lines: 10,
	}, {
		name:  "a named statement with a short projection",
		chain: []string{"changedSQL + project(\"changed\") + ` FROM changed`"},
	}, {
		name:  "a short statement",
		chain: []string{"`SELECT item_id FROM items ` + `WHERE item_id = $1`"},
	}, {
		name: "long text that is not SQL",
		chain: []string{
			"`Usage: contractor items [flags]",
			"",
			"Select the items of an Audit.",
			"` + project(\"flags\") + `",
			"Flags:",
			"  --audit   the Audit",
			"  --round   the round",
			"  --state   the item state",
			"  --limit   the page size",
			"  --after   the cursor",
			"  --format  the output format`",
		},
	}, {
		name:  "a sum of numbers",
		chain: []string{"1 + 2 + len(project(`SELECT 1`))"},
	}}
	for _, test := range cases {
		t.Run(test.name, func(t *testing.T) {
			lines := append(slices.Clone(header), "var probe = "+strings.Join(test.chain, "\n"))
			var placement sqlPlacement
			err := placement.scanPackage(map[string][]byte{
				"sample.go":     []byte(strings.Join(lines, "\n") + "\n"),
				"columns.go":    []byte(columns),
				"sample_sql.go": []byte(named),
			})
			if err != nil {
				t.Fatal(err)
			}
			var want []string
			if test.lines > 0 {
				want = []string{fmt.Sprintf("sample.go:%d: %d lines", len(header)+1, test.lines)}
			}
			if !slices.Equal(placement.violations, want) {
				t.Fatalf("violations = %q, want %q", placement.violations, want)
			}
		})
	}
}

// sqlPlacement collects the SQL statements of the packages it scans and
// reports those too long to keep inline.
type sqlPlacement struct {
	statements int      // statements measured
	violations []string // "file:line: N lines" of each statement too long
}

// scanPackage measures the statements of one package, given the source of
// each of its non-test Go files by path. Its *_sql.go files only lend their
// declarations.
func (p *sqlPlacement) scanPackage(sources map[string][]byte) error {
	fset := token.NewFileSet()
	files := map[string]*ast.File{}
	values := map[string]placementValue{}
	for _, path := range slices.Sorted(maps.Keys(sources)) {
		file, err := parser.ParseFile(fset, path, sources[path], parser.SkipObjectResolution)
		if err != nil {
			return err
		}
		files[path] = file
		inSQLFile := strings.HasSuffix(path, "_sql.go")
		for _, decl := range file.Decls {
			general, ok := decl.(*ast.GenDecl)
			if !ok || (general.Tok != token.CONST && general.Tok != token.VAR) {
				continue
			}
			for _, spec := range general.Specs {
				spec := spec.(*ast.ValueSpec)
				for index, name := range spec.Names {
					value := placementValue{inSQLFile: inSQLFile}
					if _, declared := values[name.Name]; !declared && len(spec.Values) == len(spec.Names) {
						value.expr = spec.Values[index]
					}
					values[name.Name] = value
				}
			}
		}
	}
	for _, path := range slices.Sorted(maps.Keys(files)) {
		if !strings.HasSuffix(path, "_sql.go") {
			p.scanNode(fset, path, files[path], values)
		}
	}
	return nil
}

// placementValue is the value of a package-level constant or variable; expr
// is nil when the name has no single value, as when build constraints
// declare it twice.
type placementValue struct {
	expr      ast.Expr
	inSQLFile bool
}

// scanNode measures every string literal and + chain under root that no
// longer chain includes, and the chains inside their unknown operands.
func (p *sqlPlacement) scanNode(fset *token.FileSet, path string, root ast.Node, values map[string]placementValue) {
	ast.Inspect(root, func(node ast.Node) bool {
		var operands []ast.Expr
		switch node := node.(type) {
		case *ast.BinaryExpr:
			if node.Op != token.ADD {
				return true
			}
			operands = sumOperands(node)
		case *ast.BasicLit:
			if node.Kind != token.STRING {
				return true
			}
			operands = []ast.Expr{node}
		default:
			return true
		}
		statement, known := newPlacementFold(fset, values, false).fold(operands)
		if !known {
			// No operand has a known text, as in a sum of numbers.
			return true
		}
		if sqlStatementStart.MatchString(strings.TrimLeft(statement, unknownSQL+" \t\r\n")) {
			p.statements++
			inline, _ := newPlacementFold(fset, values, true).fold(operands)
			if lines := strings.Count(strings.Trim(inline, "\n"), "\n") + 1; lines > maxInlineSQLLines {
				p.violations = append(p.violations,
					fmt.Sprintf("%s:%d: %d lines", path, fset.Position(node.Pos()).Line, lines))
			}
		}
		for _, operand := range operands {
			if _, literal := operand.(*ast.BasicLit); !literal {
				p.scanNode(fset, path, operand, values)
			}
		}
		return false
	})
}

// sumOperands returns the operands of a + chain in order, through
// parentheses; any other expression is a chain of itself.
func sumOperands(expr ast.Expr) []ast.Expr {
	for {
		paren, ok := expr.(*ast.ParenExpr)
		if !ok {
			break
		}
		expr = paren.X
	}
	if sum, ok := expr.(*ast.BinaryExpr); ok && sum.Op == token.ADD {
		return append(sumOperands(sum.X), sumOperands(sum.Y)...)
	}
	return []ast.Expr{expr}
}

// placementFold folds chains of one package as the comment at the top of
// this file describes.
type placementFold struct {
	fset   *token.FileSet
	values map[string]placementValue
	// inline makes a constant or variable of a *_sql.go file unknown text.
	inline    bool
	resolving map[string]bool
}

func newPlacementFold(fset *token.FileSet, values map[string]placementValue, inline bool) *placementFold {
	return &placementFold{fset: fset, values: values, inline: inline, resolving: map[string]bool{}}
}

// fold returns the text of a chain of operands and whether any of them has a
// known text.
func (f *placementFold) fold(operands []ast.Expr) (string, bool) {
	var text strings.Builder
	known := false
	previousLine := 0
	for index, operand := range operands {
		fragment, ok := f.operandText(operand)
		if ok {
			known = true
		} else {
			fragment = unknownSQL
		}
		if index > 0 && f.fset.Position(operand.Pos()).Line > previousLine && !breaksLine(text.String(), fragment) {
			text.WriteByte('\n')
		}
		text.WriteString(fragment)
		previousLine = f.fset.Position(operand.End()).Line
	}
	return text.String(), known
}

// operandText returns the text of one operand, or false when it is unknown.
func (f *placementFold) operandText(operand ast.Expr) (string, bool) {
	switch operand := operand.(type) {
	case *ast.BasicLit:
		if operand.Kind == token.STRING {
			value, err := strconv.Unquote(operand.Value)
			return value, err == nil
		}
	case *ast.Ident:
		value, ok := f.values[operand.Name]
		if !ok || value.expr == nil || (f.inline && value.inSQLFile) || f.resolving[operand.Name] {
			return "", false
		}
		f.resolving[operand.Name] = true
		defer delete(f.resolving, operand.Name)
		return f.fold(sumOperands(value.expr))
	}
	return "", false
}

// breaksLine reports whether text before an operand ends a line or the
// operand's text starts one, ignoring spaces and tabs; an empty text before
// it starts the statement.
func breaksLine(before, fragment string) bool {
	before = strings.TrimRight(before, " \t")
	return before == "" || strings.HasSuffix(before, "\n") || strings.HasPrefix(strings.TrimLeft(fragment, " \t"), "\n")
}
