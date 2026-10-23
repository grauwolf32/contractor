// Package statesettest holds the migrated schema's state and phase sets to
// their owning Go declarations. An owner package's test declares, for one
// table it owns, each state column with its owning Go set and the named
// subsets its partial indexes admit; Require reads the table's CHECK
// constraints and partial-index predicates from a migrated schema and reports
// every difference with the table, column or index it concerns.
//
// The comparison is scoped by relation and column: only the declared columns
// of the declared table are read, so a state word that an unrelated column or
// table also uses is never compared. Every partial index of the table must be
// declared, an index that restricts no declared column with a nil entry, so
// a new state-dependent predicate cannot appear without an explicit,
// verified subset. Production code never imports this package.
package statesettest

import (
	"context"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"maps"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"strconv"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
	"github.com/jackc/pgx/v5"
)

// Table declares the state columns of one table and every partial index on
// it.
type Table struct {
	// Name is the table, resolved through the connection's search_path.
	Name string
	// Columns are the state and phase columns compared with Go.
	Columns []Column
	// Indexes maps every partial index of the table to the subset of each
	// declared column its predicate admits, by subset name. An index whose
	// predicate restricts no declared column maps to nil.
	Indexes map[string]map[string]string
}

// Column declares one state or phase column.
type Column struct {
	// Name is the column, such as state or deletion_phase.
	Name string
	// Check names the CHECK constraint that lists every value of the column.
	Check string
	// Values is the owning Go set. DeclaredValues reads it from the owning
	// type's constants, so a constant added to the type joins it.
	Values []string
	// Subsets names the intentional subsets that partial indexes admit. Each
	// member must be one of Values.
	Subsets map[string][]string
	// Partitions lists groups of subset names that split Values exactly:
	// every value lies in exactly one subset of a group. A handwritten
	// subset is held to the declaration this way, so a value added to the
	// owning type must be classified before the test passes.
	Partitions [][]string
	// Markers are nullable columns that a shape CHECK ties to one subset.
	Markers []Marker
}

// Marker is a nullable column that the shape CHECK Check holds NOT NULL in
// exactly the values of Subset and NULL in every other value, so that a
// predicate "Column IS NOT NULL" admits Subset and "Column IS NULL" the rest.
// Check must be a disjunction whose every branch restricts the state column
// and states whether the marker is null.
type Marker struct {
	Column, Check, Subset string
}

// Catalog is what a migrated schema says about one table.
type Catalog struct {
	// Checks maps each CHECK constraint to pg_get_constraintdef.
	Checks map[string]string
	// Predicates maps each partial index to pg_get_expr of its predicate.
	Predicates map[string]string
}

// Querier is the part of a pgx pool or transaction that Read uses.
type Querier interface {
	Query(ctx context.Context, sql string, args ...any) (pgx.Rows, error)
}

// Require reads table from the migrated schema q is connected to and fails t
// with every difference Compare reports.
func Require(t testing.TB, ctx context.Context, q Querier, table Table) {
	t.Helper()
	catalog, err := Read(ctx, q, table.Name)
	if err != nil {
		t.Fatalf("read the %s catalog: %v", table.Name, err)
	}
	for _, problem := range Compare(table, catalog) {
		t.Error(problem)
	}
}

// Read returns the CHECK constraints and partial-index predicates of table.
func Read(ctx context.Context, q Querier, table string) (Catalog, error) {
	catalog := Catalog{Checks: map[string]string{}, Predicates: map[string]string{}}
	for _, read := range []struct {
		sql  string
		into map[string]string
	}{
		{sql: `
SELECT conname, pg_get_constraintdef(oid)
FROM pg_constraint
WHERE conrelid = $1::text::regclass AND contype = 'c'`, into: catalog.Checks},
		{sql: `
SELECT index.relname, pg_get_expr(i.indpred, i.indrelid)
FROM pg_index AS i
JOIN pg_class AS index ON index.oid = i.indexrelid
WHERE i.indrelid = $1::text::regclass AND i.indpred IS NOT NULL`, into: catalog.Predicates},
	} {
		rows, err := q.Query(ctx, read.sql, table)
		if err != nil {
			return Catalog{}, err
		}
		type definition struct{ Name, Text string }
		definitions, err := pgx.CollectRows(rows, pgx.RowToStructByPos[definition])
		if err != nil {
			return Catalog{}, err
		}
		for _, definition := range definitions {
			read.into[definition.Name] = definition.Text
		}
	}
	if len(catalog.Checks) == 0 {
		return Catalog{}, fmt.Errorf("table %s has no CHECK constraint", table)
	}
	return catalog, nil
}

// Compare returns every difference between the declaration of table and its
// catalog, each naming the table.column, constraint or index concerned:
//
//   - a declared value that the column's CHECK does not list (Go-only) or a
//     listed value that Go does not declare (schema-only);
//   - a value that any other CHECK, or a partial-index predicate, compares
//     the column with and Go does not declare;
//   - a subset member Go does not declare, or a partition group that does
//     not split the declared values exactly;
//   - a marker whose shape CHECK does not hold it NOT NULL in exactly its
//     subset;
//   - a partial index that is not declared, is missing, admits values other
//     than its declared subset, restricts a column it declares no subset for,
//     or restricts a declared column in a form this package cannot read.
func Compare(table Table, catalog Catalog) []string {
	var problems []string
	report := func(format string, args ...any) { problems = append(problems, fmt.Sprintf(format, args...)) }
	declared := map[string]bool{}
	for _, column := range table.Columns {
		declared[column.Name] = true
		label := table.Name + "." + column.Name
		values := map[string]bool{}
		for _, value := range column.Values {
			if values[value] {
				report("%s: the owning Go set declares %q twice", label, value)
			}
			values[value] = true
		}
		if len(values) == 0 {
			report("%s: the owning Go set is empty", label)
			continue
		}
		for _, name := range slices.Sorted(maps.Keys(column.Subsets)) {
			for _, value := range column.Subsets[name] {
				if !values[value] {
					report("%s: subset %q lists %q, which the owning Go set does not declare", label, name, value)
				}
			}
		}
		for _, group := range column.Partitions {
			count := map[string]int{}
			for _, name := range group {
				subset, ok := column.Subsets[name]
				if !ok {
					report("%s: partition %v names the unknown subset %q", label, group, name)
				}
				for _, value := range subset {
					count[value]++
				}
			}
			for _, value := range sortedSet(column.Values) {
				if count[value] != 1 {
					report("%s: %q lies in %d of the subsets %v, which must split the owning Go set exactly; classify it",
						label, value, count[value], group)
				}
			}
		}

		if definition, ok := catalog.Checks[column.Check]; !ok {
			report("%s: CHECK %s is missing", label, column.Check)
		} else if listed, err := listedValues(definition, column.Name); err != nil {
			report("%s: CHECK %s: %v", label, column.Check, err)
		} else {
			if goOnly := difference(column.Values, listed); len(goOnly) > 0 {
				report("%s: the owning Go set declares %v, which CHECK %s does not list", label, goOnly, column.Check)
			}
			if schemaOnly := difference(listed, column.Values); len(schemaOnly) > 0 {
				report("%s: CHECK %s lists %v, which the owning Go set does not declare", label, column.Check, schemaOnly)
			}
		}
		for _, name := range slices.Sorted(maps.Keys(catalog.Checks)) {
			if name == column.Check {
				continue
			}
			for _, value := range comparedValues(catalog.Checks[name], column.Name) {
				if !values[value] {
					report("%s: CHECK %s compares it with %q, which the owning Go set does not declare", label, name, value)
				}
			}
		}
		for _, marker := range column.Markers {
			for _, problem := range markerProblems(column, marker, catalog) {
				report("%s: marker %s: %s", label, marker.Column, problem)
			}
		}
	}

	for _, name := range slices.Sorted(maps.Keys(catalog.Predicates)) {
		predicate := catalog.Predicates[name]
		subsets, ok := table.Indexes[name]
		if !ok {
			report("%s: partial index %s (WHERE %s) is not declared; declare the subset of each state column it admits, or nil when it restricts none",
				table.Name, name, predicate)
			continue
		}
		for _, columnName := range slices.Sorted(maps.Keys(subsets)) {
			if !declared[columnName] {
				report("%s: partial index %s declares a subset of %s, which is not a declared state column", table.Name, name, columnName)
			}
		}
		for _, column := range table.Columns {
			label := table.Name + "." + column.Name
			for _, value := range comparedValues(predicate, column.Name) {
				if !slices.Contains(column.Values, value) {
					report("%s: partial index %s compares it with %q, which the owning Go set does not declare (WHERE %s)",
						label, name, value, predicate)
				}
			}
			admitted, restricted, err := admittedValues(predicate, column)
			if err != nil {
				report("%s: partial index %s: %v (WHERE %s)", label, name, err, predicate)
				continue
			}
			subsetName, wants := subsets[column.Name]
			switch {
			case !restricted && !wants:
			case !wants:
				report("%s: partial index %s admits %v but declares no subset (WHERE %s)", label, name, admitted, predicate)
			case !restricted:
				report("%s: partial index %s declares subset %q but does not restrict the column (WHERE %s)", label, name, subsetName, predicate)
			default:
				want, known := column.Subsets[subsetName]
				if !known {
					report("%s: partial index %s declares the unknown subset %q", label, name, subsetName)
					continue
				}
				if drift := drift(want, admitted); drift != "" {
					report("%s: partial index %s admits %v, want subset %q %v: %s (WHERE %s)",
						label, name, admitted, subsetName, sortedSet(want), drift, predicate)
				}
			}
		}
	}
	for _, name := range slices.Sorted(maps.Keys(table.Indexes)) {
		if _, ok := catalog.Predicates[name]; !ok {
			report("%s: declared partial index %s is missing or has no predicate", table.Name, name)
		}
	}
	return problems
}

// literal matches one SQL string literal with an optional type cast, as
// pg_get_constraintdef and pg_get_expr print it.
const literal = `'(?:[^']|'')*'(?:::[a-z][a-z ]*[a-z])?`

var (
	literalValue = regexp.MustCompile(`'((?:[^']|'')*)'`)
	identifier   = regexp.MustCompile(`[A-Za-z_][A-Za-z0-9_]*`)
)

// restrictionForms are the column restrictions this package reads: equality
// with one value or a list, and inequality with one value or a list.
func restrictionForms(column string) (positive, negative []*regexp.Regexp) {
	name := regexp.QuoteMeta(column)
	list := `\(ARRAY\[(` + literal + `(?:, ` + literal + `)*)\]\)`
	return []*regexp.Regexp{
			regexp.MustCompile(`^` + name + ` = (` + literal + `)$`),
			regexp.MustCompile(`^` + name + ` = ANY ` + list + `$`),
		}, []*regexp.Regexp{
			regexp.MustCompile(`^` + name + ` <> (` + literal + `)$`),
			regexp.MustCompile(`^` + name + ` <> ALL ` + list + `$`),
		}
}

// restriction returns the values a whole conjunct admits when it is a
// restriction of column in a readable form.
func restriction(conjunct, column string, values []string) ([]string, bool) {
	positive, negative := restrictionForms(column)
	for _, form := range positive {
		if match := form.FindStringSubmatch(conjunct); match != nil {
			return literalValues(match[1]), true
		}
	}
	for _, form := range negative {
		if match := form.FindStringSubmatch(conjunct); match != nil {
			return difference(values, literalValues(match[1])), true
		}
	}
	return nil, false
}

// listedValues returns the values the enumerating CHECK lists for column.
func listedValues(definition, column string) ([]string, error) {
	body, ok := strings.CutPrefix(definition, "CHECK ")
	if !ok {
		return nil, fmt.Errorf("%q is not a CHECK constraint", definition)
	}
	positive, _ := restrictionForms(column)
	expression := stripParentheses(body)
	for _, form := range positive {
		if match := form.FindStringSubmatch(expression); match != nil {
			return literalValues(match[1]), nil
		}
	}
	return nil, fmt.Errorf("%s does not list the values of %s", definition, column)
}

// comparedValues returns every value that definition compares column with,
// wherever the comparison appears.
func comparedValues(definition, column string) []string {
	name := regexp.QuoteMeta(column)
	comparison := regexp.MustCompile(`(?:^|[^A-Za-z0-9_.'"])` + name +
		` (?:=|<>) (?:(?:ANY|ALL) \(ARRAY\[((?:` + literal + `)(?:, ` + literal + `)*)\]\)|(` + literal + `))`)
	var values []string
	for _, match := range comparison.FindAllStringSubmatch(definition, -1) {
		values = append(values, literalValues(match[1]+match[2])...)
	}
	return values
}

// admittedValues returns the values of column that predicate admits and
// whether it restricts the column at all. The predicate's top-level
// conjuncts that mention the column, or one of its markers, must each be a
// readable restriction; their admitted sets intersect.
func admittedValues(predicate string, column Column) ([]string, bool, error) {
	admitted := sortedSet(column.Values)
	restricted := false
	for _, conjunct := range splitTopLevel(stripParentheses(predicate), " AND ") {
		conjunct = stripParentheses(conjunct)
		names := identifiers(conjunct)
		markers := slices.DeleteFunc(slices.Clone(column.Markers), func(marker Marker) bool { return !names[marker.Column] })
		if !names[column.Name] && len(markers) == 0 {
			continue
		}
		values, ok := restriction(conjunct, column.Name, column.Values)
		if !ok && len(markers) == 1 && !names[column.Name] {
			subset := column.Subsets[markers[0].Subset]
			switch conjunct {
			case markers[0].Column + " IS NOT NULL":
				values, ok = sortedSet(subset), true
			case markers[0].Column + " IS NULL":
				values, ok = difference(column.Values, subset), true
			}
		}
		if !ok {
			return nil, false, fmt.Errorf("the conjunct %q restricts the column in a form this check cannot read; "+
				"write it as an equality or inequality with a value list, or check it separately", conjunct)
		}
		admitted = intersection(admitted, values)
		restricted = true
	}
	return admitted, restricted, nil
}

// markerProblems checks that marker.Check holds marker.Column NOT NULL in
// exactly the values of marker.Subset and NULL in the rest.
func markerProblems(column Column, marker Marker, catalog Catalog) []string {
	subset, ok := column.Subsets[marker.Subset]
	if !ok {
		return []string{fmt.Sprintf("the unknown subset %q", marker.Subset)}
	}
	definition, ok := catalog.Checks[marker.Check]
	if !ok {
		return []string{fmt.Sprintf("CHECK %s is missing", marker.Check)}
	}
	body, ok := strings.CutPrefix(definition, "CHECK ")
	if !ok {
		return []string{fmt.Sprintf("%q is not a CHECK constraint", definition)}
	}
	var problems []string
	var notNull, null []string
	for _, branch := range splitTopLevel(stripParentheses(body), " OR ") {
		var states []string
		restricted, set, isNull := false, false, false
		for _, conjunct := range splitTopLevel(stripParentheses(branch), " AND ") {
			conjunct = stripParentheses(conjunct)
			if values, ok := restriction(conjunct, column.Name, column.Values); ok {
				if restricted {
					problems = append(problems, fmt.Sprintf("CHECK %s restricts the column twice in the branch %s", marker.Check, branch))
				}
				states, restricted = intersection(sortedSet(column.Values), values), true
				continue
			}
			switch conjunct {
			case marker.Column + " IS NOT NULL":
				set, isNull = true, false
			case marker.Column + " IS NULL":
				set, isNull = true, true
			}
		}
		switch {
		case !restricted:
			problems = append(problems, fmt.Sprintf("CHECK %s has a branch that does not restrict the column: %s", marker.Check, branch))
		case !set:
			problems = append(problems, fmt.Sprintf("CHECK %s leaves %s unconstrained in %v", marker.Check, marker.Column, states))
		case isNull:
			null = append(null, states...)
		default:
			notNull = append(notNull, states...)
		}
	}
	if both := intersection(sortedSet(notNull), sortedSet(null)); len(both) > 0 {
		problems = append(problems, fmt.Sprintf("CHECK %s allows both a null and a set %s in %v", marker.Check, marker.Column, both))
	}
	if drift := drift(subset, notNull); drift != "" {
		problems = append(problems, fmt.Sprintf("CHECK %s sets %s in %v, want subset %q %v: %s",
			marker.Check, marker.Column, sortedSet(notNull), marker.Subset, sortedSet(subset), drift))
	}
	if unconstrained := difference(column.Values, append(slices.Clone(notNull), null...)); len(unconstrained) > 0 {
		problems = append(problems, fmt.Sprintf("CHECK %s admits no row in %v", marker.Check, unconstrained))
	}
	return problems
}

// identifiers returns the identifiers of text outside its string literals.
func identifiers(text string) map[string]bool {
	names := map[string]bool{}
	for _, name := range identifier.FindAllString(literalValue.ReplaceAllString(text, " "), -1) {
		names[name] = true
	}
	return names
}

func literalValues(list string) []string {
	var values []string
	for _, match := range literalValue.FindAllStringSubmatch(list, -1) {
		values = append(values, strings.ReplaceAll(match[1], "''", "'"))
	}
	return values
}

// stripParentheses removes the parentheses that enclose all of text.
func stripParentheses(text string) string {
	text = strings.TrimSpace(text)
	for strings.HasPrefix(text, "(") && closing(text, 0) == len(text)-1 {
		text = strings.TrimSpace(text[1 : len(text)-1])
	}
	return text
}

// closing returns the index of the parenthesis that closes the one at open,
// skipping string literals, or -1.
func closing(text string, open int) int {
	depth, quoted := 0, false
	for index := open; index < len(text); index++ {
		switch character := text[index]; {
		case character == '\'':
			quoted = !quoted
		case quoted:
		case character == '(':
			depth++
		case character == ')':
			depth--
			if depth == 0 {
				return index
			}
		}
	}
	return -1
}

// splitTopLevel splits text at each separator outside parentheses and string
// literals.
func splitTopLevel(text, separator string) []string {
	var parts []string
	depth, quoted, start := 0, false, 0
	for index := 0; index < len(text); index++ {
		switch character := text[index]; {
		case character == '\'':
			quoted = !quoted
		case quoted:
		case character == '(':
			depth++
		case character == ')':
			depth--
		case depth == 0 && strings.HasPrefix(text[index:], separator):
			parts = append(parts, text[start:index])
			start = index + len(separator)
			index = start - 1
		}
	}
	return append(parts, text[start:])
}

// drift names the values of want that got lacks and those it adds, or "".
func drift(want, got []string) string {
	var parts []string
	if missing := difference(want, got); len(missing) > 0 {
		parts = append(parts, fmt.Sprintf("missing %v", missing))
	}
	if extra := difference(got, want); len(extra) > 0 {
		parts = append(parts, fmt.Sprintf("extra %v", extra))
	}
	return strings.Join(parts, ", ")
}

func sortedSet(values []string) []string {
	return slices.Compact(slices.Sorted(slices.Values(values)))
}

// difference returns the sorted values of left that right lacks.
func difference(left, right []string) []string {
	return slices.DeleteFunc(sortedSet(left), func(value string) bool { return slices.Contains(right, value) })
}

func intersection(left, right []string) []string {
	return slices.DeleteFunc(sortedSet(left), func(value string) bool { return !slices.Contains(right, value) })
}

// DeclaredValues returns the values of the constants of the string type
// typeName that the package in dir declares, where dir is a
// repository-relative path such as "internal/runstore". It reads the
// non-test Go files of the package and accepts package-level constants
// written as `Name typeName = "value"` or `Name = typeName("value")`, so the
// result is the owning declaration's whole set rather than a handwritten
// copy.
//
// It fails t instead of returning part of the set when the package does not
// declare typeName as a defined string type or declares no constant of it,
// and when the package, at package level or inside a function, declares
//
//   - an alias of typeName (`type a = typeName`), through which constants of
//     the type could be declared unseen;
//   - a constant of typeName, or a type named typeName, inside a function;
//   - a constant of typeName whose value is not a string literal, an implicit
//     repetition of one, or an untyped constant whose expression mentions
//     typeName in any other form, such as typeName("a") + "b";
//   - two constants of typeName with one value.
//
// Out of scope: a constant of typeName declared in another package through a
// qualified type, such as `const x runstore.StageExecutionState = "x"` outside
// runstore, and an alias declared there; finding them would mean parsing every
// package. Constants of a distinct type defined from typeName (`type sub
// typeName`) are values of that other type and are not read.
func DeclaredValues(t testing.TB, dir, typeName string) []string {
	t.Helper()
	values, err := declaredValues(contractstest.Path(t, strings.Split(dir, "/")...), typeName)
	if err != nil {
		t.Fatalf("read the %s constants of %s: %v", typeName, dir, err)
	}
	return values
}

func declaredValues(dir, typeName string) ([]string, error) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, err
	}
	fileSet := token.NewFileSet()
	typeDeclared := false
	var values []string
	for _, entry := range entries {
		name := entry.Name()
		if entry.IsDir() || !strings.HasSuffix(name, ".go") || strings.HasSuffix(name, "_test.go") {
			continue
		}
		file, err := parser.ParseFile(fileSet, filepath.Join(dir, name), nil, parser.SkipObjectResolution)
		if err != nil {
			return nil, err
		}
		for _, declaration := range file.Decls {
			general, ok := declaration.(*ast.GenDecl)
			if !ok {
				continue
			}
			switch general.Tok {
			case token.TYPE:
				for _, spec := range general.Specs {
					spec := spec.(*ast.TypeSpec)
					switch {
					case spec.Name.Name == typeName:
						if !isIdentifier(spec.Type, "string") || spec.Assign.IsValid() || spec.TypeParams != nil {
							return nil, fmt.Errorf("%s: %s is not a defined string type", fileSet.Position(spec.Pos()), typeName)
						}
						typeDeclared = true
					case spec.Assign.IsValid() && isIdentifier(spec.Type, typeName):
						return nil, fmt.Errorf("%s: %s is an alias of %s; declare its constants with %s itself",
							fileSet.Position(spec.Pos()), spec.Name.Name, typeName, typeName)
					}
				}
			case token.CONST:
				constants, err := typedConstants(fileSet, general, typeName)
				if err != nil {
					return nil, err
				}
				values = append(values, constants...)
			}
		}
		if err := localDeclarations(fileSet, file, typeName); err != nil {
			return nil, err
		}
	}
	switch {
	case !typeDeclared:
		return nil, fmt.Errorf("%s declares no type %s", dir, typeName)
	case len(values) == 0:
		return nil, fmt.Errorf("%s declares no constant of type %s", dir, typeName)
	}
	slices.Sort(values)
	if compacted := slices.Compact(slices.Clone(values)); len(compacted) != len(values) {
		return nil, fmt.Errorf("two constants of type %s share a value: %v", typeName, values)
	}
	return values, nil
}

// localDeclarations fails on a declaration inside a function that could
// hold a value of typeName unseen: a constant of the type, an alias of it or
// a local type of the same name.
func localDeclarations(fileSet *token.FileSet, file *ast.File, typeName string) error {
	var failure error
	ast.Inspect(file, func(node ast.Node) bool {
		statement, ok := node.(*ast.DeclStmt)
		if failure != nil || !ok {
			return failure == nil
		}
		general := statement.Decl.(*ast.GenDecl)
		switch general.Tok {
		case token.TYPE:
			for _, spec := range general.Specs {
				spec := spec.(*ast.TypeSpec)
				if spec.Name.Name == typeName || spec.Assign.IsValid() && isIdentifier(spec.Type, typeName) {
					failure = fmt.Errorf("%s: function-local type %s shadows or aliases %s", fileSet.Position(spec.Pos()), spec.Name.Name, typeName)
				}
			}
		case token.CONST:
			constants, err := typedConstants(fileSet, general, typeName)
			switch {
			case err != nil:
				failure = err
			case len(constants) > 0:
				failure = fmt.Errorf("%s: function-local constant of type %s; declare it at package level",
					fileSet.Position(general.Pos()), typeName)
			}
		}
		return failure == nil
	})
	return failure
}

// typedConstants returns the values of the constants of typeName in one
// const declaration, written as `Name typeName = "value"` or
// `Name = typeName("value")`, and fails on a constant of the type, or an
// untyped one mentioning it, written in any other form.
func typedConstants(fileSet *token.FileSet, declaration *ast.GenDecl, typeName string) ([]string, error) {
	var values []string
	previousTyped := false
	for _, spec := range declaration.Specs {
		spec := spec.(*ast.ValueSpec)
		if spec.Type == nil && len(spec.Values) == 0 {
			// An implicit repetition repeats the previous type and value.
			if previousTyped {
				return nil, fmt.Errorf("%s: constant %s repeats a %s value implicitly", fileSet.Position(spec.Pos()), spec.Names[0].Name, typeName)
			}
			continue
		}
		specTyped := isIdentifier(spec.Type, typeName)
		previousTyped = specTyped
		for index, name := range spec.Names {
			if index >= len(spec.Values) {
				return nil, fmt.Errorf("%s: constant %s has no value", fileSet.Position(name.Pos()), name.Name)
			}
			value, typed := spec.Values[index], specTyped
			if call, ok := value.(*ast.CallExpr); ok && isIdentifier(call.Fun, typeName) && len(call.Args) == 1 {
				value, typed, previousTyped = call.Args[0], true, true
			}
			if !typed {
				if spec.Type == nil && mentions(value, typeName) {
					return nil, fmt.Errorf("%s: constant %s uses %s in an expression the declaration check cannot read",
						fileSet.Position(name.Pos()), name.Name, typeName)
				}
				continue
			}
			text, ok := value.(*ast.BasicLit)
			if !ok || text.Kind != token.STRING {
				return nil, fmt.Errorf("%s: constant %s of type %s is not a string literal", fileSet.Position(name.Pos()), name.Name, typeName)
			}
			unquoted, err := strconv.Unquote(text.Value)
			if err != nil {
				return nil, err
			}
			values = append(values, unquoted)
		}
	}
	return values, nil
}

// mentions reports whether expression refers to the identifier name.
func mentions(expression ast.Expr, name string) bool {
	found := false
	ast.Inspect(expression, func(node ast.Node) bool {
		if identifier, ok := node.(*ast.Ident); ok && identifier.Name == name {
			found = true
		}
		return !found
	})
	return found
}

func isIdentifier(expression ast.Expr, name string) bool {
	identifier, ok := expression.(*ast.Ident)
	return ok && identifier.Name == name
}
