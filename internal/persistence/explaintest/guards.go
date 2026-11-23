package explaintest

import (
	"context"
	"fmt"
	"maps"
	"slices"
	"strings"
	"testing"

	"github.com/jackc/pgx/v5"
)

// Querier is the part of a pgx pool or transaction that
// RequireColumnScopedGuards uses.
type Querier interface {
	Query(ctx context.Context, sql string, args ...any) (pgx.Rows, error)
}

// Qualifiers names, for each column-scoped guard of a table, the columns its
// function or WHEN condition reads without listing them in UPDATE OF. Such a
// column may only qualify a change of a listed column, as the state does in
// which a Run's Skill snapshot may still change: the guard acts only when a
// listed column changes, so skipping it when none is assigned loses nothing.
// A column compared between NEW and OLD is never a qualifier.
type Qualifiers map[string][]string

// RequireColumnScopedGuards checks the BEFORE UPDATE row triggers of table,
// resolved through the connection's search_path, and returns how many of
// them are column-scoped (PER-R22). An UPDATE OF trigger runs only when an
// UPDATE lists one of its columns, so, on the path an UPDATE runs through its
// function and in its WHEN condition, each such guard
//
//   - reads every listed column through NEW,
//   - reads no other column through NEW or OLD, in any spelling, except the
//     qualifiers declared for it, which it must read and never compare
//     between NEW and OLD,
//   - and uses NEW and OLD only through their columns, apart from RETURN.
//
// No BEFORE UPDATE trigger of the table may assign a column of NEW, which
// would change a value without listing it. A trigger function that is not
// PL/pgSQL, or text the check cannot read, fails rather than being skipped.
// The caller states how many guards it expects, so a guard that loses its
// column list fails too.
func RequireColumnScopedGuards(t testing.TB, ctx context.Context, q Querier, table string, qualifiers Qualifiers) int {
	t.Helper()
	columns, err := tableColumns(ctx, q, table)
	if err != nil {
		t.Fatalf("list the columns of %s: %v", table, err)
	}
	rows, err := q.Query(ctx, beforeUpdateTriggersSQL, table)
	if err != nil {
		t.Fatalf("list the BEFORE UPDATE triggers of %s: %v", table, err)
	}
	triggers, err := pgx.CollectRows(rows, func(row pgx.CollectableRow) (guardTrigger, error) {
		var trigger guardTrigger
		err := row.Scan(&trigger.name, &trigger.language, &trigger.source, &trigger.definition, &trigger.updateOf)
		return trigger, err
	})
	if err != nil {
		t.Fatalf("read the BEFORE UPDATE triggers of %s: %v", table, err)
	}
	guards := 0
	for _, trigger := range triggers {
		if len(trigger.updateOf) > 0 {
			guards++
		}
		for _, problem := range checkBeforeUpdateTrigger(trigger, columns, qualifiers[trigger.name]) {
			t.Errorf("%s.%s %s", table, trigger.name, problem)
		}
	}
	for _, name := range slices.Sorted(maps.Keys(qualifiers)) {
		if !slices.ContainsFunc(triggers, func(trigger guardTrigger) bool { return trigger.name == name }) {
			t.Errorf("%s has no BEFORE UPDATE trigger %s to declare qualifiers for", table, name)
		}
	}
	return guards
}

func tableColumns(ctx context.Context, q Querier, table string) (map[string]bool, error) {
	rows, err := q.Query(ctx, tableColumnsSQL, table)
	if err != nil {
		return nil, err
	}
	names, err := pgx.CollectRows(rows, pgx.RowTo[string])
	if err != nil {
		return nil, err
	}
	return set(names), nil
}

// guardTrigger is one BEFORE UPDATE row trigger as the catalog holds it.
type guardTrigger struct {
	name, language string
	// source is the body of the trigger function.
	source string
	// definition is the trigger as pg_get_triggerdef prints it.
	definition string
	// updateOf lists its UPDATE OF columns, if any.
	updateOf []string
}

// checkBeforeUpdateTrigger returns what is wrong with trigger, whose table
// has columns, given the qualifiers declared for it.
func checkBeforeUpdateTrigger(trigger guardTrigger, columns map[string]bool, qualifiers []string) []string {
	if trigger.language != "plpgsql" {
		return []string{fmt.Sprintf("runs a %s function, which this check cannot read", trigger.language)}
	}
	body, err := tokenize(trigger.source)
	if err != nil {
		return []string{"has a function this check cannot read: " + err.Error()}
	}
	path := updatePath(body)
	var problems []string
	for _, assigned := range assignments(path) {
		problems = append(problems, fmt.Sprintf("assigns %s, which no UPDATE OF guard sees", assigned))
	}
	if len(trigger.updateOf) == 0 {
		if len(qualifiers) > 0 {
			problems = append(problems, "declares qualifiers but fires on every UPDATE")
		}
		return problems
	}
	definition, err := tokenize(trigger.definition)
	if err != nil {
		return append(problems, "has a definition this check cannot read: "+err.Error())
	}
	when, err := whenCondition(definition)
	if err != nil {
		return append(problems, "has a definition this check cannot read: "+err.Error())
	}
	var references []columnReference
	for _, part := range [][]token{path, when} {
		found, misuse := rowReferences(part)
		references, problems = append(references, found...), append(problems, misuse...)
	}
	listed, qualifying := set(trigger.updateOf), set(qualifiers)
	readThroughNew, read := map[string]bool{}, map[string]bool{}
	var unknown, uncovered []string
	for _, reference := range references {
		read[reference.column] = true
		readThroughNew[reference.column] = readThroughNew[reference.column] || reference.row == "NEW"
		switch {
		case !columns[reference.column]:
			unknown = append(unknown, reference.String())
		case !listed[reference.column] && !qualifying[reference.column]:
			uncovered = append(uncovered, reference.String())
		}
	}
	if len(unknown) > 0 {
		problems = append(problems, fmt.Sprintf("reads %v, which are no columns of its table", sortedSet(unknown)))
	}
	if len(uncovered) > 0 {
		problems = append(problems, fmt.Sprintf("fires on UPDATE OF %v but reads %v, which it neither lists nor declares as qualifiers",
			trigger.updateOf, sortedSet(uncovered)))
	}
	var unread []string
	for _, column := range trigger.updateOf {
		if !readThroughNew[column] {
			unread = append(unread, column)
		}
	}
	if len(unread) > 0 {
		problems = append(problems, fmt.Sprintf("fires on UPDATE OF %v but never reads NEW of %v", trigger.updateOf, unread))
	}
	for _, qualifier := range slices.Sorted(maps.Keys(qualifying)) {
		switch {
		case listed[qualifier]:
			problems = append(problems, fmt.Sprintf("declares its listed column %s as a qualifier", qualifier))
		case !read[qualifier]:
			problems = append(problems, fmt.Sprintf("declares the qualifier %s, which it never reads", qualifier))
		}
	}
	for _, part := range [][]token{path, when} {
		for _, compared := range comparisons(part) {
			switch {
			case compared[0] != compared[1]:
				problems = append(problems, fmt.Sprintf("compares NEW.%s with OLD.%s", compared[0], compared[1]))
			case qualifying[compared[0]]:
				problems = append(problems, fmt.Sprintf("compares its qualifier %s between NEW and OLD; list it in UPDATE OF", compared[0]))
			}
		}
	}
	return problems
}

// columnReference is NEW.column or OLD.column.
type columnReference struct {
	row, column string
}

func (r columnReference) String() string {
	return r.row + "." + r.column
}

// rowReferences returns the column references of NEW and OLD in tokens and
// a problem for each other use of the two rows; RETURN NEW and RETURN OLD
// are the only uses of a whole row it accepts.
func rowReferences(tokens []token) ([]columnReference, []string) {
	var references []columnReference
	var problems []string
	for i, current := range tokens {
		if !rowVariable(tokens, i) {
			continue
		}
		row := strings.ToUpper(current.text)
		switch {
		case i+1 < len(tokens) && tokens[i+1].is(symbolToken, "."):
			if i+2 >= len(tokens) || tokens[i+2].kind != wordToken && tokens[i+2].kind != quotedToken {
				problems = append(problems, fmt.Sprintf("uses %s. without a column name", row))
				continue
			}
			references = append(references, columnReference{row: row, column: tokens[i+2].text})
		case i > 0 && tokens[i-1].isWord("return") && i+1 < len(tokens) && tokens[i+1].is(symbolToken, ";"):
		default:
			problems = append(problems, fmt.Sprintf("uses the whole %s row, so this check cannot tell which columns it reads", row))
		}
	}
	return references, problems
}

// rowVariable reports whether tokens[i] is the NEW or OLD row rather than a
// field of that name.
func rowVariable(tokens []token, i int) bool {
	return tokens[i].isWord("new", "old") && (i == 0 || !tokens[i-1].is(symbolToken, "."))
}

// assignments returns each column of NEW, or the whole NEW row, that tokens
// assign: as the target of := or of = at the start of a statement, or as an
// INTO target.
func assignments(tokens []token) []string {
	var assigned []string
	into := false
	for i := range tokens {
		current := tokens[i]
		switch {
		case current.isWord("into"):
			into = i == 0 || !tokens[i-1].isWord("insert", "merge")
			continue
		case into && !intoTarget(current):
			into = false
		}
		if !current.isWord("new") || !rowVariable(tokens, i) {
			continue
		}
		target, after := "NEW", i+1
		if after+1 < len(tokens) && tokens[after].is(symbolToken, ".") {
			target, after = "NEW."+tokens[after+1].text, after+2
			after = skipFieldsAndSubscripts(tokens, after)
		}
		if into || after < len(tokens) && (tokens[after].is(symbolToken, ":=") ||
			tokens[after].is(symbolToken, "=") && statementStart(tokens, i)) {
			assigned = append(assigned, target)
		}
	}
	return assigned
}

// skipFieldsAndSubscripts returns the index after any .field and [subscript]
// parts at i of an assignment target.
func skipFieldsAndSubscripts(tokens []token, i int) int {
	for i < len(tokens) {
		switch {
		case tokens[i].is(symbolToken, "["):
			end, err := closing(tokens, i, "[", "]")
			if err != nil {
				return len(tokens)
			}
			i = end + 1
		case i+1 < len(tokens) && tokens[i].is(symbolToken, "."):
			i += 2
		default:
			return i
		}
	}
	return i
}

// intoTarget reports whether current can belong to the target list of INTO.
func intoTarget(current token) bool {
	switch current.kind {
	case quotedToken:
		return true
	case symbolToken:
		return current.text == "." || current.text == "," || current.text == "[" || current.text == "]"
	case wordToken:
		return !current.isWord("from", "where", "using", "group", "order", "limit", "offset", "having",
			"union", "intersect", "except", "window", "for", "fetch", "returning", "values", "select")
	}
	return false
}

// statementStart reports whether tokens[i] starts a PL/pgSQL statement, where
// = assigns rather than compares.
func statementStart(tokens []token, i int) bool {
	if i == 0 {
		return true
	}
	previous := tokens[i-1]
	return previous.is(symbolToken, ";") || previous.is(symbolToken, ">>") ||
		previous.isWord("begin", "then", "else", "loop", "declare", "diagnostics")
}

// comparisons returns the column pairs that tokens compare between NEW and
// OLD, in either order, with =, <>, != or IS [NOT] DISTINCT FROM and with
// optional casts, as [column of NEW, column of OLD]. A comparison wrapped in
// a function call is not found here, but its columns are still references.
func comparisons(tokens []token) [][2]string {
	var pairs [][2]string
	for i := range tokens {
		if !rowVariable(tokens, i) || i+2 >= len(tokens) || !tokens[i+1].is(symbolToken, ".") {
			continue
		}
		next := skipCast(tokens, i+3)
		switch {
		case next < len(tokens) && (tokens[next].is(symbolToken, "=") || tokens[next].is(symbolToken, "<>") ||
			tokens[next].is(symbolToken, "!=")):
			next++
		case next+2 < len(tokens) && tokens[next].isWord("is") && tokens[next+1].isWord("distinct") &&
			tokens[next+2].isWord("from"):
			next += 3
		case next+3 < len(tokens) && tokens[next].isWord("is") && tokens[next+1].isWord("not") &&
			tokens[next+2].isWord("distinct") && tokens[next+3].isWord("from"):
			next += 4
		default:
			continue
		}
		if next+2 >= len(tokens) || !rowVariable(tokens, next) || !tokens[next+1].is(symbolToken, ".") ||
			tokens[next].text == tokens[i].text {
			continue
		}
		pair := [2]string{tokens[i+2].text, tokens[next+2].text}
		if tokens[i].isWord("old") {
			pair[0], pair[1] = pair[1], pair[0]
		}
		pairs = append(pairs, pair)
	}
	return pairs
}

// skipCast returns the index after any ::type casts at i.
func skipCast(tokens []token, i int) int {
	for i < len(tokens) && tokens[i].is(symbolToken, "::") {
		i++
		for i < len(tokens) && (tokens[i].kind == wordToken || tokens[i].kind == quotedToken || tokens[i].is(symbolToken, ".")) {
			i++
		}
		for i < len(tokens) && (tokens[i].is(symbolToken, "(") || tokens[i].is(symbolToken, "[")) {
			closingText := ")"
			if tokens[i].text == "[" {
				closingText = "]"
			}
			end, err := closing(tokens, i, tokens[i].text, closingText)
			if err != nil {
				return len(tokens)
			}
			i = end + 1
		}
	}
	return i
}

func set(values []string) map[string]bool {
	result := map[string]bool{}
	for _, value := range values {
		result[value] = true
	}
	return result
}

func sortedSet(values []string) []string {
	return slices.Compact(slices.Sorted(slices.Values(values)))
}

// changedConstants holds, for each column type that Changed supports, a
// constant of that type.
var changedConstants = map[string]string{
	"text": `'changed'`, "bigint": `'1'`, "integer": `'1'`, "smallint": `'1'`, "boolean": `'true'`,
	"jsonb": `'["changed"]'`, "text[]": `'{changed}'`, "timestamp with time zone": `'2000-01-01 00:00:00+00'`,
}

// Changed returns an expression that, assigned to column in an UPDATE of
// table, changes the column whatever value it holds: NULL for a value and a
// constant of the column's type for NULL. A guard's tests assign it to each
// guarded column in turn to show that the guard sees each one change alone;
// a check constraint or NOT NULL that would reject the value runs only after
// the BEFORE triggers.
func Changed(ctx context.Context, q Querier, table, column string) (string, error) {
	rows, err := q.Query(ctx, columnTypeSQL, table, column)
	if err != nil {
		return "", err
	}
	columnType, err := pgx.CollectExactlyOneRow(rows, pgx.RowTo[string])
	if err != nil {
		return "", fmt.Errorf("find the column %s of %s: %w", column, table, err)
	}
	constant, ok := changedConstants[columnType]
	if !ok {
		return "", fmt.Errorf("the column %s of %s has the type %s, which Changed has no constant for", column, table, columnType)
	}
	return fmt.Sprintf("CASE WHEN %s IS NULL THEN %s::%s END", column, constant, columnType), nil
}
