package persistence

import (
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"io/fs"
	"maps"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"testing"
)

// This file finds the tables and views that the SQL text of Go code reads and
// writes, for the table ownership check (PER-R18). It reads SQL the way the
// repository writes it: statements in string literals and in package-level
// constants and variables, joined with + to other literals, constants and
// expressions. An operand it cannot resolve to text, such as a variable or a
// call, stands for unknown SQL.
//
// A relation is written by INSERT INTO, UPDATE, DELETE FROM, MERGE INTO,
// TRUNCATE, LOCK TABLE, COPY ... FROM, CREATE, ALTER or DROP TABLE, by a FOR
// UPDATE or FOR NO KEY UPDATE row lock (on the relations its OF list names,
// or on every relation of its FROM clause) and by CopyFrom into a
// pgx.Identifier. It is read by FROM, JOIN, USING and COPY ... TO, including
// FOR SHARE and FOR KEY SHARE locks. Comments, string constants and the names
// of WITH queries are not relations.
// When a declaration builds a statement whose relation is unknown SQL, as in
// "DELETE FROM " + table, every string literal of that declaration that
// names a relation counts as an access of that kind.
//
// Limits: a relation name that reaches SQL only through a function argument
// or a variable of another declaration is not seen, and neither is a lock
// clause kept apart from its FROM clause, as in query += " FOR UPDATE".

// sourceRoots are the module directories whose non-test Go files hold the
// Server's SQL, as in internal/archtest.
var sourceRoots = []string{"api", "cmd", "internal", "tools"}

// unknownSQL marks an operand whose text is not known statically.
const unknownSQL = "\x00"

// sqlAccess is one read or write of a relation by the SQL of a package.
type sqlAccess struct {
	pkg      string // package directory relative to the module root
	relation string
	write    bool
	position string // file:line of the SQL text, relative to the module root
}

// scanModuleSQL returns the relation accesses of every non-test Go file under
// the source roots of the module at root, and how many files it parsed.
func scanModuleSQL(root string, relations map[string]bool) ([]sqlAccess, int, error) {
	packages := map[string][]string{}
	for _, sourceRoot := range sourceRoots {
		start := filepath.Join(root, sourceRoot)
		err := filepath.WalkDir(start, func(path string, entry fs.DirEntry, err error) error {
			if err != nil {
				return err
			}
			if entry.IsDir() {
				name := entry.Name()
				if path != start && (name == "testdata" || name == "vendor" || name == "node_modules" ||
					strings.HasPrefix(name, ".") || strings.HasPrefix(name, "_")) {
					return filepath.SkipDir
				}
				return nil
			}
			if strings.HasSuffix(path, ".go") && !strings.HasSuffix(path, "_test.go") {
				packages[filepath.Dir(path)] = append(packages[filepath.Dir(path)], path)
			}
			return nil
		})
		if err != nil && !os.IsNotExist(err) {
			return nil, 0, err
		}
	}
	var accesses []sqlAccess
	parsed := 0
	for _, dir := range slices.Sorted(maps.Keys(packages)) {
		relative, err := filepath.Rel(root, dir)
		if err != nil {
			return nil, 0, err
		}
		sources := map[string][]byte{}
		for _, path := range packages[dir] {
			source, err := os.ReadFile(path)
			if err != nil {
				return nil, 0, err
			}
			name, err := filepath.Rel(root, path)
			if err != nil {
				return nil, 0, err
			}
			sources[filepath.ToSlash(name)] = source
		}
		found, err := scanPackageSQL(filepath.ToSlash(relative), sources, relations)
		if err != nil {
			return nil, 0, err
		}
		accesses = append(accesses, found...)
		parsed += len(sources)
	}
	return accesses, parsed, nil
}

// scanPackageSQL returns the relation accesses of one package, given the
// source of each of its files by path.
func scanPackageSQL(pkg string, sources map[string][]byte, relations map[string]bool) ([]sqlAccess, error) {
	fset := token.NewFileSet()
	var files []*ast.File
	for _, name := range slices.Sorted(maps.Keys(sources)) {
		file, err := parser.ParseFile(fset, name, sources[name], parser.SkipObjectResolution)
		if err != nil {
			return nil, err
		}
		files = append(files, file)
	}
	scanner := &packageScanner{
		pkg: pkg, fset: fset, relations: relations, constants: map[string]ast.Expr{}, resolving: map[string]bool{},
	}
	for _, file := range files {
		for _, decl := range file.Decls {
			general, ok := decl.(*ast.GenDecl)
			if !ok || (general.Tok != token.CONST && general.Tok != token.VAR) {
				continue
			}
			for _, spec := range general.Specs {
				value := spec.(*ast.ValueSpec)
				if len(value.Names) != len(value.Values) {
					continue
				}
				for index, name := range value.Names {
					scanner.constants[name.Name] = value.Values[index]
				}
			}
		}
	}
	for _, file := range files {
		for _, decl := range file.Decls {
			scanner.scanDeclaration(decl)
		}
	}
	return scanner.accesses, nil
}

type packageScanner struct {
	pkg       string
	fset      *token.FileSet
	relations map[string]bool
	// constants are the package-level constants and variables by name, so a
	// statement built from them can be read whole.
	constants map[string]ast.Expr
	resolving map[string]bool
	accesses  []sqlAccess

	// Per declaration: statements with an unknown relation, and the string
	// literals that name a relation.
	unknownRead, unknownWrite bool
	named                     []*ast.BasicLit
}

func (s *packageScanner) scanDeclaration(decl ast.Decl) {
	s.unknownRead, s.unknownWrite, s.named = false, false, nil
	s.walk(decl)
	if !s.unknownRead && !s.unknownWrite {
		return
	}
	for _, literal := range s.named {
		value, _ := strconv.Unquote(literal.Value)
		s.record(value, s.unknownWrite, literal.Pos())
	}
}

func (s *packageScanner) walk(node ast.Node) {
	ast.Inspect(node, func(node ast.Node) bool {
		switch node := node.(type) {
		case *ast.ImportSpec:
			return false
		case *ast.Field:
			// A struct tag is not SQL.
			if node.Type != nil {
				s.walk(node.Type)
			}
			return false
		case *ast.CompositeLit:
			if relation, ok := s.copyTarget(node); ok {
				s.record(relation, true, node.Pos())
			}
		case *ast.BinaryExpr:
			if node.Op != token.ADD || !s.isStringConcatenation(node) {
				return true
			}
			var text strings.Builder
			var opaque []ast.Expr
			s.flatten(node, &text, &opaque)
			s.scanText(text.String(), node.Pos())
			for _, operand := range opaque {
				s.walk(operand)
			}
			return false
		case *ast.BasicLit:
			if node.Kind != token.STRING {
				return true
			}
			value, err := strconv.Unquote(node.Value)
			if err != nil {
				return true
			}
			if s.relations[value] {
				s.named = append(s.named, node)
			}
			s.scanText(value, node.Pos())
		}
		return true
	})
}

// copyTarget returns the relation of a pgx.Identifier composite literal,
// which names the table that CopyFrom writes.
func (s *packageScanner) copyTarget(literal *ast.CompositeLit) (string, bool) {
	selector, ok := literal.Type.(*ast.SelectorExpr)
	if !ok || selector.Sel.Name != "Identifier" || len(literal.Elts) == 0 {
		return "", false
	}
	if ident, ok := selector.X.(*ast.Ident); !ok || ident.Name != "pgx" {
		return "", false
	}
	last, ok := literal.Elts[len(literal.Elts)-1].(*ast.BasicLit)
	if !ok || last.Kind != token.STRING {
		return "", false
	}
	value, err := strconv.Unquote(last.Value)
	return value, err == nil && s.relations[value]
}

// isStringConcatenation reports whether a + expression has a string operand.
func (s *packageScanner) isStringConcatenation(expr ast.Expr) bool {
	switch expr := expr.(type) {
	case *ast.ParenExpr:
		return s.isStringConcatenation(expr.X)
	case *ast.BinaryExpr:
		return expr.Op == token.ADD && (s.isStringConcatenation(expr.X) || s.isStringConcatenation(expr.Y))
	case *ast.BasicLit:
		return expr.Kind == token.STRING
	case *ast.Ident:
		value, ok := s.constants[expr.Name]
		if !ok || s.resolving[expr.Name] {
			return false
		}
		s.resolving[expr.Name] = true
		defer delete(s.resolving, expr.Name)
		return s.isStringConcatenation(value)
	}
	return false
}

// flatten writes the text of a string expression: literals and constants
// contribute their text, any other operand unknownSQL. Those other operands
// are returned in opaque, so the SQL inside them is scanned separately.
func (s *packageScanner) flatten(expr ast.Expr, text *strings.Builder, opaque *[]ast.Expr) {
	switch expr := expr.(type) {
	case *ast.ParenExpr:
		s.flatten(expr.X, text, opaque)
		return
	case *ast.BinaryExpr:
		if expr.Op == token.ADD {
			s.flatten(expr.X, text, opaque)
			s.flatten(expr.Y, text, opaque)
			return
		}
	case *ast.BasicLit:
		if value, err := strconv.Unquote(expr.Value); err == nil && expr.Kind == token.STRING {
			text.WriteString(value)
			return
		}
	case *ast.Ident:
		if value, ok := s.constants[expr.Name]; ok && !s.resolving[expr.Name] {
			s.resolving[expr.Name] = true
			var inner strings.Builder
			var innerOpaque []ast.Expr
			s.flatten(value, &inner, &innerOpaque)
			delete(s.resolving, expr.Name)
			// The constant's own operands are scanned with its declaration.
			text.WriteString(inner.String())
			return
		}
	}
	text.WriteString(unknownSQL)
	*opaque = append(*opaque, expr)
}

func (s *packageScanner) scanText(text string, at token.Pos) {
	for _, access := range sqlRelationAccesses(text, s.relations) {
		switch {
		case access.relation != "":
			s.record(access.relation, access.write, at)
		case access.write:
			s.unknownWrite = true
		default:
			s.unknownRead = true
		}
	}
}

func (s *packageScanner) record(relation string, write bool, at token.Pos) {
	position := s.fset.Position(at)
	s.accesses = append(s.accesses, sqlAccess{
		pkg: s.pkg, relation: relation, write: write,
		position: fmt.Sprintf("%s:%d", position.Filename, position.Line),
	})
}

// relationAccess is one access found in SQL text; an empty relation is a
// statement whose relation is unknown SQL.
type relationAccess struct {
	relation string
	write    bool
}

type sqlTokenKind int

const (
	sqlWord sqlTokenKind = iota
	sqlQuoted
	sqlPunct
	sqlUnknown
)

type sqlToken struct {
	kind sqlTokenKind
	text string // a word in upper case, a quoted identifier without quotes
}

// lexSQL splits SQL text into words, quoted identifiers, punctuation and
// unknown operands. Comments, string constants, numbers and bind parameters
// are dropped, and so are fmt verbs, which stand for unknown SQL.
func lexSQL(text string) []sqlToken {
	var tokens []sqlToken
	for index := 0; index < len(text); {
		char := text[index]
		switch {
		case char == ' ' || char == '\t' || char == '\n' || char == '\r':
			index++
		case strings.HasPrefix(text[index:], "--"):
			end := strings.IndexByte(text[index:], '\n')
			if end < 0 {
				return tokens
			}
			index += end
		case strings.HasPrefix(text[index:], "/*"):
			depth := 0
			for index < len(text) {
				if strings.HasPrefix(text[index:], "/*") {
					depth, index = depth+1, index+2
				} else if strings.HasPrefix(text[index:], "*/") {
					depth, index = depth-1, index+2
					if depth == 0 {
						break
					}
				} else {
					index++
				}
			}
		case char == '\'':
			// E'...' takes backslash escapes; the E was lexed as a word.
			escapes := len(tokens) > 0 && tokens[len(tokens)-1] == sqlToken{kind: sqlWord, text: "E"} &&
				index > 0 && (text[index-1] == 'E' || text[index-1] == 'e')
			index++
			for index < len(text) {
				if escapes && text[index] == '\\' {
					index += 2
					continue
				}
				if text[index] == '\'' {
					if index+1 < len(text) && text[index+1] == '\'' {
						index += 2
						continue
					}
					break
				}
				index++
			}
			index++
		case char == '"':
			end := index + 1
			var name strings.Builder
			for end < len(text) {
				if text[end] == '"' {
					if end+1 < len(text) && text[end+1] == '"' {
						name.WriteByte('"')
						end += 2
						continue
					}
					break
				}
				name.WriteByte(text[end])
				end++
			}
			tokens = append(tokens, sqlToken{kind: sqlQuoted, text: name.String()})
			index = end + 1
		case char == '$':
			end := index + 1
			if end < len(text) && text[end] >= '0' && text[end] <= '9' {
				// A bind parameter.
				for end < len(text) && text[end] >= '0' && text[end] <= '9' {
					end++
				}
				index = end
				continue
			}
			for end < len(text) && isSQLWordByte(text[end]) {
				end++
			}
			if end < len(text) && text[end] == '$' {
				// A dollar-quoted string: skip to its closing tag.
				tag := text[index : end+1]
				closing := strings.Index(text[end+1:], tag)
				if closing < 0 {
					return tokens
				}
				index = end + 1 + closing + len(tag)
				continue
			}
			tokens = append(tokens, sqlToken{kind: sqlPunct, text: "$"})
			index++
		case char == '%':
			end := index + 1
			for end < len(text) && (text[end] == '[' || text[end] == ']' || (text[end] >= '0' && text[end] <= '9')) {
				end++
			}
			if end < len(text) && strings.IndexByte("svqd", text[end]) >= 0 {
				tokens = append(tokens, sqlToken{kind: sqlUnknown})
				index = end + 1
				continue
			}
			tokens = append(tokens, sqlToken{kind: sqlPunct, text: "%"})
			index++
		case char == unknownSQL[0]:
			tokens = append(tokens, sqlToken{kind: sqlUnknown})
			index++
		case char >= '0' && char <= '9':
			for index < len(text) && (isSQLWordByte(text[index]) || text[index] == '.') {
				index++
			}
		case isSQLWordByte(char):
			end := index
			for end < len(text) && (isSQLWordByte(text[end]) || text[end] == '$') {
				end++
			}
			tokens = append(tokens, sqlToken{kind: sqlWord, text: strings.ToUpper(text[index:end])})
			index = end
		default:
			tokens = append(tokens, sqlToken{kind: sqlPunct, text: string(char)})
			index++
		}
	}
	return tokens
}

func isSQLWordByte(char byte) bool {
	return char == '_' || (char >= 'a' && char <= 'z') || (char >= 'A' && char <= 'Z') ||
		(char >= '0' && char <= '9') || char >= 0x80
}

// fromListEnd are the words that end a FROM or USING list.
var fromListEnd = map[string]bool{
	"WHERE": true, "GROUP": true, "HAVING": true, "WINDOW": true, "ORDER": true, "LIMIT": true,
	"OFFSET": true, "FETCH": true, "FOR": true, "UNION": true, "INTERSECT": true, "EXCEPT": true,
	"RETURNING": true, "SET": true, "DO": true, "VALUES": true, "SELECT": true, "WHEN": true,
}

// aliasStop are the words that may follow a relation in a FROM item and are
// not its alias.
var aliasStop = map[string]bool{
	"WHERE": true, "GROUP": true, "HAVING": true, "WINDOW": true, "ORDER": true, "LIMIT": true,
	"OFFSET": true, "FETCH": true, "FOR": true, "UNION": true, "INTERSECT": true, "EXCEPT": true,
	"RETURNING": true, "SET": true, "DO": true, "VALUES": true, "SELECT": true, "WHEN": true,
	"JOIN": true, "INNER": true, "LEFT": true, "RIGHT": true, "FULL": true, "CROSS": true,
	"NATURAL": true, "ON": true, "USING": true, "TABLESAMPLE": true, "AND": true, "OR": true,
	"THEN": true, "ELSE": true, "END": true,
}

// sqlStatement walks the tokens of SQL text and collects relation accesses.
type sqlStatement struct {
	tokens    []sqlToken
	relations map[string]bool
	ctes      map[string]bool
	scopes    [][]relationAlias
	found     []relationAccess
}

type relationAlias struct{ relation, alias string }

// sqlRelationAccesses returns the relation accesses of SQL text: relations
// it reads or writes, and an access with an empty relation for each statement
// whose relation is unknown SQL.
func sqlRelationAccesses(text string, relations map[string]bool) []relationAccess {
	statement := &sqlStatement{tokens: lexSQL(text), relations: relations, ctes: map[string]bool{}}
	statement.collectCTEs()
	statement.scopes = [][]relationAlias{nil}
	tokens := statement.tokens
	for index := 0; index < len(tokens); index++ {
		switch {
		case statement.punct(index, "("):
			statement.scopes = append(statement.scopes, nil)
		case statement.punct(index, ")"):
			if len(statement.scopes) > 1 {
				statement.scopes = statement.scopes[:len(statement.scopes)-1]
			}
		case statement.punct(index, ";"):
			statement.scopes = [][]relationAlias{nil}
		case statement.word(index, "INSERT") && statement.word(index+1, "INTO"),
			statement.word(index, "MERGE") && statement.word(index+1, "INTO"),
			statement.word(index, "DELETE") && statement.word(index+1, "FROM"):
			statement.target(index+2, true)
		case statement.word(index, "UPDATE") && !statement.wordIn(index-1, "FOR", "KEY", "DO", "ON"):
			statement.target(index+1, true)
		case statement.word(index, "TRUNCATE"), statement.word(index, "LOCK"):
			statement.list(index+1, true)
		case statement.word(index, "COPY"):
			statement.copyStatement(index + 1)
		case statement.word(index, "TABLE") && statement.wordIn(index-1, "CREATE", "ALTER", "DROP", "TEMP",
			"TEMPORARY", "UNLOGGED"):
			next := index + 1
			for statement.wordIn(next, "IF", "NOT", "EXISTS", "ONLY") {
				next++
			}
			statement.target(next, true)
		case statement.word(index, "FROM") && !statement.wordIn(index-1, "DELETE", "DISTINCT"),
			statement.word(index, "USING") && !statement.punct(index+1, "("):
			statement.list(index+1, false)
		case statement.word(index, "JOIN"):
			statement.item(index + 1)
		case statement.word(index, "FOR") && statement.word(index+1, "UPDATE"):
			statement.lock(index + 2)
		case statement.word(index, "FOR") && statement.word(index+1, "NO") && statement.word(index+2, "KEY") &&
			statement.word(index+3, "UPDATE"):
			statement.lock(index + 4)
		}
	}
	return statement.found
}

func (s *sqlStatement) word(index int, word string) bool {
	return index >= 0 && index < len(s.tokens) && s.tokens[index].kind == sqlWord && s.tokens[index].text == word
}

func (s *sqlStatement) wordIn(index int, words ...string) bool {
	return slices.ContainsFunc(words, func(word string) bool { return s.word(index, word) })
}

func (s *sqlStatement) punct(index int, text string) bool {
	return index >= 0 && index < len(s.tokens) && s.tokens[index].kind == sqlPunct && s.tokens[index].text == text
}

// name returns the relation at index, skipping a public schema qualifier,
// and the index after it.
func (s *sqlStatement) name(index int) (string, int, bool) {
	if s.word(index, "PUBLIC") && s.punct(index+1, ".") {
		index += 2
	}
	if index >= len(s.tokens) || s.punct(index+1, ".") {
		return "", index, false
	}
	token := s.tokens[index]
	name := token.text
	switch token.kind {
	case sqlWord:
		name = strings.ToLower(name)
	case sqlQuoted:
	default:
		return "", index, false
	}
	if !s.relations[name] || s.ctes[name] {
		return "", index, false
	}
	return name, index + 1, true
}

// target records the relation that a statement writes.
func (s *sqlStatement) target(index int, write bool) {
	if s.word(index, "ONLY") {
		index++
	}
	if index < len(s.tokens) && s.tokens[index].kind == sqlUnknown {
		s.found = append(s.found, relationAccess{write: write})
		return
	}
	if name, _, ok := s.name(index); ok {
		s.found = append(s.found, relationAccess{relation: name, write: write})
	}
}

// copyStatement records the relation of a COPY statement: COPY ... FROM writes it,
// COPY ... TO reads it.
func (s *sqlStatement) copyStatement(index int) {
	name, next, ok := s.name(index)
	if !ok {
		return
	}
	if s.punct(next, "(") {
		for next < len(s.tokens) && !s.punct(next, ")") {
			next++
		}
		next++
	}
	s.found = append(s.found, relationAccess{relation: name, write: s.word(next, "FROM")})
}

// item records the relation of one FROM, JOIN or USING item, with its alias.
func (s *sqlStatement) item(index int) {
	for s.wordIn(index, "LATERAL", "ONLY") {
		index++
	}
	if index < len(s.tokens) && s.tokens[index].kind == sqlUnknown {
		s.found = append(s.found, relationAccess{})
		return
	}
	name, next, ok := s.name(index)
	if !ok || s.punct(next, "(") {
		return
	}
	alias := ""
	if s.word(next, "AS") {
		next++
	}
	if next < len(s.tokens) && (s.tokens[next].kind == sqlQuoted ||
		(s.tokens[next].kind == sqlWord && !aliasStop[s.tokens[next].text])) {
		alias = strings.ToLower(s.tokens[next].text)
	}
	s.found = append(s.found, relationAccess{relation: name})
	top := len(s.scopes) - 1
	s.scopes[top] = append(s.scopes[top], relationAlias{relation: name, alias: alias})
}

// list records the items of a FROM or USING list, or the relations of a
// TRUNCATE or LOCK statement when write is set.
func (s *sqlStatement) list(index int, write bool) {
	if write {
		for s.wordIn(index, "TABLE", "ONLY") {
			index++
		}
	}
	add := func(at int) {
		if write {
			s.target(at, true)
		} else {
			s.item(at)
		}
	}
	add(index)
	depth := 0
	for at := index; at < len(s.tokens); at++ {
		switch {
		case s.punct(at, "("):
			depth++
		case s.punct(at, ")"):
			if depth == 0 {
				return
			}
			depth--
		case depth > 0:
		case s.punct(at, ";"):
			return
		case s.punct(at, ","):
			add(at + 1)
		case s.tokens[at].kind == sqlWord && fromListEnd[s.tokens[at].text],
			s.word(at, "ON") && s.word(at+1, "CONFLICT"),
			write && s.word(at, "IN"):
			return
		}
	}
}

// lock records the relations that a FOR UPDATE or FOR NO KEY UPDATE clause
// locks: those its OF list names, or every relation of its FROM clause.
func (s *sqlStatement) lock(index int) {
	var named []string
	if s.word(index, "OF") {
		for at := index + 1; at < len(s.tokens); at += 2 {
			token := s.tokens[at]
			if token.kind != sqlWord && token.kind != sqlQuoted {
				break
			}
			named = append(named, strings.ToLower(token.text))
			if !s.punct(at+1, ",") {
				break
			}
		}
	}
	for _, from := range s.scopes[len(s.scopes)-1] {
		if len(named) == 0 || slices.Contains(named, from.alias) ||
			(from.alias == "" && slices.Contains(named, from.relation)) {
			s.found = append(s.found, relationAccess{relation: from.relation, write: true})
		}
	}
}

// collectCTEs records the names of WITH queries, which shadow relations.
func (s *sqlStatement) collectCTEs() {
	for index := range s.tokens {
		if s.tokens[index].kind != sqlWord && s.tokens[index].kind != sqlQuoted {
			continue
		}
		if !s.wordIn(index-1, "WITH", "RECURSIVE") && !s.punct(index-1, ",") {
			continue
		}
		next := index + 1
		if s.punct(next, "(") {
			for next < len(s.tokens) && !s.punct(next, ")") {
				next++
			}
			next++
		}
		if !s.word(next, "AS") {
			continue
		}
		next++
		if s.word(next, "NOT") {
			next++
		}
		if s.word(next, "MATERIALIZED") {
			next++
		}
		if s.punct(next, "(") {
			s.ctes[strings.ToLower(s.tokens[index].text)] = true
		}
	}
}

// TestSQLAccessScanReadsStatementsAsTheRepositoryWritesThem checks the scanner
// on the shapes repository SQL takes: constants joined with +, comma joins
// and aliases, lock clauses, CTEs, comments and string constants, dynamic
// relation names and CopyFrom.
func TestSQLAccessScanReadsStatementsAsTheRepositoryWritesThem(t *testing.T) {
	lines := []string{
		/* 1 */ "package sample",
		/* 2 */ "",
		/* 3 */ `import "github.com/jackc/pgx/v5"`,
		/* 4 */ "",
		/* 5 */ "type row struct {",
		/* 6 */ "\tName string `db:\"FROM projects\"`",
		/* 7 */ "}",
		/* 8 */ "",
		/* 9 */ `const columns = "run_id, state"`,
		/* 10 */ "const lockClause = ` FOR UPDATE`",
		/* 11 */ "var lockRun = `SELECT ` + columns + ` FROM runs WHERE run_id = $1` + lockClause",
		/* 12 */ "const joined = `SELECT run.state FROM runs AS run, stages stage JOIN audits ON audits.run_id = run.run_id",
		/* 13 */ " WHERE run.note IS DISTINCT FROM projects.note FOR UPDATE OF stage`",
		/* 14 */ "const upsert = `INSERT INTO items (item_id) SELECT item_id FROM events",
		/* 15 */ "ON CONFLICT (item_id) DO UPDATE SET state = EXCLUDED.state, settings = 1`",
		/* 16 */ "const ignored = `WITH projects AS (SELECT 1) SELECT * FROM projects -- FROM audits",
		/* 17 */ "/* UPDATE audits */ WHERE note = 'DELETE FROM audits' AND body = $q$UPDATE audits$q$`",
		/* 18 */ "const shared = `SELECT 1 FROM public.settings FOR SHARE`",
		/* 19 */ "const deleteUsing = `DELETE FROM items USING events WHERE items.id = events.id`",
		/* 20 */ "const nested = `SELECT 1 FROM runs WHERE EXISTS (SELECT 1 FROM stages FOR NO KEY UPDATE) AND EXTRACT(EPOCH FROM now()) > 0`",
		/* 21 */ "const truncate = `TRUNCATE TABLE pairs, charts`",
		/* 22 */ "const lockTable = `LOCK TABLE items IN SHARE MODE`",
		/* 23 */ "const updateFrom = `UPDATE \"runs\" SET note = s.note FROM stages s WHERE s.run_id = runs.run_id`",
		/* 24 */ "",
		/* 25 */ "func purge(tx pgx.Tx) {",
		/* 26 */ "\tfor _, table := range []string{\"pairs\", \"charts\"} {",
		/* 27 */ "\t\t_, _ = tx.Exec(nil, `DELETE FROM `+table+` WHERE experiment_id = $1`)",
		/* 28 */ "\t}",
		/* 29 */ "\t_, _ = tx.CopyFrom(nil, pgx.Identifier{\"events\"}, nil, nil)",
		/* 30 */ "}",
		/* 31 */ "",
		/* 32 */ "func list(table string) string {",
		/* 33 */ "\t_ = []string{\"audits\"}",
		/* 34 */ "\treturn fmt.Sprintf(\"SELECT * FROM %s\", table)",
		/* 35 */ "}",
		/* 36 */ "",
		/* 37 */ "func unrelated() string { return \"projects\" }",
		/* 38 */ "",
		/* 39 */ "func update() { _ = `UPDATE ` + \"runs\" + ` SET state = ` + state() + ` WHERE run_id = $1` }",
		/* 40 */ "const copyIn = `COPY charts (chart_id) FROM STDIN`",
		/* 41 */ "const copyOut = `COPY pairs TO STDOUT`",
		/* 42 */ "const escaped = `SELECT E'it\\'s FROM audits' AS note FROM settings`",
	}
	relations := map[string]bool{}
	for _, name := range []string{"runs", "stages", "audits", "projects", "items", "events", "settings", "pairs", "charts"} {
		relations[name] = true
	}
	accesses, err := scanPackageSQL("internal/sample", map[string][]byte{
		"internal/sample/sample.go": []byte(strings.Join(lines, "\n") + "\n"),
	}, relations)
	if err != nil {
		t.Fatal(err)
	}
	found := map[string]bool{}
	for _, access := range accesses {
		if access.pkg != "internal/sample" {
			t.Fatalf("access attributed to %s", access.pkg)
		}
		mode := "read"
		if access.write {
			mode = "write"
		}
		found[access.position+" "+mode+" "+access.relation] = true
	}
	want := []string{
		"11 read runs", "11 write runs",
		"12 read runs", "12 read stages", "12 read audits", "12 write stages",
		"14 write items", "14 read events",
		"18 read settings",
		"19 write items", "19 read events",
		"20 read runs", "20 read stages", "20 write stages",
		"21 write pairs", "21 write charts",
		"22 write items",
		"23 write runs", "23 read stages",
		"26 write pairs", "26 write charts", "29 write events",
		"33 read audits",
		"39 write runs",
		"40 write charts", "41 read pairs",
		"42 read settings",
	}
	for index, entry := range want {
		want[index] = "internal/sample/sample.go:" + entry
	}
	got := slices.Sorted(maps.Keys(found))
	slices.Sort(want)
	if strings.Join(got, "\n") != strings.Join(want, "\n") {
		t.Fatalf("accesses =\n%s\nwant\n%s", strings.Join(got, "\n"), strings.Join(want, "\n"))
	}
}
