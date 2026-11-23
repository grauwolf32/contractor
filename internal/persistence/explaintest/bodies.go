package explaintest

import "slices"

// Name is a table or function name as SQL text spells it: each part
// unquoted and folded to lower case, or quoted and kept as written. Schema is
// empty for an unqualified name.
type Name struct {
	Schema, Object string
}

// Body is what the text of a PL/pgSQL or SQL routine body does, read by the
// same lexer as SplitStatements: comments are skipped, and the text of a
// string constant, plain, escape or dollar-quoted, is never read as SQL.
type Body struct {
	// Writes are the tables an INSERT INTO, UPDATE [ONLY], DELETE FROM
	// [ONLY], MERGE INTO, TRUNCATE or LOCK [TABLE] statement names, in the
	// order they appear. The UPDATE of ON CONFLICT DO UPDATE SET and of a
	// FOR UPDATE row lock names no table.
	Writes []Name
	// Locks are the relations an exclusive row lock, FOR UPDATE or FOR NO
	// KEY UPDATE, takes: those its OF list names, by alias or name, or else
	// every FROM and JOIN item of its query.
	Locks []Name
	// Reads are the FROM, JOIN and USING items: names that a parenthesis
	// does not follow. A word after the FROM of an expression, such as
	// EXTRACT(EPOCH FROM NEW.created_at), counts too, so a caller compares
	// Reads with the relations it knows.
	Reads []Name
	// Calls are the names followed by an opening parenthesis, in the order
	// they appear: every function the body calls, and also any table or
	// key word written that way, such as a column list or VALUES (.
	Calls []Name
	// Dynamic reports that the body runs EXECUTE, whose SQL may write
	// anything.
	Dynamic bool
}

// fromItem is one FROM or JOIN item of a query and its alias.
type fromItem struct {
	name  Name
	alias string
}

// listEnd are the words that end a FROM or USING list.
var listEnd = map[string]bool{
	"where": true, "group": true, "having": true, "window": true, "order": true, "limit": true,
	"offset": true, "fetch": true, "for": true, "union": true, "intersect": true, "except": true,
	"returning": true, "set": true, "do": true, "values": true, "select": true, "when": true,
	"on": true, "into": true, "loop": true, "then": true, "else": true, "end": true,
}

// notAlias are the words that may follow a FROM item without being its
// alias.
var notAlias = map[string]bool{
	"join": true, "inner": true, "left": true, "right": true, "full": true, "cross": true,
	"natural": true, "using": true, "tablesample": true, "and": true, "or": true,
}

// ReadBody reads routine text obtained with RoutineBodySQL. Text
// the lexer cannot split, such as an unterminated constant, is an error.
func ReadBody(source string) (Body, error) {
	tokens, err := tokenize(source)
	if err != nil {
		return Body{}, err
	}
	var body Body
	// scopes holds the FROM and JOIN items of each open parenthesis level,
	// for the row locks of the query at that level.
	scopes := [][]fromItem{nil}
	read := func(at int) {
		if item, ok := itemAt(tokens, at); ok {
			body.Reads = append(body.Reads, item.name)
			scopes[len(scopes)-1] = append(scopes[len(scopes)-1], item)
		}
	}
	for i, current := range tokens {
		previous := token{}
		if i > 0 {
			previous = tokens[i-1]
		}
		switch {
		case current.is(symbolToken, "("):
			if nameToken(previous) {
				name := Name{Object: previous.text}
				if i >= 3 && tokens[i-2].is(symbolToken, ".") && nameToken(tokens[i-3]) {
					name.Schema = tokens[i-3].text
				}
				body.Calls = append(body.Calls, name)
			}
			scopes = append(scopes, nil)
		case current.is(symbolToken, ")"):
			if len(scopes) > 1 {
				scopes = scopes[:len(scopes)-1]
			}
		case current.is(symbolToken, ";"):
			scopes = [][]fromItem{nil}
		case current.isWord("execute"):
			body.Dynamic = true
		case current.isWord("into") && previous.isWord("insert", "merge"),
			current.isWord("from") && previous.isWord("delete"):
			if name, _, ok := nameAt(tokens, i+1); ok {
				body.Writes = append(body.Writes, name)
			}
		case current.isWord("update") && !previous.isWord("for", "key", "do", "on") &&
			!(i+1 < len(tokens) && tokens[i+1].isWord("set", "of", "nowait", "skip")):
			if name, _, ok := nameAt(tokens, i+1); ok {
				body.Writes = append(body.Writes, name)
			}
		case current.isWord("truncate", "lock"):
			at := i + 1
			if at < len(tokens) && tokens[at].isWord("table") {
				at++
			}
			eachListItem(tokens, at, func(at int) {
				if name, _, ok := nameAt(tokens, at); ok {
					body.Writes = append(body.Writes, name)
				}
			})
		case current.isWord("from") && !previous.isWord("delete", "distinct"),
			current.isWord("using") && !(i+1 < len(tokens) && tokens[i+1].is(symbolToken, "(")):
			eachListItem(tokens, i+1, read)
		case current.isWord("join"):
			read(i + 1)
		case current.isWord("update") && previous.isWord("for", "key") &&
			(previous.isWord("for") || i >= 3 && tokens[i-2].isWord("no") && tokens[i-3].isWord("for")):
			body.Locks = append(body.Locks, locked(tokens, i+1, scopes[len(scopes)-1])...)
		}
	}
	return body, nil
}

// locked returns the relations of items that a row lock clause whose OF list
// would start at tokens[at] takes.
func locked(tokens []token, at int, items []fromItem) []Name {
	var named []string
	if at < len(tokens) && tokens[at].isWord("of") {
		for at++; at < len(tokens) && nameToken(tokens[at]); at += 2 {
			named = append(named, tokens[at].text)
			if at+1 >= len(tokens) || !tokens[at+1].is(symbolToken, ",") {
				break
			}
		}
	}
	var names []Name
	for _, item := range items {
		if named == nil || item.alias != "" && slices.Contains(named, item.alias) ||
			item.alias == "" && slices.Contains(named, item.name.Object) {
			names = append(names, item.name)
		}
	}
	return names
}

// eachListItem calls visit with the start of each item of the comma list
// that starts at tokens[at], up to a word that ends the list or a closing
// parenthesis or semicolon at its own level.
func eachListItem(tokens []token, at int, visit func(int)) {
	visit(at)
	depth := 0
	for ; at < len(tokens); at++ {
		switch current := tokens[at]; {
		case current.is(symbolToken, "("):
			depth++
		case current.is(symbolToken, ")"):
			if depth == 0 {
				return
			}
			depth--
		case depth > 0:
		case current.is(symbolToken, ";"), current.kind == wordToken && listEnd[current.text]:
			return
		case current.is(symbolToken, ","):
			visit(at + 1)
		}
	}
}

// itemAt reads the FROM or JOIN item that starts at tokens[at]: a relation
// name after optional LATERAL or ONLY, with its alias. A name followed by a
// parenthesis is a function, and a parenthesis a subquery; neither is an
// item.
func itemAt(tokens []token, at int) (fromItem, bool) {
	for at < len(tokens) && tokens[at].isWord("lateral") {
		at++
	}
	name, next, ok := nameAt(tokens, at)
	if !ok || next < len(tokens) && tokens[next].is(symbolToken, "(") {
		return fromItem{}, false
	}
	if next < len(tokens) && tokens[next].isWord("as") {
		next++
	}
	item := fromItem{name: name}
	if next < len(tokens) && (tokens[next].kind == quotedToken ||
		tokens[next].kind == wordToken && !notAlias[tokens[next].text] && !listEnd[tokens[next].text]) {
		item.alias = tokens[next].text
	}
	return item, true
}

// nameAt reads the possibly qualified name that starts at tokens[i], after an
// optional ONLY, and returns the index after it.
func nameAt(tokens []token, i int) (Name, int, bool) {
	if i < len(tokens) && tokens[i].isWord("only") && i+1 < len(tokens) && nameToken(tokens[i+1]) {
		i++
	}
	if i >= len(tokens) || !nameToken(tokens[i]) {
		return Name{}, i, false
	}
	if i+2 < len(tokens) && tokens[i+1].is(symbolToken, ".") && nameToken(tokens[i+2]) {
		return Name{Schema: tokens[i].text, Object: tokens[i+2].text}, i + 3, true
	}
	return Name{Object: tokens[i].text}, i + 1, true
}

func nameToken(t token) bool {
	return t.kind == wordToken || t.kind == quotedToken
}
