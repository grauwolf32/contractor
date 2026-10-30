package explaintest

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
	// [ONLY] or MERGE INTO statement names, in the order they appear. The
	// UPDATE of ON CONFLICT DO UPDATE SET and of a FOR UPDATE row lock names
	// no table.
	Writes []Name
	// Calls are the names followed by an opening parenthesis, in the order
	// they appear: every function the body calls, and also any table or
	// key word written that way, such as a column list or VALUES (.
	Calls []Name
	// Dynamic reports that the body runs EXECUTE, whose SQL may write
	// anything.
	Dynamic bool
}

// ReadBody reads the text of a routine body, such as pg_proc.prosrc. Text
// the lexer cannot split, such as an unterminated constant, is an error.
func ReadBody(source string) (Body, error) {
	tokens, err := tokenize(source)
	if err != nil {
		return Body{}, err
	}
	var body Body
	for i, current := range tokens {
		switch {
		case current.isWord("execute"):
			body.Dynamic = true
		case current.isWord("into") && i > 0 && tokens[i-1].isWord("insert", "merge"),
			current.isWord("from") && i > 0 && tokens[i-1].isWord("delete"):
			if name, ok := nameAt(tokens, i+1); ok {
				body.Writes = append(body.Writes, name)
			}
		case current.isWord("update") &&
			!(i+1 < len(tokens) && tokens[i+1].isWord("set", "of", "nowait", "skip")):
			if name, ok := nameAt(tokens, i+1); ok {
				body.Writes = append(body.Writes, name)
			}
		case current.is(symbolToken, "(") && i > 0 && nameToken(tokens[i-1]):
			name := Name{Object: tokens[i-1].text}
			if i >= 3 && tokens[i-2].is(symbolToken, ".") && nameToken(tokens[i-3]) {
				name.Schema = tokens[i-3].text
			}
			body.Calls = append(body.Calls, name)
		}
	}
	return body, nil
}

// nameAt reads the possibly qualified name that starts at tokens[i], after an
// optional ONLY.
func nameAt(tokens []token, i int) (Name, bool) {
	if i < len(tokens) && tokens[i].isWord("only") && i+1 < len(tokens) && nameToken(tokens[i+1]) {
		i++
	}
	if i >= len(tokens) || !nameToken(tokens[i]) {
		return Name{}, false
	}
	if i+2 < len(tokens) && tokens[i+1].is(symbolToken, ".") && nameToken(tokens[i+2]) {
		return Name{Schema: tokens[i].text, Object: tokens[i+2].text}, true
	}
	return Name{Object: tokens[i].text}, true
}

func nameToken(t token) bool {
	return t.kind == wordToken || t.kind == quotedToken
}
