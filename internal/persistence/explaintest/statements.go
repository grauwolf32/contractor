package explaintest

// Statement is one top-level statement of SQL text, read by the same lexer
// as the guard check.
type Statement struct {
	// Words are the statement's unquoted words, folded to lower case, in
	// the order they appear.
	Words []string
	// Values are the statement's quoted identifiers and the values of its
	// string constants, dollar-quoted bodies included.
	Values []string
}

// SplitStatements splits SQL text, such as a migration file that one simple
// query sends, into its top-level statements where psql would split it. A
// semicolon ends a statement unless it is inside a comment, a string
// constant, a quoted identifier, parentheses or the BEGIN ... END body of a
// CREATE [OR REPLACE] FUNCTION or PROCEDURE (BEGIN ATOMIC), in which CASE
// also ends with END. A statement with no tokens is left out. Text the lexer
// cannot split, such as an unterminated comment or constant, is an error.
func SplitStatements(source string) ([]Statement, error) {
	tokens, err := tokenize(source)
	if err != nil {
		return nil, err
	}
	var statements []Statement
	var current Statement
	var leading []string // up to four leading words, to recognise CREATE [OR REPLACE] FUNCTION
	parentheses, body, empty := 0, 0, true
	for _, next := range tokens {
		switch next.kind {
		case wordToken:
			current.Words = append(current.Words, next.text)
			if len(leading) < 4 {
				leading = append(leading, next.text)
			}
			if parentheses == 0 && routineDefinition(leading) {
				switch next.text {
				case "begin":
					body++
				case "case":
					if body > 0 {
						body++
					}
				case "end":
					if body > 0 {
						body--
					}
				}
			}
		case quotedToken, literalToken:
			current.Values = append(current.Values, next.text)
		case symbolToken:
			switch next.text {
			case "(":
				parentheses++
			case ")":
				if parentheses > 0 {
					parentheses--
				}
			case ";":
				if parentheses == 0 && body == 0 {
					if !empty {
						statements = append(statements, current)
					}
					current, leading, empty = Statement{}, nil, true
					continue
				}
			}
		}
		empty = false
	}
	if !empty {
		statements = append(statements, current)
	}
	return statements, nil
}

// routineDefinition reports whether the leading words of a statement are
// CREATE [OR REPLACE] FUNCTION or PROCEDURE.
func routineDefinition(leading []string) bool {
	routine := func(word string) bool { return word == "function" || word == "procedure" }
	switch {
	case len(leading) < 2 || leading[0] != "create":
		return false
	case routine(leading[1]):
		return true
	default:
		return len(leading) >= 4 && leading[1] == "or" && leading[2] == "replace" && routine(leading[3])
	}
}
