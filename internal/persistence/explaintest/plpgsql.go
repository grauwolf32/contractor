package explaintest

import (
	"errors"
	"fmt"
	"slices"
	"strings"
)

// The guard check reads PL/pgSQL trigger functions and trigger definitions
// as PostgreSQL's lexer splits them: comments are skipped, string constants
// (plain, escape and dollar-quoted) and quoted identifiers are single tokens,
// and an unquoted word is folded to lower case. Text it cannot split, such as
// an unterminated constant, is an error, so the check fails instead of
// skipping what it did not read.

type tokenKind int

const (
	wordToken      tokenKind = iota // an unquoted identifier or key word, folded to lower case
	quotedToken                     // a quoted identifier, as written
	literalToken                    // the value of a string constant
	numberToken                     // a numeric constant
	parameterToken                  // a positional parameter such as $1
	symbolToken                     // an operator or punctuation
)

type token struct {
	kind tokenKind
	text string
}

func (t token) is(kind tokenKind, text string) bool {
	return t.kind == kind && t.text == text
}

func (t token) isWord(words ...string) bool {
	if t.kind != wordToken {
		return false
	}
	for _, word := range words {
		if t.text == word {
			return true
		}
	}
	return false
}

// operatorCharacters may form an operator of several characters.
const operatorCharacters = "+-*/<>=~!@#%^&|`?"

// tokenize splits PL/pgSQL or SQL text into tokens.
func tokenize(source string) ([]token, error) {
	var tokens []token
	for i := 0; i < len(source); {
		c := source[i]
		switch {
		case c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v':
			i++
		case strings.HasPrefix(source[i:], "--"):
			end := strings.IndexByte(source[i:], '\n')
			if end < 0 {
				return tokens, nil
			}
			i += end + 1
		case strings.HasPrefix(source[i:], "/*"):
			end, err := blockCommentEnd(source, i)
			if err != nil {
				return nil, err
			}
			i = end
		case c == '\'':
			value, end, err := stringConstant(source, i, false)
			if err != nil {
				return nil, err
			}
			tokens, i = append(tokens, token{literalToken, value}), end
		case c == '"':
			name, end, err := quotedIdentifier(source, i)
			if err != nil {
				return nil, err
			}
			tokens, i = append(tokens, token{quotedToken, name}), end
		case c == '$':
			next, end, err := dollarToken(source, i)
			if err != nil {
				return nil, err
			}
			tokens, i = append(tokens, next), end
		case identifierStart(c):
			end := i + 1
			for end < len(source) && identifierPart(source[end]) {
				end++
			}
			word := strings.ToLower(source[i:end])
			if end < len(source) && source[end] == '\'' && (word == "e" || word == "b" || word == "x" || word == "n") {
				value, after, err := stringConstant(source, end, word == "e")
				if err != nil {
					return nil, err
				}
				tokens, i = append(tokens, token{literalToken, value}), after
				continue
			}
			tokens, i = append(tokens, token{wordToken, word}), end
		case digit(c) || c == '.' && i+1 < len(source) && digit(source[i+1]):
			end := numberEnd(source, i)
			tokens, i = append(tokens, token{numberToken, source[i:end]}), end
		case c == ':':
			text := ":"
			if i+1 < len(source) && (source[i+1] == '=' || source[i+1] == ':') {
				text = source[i : i+2]
			}
			tokens, i = append(tokens, token{symbolToken, text}), i+len(text)
		case c == '.' && i+1 < len(source) && source[i+1] == '.':
			tokens, i = append(tokens, token{symbolToken, ".."}), i+2
		case strings.IndexByte(operatorCharacters, c) >= 0:
			operator := operatorAt(source, i)
			tokens, i = append(tokens, token{symbolToken, operator}), i+len(operator)
		case strings.IndexByte("()[],;.", c) >= 0:
			tokens, i = append(tokens, token{symbolToken, string(c)}), i+1
		default:
			return nil, fmt.Errorf("unexpected character %q at offset %d", c, i)
		}
	}
	return tokens, nil
}

func identifierStart(c byte) bool {
	return c == '_' || c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z' || c >= 0x80
}

func identifierPart(c byte) bool {
	return identifierStart(c) || digit(c) || c == '$'
}

func digit(c byte) bool {
	return c >= '0' && c <= '9'
}

// blockCommentEnd returns the offset after the block comment at start;
// PostgreSQL block comments nest.
func blockCommentEnd(source string, start int) (int, error) {
	depth := 0
	for i := start; i+1 < len(source); {
		switch source[i : i+2] {
		case "/*":
			depth, i = depth+1, i+2
		case "*/":
			depth, i = depth-1, i+2
			if depth == 0 {
				return i, nil
			}
		default:
			i++
		}
	}
	return 0, fmt.Errorf("unterminated comment at offset %d", start)
}

// stringConstant reads the constant whose opening quote is at start, with
// doubled quotes and, in an escape string, backslash escapes.
func stringConstant(source string, start int, escapes bool) (string, int, error) {
	var value strings.Builder
	for i := start + 1; i < len(source); i++ {
		switch c := source[i]; {
		case escapes && c == '\\' && i+1 < len(source):
			value.WriteByte(source[i+1])
			i++
		case c == '\'' && i+1 < len(source) && source[i+1] == '\'':
			value.WriteByte('\'')
			i++
		case c == '\'':
			return value.String(), i + 1, nil
		default:
			value.WriteByte(c)
		}
	}
	return "", 0, fmt.Errorf("unterminated string constant at offset %d", start)
}

func quotedIdentifier(source string, start int) (string, int, error) {
	var name strings.Builder
	for i := start + 1; i < len(source); i++ {
		switch {
		case source[i] == '"' && i+1 < len(source) && source[i+1] == '"':
			name.WriteByte('"')
			i++
		case source[i] == '"':
			if name.Len() == 0 {
				return "", 0, fmt.Errorf("empty quoted identifier at offset %d", start)
			}
			return name.String(), i + 1, nil
		default:
			name.WriteByte(source[i])
		}
	}
	return "", 0, fmt.Errorf("unterminated quoted identifier at offset %d", start)
}

// dollarToken reads a positional parameter ($1) or a dollar-quoted string
// constant ($$...$$ or $tag$...$tag$) at start.
func dollarToken(source string, start int) (token, int, error) {
	end := start + 1
	if end < len(source) && digit(source[end]) {
		for end < len(source) && digit(source[end]) {
			end++
		}
		return token{parameterToken, source[start:end]}, end, nil
	}
	if end < len(source) && source[end] != '$' && !identifierStart(source[end]) {
		return token{}, 0, fmt.Errorf("unexpected $ at offset %d", start)
	}
	for end < len(source) && source[end] != '$' && identifierPart(source[end]) {
		end++
	}
	if end >= len(source) || source[end] != '$' {
		return token{}, 0, fmt.Errorf("unterminated dollar quote tag at offset %d", start)
	}
	tag := source[start : end+1]
	closing := strings.Index(source[end+1:], tag)
	if closing < 0 {
		return token{}, 0, fmt.Errorf("unterminated dollar-quoted string %s at offset %d", tag, start)
	}
	body := end + 1
	return token{literalToken, source[body : body+closing]}, body + closing + len(tag), nil
}

func numberEnd(source string, start int) int {
	end := start
	for end < len(source) && digit(source[end]) {
		end++
	}
	// 1..10 is a range, not the number 1. followed by .10.
	if end < len(source) && source[end] == '.' && !(end+1 < len(source) && source[end+1] == '.') {
		end++
		for end < len(source) && digit(source[end]) {
			end++
		}
	}
	if end < len(source) && (source[end] == 'e' || source[end] == 'E') {
		exponent := end + 1
		if exponent < len(source) && (source[exponent] == '+' || source[exponent] == '-') {
			exponent++
		}
		if exponent < len(source) && digit(source[exponent]) {
			end = exponent
			for end < len(source) && digit(source[end]) {
				end++
			}
		}
	}
	return end
}

// operatorAt reads the operator at start as PostgreSQL does: the longest run
// of operator characters that starts no comment, where a name of several
// characters cannot end in + or - unless it holds one of ~!@#%^&|`?.
func operatorAt(source string, start int) string {
	end := start + 1
	for end < len(source) && strings.IndexByte(operatorCharacters, source[end]) >= 0 &&
		!strings.HasPrefix(source[end:], "--") && !strings.HasPrefix(source[end:], "/*") {
		end++
	}
	operator := source[start:end]
	for len(operator) > 1 && strings.ContainsAny(operator[len(operator)-1:], "+-") &&
		!strings.ContainsAny(operator, "~!@#%^&|`?") {
		operator = operator[:len(operator)-1]
	}
	return operator
}

// whenCondition returns the tokens of the WHEN condition of a trigger
// definition as pg_get_triggerdef prints it, or none.
func whenCondition(definition []token) ([]token, error) {
	for i := 0; i+2 < len(definition); i++ {
		if !definition[i].isWord("for") || !definition[i+1].isWord("each") || !definition[i+2].isWord("row") {
			continue
		}
		rest := definition[i+3:]
		if len(rest) == 0 || !rest[0].isWord("when") {
			return nil, nil
		}
		if len(rest) < 2 || !rest[1].is(symbolToken, "(") {
			return nil, errors.New("WHEN is not followed by a parenthesised condition")
		}
		end, err := closing(rest, 1, "(", ")")
		if err != nil {
			return nil, err
		}
		return rest[2:end], nil
	}
	return nil, errors.New("the definition has no FOR EACH ROW")
}

// closing returns the index of the token that closes the bracket at open.
func closing(tokens []token, open int, opening, closingText string) (int, error) {
	depth := 0
	for i := open; i < len(tokens); i++ {
		switch {
		case tokens[i].is(symbolToken, opening):
			depth++
		case tokens[i].is(symbolToken, closingText):
			depth--
			if depth == 0 {
				return i, nil
			}
		}
	}
	return 0, fmt.Errorf("unbalanced %s", opening)
}

// updatePath returns the statements of a trigger function body that an
// UPDATE can run. A body whose first statement is
//
//	IF TG_OP = 'UPDATE' THEN ... RETURN NEW; END IF;
//
// runs only that block on UPDATE, and a body whose first statement is the
// same block for 'INSERT' runs everything after it; any other body, including
// one with ELSE or ELSIF in that first IF, is read whole. A DECLARE section
// before the body belongs to every path, since each call evaluates its
// defaults. Reading more than the UPDATE path can only add failures, never
// hide one.
func updatePath(body []token) []token {
	if len(body) > 0 && body[0].isWord("declare") {
		for i := 1; i < len(body); i++ {
			if body[i].isWord("begin") {
				return append(slices.Clip(body[:i]), updatePath(body[i:])...)
			}
		}
		return body
	}
	if len(body) < 6 || !body[0].isWord("begin") || !body[1].isWord("if") || !body[2].isWord("tg_op") ||
		!body[3].is(symbolToken, "=") || body[4].kind != literalToken || !body[5].isWord("then") {
		return body
	}
	operation := strings.ToUpper(body[4].text)
	if operation != "UPDATE" && operation != "INSERT" {
		return body
	}
	depth := 0
	for i := 6; i+1 < len(body); i++ {
		switch {
		case body[i].isWord("end") && body[i+1].isWord("if"):
			if depth > 0 {
				depth, i = depth-1, i+1
				continue
			}
			block := body[6:i]
			if len(block) < 3 || !block[len(block)-3].isWord("return") ||
				!block[len(block)-2].isWord("new", "old", "null") || !block[len(block)-1].is(symbolToken, ";") ||
				i+2 >= len(body) || !body[i+2].is(symbolToken, ";") {
				return body
			}
			if operation == "UPDATE" {
				return block
			}
			return body[i+3:]
		case body[i].isWord("if"):
			depth++
		case depth == 0 && body[i].isWord("else", "elsif", "elseif"):
			return body
		}
	}
	return body
}
