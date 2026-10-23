package evaldomain

import (
	"encoding/json"
	"reflect"
	"strings"
	"testing"
)

// strictJSONCode returns the Eval code StrictJSON answers for raw, or "" when
// it accepts raw.
func strictJSONCode(t *testing.T, raw string) string {
	t.Helper()
	_, err := StrictJSON([]byte(raw))
	if err == nil {
		return ""
	}
	safe, ok := err.(*Error)
	if !ok {
		t.Fatalf("StrictJSON returned %T, want *Error", err)
	}
	return safe.Code
}

func nestedArrays(levels int, inner string) string {
	return strings.Repeat("[", levels) + inner + strings.Repeat("]", levels)
}

func nestedObjects(levels int, inner string) string {
	return strings.Repeat(`{"a":`, levels) + inner + strings.Repeat("}", levels)
}

// TestStrictJSONVerdictsAndCodes pins the exact verdict and code of each rule
// StrictJSON applies (EVF-R15, EVF-R30): the top-level value is at depth 0
// and a value deeper than MaxDepth exceeds the limit; the size bound and
// invalid JSON win over depth; inside a valid document the first violation
// in document order decides the code.
func TestStrictJSONVerdictsAndCodes(t *testing.T) {
	cases := []struct {
		name, raw, code string
	}{
		{"deepest array", nestedArrays(MaxDepth+1, ""), ""},
		{"deepest scalar", nestedArrays(MaxDepth, "1"), ""},
		{"deepest object", nestedObjects(MaxDepth, "1"), ""},
		{"array too deep", nestedArrays(MaxDepth+2, ""), "eval_limit_exceeded"},
		{"scalar too deep", nestedArrays(MaxDepth+1, "1"), "eval_limit_exceeded"},
		{"object too deep", nestedObjects(MaxDepth+1, "1"), "eval_limit_exceeded"},
		{"too deep and invalid", nestedArrays(MaxDepth+2, "") + ",", "eval_invalid"},
		{"too deep and trailing", nestedArrays(MaxDepth+2, "") + " {}", "eval_invalid"},
		{"too deep with lone surrogate", nestedArrays(MaxDepth+2, `"\ud800"`), "eval_invalid"},
		{"duplicate before too deep", `{"x":1,"x":` + nestedArrays(MaxDepth+2, "") + `}`, "eval_invalid"},
		{"too deep before duplicate", `{"x":` + nestedArrays(MaxDepth+2, "") + `,"x":1}`, "eval_limit_exceeded"},
		{"non-finite before too deep", `[1e999,` + nestedArrays(MaxDepth+2, "") + `]`, "eval_invalid"},
		{"too deep before non-finite", `[` + nestedArrays(MaxDepth+2, "") + `,1e999]`, "eval_limit_exceeded"},
		{"oversized and invalid", strings.Repeat("{", MaxDocumentBytes+1), "eval_limit_exceeded"},
		{"empty", "", "eval_invalid"},
		{"whitespace", " \n\t", "eval_invalid"},
		{"duplicate key", `{"x":1,"x":2}`, "eval_invalid"},
		{"escaped duplicate key", `{"x":{"a":1,"a":2}}`, "eval_invalid"},
		{"lone high surrogate", `{"x":"\ud800"}`, "eval_invalid"},
		{"lone low surrogate", `{"x":"\udc00"}`, "eval_invalid"},
		{"surrogate pair", `{"x":"😀"}`, ""},
		{"escaped backslash before u", `{"x":"literal \\ud800"}`, ""},
		{"trailing value", `{} {}`, "eval_invalid"},
		{"trailing comma", `[1,]`, "eval_invalid"},
		{"NaN", `{"x":NaN}`, "eval_invalid"},
		{"positive overflow", `{"x":1e999}`, "eval_invalid"},
		{"negative overflow", `{"x":-1e999}`, "eval_invalid"},
		{"overflow member name", `{"1e999":1}`, ""},
		{"overflow string", `{"x":"1e999"}`, ""},
		{"invalid UTF-8", "{\"x\":\"\xff\"}", "eval_invalid"},
		{"top-level scalar", `"text"`, ""},
		{"top-level null", `null`, ""},
		{"surrounding whitespace", " {} \n", ""},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := strictJSONCode(t, c.raw); got != c.code {
				t.Fatalf("code = %q, want %q", got, c.code)
			}
		})
	}
}

// TestStrictJSONDocumentSizeBound pins that the byte bound, not a node count,
// limits a document: a 1 MiB array of one-digit numbers is accepted.
func TestStrictJSONDocumentSizeBound(t *testing.T) {
	elements := (MaxDocumentBytes - 1) / 2
	raw := "[" + strings.Repeat("1,", elements-1) + "1]"
	if len(raw) > MaxDocumentBytes || MaxDocumentBytes-len(raw) > 2 {
		t.Fatalf("fixture is %d bytes", len(raw))
	}
	value, err := StrictJSON([]byte(raw))
	if err != nil {
		t.Fatalf("1 MiB of nodes rejected: %v", err)
	}
	if got := len(value.([]any)); got != elements {
		t.Fatalf("decoded %d elements, want %d", got, elements)
	}
	exact := `"` + strings.Repeat("a", MaxDocumentBytes-2) + `"`
	if code := strictJSONCode(t, exact); code != "" {
		t.Fatalf("document of exactly MaxDocumentBytes: %s", code)
	}
	if code := strictJSONCode(t, exact+" "); code != "eval_limit_exceeded" {
		t.Fatalf("document one byte over MaxDocumentBytes: %q", code)
	}
}

// TestStrictJSONKeepsNumberLiterals pins the decoded tree: numbers stay
// json.Number with their literal, and an empty array is an empty, non-nil
// slice.
func TestStrictJSONKeepsNumberLiterals(t *testing.T) {
	value, err := StrictJSON([]byte(`{"n":[1.50,-0,1e2,12345678901234567890,1e-999],"s":"x","b":true,"z":null,"e":[],"o":{}}`))
	if err != nil {
		t.Fatal(err)
	}
	want := map[string]any{
		"n": []any{json.Number("1.50"), json.Number("-0"), json.Number("1e2"), json.Number("12345678901234567890"), json.Number("1e-999")},
		"s": "x",
		"b": true,
		"z": nil,
		"e": []any{},
		"o": map[string]any{},
	}
	if !reflect.DeepEqual(value, want) {
		t.Fatalf("decoded %#v, want %#v", value, want)
	}
	if value.(map[string]any)["e"].([]any) == nil {
		t.Fatal("empty array decoded to a nil slice")
	}
	encoded, err := json.Marshal(value)
	if err != nil {
		t.Fatal(err)
	}
	if string(encoded) != `{"b":true,"e":[],"n":[1.50,-0,1e2,12345678901234567890,1e-999],"o":{},"s":"x","z":null}` {
		t.Fatalf("re-encoded %s", encoded)
	}
}
