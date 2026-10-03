package strictjson

import (
	"encoding/json"
	"errors"
	"strings"
	"testing"
)

func TestDecodeRejectsUnknownFieldsAndTrailingData(t *testing.T) {
	var target struct {
		Name string `json:"name"`
	}
	if err := Decode([]byte(`{"name":"a"}`), &target); err != nil || target.Name != "a" {
		t.Fatalf("Decode = %v, %+v", err, target)
	}
	if err := Decode([]byte(`{"name":"a","other":1}`), &target); err == nil || errors.Is(err, ErrTrailingData) {
		t.Fatalf("unknown field error = %v", err)
	}
	for _, input := range []string{`{"name":"a"}{}`, `{"name":"a"} x`} {
		if err := Decode([]byte(input), &target); !errors.Is(err, ErrTrailingData) {
			t.Fatalf("Decode(%s) error = %v", input, err)
		}
	}
}

func TestRequireEOFSeparatesTrailingValuesFromSyntaxErrors(t *testing.T) {
	decode := func(input string) error {
		decoder := json.NewDecoder(strings.NewReader(input))
		var value any
		if err := decoder.Decode(&value); err != nil {
			t.Fatal(err)
		}
		return RequireEOF(decoder)
	}
	if err := decode(`{} `); err != nil {
		t.Fatalf("single value error = %v", err)
	}
	if err := decode(`{} []`); !errors.Is(err, ErrTrailingData) {
		t.Fatalf("trailing value error = %v", err)
	}
	if err := decode(`{} x`); err == nil || errors.Is(err, ErrTrailingData) {
		t.Fatalf("trailing syntax error = %v", err)
	}
}

func TestRejectDuplicateKeys(t *testing.T) {
	for input, want := range map[string]error{
		`{"a":1,"b":[{"a":1},{"a":2}]}`: nil,
		`"scalar"`:                      nil,
		`{"a":1,"a":2}`:                 ErrDuplicateKey,
		`{"a":{"b":1,"b":2}}`:           ErrDuplicateKey,
		`[{"a":1,"a":1}]`:               ErrDuplicateKey,
		`{"a":1} {}`:                    ErrTrailingData,
		`{"a":1e400}`:                   nil,
	} {
		if err := RejectDuplicateKeys([]byte(input)); !errors.Is(err, want) {
			t.Errorf("RejectDuplicateKeys(%s) = %v, want %v", input, err, want)
		}
	}
	for _, input := range []string{``, `{"a":}`, `]`, `{"a":1`} {
		if err := RejectDuplicateKeys([]byte(input)); err == nil {
			t.Errorf("RejectDuplicateKeys(%q) accepted invalid JSON", input)
		}
	}
}

func TestValidUnicodeEscapes(t *testing.T) {
	// %u stands for a backslash-u escape, %% for an escaped backslash.
	for _, tc := range []struct {
		input string
		want  bool
	}{
		{`"plain"`, true},
		{`"%u00e9"`, true},
		{`"%ud83d%ude00"`, true},
		{`"%uD83D%uDE00"`, true},
		{`"%%ud800"`, true},
		{`"%ud800"`, false},
		{`"%ude00"`, false},
		{`"%ud800%u0041"`, false},
		{`"%ud800x"`, false},
		{`{"%ud800":1}`, false},
		{`"%u12"`, true},
		{`"%uZZZZ"`, true},
		{`"%ud800%uZZZZ"`, false},
		{`"%"%ude00"`, false},
	} {
		input := strings.ReplaceAll(tc.input, "%", `\`)
		if got := ValidUnicodeEscapes([]byte(input)); got != tc.want {
			t.Errorf("ValidUnicodeEscapes(%s) = %v, want %v", input, got, tc.want)
		}
	}
}

func TestCanonicalUsesJCS(t *testing.T) {
	got, err := Canonical(map[string]any{"b": 1.0, "a": []any{"x", 2.5}})
	if err != nil || string(got) != `{"a":["x",2.5],"b":1}` {
		t.Fatalf("Canonical = %s, %v", got, err)
	}
	if _, err := Canonical(func() {}); err == nil {
		t.Fatal("Canonical accepted an unencodable value")
	}
}
