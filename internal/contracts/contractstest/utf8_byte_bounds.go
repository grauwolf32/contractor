package contractstest

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"
	"unicode/utf8"
)

// UTF8ByteBound is one public string of api/testdata/public/utf8-byte-bound-cases.json:
// the OpenAPI component schema and the path below it that declare it (empty
// for the component itself), its bound in UTF-8 bytes and the limits.md rule
// that owns the bound.
type UTF8ByteBound struct {
	Schema   string   `json:"schema"`
	Path     []string `json:"path"`
	MaxBytes int      `json:"maxBytes"`
	Rule     string   `json:"rule"`
	Prefix   string   `json:"prefix,omitempty"`
}

// UTF8ByteBoundValue is a string of one character at or one byte over a
// bound.
type UTF8ByteBoundValue struct {
	Character string
	Value     string
	Within    bool
}

// Values returns, for each shared character, the string of exactly MaxBytes
// bytes (after the optional structured prefix, the character repeated, padded with "a") and the same string with
// one more "a".
func (b UTF8ByteBound) Values(characters []string) []UTF8ByteBoundValue {
	return b.EmbeddedValues(characters, b.Prefix, "")
}

// EmbeddedValues is Values for a value with structure, such as a URL or an
// SSH key: the run of each character fills the bytes between prefix and
// suffix, so the value is exactly MaxBytes bytes, and the value one byte
// over the bound has one more "a" at the end of the run.
func (b UTF8ByteBound) EmbeddedValues(characters []string, prefix, suffix string) []UTF8ByteBoundValue {
	values := make([]UTF8ByteBoundValue, 0, 2*len(characters))
	room := b.MaxBytes - len(prefix) - len(suffix)
	for _, character := range characters {
		run := strings.Repeat(character, room/len(character)) + strings.Repeat("a", room%len(character))
		values = append(values,
			UTF8ByteBoundValue{Character: character, Value: prefix + run + suffix, Within: true},
			UTF8ByteBoundValue{Character: character, Value: prefix + run + "a" + suffix, Within: false},
		)
	}
	return values
}

// ReadUTF8ByteBoundCases strictly reads the shared UTF-8 byte bounds that the
// UI reads too (ui/src/api/contract.test.ts). It returns the characters the
// boundary strings are made of and every bounded field; an unknown key, an
// empty list or a field without a bound fails the test.
func ReadUTF8ByteBoundCases(t testing.TB) ([]string, []UTF8ByteBound) {
	t.Helper()
	var fixture struct {
		Description string          `json:"description"`
		Characters  []string        `json:"characters"`
		Fields      []UTF8ByteBound `json:"fields"`
	}
	decoder := json.NewDecoder(bytes.NewReader(ReadFile(t, "api", "testdata", "public", "utf8-byte-bound-cases.json")))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&fixture); err != nil {
		t.Fatalf("decode utf8-byte-bound-cases.json: %v", err)
	}
	if len(fixture.Characters) == 0 || len(fixture.Fields) == 0 {
		t.Fatal("utf8-byte-bound-cases.json needs characters and fields")
	}
	widths := map[int]bool{}
	for _, character := range fixture.Characters {
		if utf8.RuneCountInString(character) != 1 || !utf8.ValidString(character) {
			t.Fatalf("shared character %q is not one valid code point", character)
		}
		widths[len(character)] = true
	}
	if len(widths) < 2 {
		t.Fatal("utf8-byte-bound-cases.json needs characters of different UTF-8 widths")
	}
	for _, field := range fixture.Fields {
		if field.Schema == "" || field.Path == nil || field.MaxBytes <= 0 || field.Rule == "" || !utf8.ValidString(field.Prefix) || len(field.Prefix) >= field.MaxBytes {
			t.Fatalf("incomplete UTF-8 byte bound %+v", field)
		}
	}
	return fixture.Characters, fixture.Fields
}
