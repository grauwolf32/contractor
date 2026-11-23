package contractstest

import (
	"bytes"
	"encoding/json"
	"fmt"
	"maps"
	"slices"
	"strings"
	"testing"
)

// CredentialFieldValueCase is one value of
// api/testdata/v1alpha1/credential-field-value-cases.json and the verdict of
// the credential field-value grammar on it.
type CredentialFieldValueCase struct {
	Name  string
	Value string
	Valid bool
}

// CredentialFieldValueCredential names a Runtime credential kind and the JSON
// pointer of the value inside its creation material.
type CredentialFieldValueCredential struct {
	Kind   string `json:"kind"`
	Member string `json:"member"`
}

// CredentialFieldValueSettings names where RuntimeSettings carries the value:
// Set holds JSON pointers applied to the base document first.
type CredentialFieldValueSettings struct {
	Pointer string                     `json:"pointer"`
	Set     map[string]json.RawMessage `json:"set"`
}

// CredentialFieldValueField is one place that applies the grammar. MaxBytes
// is zero when the place has no byte bound.
type CredentialFieldValueField struct {
	Name            string                          `json:"name"`
	MaxBytes        int                             `json:"maxBytes"`
	Rule            string                          `json:"rule"`
	Credential      *CredentialFieldValueCredential `json:"credential"`
	LLMCredential   bool                            `json:"llmCredential"`
	RuntimeSettings *CredentialFieldValueSettings   `json:"runtimeSettings"`
}

// Cases returns the grammar cases and, when the field has a byte bound, its
// bound cases: MaxBytes visible characters, one more, and MaxBytes bytes of
// an "a", spaces and a "b".
func (f CredentialFieldValueField) Cases(grammar []CredentialFieldValueCase) []CredentialFieldValueCase {
	cases := slices.Clone(grammar)
	if f.MaxBytes > 0 {
		at := strings.Repeat("a", f.MaxBytes)
		cases = append(cases,
			CredentialFieldValueCase{Name: fmt.Sprintf("%d visible bytes", f.MaxBytes), Value: at, Valid: true},
			CredentialFieldValueCase{Name: fmt.Sprintf("%d visible bytes", f.MaxBytes+1), Value: at + "a", Valid: false},
			CredentialFieldValueCase{
				Name:  fmt.Sprintf("%d bytes with an inner space run", f.MaxBytes),
				Value: "a" + strings.Repeat(" ", f.MaxBytes-2) + "b", Valid: true,
			},
		)
	}
	return cases
}

// CredentialFieldValueCases is the expanded shared fixture: the grammar
// corpus, every field that applies it and the RuntimeSettings base document.
type CredentialFieldValueCases struct {
	Grammar []CredentialFieldValueCase
	Fields  []CredentialFieldValueField
	Base    []byte
}

// ReadCredentialFieldValueCases strictly reads the shared credential
// field-value grammar cases that the Runtime reads too
// (runtime/tests/credential_field_values.py). It places every
// ASCII character at each position template and adds the listed values; a
// character range that is not exactly 0 through 127 or an incomplete field
// fails the test.
func ReadCredentialFieldValueCases(t testing.TB) CredentialFieldValueCases {
	t.Helper()
	var fixture struct {
		Description string            `json:"description"`
		Base        string            `json:"base"`
		Positions   map[string]string `json:"positions"`
		Characters  []struct {
			From  int      `json:"from"`
			To    int      `json:"to"`
			Valid []string `json:"valid"`
		} `json:"characters"`
		Values []struct {
			Name  string `json:"name"`
			Value string `json:"value"`
			Valid *bool  `json:"valid"`
		} `json:"values"`
		Fields []CredentialFieldValueField `json:"fields"`
	}
	decoder := json.NewDecoder(bytes.NewReader(ReadFile(t, "api", "testdata", "v1alpha1", "credential-field-value-cases.json")))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&fixture); err != nil {
		t.Fatalf("decode credential-field-value-cases.json: %v", err)
	}
	if len(fixture.Positions) == 0 || len(fixture.Values) == 0 || len(fixture.Fields) == 0 ||
		!strings.HasPrefix(fixture.Base, "valid/") {
		t.Fatal("credential-field-value-cases.json needs positions, values, fields and a valid/ base")
	}
	var cases CredentialFieldValueCases
	next := 0
	for _, characters := range fixture.Characters {
		if characters.From != next || characters.To < characters.From {
			t.Fatalf("character range %d-%d does not continue at %d", characters.From, characters.To, next)
		}
		for _, position := range characters.Valid {
			if _, ok := fixture.Positions[position]; !ok {
				t.Fatalf("character range %d-%d names unknown position %q", characters.From, characters.To, position)
			}
		}
		for code := characters.From; code <= characters.To; code++ {
			for _, position := range slices.Sorted(maps.Keys(fixture.Positions)) {
				template := fixture.Positions[position]
				if strings.Count(template, "{c}") != 1 {
					t.Fatalf("position %q must hold {c} once", position)
				}
				cases.Grammar = append(cases.Grammar, CredentialFieldValueCase{
					Name:  fmt.Sprintf("U+%04X %s", code, position),
					Value: strings.Replace(template, "{c}", string(rune(code)), 1),
					Valid: slices.Contains(characters.Valid, position),
				})
			}
		}
		next = characters.To + 1
	}
	if next != 128 {
		t.Fatalf("character ranges end at %d, want every ASCII character", next)
	}
	for _, value := range fixture.Values {
		if value.Name == "" || value.Valid == nil {
			t.Fatalf("value %q needs a name and a verdict", value.Value)
		}
		cases.Grammar = append(cases.Grammar, CredentialFieldValueCase{Name: value.Name, Value: value.Value, Valid: *value.Valid})
	}
	for _, field := range fixture.Fields {
		if field.Name == "" || field.MaxBytes < 0 || (field.MaxBytes > 0) == (field.Rule == "") ||
			field.Credential == nil && !field.LLMCredential && field.RuntimeSettings == nil {
			t.Fatalf("incomplete credential field-value field %+v", field)
		}
		if field.Credential != nil && (field.Credential.Kind == "" || !strings.HasPrefix(field.Credential.Member, "/")) ||
			field.RuntimeSettings != nil && !strings.HasPrefix(field.RuntimeSettings.Pointer, "/") {
			t.Fatalf("credential field-value field %q needs a kind, member and pointer", field.Name)
		}
	}
	cases.Fields = fixture.Fields
	cases.Base = ReadFixture(t, "valid", strings.TrimPrefix(fixture.Base, "valid/"))
	return cases
}

// SetJSONPointer sets the member that pointer names in document to value;
// every segment before the last must name an object, which is created when it
// is absent.
func SetJSONPointer(t testing.TB, document map[string]any, pointer string, value any) {
	t.Helper()
	segments := strings.Split(strings.TrimPrefix(pointer, "/"), "/")
	if !strings.HasPrefix(pointer, "/") || slices.Contains(segments, "") {
		t.Fatalf("invalid JSON pointer %q", pointer)
	}
	current := document
	for _, segment := range segments[:len(segments)-1] {
		child, exists := current[segment]
		if !exists {
			child = map[string]any{}
			current[segment] = child
		}
		next, ok := child.(map[string]any)
		if !ok {
			t.Fatalf("JSON pointer %q does not name an object at %q", pointer, segment)
		}
		current = next
	}
	current[segments[len(segments)-1]] = value
}
