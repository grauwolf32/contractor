package contractstest

import (
	"bytes"
	"encoding/hex"
	"encoding/json"
	"strings"
	"testing"
	"unicode/utf8"
)

// TargetCredentialCase is one case of
// api/testdata/v1alpha1/target-credential-cases.json: the exact bytes, the
// verdict of the decoders and constructors and, for an invalid case that a
// JSON Schema cannot reject, SchemaAccepts.
type TargetCredentialCase struct {
	Name          string
	Kind          string
	Data          []byte
	Valid         bool
	SchemaAccepts bool
}

// SchemaDocument reports whether the case's bytes are one UTF-8 JSON value a
// schema can judge. Bytes that are not count as rejected by every schema.
func (c TargetCredentialCase) SchemaDocument() bool {
	return utf8.Valid(c.Data) && json.Valid(c.Data)
}

// TargetCredentialCases are the exact reference cases, which the Server, the
// Runtime and the private schema read, and the creation material cases, which
// only the Server and the public schema read.
type TargetCredentialCases struct {
	References []TargetCredentialCase
	Materials  []TargetCredentialCase
}

type targetCredentialFixtureCase struct {
	Name     string          `json:"name"`
	Kind     string          `json:"kind"`
	Text     *string         `json:"text"`
	Hex      string          `json:"hex"`
	Material json.RawMessage `json:"material"`
	Fill     []struct {
		Pointer string `json:"pointer"`
		Unit    string `json:"unit"`
		Count   int    `json:"count"`
	} `json:"fill"`
	Valid  *bool  `json:"valid"`
	Schema string `json:"schema"`
}

// ReadTargetCredentialCases strictly reads the shared target credential cases
// that the Runtime reads too (runtime/tests/test_shared_target_credential_references.py).
// A case needs a unique name, a verdict and exactly one of text, hex or
// material; fill applies only to material, a material case names a target
// kind and schema may only record that a schema accepts an invalid case.
func ReadTargetCredentialCases(t testing.TB) TargetCredentialCases {
	t.Helper()
	var fixture struct {
		Description string                        `json:"description"`
		References  []targetCredentialFixtureCase `json:"references"`
		Materials   []targetCredentialFixtureCase `json:"materials"`
	}
	decoder := json.NewDecoder(bytes.NewReader(ReadFile(t, "api", "testdata", "v1alpha1", "target-credential-cases.json")))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&fixture); err != nil {
		t.Fatalf("decode target-credential-cases.json: %v", err)
	}
	if fixture.Description == "" || len(fixture.References) == 0 || len(fixture.Materials) == 0 {
		t.Fatal("target-credential-cases.json needs a description, references and materials")
	}
	names := map[string]bool{}
	read := func(section string, raw targetCredentialFixtureCase, material bool) TargetCredentialCase {
		t.Helper()
		if raw.Name == "" || names[section+raw.Name] || raw.Valid == nil {
			t.Fatalf("%s case %q needs a unique name and a verdict", section, raw.Name)
		}
		names[section+raw.Name] = true
		sources := 0
		for _, present := range []bool{raw.Text != nil, raw.Hex != "", raw.Material != nil} {
			if present {
				sources++
			}
		}
		if sources != 1 || len(raw.Fill) != 0 && raw.Material == nil {
			t.Fatalf("%s case %q needs exactly one of text, hex and material, and fill only with material", section, raw.Name)
		}
		if material != strings.HasPrefix(raw.Kind, "target-") {
			t.Fatalf("%s case %q: only a material case names a target kind", section, raw.Name)
		}
		if raw.Schema != "" && (raw.Schema != "accepts" || *raw.Valid) {
			t.Fatalf("%s case %q: schema may only record that a schema accepts an invalid case", section, raw.Name)
		}
		result := TargetCredentialCase{
			Name: raw.Name, Kind: raw.Kind, Valid: *raw.Valid, SchemaAccepts: raw.Schema == "accepts",
		}
		switch {
		case raw.Text != nil:
			result.Data = []byte(*raw.Text)
		case raw.Hex != "":
			data, err := hex.DecodeString(raw.Hex)
			if err != nil {
				t.Fatalf("%s case %q hex: %v", section, raw.Name, err)
			}
			result.Data = data
		case len(raw.Fill) == 0:
			result.Data = append([]byte(nil), raw.Material...)
		default:
			var document map[string]any
			if err := json.Unmarshal(raw.Material, &document); err != nil {
				t.Fatalf("%s case %q: a filled material must be an object: %v", section, raw.Name, err)
			}
			for _, fill := range raw.Fill {
				if fill.Unit == "" || fill.Count < 1 {
					t.Fatalf("%s case %q: fill needs a unit and a positive count", section, raw.Name)
				}
				SetJSONPointer(t, document, fill.Pointer, strings.Repeat(fill.Unit, fill.Count))
			}
			var encoded bytes.Buffer
			encoder := json.NewEncoder(&encoded)
			encoder.SetEscapeHTML(false)
			if err := encoder.Encode(document); err != nil {
				t.Fatal(err)
			}
			result.Data = bytes.TrimSuffix(encoded.Bytes(), []byte("\n"))
		}
		return result
	}
	var cases TargetCredentialCases
	for _, raw := range fixture.References {
		cases.References = append(cases.References, read("reference", raw, false))
	}
	for _, raw := range fixture.Materials {
		cases.Materials = append(cases.Materials, read("material", raw, true))
	}
	return cases
}
