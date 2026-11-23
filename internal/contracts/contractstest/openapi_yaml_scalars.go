package contractstest

import (
	"bytes"
	"encoding/json"
	"math"
	"testing"
)

// OpenAPIYAMLCase is one document of api/testdata/openapi-yaml-scalar-cases.json,
// rendered as the fixture's description says. Accepted documents read as
// Value; rejected documents are rejected.
type OpenAPIYAMLCase struct {
	Name     string
	Document []byte
	Accepted bool
	Value    map[string]any
}

// ReadOpenAPIYAMLCases strictly reads the shared YAML scalar and key cases
// (SCN-R124) that the Runtime OpenAPI authoring Toolset reads too
// (runtime/tests/test_openapi_toolset.py). Each write case becomes two
// accepted documents, the written scalar as a member value and as a member
// name, which read back as the original string. An unknown key, an unknown
// verdict or an empty group fails the test.
func ReadOpenAPIYAMLCases(t testing.TB) []OpenAPIYAMLCase {
	t.Helper()
	type scalarCase struct {
		Name    string          `json:"name"`
		YAML    string          `json:"yaml"`
		Verdict string          `json:"verdict"`
		Value   json.RawMessage `json:"value"`
		Member  *string         `json:"member"`
	}
	var fixture struct {
		Description string       `json:"description"`
		Values      []scalarCase `json:"values"`
		Keys        []scalarCase `json:"keys"`
		Documents   []scalarCase `json:"documents"`
		Writes      []struct {
			String string `json:"string"`
			Scalar string `json:"scalar"`
		} `json:"writes"`
	}
	decoder := json.NewDecoder(bytes.NewReader(ReadFile(t, "api", "testdata", "openapi-yaml-scalar-cases.json")))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&fixture); err != nil {
		t.Fatalf("decode shared OpenAPI YAML scalar cases: %v", err)
	}
	if len(fixture.Values) == 0 || len(fixture.Keys) == 0 || len(fixture.Documents) == 0 || len(fixture.Writes) == 0 {
		t.Fatal("shared OpenAPI YAML scalar cases need values, keys, documents and writes")
	}
	var cases []OpenAPIYAMLCase
	add := func(group string, item scalarCase, document string, value func() map[string]any) {
		label := group + "/" + item.Name
		if item.Name == "" {
			label = group + "/" + item.YAML
		}
		switch item.Verdict {
		case "accepted":
			cases = append(cases, OpenAPIYAMLCase{Name: label, Document: []byte(document), Accepted: true, Value: value()})
		case "rejected":
			if item.Value != nil || item.Member != nil {
				t.Fatalf("%s: a rejected case has no value", label)
			}
			cases = append(cases, OpenAPIYAMLCase{Name: label, Document: []byte(document)})
		default:
			t.Fatalf("%s: unknown verdict %q", label, item.Verdict)
		}
	}
	decode := func(label string, raw json.RawMessage) any {
		if raw == nil {
			t.Fatalf("%s: an accepted case needs a value", label)
		}
		var value any
		if err := json.Unmarshal(raw, &value); err != nil {
			t.Fatalf("%s: %v", label, err)
		}
		return value
	}
	for _, item := range fixture.Values {
		add("values", item, "v: "+item.YAML+"\n", func() map[string]any {
			return map[string]any{"v": decode(item.YAML, item.Value)}
		})
	}
	for _, item := range fixture.Keys {
		add("keys", item, item.YAML+": v\n", func() map[string]any {
			if item.Member == nil {
				t.Fatalf("%s: an accepted key case needs a member", item.YAML)
			}
			return map[string]any{*item.Member: "v"}
		})
	}
	for _, item := range fixture.Documents {
		add("documents", item, item.YAML, func() map[string]any {
			value, ok := decode(item.Name, item.Value).(map[string]any)
			if !ok {
				t.Fatalf("%s: an accepted document reads as an object", item.Name)
			}
			return value
		})
	}
	for _, write := range fixture.Writes {
		cases = append(cases,
			OpenAPIYAMLCase{Name: "writes/value/" + write.String, Document: []byte("v: " + write.Scalar + "\n"), Accepted: true, Value: map[string]any{"v": write.String}},
			OpenAPIYAMLCase{Name: "writes/member/" + write.String, Document: []byte(write.Scalar + ": v\n"), Accepted: true, Value: map[string]any{write.String: "v"}},
		)
	}
	return cases
}

// SameJSONValue reports whether two decoded JSON values are equal. Numbers
// compare as float64 bits, so -0 and 0 differ.
func SameJSONValue(left, right any) bool {
	switch left := left.(type) {
	case map[string]any:
		right, ok := right.(map[string]any)
		if !ok || len(left) != len(right) {
			return false
		}
		for key, value := range left {
			other, found := right[key]
			if !found || !SameJSONValue(value, other) {
				return false
			}
		}
		return true
	case []any:
		right, ok := right.([]any)
		if !ok || len(left) != len(right) {
			return false
		}
		for index := range left {
			if !SameJSONValue(left[index], right[index]) {
				return false
			}
		}
		return true
	case float64:
		right, ok := right.(float64)
		return ok && math.Float64bits(left) == math.Float64bits(right)
	default:
		return left == right
	}
}
