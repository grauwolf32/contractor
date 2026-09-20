package scanplan

import (
	"errors"
	"reflect"
	"strings"
	"testing"
)

func assertDocumentError(t *testing.T, data []byte, mediaType, code string) {
	t.Helper()
	document, err := parseDocument(data, mediaType)
	var preparation *PreparationError
	if !errors.As(err, &preparation) || preparation.Code != code {
		t.Fatalf("error = %v, want preparation code %s", err, code)
	}
	if document != nil || strings.Contains(err.Error(), "canary") {
		t.Fatalf("failed parse returned document or exposed source: %v", err)
	}
}

func TestParseDocumentJSONAndYAMLShareLosslessScalarRepresentation(t *testing.T) {
	jsonSource := `{"openapi":"3.0.4","quoted":"9007199254740993","empty":"",` +
		`"boolean":true,"null":null,"integer":9007199254740991,"float":1.25,` +
		`"array":[false,"ключ 🔑",0],"escaped":"line\nnext\u0000",` +
		`"body":{"$ref":"literal payload reference"}}`
	yamlSource := "openapi: '3.0.4'\nquoted: '9007199254740993'\nempty: ''\n" +
		"boolean: true\nnull: null\ninteger: 9007199254740991\nfloat: 1.25\n" +
		"array: [false, 'ключ 🔑', 0]\nescaped: \"line\\nnext\\u0000\"\n" +
		"body:\n  $ref: literal payload reference\n"
	// A YAML null key is not a string, even though it is a string key in JSON.
	yamlSource = strings.Replace(yamlSource, "\nnull:", "\n'null':", 1)
	expected := map[string]any{
		"openapi": "3.0.4", "quoted": "9007199254740993", "empty": "",
		"boolean": true, "null": nil, "integer": float64(9007199254740991), "float": 1.25,
		"array": []any{false, "ключ 🔑", float64(0)}, "escaped": "line\nnext\x00",
		"body": map[string]any{"$ref": "literal payload reference"},
	}
	for _, mediaType := range []string{"application/json", "application/yaml", "application/x-yaml", "text/yaml"} {
		t.Run(mediaType, func(t *testing.T) {
			source := yamlSource
			if mediaType == "application/json" {
				source = jsonSource
			}
			document, err := parseDocument([]byte(source), mediaType)
			if err != nil || !reflect.DeepEqual(document, expected) {
				t.Fatalf("parse = %#v, %v; want %#v", document, err, expected)
			}
		})
	}
}

func TestParseDocumentRejectsAmbiguousOrNonJSONDocuments(t *testing.T) {
	tests := []struct {
		name, mediaType, source string
	}{
		{"JSON duplicate", "application/json", `{"canary":1,"canary":2}`},
		{"nested duplicate", "application/json", `{"one":{"canary":1,"canary":2}}`},
		{"escaped duplicate", "application/json", `{"canary":1,"\u0063anary":2}`},
		{"JSON trailing document", "application/json", `{} {"canary":1}`},
		{"JSON YAML fallback", "application/json", "canary: value"},
		{"JSON trailing comma", "application/json", `{"canary":1,}`},
		{"JSON comments", "application/json", "{} // canary"},
		{"JSON array", "application/json", `["canary"]`},
		{"JSON null", "application/json", "null"},
		{"JSON high surrogate", "application/json", `{"canary":"\ud800"}`},
		{"JSON low surrogate", "application/json", `{"canary":"\udfff"}`},
		{"JSON mismatched surrogate", "application/json", `{"canary":"\ud800\u0041"}`},
		{"JSON scalar invalid escape", "application/json", `{"canary":"\x20"}`},
		{"YAML duplicate", "application/yaml", "canary: 1\ncanary: 2"},
		{"YAML equivalent duplicate", "application/yaml", "canary: 1\n'canary': 2"},
		{"YAML numeric key", "application/yaml", "123: canary"},
		{"YAML bool key", "application/yaml", "true: canary"},
		{"YAML null key", "application/yaml", "null: canary"},
		{"YAML composite key", "application/yaml", "? [a, b]\n: canary"},
		{"YAML nested numeric key", "application/yaml", "outer: {123: canary}"},
		{"YAML anchor", "application/yaml", "canary: &secret value"},
		{"YAML alias", "application/yaml", "canary: &secret value\nother: *secret"},
		{"YAML recursive alias", "application/yaml", "canary: &secret [*secret]"},
		{"YAML merge", "application/yaml", "canary: {<<: {a: b}}"},
		{"YAML custom scalar tag", "application/yaml", "canary: !secret value"},
		{"YAML custom mapping tag", "application/yaml", "canary: !secret {a: b}"},
		{"YAML binary", "application/yaml", "canary: !!binary c2VjcmV0"},
		{"YAML set", "application/yaml", "canary: !!set {a: null}"},
		{"YAML timestamp", "application/yaml", "canary: 2026-09-20"},
		{"YAML malformed explicit null", "application/yaml", "canary: !!null invalid"},
		{"YAML malformed explicit bool", "application/yaml", "canary: !!bool invalid"},
		{"YAML multi document", "application/yaml", "canary: value\n---\nother: value"},
		{"YAML trailing empty document", "application/yaml", "canary: value\n---\n"},
		{"YAML list", "application/yaml", "- canary"},
		{"YAML scalar", "application/yaml", "canary"},
		{"empty", "application/yaml", ""},
		{"comment only", "application/yaml", "# canary"},
		{"invalid UTF8", "application/yaml", "canary: \xff"},
		{"invalid JSON UTF8", "application/json", "{\"canary\":\"\xff\"}"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			assertDocumentError(t, []byte(test.source), test.mediaType, "invalid_document")
		})
	}
}

func TestParseDocumentPreservesExplicitStringsAndSupplementaryUnicode(t *testing.T) {
	for _, source := range []string{
		`{"number":"1e9999","surrogate":"\ud83d\udd11","literal":"\\ud800"}`,
		"number: !!str 1e9999\nsurrogate: '🔑'\nliteral: '\\ud800'\n",
	} {
		document, err := parseDocument([]byte(source), "application/yaml")
		if err != nil || document["number"] != "1e9999" || document["surrogate"] != "🔑" || document["literal"] != `\ud800` {
			t.Fatalf("parse = %#v, %v", document, err)
		}
	}
}

func TestParseDocumentNumericBoundsBeforeFloatRounding(t *testing.T) {
	for _, number := range []string{
		"9007199254740992", "-9007199254740992", "9007199254740993", "9007199254740993.0",
		"9007199254740991.1", "-9007199254740991.01", "90071992547409910.1e-1",
		"9.0071992547409911e15", "1e9999", "1e-9999", strings.Repeat("9", 400),
	} {
		for _, mediaType := range []string{"application/json", "application/yaml"} {
			t.Run(mediaType+"/"+number, func(t *testing.T) {
				assertDocumentError(t, []byte(`{"number":`+number+`,"secret":"canary"}`), mediaType, "invalid_document")
			})
		}
	}
	for _, number := range []string{"0", "-0", "1.25", "1e-300", "9007199254740991", "-9007199254740991", "9007199254740991.0", "90071992547409910e-1", "0.09007199254740991e17"} {
		document, err := parseDocument([]byte(`{"number":`+number+`}`), "application/json")
		if err != nil {
			t.Fatalf("safe number %s: %v", number, err)
		}
		if _, ok := document["number"].(float64); !ok {
			t.Fatalf("number %s became %T", number, document["number"])
		}
	}
}

func TestParseDocumentYAMLNumbersRemainJSONCompatible(t *testing.T) {
	for _, number := range []string{".nan", ".inf", "-.inf", "0x20000000000000", "0xFFFFFFFFFFFFFFFFFFFFFFFF", "9_007_199_254_740_992"} {
		t.Run(number, func(t *testing.T) {
			assertDocumentError(t, []byte("number: "+number+"\nsecret: canary"), "application/yaml", "invalid_document")
		})
	}
	document, err := parseDocument([]byte("hex: 0x10\noctal: 0o10\nbinary: 0b10\nfloat: .5\nunderscores: 1_000\n"), "application/yaml")
	want := map[string]any{"hex": float64(16), "octal": float64(8), "binary": float64(2), "float": 0.5, "underscores": float64(1000)}
	if err != nil || !reflect.DeepEqual(document, want) {
		t.Fatalf("parse = %#v, %v; want %#v", document, err, want)
	}
}

func TestParseDocumentBounds(t *testing.T) {
	for _, mediaType := range []string{"application/json", "application/yaml"} {
		t.Run(mediaType+"/bytes", func(t *testing.T) {
			source := `{"value":"` + strings.Repeat("x", MaxSourceBytes-len(`{"value":""}`)) + `"}`
			if mediaType == "application/yaml" {
				source = `value: "` + strings.Repeat("x", MaxSourceBytes-len(`value: ""`)) + `"`
			}
			if _, err := parseDocument([]byte(source), mediaType); err != nil {
				t.Fatalf("exact byte limit: %v", err)
			}
			assertDocumentError(t, []byte(source+" "), mediaType, "source_limit_exceeded")
		})
		t.Run(mediaType+"/depth", func(t *testing.T) {
			prefix, suffix := `{"value":`, "}"
			if mediaType == "application/yaml" {
				prefix, suffix = "value: ", ""
			}
			source := prefix + strings.Repeat("[", MaxDepth-2) + "0" + strings.Repeat("]", MaxDepth-2) + suffix
			if _, err := parseDocument([]byte(source), mediaType); err != nil {
				t.Fatalf("exact depth limit: %v", err)
			}
			deep := prefix + strings.Repeat("[", MaxDepth*20) + "0" + strings.Repeat("]", MaxDepth*20) + suffix
			assertDocumentError(t, []byte(deep), mediaType, "source_limit_exceeded")
		})
		t.Run(mediaType+"/nodes", func(t *testing.T) {
			// Root mapping, string key and sequence consume the first three nodes.
			prefix, suffix := `{"value":[`, "]}"
			if mediaType == "application/yaml" {
				prefix, suffix = "value: [", "]"
			}
			source := prefix + strings.Repeat("0,", MaxNodes-4) + "0" + suffix
			if _, err := parseDocument([]byte(source), mediaType); err != nil {
				t.Fatalf("exact node limit: %v", err)
			}
			over := prefix + strings.Repeat("0,", MaxNodes-3) + "0" + suffix
			assertDocumentError(t, []byte(over), mediaType, "source_limit_exceeded")
		})
	}
	t.Run("block YAML depth", func(t *testing.T) {
		var source strings.Builder
		for i := 0; i < MaxDepth*2; i++ {
			source.WriteString(strings.Repeat(" ", i) + "key:\n")
		}
		source.WriteString(strings.Repeat(" ", MaxDepth*2) + "canary\n")
		assertDocumentError(t, []byte(source.String()), "application/yaml", "source_limit_exceeded")
	})
}

func TestParseDocumentRejectsUnsupportedMediaTypes(t *testing.T) {
	for _, mediaType := range []string{"", "text/plain", "text/json", "application/json; charset=utf-8"} {
		assertDocumentError(t, []byte(`{"secret":"canary"}`), mediaType, "unsupported_media_type")
	}
}
