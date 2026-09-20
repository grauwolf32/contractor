package scanplan

import (
	"encoding/json"
	"math"
	"reflect"
	"strconv"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts"
)

func extractionJSON(t *testing.T, input string) any {
	t.Helper()
	var value any
	if err := json.Unmarshal([]byte(input), &value); err != nil {
		t.Fatal(err)
	}
	return value
}

func TestConcreteExamplePrecedenceAndNoValidation(t *testing.T) {
	tests := []struct{ name, object, schema, want string }{
		{"object null beats conflict", `{"example":null,"examples":{"a":{"value":"named"}}}`, `{"example":"schema"}`, `null`},
		{"named sorted", `{"examples":{"z":{"value":"last"},"a":{"value":"first"}}}`, `{"example":"schema"}`, `"first"`},
		{"named skips unusable", `{"examples":{"a":{"$ref":"#/missing"},"b":{"externalValue":"file:///private"},"c":{},"d":{"value":null}}}`, `{"example":"schema"}`, `null`},
		{"schema example", `{}`, `{"example":"example","examples":["examples"],"default":"default","const":"const","enum":["enum"]}`, `"example"`},
		{"schema examples", `{}`, `{"examples":[null,"unused"],"default":"default"}`, `null`},
		{"default", `{}`, `{"default":false,"const":true,"enum":["enum"]}`, `false`},
		{"const", `{}`, `{"const":0,"enum":["enum"]}`, `0`},
		{"enum", `{}`, `{"enum":[null,"unused"]}`, `null`},
		{"empty examples fall through", `{}`, `{"examples":[],"enum":[],"default":"fallback"}`, `"fallback"`},
		{"malformed named examples fall through", `{"examples":42}`, `{"default":"fallback"}`, `"fallback"`},
		{"constraints ignored", `{}`, `{"type":["integer","null"],"format":"byte","pattern":"[","minimum":99,"required":["missing"],"example":{"anything":"payload"}}`, `{"anything":"payload"}`},
		{"payload refs are data", `{}`, `{"default":{"$ref":"https://private.invalid","nested":{"$ref":"#/missing"}}}`, `{"$ref":"https://private.invalid","nested":{"$ref":"#/missing"}}`},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p := &preparer{root: map[string]any{}}
			value, found, code := p.example(extractionJSON(t, tt.object).(map[string]any), extractionJSON(t, tt.schema))
			if !found || code != "" || !reflect.DeepEqual(value, extractionJSON(t, tt.want)) {
				t.Fatalf("value=%#v found=%v code=%q", value, found, code)
			}
		})
	}
}

func TestConcreteHintsAvoidUnusedReferences(t *testing.T) {
	for _, raw := range []string{
		`{"$ref":"file:///private","example":"local"}`,
		`{"$ref":"#/missing","default":"local"}`,
		`{"$ref":42,"const":"local"}`,
	} {
		p := &preparer{root: map[string]any{}}
		value, found, code := p.example(nil, extractionJSON(t, raw))
		if value != "local" || !found || code != "" || p.refs != 0 {
			t.Fatalf("schema=%s value=%v found=%v code=%q refs=%d", raw, value, found, code, p.refs)
		}
	}
	p := &preparer{}
	value, found, code := p.example(map[string]any{"example": "direct"}, map[string]any{"$ref": "#/missing"})
	if value != "direct" || !found || code != "" || p.refs != 0 || p.schemaNodes != 0 {
		t.Fatal("direct object example traversed schema")
	}
}

func TestConcretePropertiesAndComposition(t *testing.T) {
	tests := []struct {
		name, schema, want, code string
	}{
		{"available properties", `{"properties":{"known":{"default":0},"nil":{"const":null},"optional":{"type":"string"},"broken":{"$ref":"https://private.invalid"}},"required":["known","nil"]}`, `{"known":0,"nil":null}`, ""},
		{"readOnly omitted", `{"properties":{"id":{"readOnly":true,"example":4},"name":{"example":"x"}},"required":["id","name"]}`, `{"name":"x"}`, ""},
		{"missing required value", `{"properties":{"known":{"const":1},"missing":{"type":"integer"}},"required":["missing"]}`, "", "missing_required_property_example"},
		{"missing required definition", `{"properties":{"known":{"const":1}},"required":["missing"]}`, "", "missing_required_property_example"},
		{"no invented object", `{"type":"object","properties":{"x":{"type":"string"}}}`, "", ""},
		{"boolean true has no hint", `true`, "", ""},
		{"boolean false has no hint", `false`, "", ""},
		{"oneOf first usable", `{"oneOf":[{"$ref":"#/missing"},{"type":"string"},{"const":null},{"example":"later"}]}`, `null`, ""},
		{"anyOf first usable", `{"anyOf":[{"type":"integer"},{"default":"actual"},{"const":"later"}]}`, `"actual"`, ""},
		{"allOf and own properties", `{"allOf":[{"example":{"a":1,"shared":"first"}},{"properties":{"b":{"default":2},"shared":{"const":"second"}}}],"properties":{"shared":{"example":"local"},"c":{"enum":[3]}},"required":["a","b","c"]}`, `{"a":1,"b":2,"c":3,"shared":"local"}`, ""},
		{"allOf missing required blocks synthesis", `{"allOf":[{"properties":{"x":{"type":"integer"}},"required":["x"]},{"example":{"y":2}}]}`, "", "missing_required_property_example"},
		{"unused items are not guessed", `{"type":"array","items":{"default":"item"}}`, "", ""},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p := &preparer{root: map[string]any{}}
			value, found, code := p.example(nil, extractionJSON(t, tt.schema))
			if code != tt.code || found != (tt.want != "") {
				t.Fatalf("found=%v code=%q; want found=%v code=%q", found, code, tt.want != "", tt.code)
			}
			if found && !reflect.DeepEqual(value, extractionJSON(t, tt.want)) {
				t.Fatalf("value=%#v want=%s", value, tt.want)
			}
		})
	}
}

func TestConcreteLocalReferencesAndStructuralSiblings(t *testing.T) {
	root := extractionJSON(t, `{"schemas":{"value":{"example":"base"},"readOnly":{"readOnly":true,"example":"id"},"cycle":{"$ref":"#/schemas/cycle"}},"params":{"base":{"name":"base","schema":{"default":"value"}},"middle":{"$ref":"#/params/base","name":"middle"}},"a/b~c":[{"const":"escaped"}]}`).(map[string]any)
	before, _ := json.Marshal(root)
	p := &preparer{root: root}
	for _, tt := range []struct{ schema, want string }{
		{`{"$ref":"#/schemas/value","default":"sibling"}`, `"sibling"`},
		{`{"properties":{"id":{"$ref":"#/schemas/readOnly"},"known":{"$ref":"#/schemas/value"},"optional":{"$ref":"#/schemas/cycle"}},"required":["id","known"]}`, `{"known":"base"}`},
		{`{"$ref":"#/a~1b~0c/0"}`, `"escaped"`},
		{`{"$ref":"#/missing","properties":{"local":{"const":"usable"}}}`, `{"local":"usable"}`},
	} {
		value, found, code := p.example(nil, extractionJSON(t, tt.schema))
		if !found || code != "" || !reflect.DeepEqual(value, extractionJSON(t, tt.want)) {
			t.Fatalf("schema=%s value=%#v found=%v code=%q", tt.schema, value, found, code)
		}
	}
	resolved, code := p.resolve(map[string]any{"$ref": "#/params/middle", "name": "outer"}, nil)
	if code != "" || resolved["name"] != "outer" || resolved["schema"] == nil {
		t.Fatalf("resolved=%#v code=%q", resolved, code)
	}
	after, _ := json.Marshal(root)
	if string(before) != string(after) {
		t.Fatal("reference resolution changed source document")
	}
}

func TestConcreteReferenceFailuresAndWorkBounds(t *testing.T) {
	for _, tt := range []struct{ ref, code string }{
		{"https://private.invalid", "external_reference"},
		{"file:///private", "external_reference"},
		{"#/missing", "missing_reference"},
		{"#/bad~3token", "invalid_reference"},
		{"#/%zz", "invalid_reference"},
		{"#/values/01", "missing_reference"},
		{"#/cycle", "cyclic_reference"},
	} {
		p := &preparer{root: extractionJSON(t, `{"values":[{"const":1}],"cycle":{"$ref":"#/cycle"}}`).(map[string]any)}
		_, found, code := p.example(nil, map[string]any{"$ref": tt.ref})
		if found || code != tt.code {
			t.Fatalf("ref=%q found=%v code=%q want=%q", tt.ref, found, code, tt.code)
		}
	}
	p := &preparer{root: map[string]any{}, refs: MaxReferences}
	if _, found, code := p.example(nil, map[string]any{"$ref": "#/missing"}); found || code != "reference_limit_exceeded" || !p.exhausted {
		t.Fatalf("reference bound: found=%v code=%q exhausted=%v", found, code, p.exhausted)
	}
	p = &preparer{schemaNodes: MaxNodes}
	if _, found, code := p.example(nil, map[string]any{"default": "x"}); found || code != "schema_work_limit_exceeded" || !p.schemaExhausted {
		t.Fatalf("schema bound: found=%v code=%q exhausted=%v", found, code, p.schemaExhausted)
	}
	var nested any = map[string]any{"const": "leaf"}
	for range MaxDepth {
		nested = map[string]any{"properties": map[string]any{"child": nested}}
	}
	p = &preparer{}
	if _, found, code := p.example(nil, nested); found || code != "schema_depth_exceeded" {
		t.Fatalf("depth bound: found=%v code=%q", found, code)
	}
	chain := make([]string, MaxReferenceDepth)
	p = &preparer{root: map[string]any{}}
	if _, _, code := p.reference("#/missing", chain); code != "reference_depth_exceeded" {
		t.Fatalf("reference depth bound: %q", code)
	}
}

func TestConcreteValueAmplificationIsBoundedBeforeEncoding(t *testing.T) {
	large := strings.Repeat("a", 40000)
	p := &preparer{root: map[string]any{"shared": map[string]any{"example": large}}}
	schema := extractionJSON(t, `{"properties":{"a":{"$ref":"#/shared"},"b":{"$ref":"#/shared"}}}`)
	if _, found, code := p.example(nil, schema); found || code != "concrete_value_limit_exceeded" {
		t.Fatalf("amplification: found=%v code=%q", found, code)
	}
	p = &preparer{}
	object := map[string]any{"examples": map[string]any{
		"a": map[string]any{"value": strings.Repeat("a", contracts.MaxHTTPRequestBodyBytes+1)},
		"b": map[string]any{"value": "small"},
	}}
	if value, found, code := p.example(object, nil); !found || code != "" || value != "small" {
		t.Fatalf("oversized named example did not fall through: value=%v found=%v code=%q", value, found, code)
	}
	if !concreteValueFits(strings.Repeat("a", contracts.MaxHTTPRequestBodyBytes)) {
		t.Fatal("exact plain-text byte bound rejected")
	}
	if concreteValueFits(map[string]any{"x": strings.Repeat("\x00", contracts.MaxHTTPRequestBodyBytes/6)}) {
		t.Fatal("escaped JSON bytes were not bounded")
	}
	cyclic := map[string]any{}
	cyclic["self"] = cyclic
	if concreteValueFits(cyclic) {
		t.Fatal("cyclic concrete value accepted")
	}
}

func TestConcreteCanonicalNegativeZeroSize(t *testing.T) {
	values := make([]any, 22000)
	for i := range values {
		values[i] = math.Copysign(0, -1)
	}
	canonical, err := contracts.MarshalPrivateCanonical(values)
	if err != nil || len(canonical) != 44001 {
		t.Fatalf("canonical negative-zero fixture length=%d err=%v", len(canonical), err)
	}
	if !concreteValueFits(values) {
		t.Fatal("canonical negative zeros inside byte bound were rejected")
	}
}

func TestConcreteMergeWorkIsChargedBeforeCopying(t *testing.T) {
	sample := map[string]any{}
	for i := range 100 {
		sample[strconv.Itoa(i)] = true
	}
	p := &preparer{schemaNodes: MaxNodes - 150}
	schema := map[string]any{"allOf": []any{
		map[string]any{"example": sample},
		map[string]any{"example": sample},
	}}
	if _, found, code := p.example(nil, schema); found || code != "schema_work_limit_exceeded" || !p.schemaExhausted {
		t.Fatalf("repeated allOf copies were not charged: found=%v code=%q exhausted=%v", found, code, p.schemaExhausted)
	}
	p = &preparer{root: map[string]any{"large": sample}, schemaNodes: MaxNodes - 10}
	if _, code := p.resolve(map[string]any{"$ref": "#/large", "sibling": true}, nil); code != "schema_work_limit_exceeded" || !p.schemaExhausted {
		t.Fatalf("reference overlay copy was not charged: code=%q exhausted=%v", code, p.schemaExhausted)
	}
}
