package scanplan

import (
	"encoding/json"
	"math"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts"
)

var optionsMarshalCalls int

type optionsCustomMarshaler string

func (value optionsCustomMarshaler) MarshalJSON() ([]byte, error) {
	optionsMarshalCalls++
	return []byte(`"unexpected"`), nil
}

type optionsPointerMarshaler int

func (value *optionsPointerMarshaler) MarshalJSON() ([]byte, error) {
	optionsMarshalCalls++
	return []byte(`"unexpected"`), nil
}

type optionsTextKey string

func (value optionsTextKey) MarshalText() ([]byte, error) {
	optionsMarshalCalls++
	return []byte("unexpected"), nil
}

func optionsWithBody(value any) Options {
	return Options{Operations: map[string]OperationInput{
		"#/paths/~1items/post": {Body: &BodyInput{MediaType: "application/json", Value: value}},
	}}
}

func TestValidateOptionsValuesPreservesUnicodeAndSimpleGoTypes(t *testing.T) {
	type name string
	type count int64
	options := Options{
		Server:          "https://example.invalid/ключ",
		ServerVariables: map[string]string{"имя": "значение"},
		Authentication:  map[string]contracts.SecretString{"auth": contracts.NewSecretString("секрет 🔑")},
		Operations: map[string]OperationInput{
			"#/paths/~1ключ/post": {
				Parameters: map[string]any{"query:имя": "значение", "query:count": count(7)},
				Body: &BodyInput{MediaType: "application/json", Value: map[name]any{
					"ключ":    []string{"значение", "🔑"},
					"numbers": []int{1, 2, 3}, "array": [2]uint8{1, 2},
					"bool": true, "null": nil, "float": float32(1.25), "number": json.Number("1e2"),
				}},
			},
		},
	}
	if !validateOptionsValues(options) || !validateOptionsValues(Options{}) {
		t.Fatal("valid JSON-compatible options rejected")
	}
}

func TestValidateOptionsValuesRejectsInvalidUTF8Everywhere(t *testing.T) {
	bad := "secret-canary\xff"
	tests := map[string]Options{
		"server":               {Server: bad},
		"variable key":         {ServerVariables: map[string]string{bad: "value"}},
		"variable value":       {ServerVariables: map[string]string{"name": bad}},
		"authentication key":   {Authentication: map[string]contracts.SecretString{bad: contracts.NewSecretString("value")}},
		"authentication value": {Authentication: map[string]contracts.SecretString{"name": contracts.NewSecretString(bad)}},
		"operation pointer":    {Operations: map[string]OperationInput{bad: {}}},
		"parameter key":        {Operations: map[string]OperationInput{"operation": {Parameters: map[string]any{bad: "value"}}}},
		"parameter value":      {Operations: map[string]OperationInput{"operation": {Parameters: map[string]any{"name": bad}}}},
		"body media type":      {Operations: map[string]OperationInput{"operation": {Body: &BodyInput{MediaType: bad}}}},
		"body scalar":          optionsWithBody(bad),
		"body key":             optionsWithBody(map[string]any{bad: "value"}),
		"body nested value":    optionsWithBody([]any{map[string]any{"name": bad}}),
	}
	for name, options := range tests {
		t.Run(name, func(t *testing.T) {
			if validateOptionsValues(options) {
				t.Fatal("invalid UTF-8 accepted")
			}
		})
	}
}

func TestValidateOptionsValuesRejectsCustomMarshalingAndNonJSONShapes(t *testing.T) {
	value := "text"
	tests := map[string]any{
		"custom value marshaler":  optionsCustomMarshaler("secret"),
		"custom pointer receiver": optionsPointerMarshaler(1),
		"custom typed map value":  map[string]optionsCustomMarshaler{"key": "secret"},
		"custom map key":          map[optionsTextKey]string{"key": "secret"},
		"empty custom slice":      []optionsCustomMarshaler{},
		"empty custom map":        map[string]optionsCustomMarshaler{},
		"arbitrary struct":        struct{ Value string }{"secret"},
		"nested secret":           contracts.NewSecretString("secret"),
		"pointer":                 &value,
		"nil pointer":             (*string)(nil),
		"pointer slice":           []*string{},
		"integer keys":            map[int]string{1: "value"},
		"empty integer keys":      map[int]string{},
		"byte slice":              []byte("secret"),
		"nil byte slice":          []byte(nil),
		"raw JSON":                json.RawMessage(`{"key":"value"}`),
		"function":                func() {},
		"channel":                 make(chan int),
		"complex":                 complex(1, 2),
		"NaN":                     math.NaN(),
		"infinity":                math.Inf(1),
	}
	optionsMarshalCalls = 0
	for name, value := range tests {
		t.Run(name, func(t *testing.T) {
			if validateOptionsValues(optionsWithBody(value)) {
				t.Fatal("non-JSON value shape accepted")
			}
		})
	}
	if optionsMarshalCalls != 0 {
		t.Fatal("validation executed caller-supplied serialization")
	}
}

func TestValidateOptionsValuesBoundsCyclesAndDepth(t *testing.T) {
	cycle := map[string]any{}
	cycle["cycle"] = cycle
	if validateOptionsValues(optionsWithBody(cycle)) {
		t.Fatal("cyclic map accepted")
	}
	slice := make([]any, 1)
	slice[0] = slice
	if validateOptionsValues(optionsWithBody(slice)) {
		t.Fatal("cyclic slice accepted")
	}
	var value any = "leaf"
	// Options/body value starts at depth five.
	for i := 0; i < MaxDepth-5; i++ {
		value = []any{value}
	}
	if !validateOptionsValues(optionsWithBody(value)) {
		t.Fatal("exact maximum depth rejected")
	}
	if validateOptionsValues(optionsWithBody([]any{value})) {
		t.Fatal("excessive depth accepted")
	}
}

func TestValidateOptionsValuesBoundsNodeCountBeforeMarshaling(t *testing.T) {
	// Fixed options fields and one operation/body consume 21 nodes, including
	// the array itself; each array entry consumes one additional node.
	values := make([]int, MaxNodes-21)
	if !validateOptionsValues(optionsWithBody(values)) {
		t.Fatal("exact maximum node count rejected")
	}
	if validateOptionsValues(optionsWithBody(append(values, 0))) {
		t.Fatal("excessive node count accepted")
	}
}

func TestValidateOptionsValuesBoundsAggregateStringBytes(t *testing.T) {
	// Each field is below the byte limit, but their aggregate is not.
	part := strings.Repeat("x", MaxOptionsBytes/3+1)
	options := Options{
		Server:          part,
		ServerVariables: map[string]string{"name": part},
		Authentication:  map[string]contracts.SecretString{"auth": contracts.NewSecretString(part)},
	}
	if validateOptionsValues(options) {
		t.Fatal("aggregate string byte overflow accepted")
	}
	if validateOptionsValues(optionsWithBody(strings.Repeat("ключ", MaxOptionsBytes/4))) {
		t.Fatal("UTF-8 byte count was treated as rune count")
	}
	if validateOptionsValues(optionsWithBody(map[string]any{strings.Repeat("x", MaxOptionsBytes): nil})) {
		t.Fatal("map key bytes were excluded from aggregate limit")
	}
	if !validateOptionsValues(optionsWithBody(strings.Repeat("x", MaxOptionsBytes/2))) {
		t.Fatal("bounded string unexpectedly rejected")
	}
}
