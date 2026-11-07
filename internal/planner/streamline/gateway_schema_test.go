package streamline

import (
	"encoding/json"
	"reflect"
	"testing"

	"google.golang.org/adk/v2/model"
	"google.golang.org/genai"
)

func TestGatewayResponseSchemaNormalization(t *testing.T) {
	nullable := true
	schema := &genai.Schema{
		Type: genai.TypeObject, PropertyOrdering: []string{"rows"}, Required: []string{"rows"},
		Properties: map[string]*genai.Schema{
			"rows": {Type: genai.TypeArray, Items: &genai.Schema{
				Type: genai.TypeObject, Properties: map[string]*genai.Schema{
					"type": {Type: genai.TypeString, Enum: []string{"OBJECT"}},
					"note": {Type: genai.TypeString, Nullable: &nullable},
				},
			}},
			"variant": {AnyOf: []*genai.Schema{{Type: genai.TypeString}, {Type: genai.TypeInteger}}},
			"empty":   {Type: genai.TypeObject},
		},
	}
	before, _ := json.Marshal(schema)
	format, err := encodeResponseFormat(&genai.GenerateContentConfig{ResponseSchema: schema})
	if err != nil {
		t.Fatal(err)
	}
	encoded, _ := json.Marshal(format)
	want := `{
		"type":"json_schema","json_schema":{"name":"response","strict":true,"schema":{
			"type":"object","additionalProperties":false,"required":["empty","rows","variant"],"properties":{
				"empty":{"type":"object","properties":{},"additionalProperties":false,"required":[]},
				"rows":{"type":"array","items":{"type":"object","additionalProperties":false,"required":["note","type"],"properties":{
					"type":{"type":"string","enum":["OBJECT"]},"note":{"type":["string","null"]}
				}}},
				"variant":{"anyOf":[{"type":"string"},{"type":"integer"}]}
			}
		}}
	}`
	var gotValue, wantValue any
	if err := json.Unmarshal(encoded, &gotValue); err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal([]byte(want), &wantValue); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(gotValue, wantValue) {
		t.Fatalf("format=%s, want=%s", encoded, want)
	}
	after, _ := json.Marshal(schema)
	if string(before) != string(after) {
		t.Fatal("adapter mutated the ADK output schema")
	}
}

func TestGatewayResponseFormatOptions(t *testing.T) {
	for _, test := range []struct {
		name    string
		config  *genai.GenerateContentConfig
		want    string
		invalid bool
	}{
		{name: "none", config: &genai.GenerateContentConfig{}, want: "null"},
		{name: "json", config: &genai.GenerateContentConfig{ResponseMIMEType: "application/json"}, want: `{"type":"json_object"}`},
		{name: "raw-schema", config: &genai.GenerateContentConfig{
			ResponseSchema:     compactorOutputSchema(),
			ResponseJsonSchema: map[string]any{"type": "object", "properties": map[string]any{"value": map[string]any{"type": "integer", "maximum": int64(9007199254740993)}}},
		}, want: `{"type":"json_schema","json_schema":{"name":"response","strict":true,"schema":{"additionalProperties":false,"properties":{"value":{"maximum":9007199254740993,"type":"integer"}},"required":["value"],"type":"object"}}}`},
		{name: "invalid-schema", config: &genai.GenerateContentConfig{ResponseJsonSchema: []string{"invalid"}}, invalid: true},
		{name: "unencodable-schema", config: &genai.GenerateContentConfig{ResponseJsonSchema: make(chan int)}, invalid: true},
	} {
		t.Run(test.name, func(t *testing.T) {
			format, err := encodeResponseFormat(test.config)
			if test.invalid {
				if _, err := encodeChatRequest(&model.LLMRequest{Config: test.config}, "model", 10); err == nil {
					t.Fatal("request silently dropped an invalid schema")
				}
				if err == nil {
					t.Fatal("invalid schema accepted")
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			encoded, _ := json.Marshal(format)
			if string(encoded) != test.want {
				t.Fatalf("format=%s, want=%s", encoded, test.want)
			}
		})
	}
}
