package streamline

import (
	"bytes"
	"encoding/json"
	"fmt"
	"slices"
	"strings"

	"google.golang.org/genai"
)

type chatResponseFormat struct {
	Type       string            `json:"type"`
	JSONSchema *chatOutputSchema `json:"json_schema,omitempty"`
}

type chatOutputSchema struct {
	Name   string         `json:"name"`
	Strict bool           `json:"strict"`
	Schema map[string]any `json:"schema"`
}

// encodeResponseFormat translates the public ADK generation options at the
// provider boundary, as the Python Gateway adapter does. Copy before normalizing:
// an agent can reuse its schema for subsequent requests.
func encodeResponseFormat(config *genai.GenerateContentConfig) (*chatResponseFormat, error) {
	var schema any = config.ResponseJsonSchema
	if schema == nil && config.ResponseSchema != nil {
		schema = config.ResponseSchema
	}
	if schema == nil {
		if config.ResponseMIMEType == "application/json" {
			return &chatResponseFormat{Type: "json_object"}, nil
		}
		return nil, nil
	}
	encoded, err := json.Marshal(schema)
	if err != nil {
		return nil, fmt.Errorf("encode Planner response schema")
	}
	var normalized map[string]any
	// This is our freshly serialized configuration, not an untrusted response.
	decoder := json.NewDecoder(bytes.NewReader(encoded))
	decoder.UseNumber()
	if err := decoder.Decode(&normalized); err != nil || normalized == nil {
		return nil, fmt.Errorf("Planner response schema must be an object")
	}
	normalizeResponseSchema(normalized)
	return &chatResponseFormat{Type: "json_schema", JSONSchema: &chatOutputSchema{
		Name: "response", Strict: true, Schema: normalized,
	}}, nil
}

func normalizeResponseSchema(schema map[string]any) {
	delete(schema, "propertyOrdering")
	object := false
	if kind, ok := schema["type"].(string); ok {
		kind = strings.ToLower(kind)
		object = kind == "object"
		schema["type"] = kind
		if schema["nullable"] == true {
			schema["type"] = []string{kind, "null"}
		}
	}
	delete(schema, "nullable")
	if object && schema["properties"] == nil {
		schema["properties"] = map[string]any{}
	}
	if properties, ok := schema["properties"].(map[string]any); ok {
		schema["additionalProperties"] = false
		required := make([]string, 0, len(properties))
		for name, child := range properties {
			required = append(required, name)
			if nested, ok := child.(map[string]any); ok {
				normalizeResponseSchema(nested)
			}
		}
		slices.Sort(required)
		schema["required"] = required
	}
	if items, ok := schema["items"].(map[string]any); ok {
		normalizeResponseSchema(items)
	}
	for _, keyword := range []string{"anyOf", "oneOf", "allOf"} {
		if alternatives, ok := schema[keyword].([]any); ok {
			for _, alternative := range alternatives {
				if nested, ok := alternative.(map[string]any); ok {
					normalizeResponseSchema(nested)
				}
			}
		}
	}
	for _, keyword := range []string{"$defs", "definitions"} {
		if definitions, ok := schema[keyword].(map[string]any); ok {
			for _, child := range definitions {
				if nested, ok := child.(map[string]any); ok {
					normalizeResponseSchema(nested)
				}
			}
		}
	}
}
