package scanplan

import (
	"context"
	"encoding/json"
	"net/url"
	"strconv"
	"strings"

	"github.com/getkin/kin-openapi/openapi3"
)

func (p *preparer) resolve(value any, chain []string) (map[string]any, string) {
	object, _, code := p.dereference(value, chain)
	return object, code
}

func (p *preparer) dereference(value any, chain []string) (map[string]any, []string, string) {
	for {
		object, ok := value.(map[string]any)
		if !ok {
			return nil, chain, "invalid_reference_object"
		}
		raw, exists := object["$ref"]
		if !exists {
			return object, chain, ""
		}
		if len(object) != 1 {
			return nil, chain, "unsupported_reference_siblings"
		}
		ref, ok := raw.(string)
		if !ok {
			return nil, chain, "invalid_reference"
		}
		if !strings.HasPrefix(ref, "#/") {
			return nil, chain, "external_reference"
		}
		decoded, err := url.PathUnescape(ref[1:])
		if err != nil {
			return nil, chain, "invalid_reference"
		}
		for _, prior := range chain {
			if prior == decoded {
				return nil, chain, "cyclic_reference"
			}
		}
		if len(chain) >= MaxReferenceDepth {
			return nil, chain, "reference_depth_exceeded"
		}
		p.refs++
		if p.refs > MaxReferences {
			p.exhausted = true
			return nil, chain, "reference_limit_exceeded"
		}
		chain = append(append([]string{}, chain...), decoded)
		value = p.root
		for _, part := range strings.Split(decoded[1:], "/") {
			for i := 0; i < len(part); i++ {
				if part[i] == '~' {
					if i+1 >= len(part) || part[i+1] != '0' && part[i+1] != '1' {
						return nil, chain, "invalid_reference"
					}
					i++
				}
			}
			part = strings.ReplaceAll(strings.ReplaceAll(part, "~1", "/"), "~0", "~")
			switch container := value.(type) {
			case map[string]any:
				var exists bool
				value, exists = container[part]
				if !exists {
					return nil, chain, "missing_reference"
				}
			case []any:
				index, err := strconv.Atoi(part)
				if err != nil || index < 0 || index >= len(container) || strconv.Itoa(index) != part {
					return nil, chain, "missing_reference"
				}
				value = container[index]
			default:
				return nil, chain, "missing_reference"
			}
		}
	}
}

// Only schema-bearing fields are traversed. In particular, example/default/enum
// payloads may contain ordinary "$ref" properties and are never dereferenced.
func (p *preparer) schema(value any, chain []string, depth int) (map[string]any, string) {
	if depth > MaxDepth {
		return nil, "schema_depth_exceeded"
	}
	p.schemaNodes++
	if p.schemaNodes > MaxNodes {
		p.schemaExhausted = true
		return nil, "schema_work_limit_exceeded"
	}
	object, next, code := p.dereference(value, chain)
	if code != "" {
		return nil, code
	}
	result := map[string]any{}
	for _, key := range keys(object) {
		value := object[key]
		switch key {
		case "properties":
			properties, ok := value.(map[string]any)
			if !ok {
				return nil, "invalid_schema"
			}
			resolved := map[string]any{}
			for _, name := range keys(properties) {
				child, code := p.schema(properties[name], next, depth+1)
				if code != "" {
					return nil, code
				}
				resolved[name] = child
			}
			result[key] = resolved
		case "items", "not", "additionalProperties":
			if _, ok := value.(bool); ok && key == "additionalProperties" {
				result[key] = value
				continue
			}
			child, code := p.schema(value, next, depth+1)
			if code != "" {
				return nil, code
			}
			result[key] = child
		case "allOf", "anyOf", "oneOf":
			values, ok := value.([]any)
			if !ok {
				return nil, "invalid_schema"
			}
			resolved := []any{}
			for _, entry := range values {
				child, code := p.schema(entry, next, depth+1)
				if code != "" {
					return nil, code
				}
				resolved = append(resolved, child)
			}
			result[key] = resolved
		case "type":
			if _, ok := value.(string); !ok {
				return nil, "unsupported_schema"
			}
			result[key] = value
		case "format":
			if value == "binary" || value == "byte" {
				return nil, "unsupported_binary_schema"
			}
			result[key] = value
		case "title", "description", "example", "default", "enum", "nullable", "readOnly", "writeOnly", "deprecated",
			"minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf", "minLength", "maxLength", "pattern",
			"minItems", "maxItems", "uniqueItems", "minProperties", "maxProperties", "required":
			result[key] = value
		default:
			if !strings.HasPrefix(key, "x-") {
				return nil, "unsupported_schema"
			}
		}
	}
	return result, ""
}

func validateExample(value any, schema map[string]any) string {
	encoded, err := json.Marshal(schema)
	if err != nil {
		return "invalid_schema"
	}
	var parsed openapi3.Schema
	if json.Unmarshal(encoded, &parsed) != nil || parsed.Validate(context.Background()) != nil {
		return "invalid_schema"
	}
	if parsed.VisitJSON(value, openapi3.VisitAsRequest(), openapi3.EnableFormatValidation()) != nil {
		return "invalid_example"
	}
	return ""
}

func (p *preparer) example(object, schema map[string]any) (any, bool, string) {
	if value, exists := object["example"]; exists {
		if _, conflict := object["examples"]; conflict {
			return nil, false, "ambiguous_examples"
		}
		return value, true, ""
	}
	if raw, exists := object["examples"]; exists {
		examples, ok := raw.(map[string]any)
		if !ok {
			return nil, false, "invalid_examples"
		}
		if len(examples) > 0 {
			example, code := p.resolve(examples[keys(examples)[0]], nil)
			if code != "" {
				return nil, false, code
			}
			if _, exists := example["externalValue"]; exists {
				return nil, false, "external_example"
			}
			value, exists := example["value"]
			if !exists {
				return nil, false, "invalid_examples"
			}
			return value, true, ""
		}
	}
	value, exists := schema["example"]
	return value, exists, ""
}
