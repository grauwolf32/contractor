package scanplan

import (
	"encoding/json"
	"net/url"
	"strconv"
	"strings"

	"github.com/grauwolf32/contractor/internal/contracts"
)

func (p *preparer) resolve(value any, chain []string) (map[string]any, string) {
	object, _, code := p.dereference(value, chain)
	return object, code
}

// Structural references keep the strict local JSON Pointer resolver. Sibling
// fields override referenced fields without mutating the source document.
func (p *preparer) dereference(value any, chain []string) (map[string]any, []string, string) {
	siblings := map[string]any{}
	for {
		object, ok := value.(map[string]any)
		if !ok {
			return nil, chain, "invalid_reference_object"
		}
		raw, exists := object["$ref"]
		if !exists {
			if len(siblings) == 0 {
				return object, chain, ""
			}
			merged, code := p.overlay(object, siblings)
			return merged, chain, code
		}
		if code := p.reserveValueWork(len(object) - 1); code != "" {
			return nil, chain, code
		}
		for key, value := range object {
			if _, exists := siblings[key]; key != "$ref" && !exists {
				siblings[key] = value
			}
		}
		var code string
		value, chain, code = p.reference(raw, chain)
		if code != "" {
			return nil, chain, code
		}
	}
}

func (p *preparer) reference(raw any, chain []string) (any, []string, string) {
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
	var value any = p.root
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
	return value, chain, ""
}

func (p *preparer) overlay(base, sibling map[string]any) (map[string]any, string) {
	if code := p.reserveValueWork(len(base) + len(sibling)); code != "" {
		return nil, code
	}
	merged := make(map[string]any, len(base)+len(sibling))
	for key, value := range base {
		merged[key] = value
	}
	for key, value := range sibling {
		if key != "$ref" {
			merged[key] = value
		}
	}
	return merged, ""
}

type concreteValue struct {
	value    any
	found    bool
	readOnly bool
	code     string
}

// Examples are concrete request data, not assertions about schema validity.
// Presence includes explicit null. Hints never cause traversal of payload data.
func (p *preparer) example(object map[string]any, schema any) (resultValue any, found bool, resultCode string) {
	defer func() {
		if found && !concreteValueFits(resultValue) {
			resultValue, found, resultCode = nil, false, "concrete_value_limit_exceeded"
		}
	}()
	if value, exists := object["example"]; exists {
		return value, true, ""
	}
	code := ""
	if raw, exists := object["examples"]; exists {
		if examples, ok := raw.(map[string]any); ok {
			for _, name := range keys(examples) {
				if failure := p.valueWork(1); failure != "" {
					return nil, false, failure
				}
				candidate, ok := examples[name].(map[string]any)
				if ok {
					if value, exists := candidate["value"]; exists {
						if concreteValueFits(value) {
							return value, true, ""
						}
						if code == "" {
							code = "concrete_value_limit_exceeded"
						}
						continue
					}
				}
				candidate, failure := p.resolve(examples[name], nil)
				if failure == "" {
					if value, exists := candidate["value"]; exists {
						if concreteValueFits(value) {
							return value, true, ""
						}
						failure = "concrete_value_limit_exceeded"
					} else {
						failure = "invalid_examples"
						if _, exists := candidate["externalValue"]; exists {
							failure = "external_example"
						}
					}
				}
				if code == "" {
					code = failure
				}
				if p.exhausted {
					return nil, false, failure
				}
			}
		} else {
			code = "invalid_examples"
		}
	}
	result := p.schemaValue(schema, nil, 1, false)
	if result.found || result.code != "" {
		return result.value, result.found, result.code
	}
	return nil, false, code
}

// Reused examples are shared while assembling objects. Measure their expanded
// size before any encoder follows those aliases and allocates the JSON output.
// Strings can be plain-text bodies, so their top-level bound uses raw UTF-8.
// Structured values count JSON punctuation and escaped strings. Final wire
// serialization still enforces its byte limit, including top-level JSON strings.
func concreteValueFits(value any) bool {
	if text, ok := value.(string); ok {
		return len(text) <= contracts.MaxHTTPRequestBodyBytes
	}
	remaining, nodes := contracts.MaxHTTPRequestBodyBytes, 0
	spend := func(size int) bool {
		remaining -= size
		return remaining >= 0
	}
	stringSize := func(text string) bool {
		if !spend(len(text) + 2) {
			return false
		}
		for i := 0; i < len(text); i++ {
			extra := 0
			switch text[i] {
			case '"', '\\', '\b', '\f', '\n', '\r', '\t':
				extra = 1
			default:
				if text[i] < 0x20 {
					extra = 5
				}
			}
			if !spend(extra) {
				return false
			}
		}
		return true
	}
	var visit func(any, int) bool
	visit = func(value any, depth int) bool {
		nodes++
		if nodes > MaxNodes || depth > MaxDepth {
			return false
		}
		switch value := value.(type) {
		case map[string]any:
			if !spend(2) {
				return false
			}
			first := true
			for key, item := range value {
				if !first && !spend(1) {
					return false
				}
				first = false
				if !stringSize(key) || !spend(1) || !visit(item, depth+1) {
					return false
				}
			}
			return true
		case []any:
			if !spend(2) {
				return false
			}
			for i, item := range value {
				if i > 0 && !spend(1) || !visit(item, depth+1) {
					return false
				}
			}
			return true
		case string:
			return stringSize(value)
		case float64:
			// JSON's -0 becomes 0 in the canonical wire representation.
			if value == 0 {
				return spend(1)
			}
			encoded, err := json.Marshal(value)
			return err == nil && spend(len(encoded))
		case nil, bool:
			encoded, err := json.Marshal(value)
			return err == nil && spend(len(encoded))
		default:
			return false
		}
	}
	return visit(value, 1)
}

func (p *preparer) valueWork(depth int) string {
	if depth > MaxDepth {
		return "schema_depth_exceeded"
	}
	return p.reserveValueWork(1)
}

func (p *preparer) reserveValueWork(count int) string {
	if count > MaxNodes-p.schemaNodes {
		p.schemaNodes = MaxNodes + 1
		p.schemaExhausted = true
		return "schema_work_limit_exceeded"
	}
	p.schemaNodes += count
	return ""
}

func schemaHint(object map[string]any) (any, bool) {
	if value, exists := object["example"]; exists {
		return value, true
	}
	if values, ok := object["examples"].([]any); ok && len(values) > 0 {
		return values[0], true
	}
	for _, key := range []string{"default", "const"} {
		if value, exists := object[key]; exists {
			return value, true
		}
	}
	if values, ok := object["enum"].([]any); ok && len(values) > 0 {
		return values[0], true
	}
	return nil, false
}

func (p *preparer) schemaValue(value any, chain []string, depth int, property bool) concreteValue {
	if code := p.valueWork(depth); code != "" {
		return concreteValue{code: code}
	}
	object, ok := value.(map[string]any)
	if !ok {
		// Boolean schemas and type-only schemas contain no concrete data.
		return concreteValue{}
	}
	if property && object["readOnly"] == true {
		return concreteValue{readOnly: true}
	}
	if value, exists := schemaHint(object); exists {
		return concreteValue{value: value, found: true}
	}
	if raw, exists := object["$ref"]; exists {
		target, next, code := p.reference(raw, chain)
		if code == "" {
			if targetObject, ok := target.(map[string]any); ok {
				target, code = p.overlay(targetObject, object)
			} else if len(object) > 1 {
				target, code = p.overlay(nil, object)
			}
			if code != "" {
				return concreteValue{code: code}
			}
			return p.schemaValue(target, next, depth+1, property)
		}
		if p.exhausted {
			return concreteValue{code: code}
		}
		// A broken reference cannot hide usable sibling property/branch hints.
		result := p.objectValue(object, chain, depth)
		if result.found || result.code != "" {
			return result
		}
		return concreteValue{code: code}
	}
	return p.objectValue(object, chain, depth)
}

// allOf merges available object samples in array order; later branches win,
// then local properties win. This is data extraction, not schema intersection.
// Missing required data in a branch conservatively prevents synthesis.
func (p *preparer) objectValue(object map[string]any, chain []string, depth int) concreteValue {
	result := map[string]any{}
	readOnly := map[string]bool{}
	found, missingRequired := false, false
	code := ""
	if branches, ok := object["allOf"].([]any); ok {
		for _, branch := range branches {
			value := p.schemaValue(branch, chain, depth+1, false)
			if p.schemaExhausted || p.exhausted {
				return value
			}
			if sample, ok := value.value.(map[string]any); value.found && ok {
				if code := p.reserveValueWork(len(sample)); code != "" {
					return concreteValue{code: code}
				}
				for name, entry := range sample {
					result[name] = entry
				}
				found = true
			}
			if value.code == "missing_required_property_example" {
				missingRequired = true
			}
			if code == "" {
				code = value.code
			}
		}
	}
	if properties, ok := object["properties"].(map[string]any); ok {
		for _, name := range keys(properties) {
			value := p.schemaValue(properties[name], chain, depth+1, true)
			if p.schemaExhausted || p.exhausted {
				return value
			}
			if value.readOnly {
				readOnly[name] = true
				delete(result, name)
			} else if value.found {
				result[name] = value.value
				found = true
			}
			if code == "" {
				code = value.code
			}
		}
	}
	if required, ok := object["required"].([]any); ok {
		for _, raw := range required {
			if name, ok := raw.(string); ok && !readOnly[name] {
				if _, exists := result[name]; !exists {
					missingRequired = true
				}
			}
		}
	}
	if found && !missingRequired {
		return concreteValue{value: result, found: true}
	}
	for _, keyword := range []string{"oneOf", "anyOf"} {
		if branches, ok := object[keyword].([]any); ok {
			for _, branch := range branches {
				value := p.schemaValue(branch, chain, depth+1, false)
				if value.found || p.schemaExhausted || p.exhausted {
					return value
				}
				if code == "" {
					code = value.code
				}
			}
		}
	}
	if missingRequired {
		return concreteValue{code: "missing_required_property_example"}
	}
	return concreteValue{code: code}
}
