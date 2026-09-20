package scanplan

import (
	"bytes"
	"encoding/json"
	"io"
	"math"
	"regexp"
	"strconv"
	"strings"
	"unicode/utf8"

	"go.yaml.in/yaml/v4"
	"go.yaml.in/yaml/v4/plugin/limit"
)

const maximumDocumentNumber = 1<<53 - 1

var numericYAMLScalar = regexp.MustCompile(`^[+-]?(?:0[xX][0-9a-fA-F_]+|0[oO][0-7_]+|0[bB][01_]+|(?:[0-9][0-9_]*(?:\.[0-9_]*)?|\.[0-9][0-9_]*)(?:[eE][+-]?[0-9]+)?)$`)

// parseDocument accepts one bounded JSON-compatible mapping. Both parsers
// retain numeric spelling until bounds have been checked, so unsafe
// integer examples never pass through an already-rounded float64.
func parseDocument(data []byte, mediaType string) (map[string]any, error) {
	switch mediaType {
	case "application/json", "application/yaml", "application/x-yaml", "text/yaml":
	default:
		return nil, &PreparationError{Code: "unsupported_media_type"}
	}
	if len(data) > MaxSourceBytes {
		return nil, &PreparationError{Code: "source_limit_exceeded"}
	}
	if !utf8.Valid(data) {
		return nil, &PreparationError{Code: "invalid_document"}
	}
	if mediaType == "application/json" || json.Valid(data) {
		return parseJSONDocument(data)
	}

	depthExceeded := false
	loader, err := yaml.NewLoader(bytes.NewReader(data), yaml.WithUniqueKeys(),
		yaml.WithPlugin(limit.New(limit.AliasValue(0), limit.DepthFunc(
			func(depth int, _ *yaml.DepthContext) error {
				if depth > MaxDepth {
					depthExceeded = true
					return &PreparationError{Code: "source_limit_exceeded"}
				}
				return nil
			},
		))),
	)
	if err != nil {
		return nil, &PreparationError{Code: "invalid_document"}
	}
	var document yaml.Node
	if err := loader.Load(&document); err != nil {
		if depthExceeded {
			return nil, &PreparationError{Code: "source_limit_exceeded"}
		}
		return nil, &PreparationError{Code: "invalid_document"}
	}
	var trailing yaml.Node
	if err := loader.Load(&trailing); err != io.EOF {
		return nil, &PreparationError{Code: "invalid_document"}
	}
	if document.Kind != yaml.DocumentNode || len(document.Content) != 1 ||
		document.Content[0].Kind != yaml.MappingNode {
		return nil, &PreparationError{Code: "invalid_document"}
	}
	reader := documentReader{}
	value, err := reader.read(document.Content[0], 1)
	if err != nil {
		return nil, err
	}
	return value.(map[string]any), nil
}

type documentReader struct{ nodes int }

func (r *documentReader) count(depth int) error {
	r.nodes++
	if depth > MaxDepth || r.nodes > MaxNodes {
		return &PreparationError{Code: "source_limit_exceeded"}
	}
	return nil
}

func (r *documentReader) read(node *yaml.Node, depth int) (any, error) {
	if err := r.count(depth); err != nil {
		return nil, err
	}
	if node.Anchor != "" || node.Alias != nil || node.Kind == yaml.AliasNode {
		return nil, &PreparationError{Code: "invalid_document"}
	}
	if len(node.Content) > MaxNodes-r.nodes {
		return nil, &PreparationError{Code: "source_limit_exceeded"}
	}
	switch node.Kind {
	case yaml.MappingNode:
		if node.ShortTag() != "!!map" || len(node.Content)%2 != 0 {
			break
		}
		result := make(map[string]any, len(node.Content)/2)
		for i := 0; i < len(node.Content); i += 2 {
			key := node.Content[i]
			if key.Kind != yaml.ScalarNode || key.ShortTag() != "!!str" {
				return nil, &PreparationError{Code: "invalid_document"}
			}
			if _, err := r.read(key, depth+1); err != nil {
				return nil, err
			}
			if _, duplicate := result[key.Value]; duplicate {
				return nil, &PreparationError{Code: "invalid_document"}
			}
			value, err := r.read(node.Content[i+1], depth+1)
			if err != nil {
				return nil, err
			}
			result[key.Value] = value
		}
		return result, nil
	case yaml.SequenceNode:
		if node.ShortTag() != "!!seq" {
			break
		}
		result := make([]any, 0, len(node.Content))
		for _, child := range node.Content {
			value, err := r.read(child, depth+1)
			if err != nil {
				return nil, err
			}
			result = append(result, value)
		}
		return result, nil
	case yaml.ScalarNode:
		switch node.ShortTag() {
		case "!!str":
			// The YAML resolver falls back to a string when numeric conversion
			// overflows. Quoted and explicitly tagged strings are ordinary data.
			if node.Style == 0 && numericYAMLScalar.MatchString(node.Value) {
				break
			}
			return node.Value, nil
		case "!!null":
			switch node.Value {
			case "", "~", "null", "Null", "NULL":
				return nil, nil
			}
		case "!!bool":
			value, err := strconv.ParseBool(node.Value)
			if err == nil {
				return value, nil
			}
		case "!!int":
			value, err := strconv.ParseInt(strings.ReplaceAll(node.Value, "_", ""), 0, 64)
			if err == nil && value >= -maximumDocumentNumber && value <= maximumDocumentNumber {
				return float64(value), nil
			}
		case "!!float":
			if value, ok := documentFloat(strings.ReplaceAll(node.Value, "_", "")); ok {
				return value, nil
			}
		}
	}
	return nil, &PreparationError{Code: "invalid_document"}
}

func parseJSONDocument(data []byte) (map[string]any, error) {
	if !validJSONUnicode(data) {
		return nil, &PreparationError{Code: "invalid_document"}
	}
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.UseNumber()
	reader := documentReader{}
	value, err := reader.readJSON(decoder, 1)
	if err != nil {
		return nil, err
	}
	if _, err := decoder.Token(); err != io.EOF {
		return nil, &PreparationError{Code: "invalid_document"}
	}
	result, ok := value.(map[string]any)
	if !ok {
		return nil, &PreparationError{Code: "invalid_document"}
	}
	return result, nil
}

func (r *documentReader) readJSON(decoder *json.Decoder, depth int) (any, error) {
	if err := r.count(depth); err != nil {
		return nil, err
	}
	token, err := decoder.Token()
	if err != nil {
		return nil, &PreparationError{Code: "invalid_document"}
	}
	switch token := token.(type) {
	case json.Delim:
		switch token {
		case '{':
			result := make(map[string]any)
			for decoder.More() {
				if err := r.count(depth + 1); err != nil {
					return nil, err
				}
				keyToken, err := decoder.Token()
				key, ok := keyToken.(string)
				if err != nil || !ok {
					return nil, &PreparationError{Code: "invalid_document"}
				}
				if _, duplicate := result[key]; duplicate {
					return nil, &PreparationError{Code: "invalid_document"}
				}
				value, err := r.readJSON(decoder, depth+1)
				if err != nil {
					return nil, err
				}
				result[key] = value
			}
			if closing, err := decoder.Token(); err == nil && closing == json.Delim('}') {
				return result, nil
			}
		case '[':
			result := make([]any, 0)
			for decoder.More() {
				value, err := r.readJSON(decoder, depth+1)
				if err != nil {
					return nil, err
				}
				result = append(result, value)
			}
			if closing, err := decoder.Token(); err == nil && closing == json.Delim(']') {
				return result, nil
			}
		}
	case json.Number:
		if value, ok := documentFloat(token.String()); ok {
			return value, nil
		}
	case string, bool, nil:
		return token, nil
	}
	return nil, &PreparationError{Code: "invalid_document"}
}

// encoding/json replaces unpaired UTF-16 escapes with U+FFFD. Reject them
// before decoding while preserving valid pairs and literal escaped backslashes.
func validJSONUnicode(data []byte) bool {
	for i := 0; i < len(data); i++ {
		if data[i] != '\\' {
			continue
		}
		if i+1 >= len(data) {
			return false
		}
		if data[i+1] != 'u' {
			i++
			continue
		}
		if i+6 > len(data) {
			return false
		}
		code, err := strconv.ParseUint(string(data[i+2:i+6]), 16, 16)
		if err != nil || code >= 0xdc00 && code <= 0xdfff {
			return false
		}
		if code >= 0xd800 && code <= 0xdbff {
			if i+12 > len(data) || data[i+6] != '\\' || data[i+7] != 'u' {
				return false
			}
			low, err := strconv.ParseUint(string(data[i+8:i+12]), 16, 16)
			if err != nil || low < 0xdc00 || low > 0xdfff {
				return false
			}
			i += 6
		}
		i += 5
	}
	return true
}

// documentFloat checks the exact decimal magnitude before accepting the
// float64. Comparing only the rounded value would admit 9007199254740991.1.
func documentFloat(text string) (float64, bool) {
	if !numericYAMLScalar.MatchString(text) {
		return 0, false
	}
	value, err := strconv.ParseFloat(text, 64)
	if err != nil || math.IsNaN(value) || math.IsInf(value, 0) || math.Abs(value) > maximumDocumentNumber {
		return 0, false
	}
	abs := strings.TrimPrefix(strings.TrimPrefix(text, "+"), "-")
	mantissa, exponentText, _ := strings.Cut(strings.ToLower(abs), "e")
	integer, fraction, _ := strings.Cut(mantissa, ".")
	digits := strings.TrimLeft(integer+fraction, "0")
	if digits == "" {
		return value, true
	}
	// Underflow would silently turn an explicit nonzero example into zero.
	if value == 0 {
		return 0, false
	}
	exponent := int64(0)
	if exponentText != "" {
		exponent, err = strconv.ParseInt(exponentText, 10, 32)
		if err != nil {
			return 0, false
		}
	}
	integerDigits := int64(len(digits)-len(fraction)) + exponent
	if integerDigits > 16 {
		return 0, false
	}
	if integerDigits == 16 {
		whole := digits
		if len(whole) < 16 {
			whole += strings.Repeat("0", 16-len(whole))
		} else {
			whole = whole[:16]
		}
		if whole > "9007199254740991" ||
			(whole == "9007199254740991" && len(digits) > 16 && strings.Trim(digits[16:], "0") != "") {
			return 0, false
		}
	}
	return value, true
}
