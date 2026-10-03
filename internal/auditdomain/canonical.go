package auditdomain

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"mime"
	"regexp"
	"strconv"
	"strings"
	"unicode/utf8"

	"github.com/grauwolf32/contractor/internal/documentnumber"
	"github.com/grauwolf32/contractor/internal/strictjson"
	"github.com/ucarion/jcs"
	"go.yaml.in/yaml/v4"
)

var identifierPattern = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._:-]*$`)

func canonicalJSON(value any) ([]byte, error) {
	encoded, err := json.Marshal(value)
	if err != nil {
		return nil, err
	}
	normalized, err := parseStrictJSON(encoded)
	if err != nil {
		return nil, err
	}
	formatted, err := jcs.Format(normalized)
	if err != nil {
		return nil, err
	}
	return []byte(formatted), nil
}

func decodeStrictJSON(data []byte, target any) (any, error) {
	value, err := parseStrictJSON(data)
	if err != nil {
		return nil, invalid(CodeInvalid, "json")
	}
	normalized, err := json.Marshal(value)
	if err != nil {
		return nil, invalid(CodeInvalid, "json")
	}
	decoder := json.NewDecoder(bytes.NewReader(normalized))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(target); err != nil || strictjson.RequireEOF(decoder) != nil {
		return nil, invalid(CodeInvalid, "json")
	}
	return value, nil
}

func parseStrictJSON(data []byte) (any, error) {
	return parseStrictJSONNumbers(data, finiteJSONNumber)
}

// parseStrictJSONNumbers lets source documents apply the same safe-integer
// number bound as their YAML form, while canonical encodings of Server values
// keep accepting every finite float64.
func parseStrictJSONNumbers(data []byte, number func(string) (float64, bool)) (any, error) {
	if len(data) == 0 || len(data) > MaximumDocumentBytes || !utf8.Valid(data) || bytes.IndexByte(data, 0) >= 0 || !strictjson.ValidUnicodeEscapes(data) {
		return nil, errors.New("invalid JSON bytes")
	}
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.UseNumber()
	nodes := 0
	value, err := readJSONValue(decoder, 1, &nodes, number)
	if err != nil {
		return nil, err
	}
	if err := strictjson.RequireEOF(decoder); err != nil {
		return nil, err
	}
	return value, nil
}

func finiteJSONNumber(text string) (float64, bool) {
	value, err := strconv.ParseFloat(text, 64)
	return value, err == nil && !math.IsInf(value, 0) && !math.IsNaN(value)
}

func readJSONValue(decoder *json.Decoder, depth int, nodes *int, number func(string) (float64, bool)) (any, error) {
	*nodes++
	if depth > MaximumJSONDepth || *nodes > MaximumJSONNodes {
		return nil, errors.New("JSON limit exceeded")
	}
	token, err := decoder.Token()
	if err != nil {
		return nil, err
	}
	switch typed := token.(type) {
	case json.Delim:
		switch typed {
		case '{':
			result := make(map[string]any)
			for decoder.More() {
				keyToken, keyErr := decoder.Token()
				if keyErr != nil {
					return nil, keyErr
				}
				key, ok := keyToken.(string)
				if !ok || !utf8.ValidString(key) || len([]byte(key)) > MaximumStringBytes {
					return nil, errors.New("invalid JSON key")
				}
				if _, duplicate := result[key]; duplicate {
					return nil, errors.New("duplicate JSON key")
				}
				value, valueErr := readJSONValue(decoder, depth+1, nodes, number)
				if valueErr != nil {
					return nil, valueErr
				}
				result[key] = value
			}
			closing, closeErr := decoder.Token()
			if closeErr != nil || closing != json.Delim('}') {
				return nil, errors.New("invalid JSON object")
			}
			return result, nil
		case '[':
			result := make([]any, 0)
			for decoder.More() {
				value, valueErr := readJSONValue(decoder, depth+1, nodes, number)
				if valueErr != nil {
					return nil, valueErr
				}
				result = append(result, value)
			}
			closing, closeErr := decoder.Token()
			if closeErr != nil || closing != json.Delim(']') {
				return nil, errors.New("invalid JSON array")
			}
			return result, nil
		default:
			return nil, errors.New("unexpected JSON delimiter")
		}
	case json.Number:
		value, ok := number(typed.String())
		if !ok {
			return nil, errors.New("invalid JSON number")
		}
		return value, nil
	case string:
		if !utf8.ValidString(typed) || len([]byte(typed)) > MaximumStringBytes {
			return nil, errors.New("invalid JSON string")
		}
		return typed, nil
	case bool, nil:
		return typed, nil
	default:
		return nil, errors.New("invalid JSON token")
	}
}

func parseJSONOrYAML(data []byte, mediaType string) (map[string]any, error) {
	if len(data) == 0 || len(data) > MaximumDocumentBytes || !utf8.Valid(data) || bytes.IndexByte(data, 0) >= 0 {
		return nil, invalid(CodeLimitExceeded, "document")
	}
	var value any
	var err error
	switch normalizedMediaType(mediaType) {
	case "application/json":
		value, err = parseStrictJSONNumbers(data, documentnumber.Float)
	case "application/yaml", "application/x-yaml", "text/yaml":
		value, err = parseStrictYAML(data)
	default:
		return nil, invalid(CodeInvalid, "media_type")
	}
	if err != nil {
		return nil, invalid(CodeInvalid, "document")
	}
	root, ok := value.(map[string]any)
	if !ok {
		return nil, invalid(CodeInvalid, "document")
	}
	return root, nil
}

func parseStrictYAML(data []byte) (any, error) {
	decoder := yaml.NewDecoder(bytes.NewReader(data))
	var document yaml.Node
	if err := decoder.Decode(&document); err != nil {
		return nil, err
	}
	var trailing yaml.Node
	if err := decoder.Decode(&trailing); !errors.Is(err, io.EOF) {
		return nil, errors.New("multiple YAML documents")
	}
	if document.Kind != yaml.DocumentNode || len(document.Content) != 1 {
		return nil, errors.New("invalid YAML document")
	}
	nodes := 0
	return yamlNodeValue(document.Content[0], 1, &nodes)
}

func yamlNodeValue(node *yaml.Node, depth int, nodes *int) (any, error) {
	*nodes++
	if depth > MaximumJSONDepth || *nodes > MaximumJSONNodes || node == nil {
		return nil, errors.New("YAML limit exceeded")
	}
	if node.Alias != nil || node.Anchor != "" || node.Kind == yaml.AliasNode {
		return nil, errors.New("YAML aliases are forbidden")
	}
	switch node.Kind {
	case yaml.MappingNode:
		if node.Tag != "!!map" && node.Tag != "tag:yaml.org,2002:map" || len(node.Content)%2 != 0 {
			return nil, errors.New("invalid YAML mapping")
		}
		result := make(map[string]any, len(node.Content)/2)
		for index := 0; index < len(node.Content); index += 2 {
			keyNode := node.Content[index]
			if keyNode.Kind != yaml.ScalarNode || keyNode.Tag != "!!str" && keyNode.Tag != "tag:yaml.org,2002:str" || len([]byte(keyNode.Value)) > MaximumStringBytes {
				return nil, errors.New("YAML mapping keys must be strings")
			}
			if _, duplicate := result[keyNode.Value]; duplicate {
				return nil, errors.New("duplicate YAML key")
			}
			value, err := yamlNodeValue(node.Content[index+1], depth+1, nodes)
			if err != nil {
				return nil, err
			}
			result[keyNode.Value] = value
		}
		return result, nil
	case yaml.SequenceNode:
		if node.Tag != "!!seq" && node.Tag != "tag:yaml.org,2002:seq" {
			return nil, errors.New("invalid YAML sequence")
		}
		result := make([]any, 0, len(node.Content))
		for _, child := range node.Content {
			value, err := yamlNodeValue(child, depth+1, nodes)
			if err != nil {
				return nil, err
			}
			result = append(result, value)
		}
		return result, nil
	case yaml.ScalarNode:
		return yamlScalarValue(node)
	default:
		return nil, errors.New("invalid YAML node")
	}
}

func yamlScalarValue(node *yaml.Node) (any, error) {
	switch node.Tag {
	case "!!str", "tag:yaml.org,2002:str", "!!timestamp", "tag:yaml.org,2002:timestamp":
		if !utf8.ValidString(node.Value) || len([]byte(node.Value)) > MaximumStringBytes {
			return nil, errors.New("invalid YAML string")
		}
		if node.ShortTag() == "!!str" && node.Style == 0 && documentnumber.LooksNumeric(node.Value) {
			return nil, errors.New("invalid YAML number")
		}
		return node.Value, nil
	case "!!null", "tag:yaml.org,2002:null":
		return nil, nil
	case "!!bool", "tag:yaml.org,2002:bool":
		value, err := strconv.ParseBool(strings.ToLower(node.Value))
		if err != nil {
			return nil, err
		}
		return value, nil
	case "!!int", "tag:yaml.org,2002:int":
		if value, ok := documentnumber.Integer(node.Value); ok {
			return value, nil
		}
		return nil, errors.New("invalid YAML number")
	case "!!float", "tag:yaml.org,2002:float":
		if value, ok := documentnumber.Float(strings.ReplaceAll(node.Value, "_", "")); ok {
			return value, nil
		}
		return nil, errors.New("invalid YAML number")
	default:
		return nil, fmt.Errorf("unsupported YAML tag")
	}
}

func normalizedMediaType(value string) string {
	mediaType, _, err := mime.ParseMediaType(value)
	if err != nil {
		return ""
	}
	return strings.ToLower(mediaType)
}

func validateIdentifier(value, field string) error {
	if len(value) == 0 || len([]byte(value)) > MaximumIdentifierBytes || !utf8.ValidString(value) || !identifierPattern.MatchString(value) {
		return invalid(CodeInvalid, field)
	}
	return nil
}

func validateText(value, field string, required bool) error {
	if !utf8.ValidString(value) || strings.ContainsRune(value, 0) || len([]byte(value)) > MaximumStringBytes || required && strings.TrimSpace(value) == "" {
		return invalid(CodeInvalid, field)
	}
	return nil
}

func copyStringMap(source map[string]string) map[string]string {
	if len(source) == 0 {
		return nil
	}
	result := make(map[string]string, len(source))
	for key, value := range source {
		result[key] = value
	}
	return result
}
