package auditdomain

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
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

	"github.com/ucarion/jcs"
	"go.yaml.in/yaml/v4"
)

var identifierPattern = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._:-]*$`)

func digestBytes(data []byte) string {
	sum := sha256.Sum256(data)
	return "sha256:" + hex.EncodeToString(sum[:])
}

func validDigest(value string) bool {
	if len(value) != len("sha256:")+sha256.Size*2 || !strings.HasPrefix(value, "sha256:") {
		return false
	}
	suffix := strings.TrimPrefix(value, "sha256:")
	decoded, err := hex.DecodeString(suffix)
	return err == nil && hex.EncodeToString(decoded) == suffix
}

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
	if err := decoder.Decode(target); err != nil || requireJSONEOF(decoder) != nil {
		return nil, invalid(CodeInvalid, "json")
	}
	return value, nil
}

func parseStrictJSON(data []byte) (any, error) {
	if len(data) == 0 || len(data) > MaximumDocumentBytes || !utf8.Valid(data) || bytes.IndexByte(data, 0) >= 0 || !validJSONUnicodeEscapes(data) {
		return nil, errors.New("invalid JSON bytes")
	}
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.UseNumber()
	nodes := 0
	value, err := readJSONValue(decoder, 1, &nodes)
	if err != nil {
		return nil, err
	}
	if err := requireJSONEOF(decoder); err != nil {
		return nil, err
	}
	return value, nil
}

func validJSONUnicodeEscapes(data []byte) bool {
	inString := false
	for index := 0; index < len(data); index++ {
		switch data[index] {
		case '"':
			inString = !inString
		case '\\':
			if !inString {
				continue
			}
			if index+1 >= len(data) {
				return false
			}
			if data[index+1] != 'u' {
				index++
				continue
			}
			if index+5 >= len(data) {
				return false
			}
			code, ok := parseHex16(data[index+2 : index+6])
			if !ok {
				return false
			}
			if code >= 0xdc00 && code <= 0xdfff {
				return false
			}
			if code >= 0xd800 && code <= 0xdbff {
				if index+11 >= len(data) || data[index+6] != '\\' || data[index+7] != 'u' {
					return false
				}
				low, valid := parseHex16(data[index+8 : index+12])
				if !valid || low < 0xdc00 || low > 0xdfff {
					return false
				}
				index += 11
				continue
			}
			index += 5
		}
	}
	return !inString
}

func parseHex16(value []byte) (uint16, bool) {
	if len(value) != 4 {
		return 0, false
	}
	var result uint16
	for _, character := range value {
		result <<= 4
		switch {
		case character >= '0' && character <= '9':
			result |= uint16(character - '0')
		case character >= 'a' && character <= 'f':
			result |= uint16(character-'a') + 10
		case character >= 'A' && character <= 'F':
			result |= uint16(character-'A') + 10
		default:
			return 0, false
		}
	}
	return result, true
}

func readJSONValue(decoder *json.Decoder, depth int, nodes *int) (any, error) {
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
				value, valueErr := readJSONValue(decoder, depth+1, nodes)
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
				value, valueErr := readJSONValue(decoder, depth+1, nodes)
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
		value, parseErr := strconv.ParseFloat(typed.String(), 64)
		if parseErr != nil || math.IsInf(value, 0) || math.IsNaN(value) {
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

func requireJSONEOF(decoder *json.Decoder) error {
	var trailing any
	if err := decoder.Decode(&trailing); !errors.Is(err, io.EOF) {
		if err != nil {
			return err
		}
		return errors.New("multiple JSON values")
	}
	return nil
}

func parseJSONOrYAML(data []byte, mediaType string) (map[string]any, error) {
	if len(data) == 0 || len(data) > MaximumDocumentBytes || !utf8.Valid(data) || bytes.IndexByte(data, 0) >= 0 {
		return nil, invalid(CodeLimitExceeded, "document")
	}
	var value any
	var err error
	switch normalizedMediaType(mediaType) {
	case "application/json":
		value, err = parseStrictJSON(data)
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
	case "!!str", "tag:yaml.org,2002:str":
		if !utf8.ValidString(node.Value) || len([]byte(node.Value)) > MaximumStringBytes {
			return nil, errors.New("invalid YAML string")
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
	case "!!int", "tag:yaml.org,2002:int", "!!float", "tag:yaml.org,2002:float":
		var value float64
		if err := node.Decode(&value); err != nil || math.IsInf(value, 0) || math.IsNaN(value) {
			return nil, errors.New("invalid YAML number")
		}
		return value, nil
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
