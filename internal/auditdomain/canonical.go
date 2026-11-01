package auditdomain

import (
	"bytes"
	"encoding/json"
	"errors"
	"io"
	"math"
	"mime"
	"regexp"
	"strconv"
	"strings"
	"unicode/utf8"

	"github.com/grauwolf32/contractor/internal/helpers/documentnumber"
	"github.com/grauwolf32/contractor/internal/helpers/documentyaml"
	"github.com/grauwolf32/contractor/internal/helpers/strictjson"
	"github.com/ucarion/jcs"
	"go.yaml.in/yaml/v4"
)

var identifierPattern = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._:-]*$`)

func canonicalJSON(value any) ([]byte, error) {
	return canonicalJSONWithBounds(value, strictJSONBounds)
}

// canonicalOpenAPIJSON preserves source strings in generated operation tasks
// and inventory bases. Their enclosing Audit document bounds still apply.
func canonicalOpenAPIJSON(value any) ([]byte, error) {
	if err := checkOpenAPIEncoding(value); err != nil {
		return nil, err
	}
	return canonicalJSONWithBounds(value, openAPIOutputBounds)
}

func canonicalJSONWithBounds(value any, bounds strictjson.Bounds) ([]byte, error) {
	var buffer bytes.Buffer
	encoder := json.NewEncoder(&buffer)
	// HTML escaping would inflate an otherwise bounded OpenAPI description
	// before JCS removes those escapes again.
	encoder.SetEscapeHTML(false)
	if err := encoder.Encode(value); err != nil {
		return nil, err
	}
	normalized, err := parseJSONWithBounds(bytes.TrimSuffix(buffer.Bytes(), []byte("\n")), bounds, finiteJSONNumber)
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
	return decodeJSONWithBounds(data, target, strictJSONBounds)
}

func decodeJSONWithBounds(data []byte, target any, bounds strictjson.Bounds) (any, error) {
	value, err := parseJSONWithBounds(data, bounds, finiteJSONNumber)
	if err != nil {
		return nil, invalid(CodeInvalid, "json")
	}
	normalized, err := json.Marshal(value)
	if err != nil {
		return nil, invalid(CodeInvalid, "json")
	}
	if err := strictjson.Decode(normalized, target); err != nil {
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
	return parseJSONWithBounds(data, strictJSONBounds, number)
}

func parseJSONWithBounds(data []byte, bounds strictjson.Bounds, number func(string) (float64, bool)) (any, error) {
	if len(data) == 0 || len(data) > MaximumDocumentBytes || !utf8.Valid(data) || bytes.IndexByte(data, 0) >= 0 || !strictjson.ValidUnicodeEscapes(data) {
		return nil, errors.New("invalid JSON bytes")
	}
	return strictjson.DecodeBounded(data, bounds, number)
}

// strictJSONBounds counts values, not member names, as jsonExtentOf does.
var strictJSONBounds = strictjson.Bounds{
	MaxDepth: MaximumJSONDepth, MaxNodes: MaximumJSONNodes, MaxStringBytes: MaximumStringBytes,
}

// Source documents use scanplan's shared source bounds. After resolution,
// generated Audit documents retain LIM-AUD-14/17/30 without a string cap.
var openAPIOutputBounds = strictjson.Bounds{
	MaxDepth: MaximumJSONDepth, MaxNodes: MaximumJSONNodes,
}

func finiteJSONNumber(text string) (float64, bool) {
	value, err := strconv.ParseFloat(text, 64)
	return value, err == nil && !math.IsInf(value, 0) && !math.IsNaN(value)
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
		if !documentyaml.Collection(node) || len(node.Content)%2 != 0 {
			return nil, errors.New("invalid YAML mapping")
		}
		result := make(map[string]any, len(node.Content)/2)
		for index := 0; index < len(node.Content); index += 2 {
			keyNode := node.Content[index]
			if keyNode.Anchor != "" {
				return nil, errors.New("YAML aliases are forbidden")
			}
			name, ok := documentyaml.Key(keyNode)
			if !ok || len([]byte(name)) > MaximumStringBytes {
				return nil, errors.New("YAML mapping keys must be strings")
			}
			if _, duplicate := result[name]; duplicate {
				return nil, errors.New("duplicate YAML key")
			}
			value, err := yamlNodeValue(node.Content[index+1], depth+1, nodes)
			if err != nil {
				return nil, err
			}
			result[name] = value
		}
		return result, nil
	case yaml.SequenceNode:
		if !documentyaml.Collection(node) {
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
		value, ok := documentyaml.Scalar(node)
		if !ok {
			return nil, errors.New("invalid YAML scalar")
		}
		if text, isString := value.(string); isString && (!utf8.ValidString(text) || len([]byte(text)) > MaximumStringBytes) {
			return nil, errors.New("invalid YAML string")
		}
		return value, nil
	default:
		return nil, errors.New("invalid YAML node")
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

func validateOpenAPISourceText(value, field string, required bool) error {
	if !utf8.ValidString(value) || strings.ContainsRune(value, 0) || len([]byte(value)) > MaximumDocumentBytes || required && strings.TrimSpace(value) == "" {
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
