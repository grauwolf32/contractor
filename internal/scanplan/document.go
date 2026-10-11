package scanplan

import (
	"bytes"
	"encoding/json"
	"errors"
	"io"
	"strconv"
	"strings"
	"unicode/utf8"

	"github.com/grauwolf32/contractor/internal/documentnumber"
	"github.com/grauwolf32/contractor/internal/strictjson"
	"go.yaml.in/yaml/v4"
	"go.yaml.in/yaml/v4/plugin/limit"
)

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
			if node.Style == 0 && documentnumber.LooksNumeric(node.Value) {
				break
			}
			return node.Value, nil
		case "!!timestamp":
			// OpenAPI/JSON carry dates as strings. Preserve the source spelling.
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
			if value, ok := documentnumber.Integer(node.Value); ok {
				return value, nil
			}
		case "!!float":
			if value, ok := documentnumber.Float(strings.ReplaceAll(node.Value, "_", "")); ok {
				return value, nil
			}
		}
	}
	return nil, &PreparationError{Code: "invalid_document"}
}

// parseJSONDocument applies the bounds the YAML reader applies: member names
// count as nodes, as YAML keys do.
func parseJSONDocument(data []byte) (map[string]any, error) {
	if !strictjson.ValidUnicodeEscapes(data) {
		return nil, &PreparationError{Code: "invalid_document"}
	}
	value, err := strictjson.DecodeBounded(data, strictjson.Bounds{
		MaxDepth: MaxDepth, MaxNodes: MaxNodes, NamesAreNodes: true,
	}, documentnumber.Float)
	if errors.Is(err, strictjson.ErrBounds) {
		return nil, &PreparationError{Code: "source_limit_exceeded"}
	}
	if err != nil {
		return nil, &PreparationError{Code: "invalid_document"}
	}
	result, ok := value.(map[string]any)
	if !ok {
		return nil, &PreparationError{Code: "invalid_document"}
	}
	return result, nil
}
