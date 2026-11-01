package strictjson

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
)

// Bounds limits the JSON value DecodeBounded reads.
type Bounds struct {
	// MaxDepth is the deepest nesting allowed: the top-level value is at
	// depth 1, and a member value or array element is one deeper than its
	// object or array.
	MaxDepth int
	// MaxNodes bounds the values read, objects and arrays included, and the
	// member names too when NamesAreNodes is set.
	MaxNodes int
	// NamesAreNodes counts each member name as a node at the depth of its
	// value.
	NamesAreNodes bool
	// MaxStringBytes bounds the decoded bytes of every string and member
	// name; zero leaves them unbounded.
	MaxStringBytes int
}

// ErrBounds reports a JSON value nested deeper, with more nodes or with a
// longer string than its Bounds allow.
var ErrBounds = errors.New("JSON value exceeds its bounds")

// DecodeBounded reads exactly one JSON value into a generic tree of
// map[string]any, []any, string, bool and nil, and of number's result for
// each number. It reads token by token and checks the bounds of each node
// before reading it, so it stops at the first node beyond them (ErrBounds)
// without reading the rest. It also rejects a repeated member name after
// unescaping (ErrDuplicateKey), anything after the value (ErrTrailingData), a
// number whose literal number rejects (ErrInvalid) and invalid JSON, which
// keeps encoding/json's error. Strings and names decode as encoding/json
// decodes them; a caller that must reject lone UTF-16 surrogate escapes
// checks ValidUnicodeEscapes first. An empty array decodes to an empty,
// non-nil slice.
func DecodeBounded[N any](data []byte, bounds Bounds, number func(literal string) (N, bool)) (any, error) {
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.UseNumber()
	reader := boundedReader[N]{decoder: decoder, bounds: bounds, number: number}
	value, err := reader.value(1)
	if err != nil {
		return nil, err
	}
	if _, err := decoder.Token(); err != io.EOF {
		if err == nil {
			return nil, ErrTrailingData
		}
		return nil, err
	}
	return value, nil
}

type boundedReader[N any] struct {
	decoder *json.Decoder
	bounds  Bounds
	number  func(string) (N, bool)
	nodes   int
}

// count records one node at depth and reports ErrBounds beyond the bounds.
func (r *boundedReader[N]) count(depth int) error {
	r.nodes++
	if depth > r.bounds.MaxDepth || r.nodes > r.bounds.MaxNodes {
		return ErrBounds
	}
	return nil
}

func (r *boundedReader[N]) text(value string) error {
	if r.bounds.MaxStringBytes > 0 && len(value) > r.bounds.MaxStringBytes {
		return ErrBounds
	}
	return nil
}

func (r *boundedReader[N]) value(depth int) (any, error) {
	if err := r.count(depth); err != nil {
		return nil, err
	}
	token, err := r.decoder.Token()
	if err != nil {
		return nil, err
	}
	switch typed := token.(type) {
	case json.Delim:
		if typed == '{' {
			return r.object(depth)
		}
		if typed == '[' {
			return r.array(depth)
		}
		return nil, fmt.Errorf("%w: unexpected %q", ErrInvalid, rune(typed))
	case json.Number:
		value, ok := r.number(typed.String())
		if !ok {
			return nil, fmt.Errorf("%w: number %s is not accepted", ErrInvalid, typed)
		}
		return value, nil
	case string:
		if err := r.text(typed); err != nil {
			return nil, err
		}
		return typed, nil
	case bool, nil:
		return typed, nil
	default:
		return nil, fmt.Errorf("%w: unexpected token %T", ErrInvalid, token)
	}
}

func (r *boundedReader[N]) object(depth int) (any, error) {
	result := make(map[string]any)
	for r.decoder.More() {
		if r.bounds.NamesAreNodes {
			if err := r.count(depth + 1); err != nil {
				return nil, err
			}
		}
		token, err := r.decoder.Token()
		if err != nil {
			return nil, err
		}
		name, ok := token.(string)
		if !ok {
			return nil, fmt.Errorf("%w: member name is %T", ErrInvalid, token)
		}
		if err := r.text(name); err != nil {
			return nil, err
		}
		if _, duplicate := result[name]; duplicate {
			return nil, ErrDuplicateKey
		}
		member, err := r.value(depth + 1)
		if err != nil {
			return nil, err
		}
		result[name] = member
	}
	if err := r.closing('}'); err != nil {
		return nil, err
	}
	return result, nil
}

func (r *boundedReader[N]) array(depth int) (any, error) {
	result := make([]any, 0)
	for r.decoder.More() {
		element, err := r.value(depth + 1)
		if err != nil {
			return nil, err
		}
		result = append(result, element)
	}
	if err := r.closing(']'); err != nil {
		return nil, err
	}
	return result, nil
}

func (r *boundedReader[N]) closing(want json.Delim) error {
	token, err := r.decoder.Token()
	if err != nil {
		return err
	}
	if token != want {
		return fmt.Errorf("%w: expected %q", ErrInvalid, rune(want))
	}
	return nil
}
