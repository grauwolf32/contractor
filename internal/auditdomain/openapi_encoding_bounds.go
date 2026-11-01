package auditdomain

import (
	"encoding"
	"encoding/json"
	"errors"
	"reflect"
	"strings"
	"unicode/utf8"

	"github.com/grauwolf32/contractor/internal/helpers/strictjson"
)

// checkOpenAPIEncoding bounds the tree before encoding/json builds its internal
// buffer. Reference expansion can repeat a large source string many times while
// using few nodes. Counting the encoded bytes without materializing them stops
// that work at the output-document bound, including repeated names and strings.
func checkOpenAPIEncoding(value any) error {
	budget := openAPIEncodingBudget{remaining: MaximumDocumentBytes}
	return budget.value(reflect.ValueOf(value), 1)
}

type openAPIEncodingBudget struct {
	remaining int
	nodes     int
}

func (b *openAPIEncodingBudget) charge(bytes int) error {
	if bytes > b.remaining {
		return strictjson.ErrBounds
	}
	b.remaining -= bytes
	return nil
}

// text counts exactly the escaping used by json.Encoder with EscapeHTML=false.
// It does not allocate an encoded copy, and stops scanning as soon as it cannot fit.
func (b *openAPIEncodingBudget) text(text string) error {
	if err := b.charge(2); err != nil {
		return err
	}
	for len(text) > 0 {
		size, width := 1, 1
		switch text[0] {
		case '"', '\\', '\b', '\f', '\n', '\r', '\t':
			size = 2
		default:
			if text[0] < 0x20 {
				size = 6
			} else if text[0] >= utf8.RuneSelf {
				runeValue, n := utf8.DecodeRuneInString(text)
				width, size = n, n
				if runeValue == '\u2028' || runeValue == '\u2029' || runeValue == utf8.RuneError && n == 1 {
					size = 6
				}
			}
		}
		if err := b.charge(size); err != nil {
			return err
		}
		text = text[width:]
	}
	return nil
}

func (b *openAPIEncodingBudget) value(value reflect.Value, depth int) error {
	for indirections := 0; value.IsValid() && (value.Kind() == reflect.Interface || value.Kind() == reflect.Pointer); indirections++ {
		if value.IsNil() {
			return b.scalar(4, depth)
		}
		if indirections > MaximumJSONDepth {
			return strictjson.ErrBounds
		}
		if customJSONEncoding(value.Type()) {
			return errors.New("unsupported OpenAPI JSON marshaler")
		}
		value = value.Elem()
	}
	b.nodes++
	if depth > MaximumJSONDepth || b.nodes > MaximumJSONNodes {
		return strictjson.ErrBounds
	}
	if !value.IsValid() {
		return b.charge(4)
	}
	// The OpenAPI tree and its typed wrappers contain only ordinary JSON values.
	// Never invoke user-controlled/custom marshalers during a size check.
	if customJSONEncoding(value.Type()) {
		return errors.New("unsupported OpenAPI JSON marshaler")
	}
	switch value.Kind() {
	case reflect.String:
		return b.text(value.String())
	case reflect.Bool:
		if value.Bool() {
			return b.charge(4)
		}
		return b.charge(5)
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
		reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64,
		reflect.Float32, reflect.Float64:
		// A primitive numeric token has a fixed, small maximum allocation. Use the
		// real encoder's spelling (not a second approximation of its float rules).
		token, err := json.Marshal(value.Interface())
		if err != nil {
			return err
		}
		return b.charge(len(token))
	case reflect.Map:
		if value.Type().Key().Kind() != reflect.String || customJSONEncoding(value.Type().Key()) {
			return errors.New("invalid OpenAPI JSON key")
		}
		if value.IsNil() {
			return b.charge(4)
		}
		if err := b.charge(2); err != nil {
			return err
		}
		iterator := value.MapRange()
		first := true
		for iterator.Next() {
			if !first {
				if err := b.charge(1); err != nil {
					return err
				}
			}
			first = false
			if err := b.text(iterator.Key().String()); err != nil {
				return err
			}
			if err := b.charge(1); err != nil {
				return err
			}
			if err := b.value(iterator.Value(), depth+1); err != nil {
				return err
			}
		}
		return nil
	case reflect.Slice, reflect.Array:
		if value.Kind() == reflect.Slice && value.IsNil() {
			return b.charge(4)
		}
		// []byte is not a source JSON array and has base64 encoding semantics.
		if value.Type().Elem().Kind() == reflect.Uint8 {
			return errors.New("invalid OpenAPI JSON array")
		}
		if err := b.charge(2); err != nil {
			return err
		}
		for i := 0; i < value.Len(); i++ {
			if i > 0 {
				if err := b.charge(1); err != nil {
					return err
				}
			}
			if err := b.value(value.Index(i), depth+1); err != nil {
				return err
			}
		}
		return nil
	case reflect.Struct:
		if err := b.charge(2); err != nil {
			return err
		}
		first := true
		for i := 0; i < value.NumField(); i++ {
			field := value.Type().Field(i)
			if !field.IsExported() {
				continue
			}
			name, options, _ := strings.Cut(field.Tag.Get("json"), ",")
			if name == "-" {
				continue
			}
			if name == "" || field.Anonymous || options != "" && options != "omitempty" {
				return errors.New("unsupported OpenAPI JSON field")
			}
			child := value.Field(i)
			if options == "omitempty" && emptyJSONValue(child) {
				continue
			}
			if !first {
				if err := b.charge(1); err != nil {
					return err
				}
			}
			first = false
			if err := b.text(name); err != nil {
				return err
			}
			if err := b.charge(1); err != nil {
				return err
			}
			if err := b.value(child, depth+1); err != nil {
				return err
			}
		}
		return nil
	default:
		return errors.New("invalid OpenAPI JSON value")
	}
}

func (b *openAPIEncodingBudget) scalar(bytes, depth int) error {
	b.nodes++
	if depth > MaximumJSONDepth || b.nodes > MaximumJSONNodes {
		return strictjson.ErrBounds
	}
	return b.charge(bytes)
}

func emptyJSONValue(value reflect.Value) bool {
	switch value.Kind() {
	case reflect.Array, reflect.Map, reflect.Slice, reflect.String:
		return value.Len() == 0
	case reflect.Bool, reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
		reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64,
		reflect.Float32, reflect.Float64, reflect.Interface, reflect.Pointer:
		return value.IsZero()
	default:
		return false
	}
}

func customJSONEncoding(value reflect.Type) bool {
	pointer := reflect.PointerTo(value)
	return value.Implements(reflect.TypeFor[json.Marshaler]()) || value.Implements(reflect.TypeFor[encoding.TextMarshaler]()) ||
		pointer.Implements(reflect.TypeFor[json.Marshaler]()) || pointer.Implements(reflect.TypeFor[encoding.TextMarshaler]())
}
