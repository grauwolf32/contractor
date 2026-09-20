package scanplan

import (
	"encoding"
	"encoding/json"
	"math"
	"reflect"
	"unicode/utf8"
)

var (
	optionJSONMarshaler = reflect.TypeFor[json.Marshaler]()
	optionTextMarshaler = reflect.TypeFor[encoding.TextMarshaler]()
)

// validateOptionsValues runs before json.Marshal, which would otherwise replace
// invalid UTF-8 or invoke caller-supplied marshaling methods. Only the fixed
// authentication slot may contain SecretString; generic values are JSON data.
func validateOptionsValues(options Options) bool {
	r := optionValues{types: make(map[reflect.Type]bool)}
	if !r.node(1) || !r.field("server", options.Server, 2) ||
		!r.field("serverVariables", options.ServerVariables, 2) ||
		!r.text("authentication", 2) || !r.node(2) {
		return false
	}
	for name, secret := range options.Authentication {
		if !r.text(name, 3) || !r.text(secret.Reveal(), 3) {
			return false
		}
	}
	if !r.text("operations", 2) || !r.node(2) {
		return false
	}
	for pointer, operation := range options.Operations {
		if !r.text(pointer, 3) || !r.node(3) ||
			!r.field("parameters", operation.Parameters, 4) || !r.text("body", 4) || !r.node(4) {
			return false
		}
		if operation.Body != nil && (!r.field("mediaType", operation.Body.MediaType, 5) ||
			!r.field("value", operation.Body.Value, 5)) {
			return false
		}
	}
	return r.field("maxRequests", options.MaxRequests, 2)
}

type optionValues struct {
	nodes, stringBytes int
	types              map[reflect.Type]bool
}

func (r *optionValues) node(depth int) bool {
	if depth > MaxDepth || r.nodes >= MaxNodes {
		return false
	}
	r.nodes++
	return true
}

func (r *optionValues) stringValue(value string) bool {
	if len(value) > MaxOptionsBytes-r.stringBytes || !utf8.ValidString(value) {
		return false
	}
	r.stringBytes += len(value)
	return true
}

func (r *optionValues) text(value string, depth int) bool {
	return r.node(depth) && r.stringValue(value)
}

func (r *optionValues) field(name string, value any, depth int) bool {
	return r.text(name, depth) && r.value(reflect.ValueOf(value), depth)
}

func (r *optionValues) value(value reflect.Value, depth int) bool {
	if !r.node(depth) {
		return false
	}
	for value.IsValid() && value.Kind() == reflect.Interface {
		if value.IsNil() {
			return true
		}
		value = value.Elem()
	}
	if !value.IsValid() {
		return true
	}
	if !r.allowedType(value.Type(), 1) {
		return false
	}
	switch value.Kind() {
	case reflect.String:
		return r.stringValue(value.String())
	case reflect.Bool, reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
		reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr:
		return true // Exact numeric range is checked by parseDocument after marshaling.
	case reflect.Float32, reflect.Float64:
		return !math.IsNaN(value.Float()) && !math.IsInf(value.Float(), 0)
	case reflect.Map:
		if value.Len() > (MaxNodes-r.nodes)/2 {
			return false
		}
		iterator := value.MapRange()
		for iterator.Next() {
			if !r.text(iterator.Key().String(), depth+1) || !r.value(iterator.Value(), depth+1) {
				return false
			}
		}
		return true
	case reflect.Slice, reflect.Array:
		if value.Len() > MaxNodes-r.nodes {
			return false
		}
		for i := 0; i < value.Len(); i++ {
			if !r.value(value.Index(i), depth+1) {
				return false
			}
		}
		return true
	}
	return false
}

func (r *optionValues) allowedType(value reflect.Type, depth int) bool {
	if depth > MaxDepth {
		return false
	}
	if allowed, known := r.types[value]; known {
		return allowed
	}
	if value.Implements(optionJSONMarshaler) || value.Implements(optionTextMarshaler) ||
		reflect.PointerTo(value).Implements(optionJSONMarshaler) ||
		reflect.PointerTo(value).Implements(optionTextMarshaler) {
		r.types[value] = false
		return false
	}
	// Named maps/slices may have recursive element types. Actual value cycles are
	// still bounded by node/depth accounting when traversing their contents.
	r.types[value] = true
	allowed := false
	switch value.Kind() {
	case reflect.String, reflect.Bool, reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32,
		reflect.Int64, reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32,
		reflect.Uint64, reflect.Uintptr, reflect.Float32, reflect.Float64, reflect.Interface:
		allowed = true
	case reflect.Map:
		allowed = value.Key().Kind() == reflect.String && r.allowedType(value.Key(), depth+1) &&
			r.allowedType(value.Elem(), depth+1)
	case reflect.Array:
		allowed = r.allowedType(value.Elem(), depth+1)
	case reflect.Slice:
		// encoding/json implicitly converts byte slices to base64 strings.
		allowed = value.Elem().Kind() != reflect.Uint8 && r.allowedType(value.Elem(), depth+1)
	}
	r.types[value] = allowed
	return allowed
}
