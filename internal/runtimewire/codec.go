package runtimewire

import (
	"encoding/json"
	"math"
	"strings"

	"github.com/grauwolf32/contractor/internal/helpers/strictjson"
	"google.golang.org/protobuf/reflect/protoreflect"
)

// The projection is not a second wire parser. Protobuf owns binary parsing;
// domain validators consume exact JSON numbers and only fields actually present.
// ListOf*/Nullable* are private schema conventions declared in runtime.proto.
func decodeMessage(message protoreflect.Message) ([]byte, error) {
	if len(message.GetUnknown()) != 0 {
		return nil, ErrInvalid
	}
	descriptor := message.Descriptor()
	name := string(descriptor.Name())
	if strings.HasSuffix(name, "Document") {
		field := descriptor.Fields().ByName("json")
		if !message.Has(field) {
			return nil, ErrInvalid
		}
		data := message.Get(field).Bytes()
		var value map[string]json.RawMessage
		if strictjson.DecodeNonNull(data, &value) != nil {
			return nil, ErrInvalid
		}
		return data, nil
	}
	if name == "MapOfString" {
		values := map[string]string{}
		message.Get(descriptor.Fields().ByName("values")).Map().Range(func(key protoreflect.MapKey, value protoreflect.Value) bool {
			values[key.String()] = value.String()
			return true
		})
		return json.Marshal(values)
	}
	if strings.HasPrefix(name, "Nullable") {
		choice := message.WhichOneof(descriptor.Oneofs().ByName("kind"))
		if choice == nil {
			return nil, ErrInvalid
		}
		if choice.Name() == "null" {
			if message.Get(choice).Enum() != 0 {
				return nil, ErrInvalid
			}
			return []byte("null"), nil
		}
		return decodeMessage(message.Get(choice).Message())
	}
	if strings.HasPrefix(name, "ListOf") {
		field := descriptor.Fields().ByName("values")
		list := message.Get(field).List()
		values := make([]json.RawMessage, list.Len())
		for index := range values {
			value, err := decodeValue(field, list.Get(index))
			if err != nil {
				return nil, err
			}
			values[index] = value
		}
		return json.Marshal(values)
	}
	values := make(map[string]json.RawMessage)
	var failure error
	message.Range(func(field protoreflect.FieldDescriptor, value protoreflect.Value) bool {
		encoded, err := decodeValue(field, value)
		if err != nil {
			failure = err
			return false
		}
		values[field.JSONName()] = encoded
		return true
	})
	if failure != nil {
		return nil, failure
	}
	return json.Marshal(values)
}

func decodeValue(field protoreflect.FieldDescriptor, value protoreflect.Value) ([]byte, error) {
	switch field.Kind() {
	case protoreflect.MessageKind:
		return decodeMessage(value.Message())
	case protoreflect.StringKind:
		return json.Marshal(value.String())
	case protoreflect.BoolKind:
		return json.Marshal(value.Bool())
	case protoreflect.Int64Kind:
		return json.Marshal(value.Int())
	case protoreflect.Uint64Kind:
		return json.Marshal(value.Uint())
	case protoreflect.DoubleKind:
		if math.IsNaN(value.Float()) || math.IsInf(value.Float(), 0) {
			return nil, ErrInvalid
		}
		return json.Marshal(value.Float())
	default:
		return nil, ErrInvalid
	}
}

func encodeMessage(data []byte, message protoreflect.Message) error {
	descriptor := message.Descriptor()
	if strings.HasSuffix(string(descriptor.Name()), "Document") {
		var value map[string]json.RawMessage
		if strictjson.DecodeNonNull(data, &value) != nil {
			return ErrInvalid
		}
		message.Set(descriptor.Fields().ByName("json"), protoreflect.ValueOfBytes(append([]byte(nil), data...)))
		return nil
	}
	if descriptor.Name() == "MapOfString" {
		var values map[string]string
		if strictjson.DecodeNonNull(data, &values) != nil {
			return ErrInvalid
		}
		target := message.Mutable(descriptor.Fields().ByName("values")).Map()
		for key, value := range values {
			target.Set(protoreflect.ValueOfString(key).MapKey(), protoreflect.ValueOfString(value))
		}
		return nil
	}
	if strings.HasPrefix(string(descriptor.Name()), "Nullable") {
		if string(data) == "null" {
			message.Set(descriptor.Fields().ByName("null"), protoreflect.ValueOfEnum(0))
			return nil
		}
		return encodeMessage(data, message.Mutable(descriptor.Fields().ByName("value")).Message())
	}
	if strings.HasPrefix(string(descriptor.Name()), "ListOf") {
		var values []json.RawMessage
		if err := strictjson.DecodeNonNull(data, &values); err != nil {
			return ErrInvalid
		}
		field := descriptor.Fields().ByName("values")
		list := message.Mutable(field).List()
		for _, raw := range values {
			if field.Kind() == protoreflect.MessageKind {
				value := list.NewElement()
				if err := encodeMessage(raw, value.Message()); err != nil {
					return err
				}
				list.Append(value)
				continue
			}
			value, err := encodeScalar(raw, field)
			if err != nil {
				return err
			}
			list.Append(value)
		}
		return nil
	}
	var values map[string]json.RawMessage
	if err := strictjson.DecodeNonNull(data, &values); err != nil {
		return ErrInvalid
	}
	for name, raw := range values {
		field := descriptor.Fields().ByJSONName(name)
		if field == nil {
			return ErrInvalid
		}
		if field.Kind() == protoreflect.MessageKind {
			if err := encodeMessage(raw, message.Mutable(field).Message()); err != nil {
				return err
			}
			continue
		}
		value, err := encodeScalar(raw, field)
		if err != nil {
			return err
		}
		message.Set(field, value)
	}
	return nil
}

func encodeScalar(data []byte, field protoreflect.FieldDescriptor) (protoreflect.Value, error) {
	switch field.Kind() {
	case protoreflect.Uint64Kind:
		var value uint64
		if strictjson.DecodeNonNull(data, &value) == nil {
			return protoreflect.ValueOfUint64(value), nil
		}
	case protoreflect.StringKind:
		var value string
		if strictjson.DecodeNonNull(data, &value) == nil {
			return protoreflect.ValueOfString(value), nil
		}
	case protoreflect.BoolKind:
		var value bool
		if strictjson.DecodeNonNull(data, &value) == nil {
			return protoreflect.ValueOfBool(value), nil
		}
	case protoreflect.DoubleKind:
		var value float64
		if strictjson.DecodeNonNull(data, &value) == nil && !math.IsNaN(value) && !math.IsInf(value, 0) {
			return protoreflect.ValueOfFloat64(value), nil
		}
	case protoreflect.Int64Kind:
		var value int64
		if strictjson.DecodeNonNull(data, &value) == nil {
			return protoreflect.ValueOfInt64(value), nil
		}
	}
	return protoreflect.Value{}, ErrInvalid
}
