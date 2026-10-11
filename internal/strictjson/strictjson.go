// Package strictjson adds the checks encoding/json does not make on its own:
// duplicate object keys, member names that match a field only when case is
// ignored, unknown members, trailing values, lone UTF-16 surrogate escapes,
// and RFC 8785 canonical encoding. Callers map the errors to their own codes.
package strictjson

import (
	"bytes"
	"encoding/json"
	"errors"
	"io"
	"reflect"

	"github.com/ucarion/jcs"
)

var (
	ErrDuplicateKey = errors.New("JSON object has a duplicate key")
	ErrTrailingData = errors.New("JSON value has trailing data")
	ErrInvalid      = errors.New("JSON value is invalid")
	// ErrUnknownField reports an object member whose name matches no field
	// of the target struct exactly, including a name encoding/json would
	// match case-insensitively.
	ErrUnknownField = errors.New("JSON object has an unknown member")
)

// Decode decodes exactly one JSON value into target. Beyond encoding/json with
// unknown fields disallowed, it rejects anywhere in the value:
//   - a repeated object member name, after unescaping (ErrDuplicateKey);
//   - a member whose name matches no field of its target struct exactly,
//     such as a case alias encoding/json would accept (ErrUnknownField);
//   - anything after the value, valid or not (ErrTrailingData).
//
// A duplicate key is reported before member errors. A type that decodes
// itself with UnmarshalJSON or UnmarshalText is checked for duplicate keys
// only; its method checks its own members, normally with Decode. Syntax and
// type errors are returned as encoding/json reports them.
func Decode(data []byte, target any) error {
	var info *typeInfo
	if value := reflect.ValueOf(target); value.Kind() == reflect.Pointer && !value.IsNil() {
		info = infoFor(value.Type().Elem())
	} else {
		info = anyInfo
	}
	err := check(data, info)
	if err != nil && !errors.Is(err, errSyntax) {
		return err
	}
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	if decodeErr := decoder.Decode(target); decodeErr != nil {
		return decodeErr
	}
	if err != nil {
		// encoding/json accepted what the scanner could not read: fail
		// closed rather than trust either.
		return ErrInvalid
	}
	return nil
}

// RequireEOF reports whether decoder has consumed its input: nil at EOF,
// ErrTrailingData for another JSON value and the decoder error otherwise.
func RequireEOF(decoder *json.Decoder) error {
	var trailing json.RawMessage
	err := decoder.Decode(&trailing)
	switch {
	case errors.Is(err, io.EOF):
		return nil
	case err != nil:
		return err
	default:
		return ErrTrailingData
	}
}

// RejectDuplicateKeys reads one JSON value and reports ErrDuplicateKey for a
// repeated object member name and ErrTrailingData for anything after the
// value. Invalid JSON reports ErrInvalid.
func RejectDuplicateKeys(data []byte) error {
	if err := check(data, anyInfo); err != nil {
		if errors.Is(err, errSyntax) {
			return ErrInvalid
		}
		return err
	}
	return nil
}

// ValidUnicodeEscapes reports whether every complete \uXXXX escape in data
// forms a Unicode scalar value. encoding/json silently replaces lone UTF-16
// surrogate escapes with U+FFFD, which would merge distinct byte identities.
// Malformed escapes are left for the JSON decoder to reject.
func ValidUnicodeEscapes(data []byte) bool {
	for index := 0; index < len(data); index++ {
		if data[index] != '\\' {
			continue
		}
		index++
		if index >= len(data) || data[index] != 'u' {
			continue
		}
		code, ok := hexQuad(data, index+1)
		if !ok {
			continue
		}
		index += 4
		switch {
		case code >= 0xdc00 && code <= 0xdfff:
			return false
		case code >= 0xd800 && code <= 0xdbff:
			if index+2 >= len(data) || data[index+1] != '\\' || data[index+2] != 'u' {
				return false
			}
			low, ok := hexQuad(data, index+3)
			if !ok || low < 0xdc00 || low > 0xdfff {
				return false
			}
			index += 6
		}
	}
	return true
}

func hexQuad(data []byte, start int) (uint16, bool) {
	if start+4 > len(data) {
		return 0, false
	}
	var result uint16
	for _, digit := range data[start : start+4] {
		result <<= 4
		switch {
		case digit >= '0' && digit <= '9':
			result |= uint16(digit - '0')
		case digit >= 'a' && digit <= 'f':
			result |= uint16(digit-'a') + 10
		case digit >= 'A' && digit <= 'F':
			result |= uint16(digit-'A') + 10
		default:
			return 0, false
		}
	}
	return result, true
}

// Canonical returns the RFC 8785 (JCS) encoding of value's encoding/json form.
func Canonical(value any) ([]byte, error) {
	encoded, err := json.Marshal(value)
	if err != nil {
		return nil, err
	}
	var generic any
	if err := json.Unmarshal(encoded, &generic); err != nil {
		return nil, err
	}
	formatted, err := jcs.Format(generic)
	if err != nil {
		return nil, err
	}
	return []byte(formatted), nil
}
