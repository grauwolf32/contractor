// Package strictjson adds the checks encoding/json does not make on its own:
// unknown fields and trailing values, duplicate object keys, lone UTF-16
// surrogate escapes, and RFC 8785 canonical encoding. Callers map the errors
// to their own codes.
package strictjson

import (
	"bytes"
	"encoding/json"
	"errors"
	"io"

	"github.com/ucarion/jcs"
)

var (
	ErrDuplicateKey = errors.New("JSON object has a duplicate key")
	ErrTrailingData = errors.New("JSON value has trailing data")
	ErrInvalid      = errors.New("JSON value is invalid")
)

// Decode decodes exactly one JSON value into target, rejecting unknown object
// fields. A decode error is returned unchanged; anything after the value,
// valid or not, reports ErrTrailingData.
func Decode(data []byte, target any) error {
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(target); err != nil {
		return err
	}
	if RequireEOF(decoder) != nil {
		return ErrTrailingData
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
// value. Invalid JSON reports ErrInvalid or the decoder's syntax error.
func RejectDuplicateKeys(data []byte) error {
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.UseNumber()
	if err := scanValue(decoder); err != nil {
		return err
	}
	if _, err := decoder.Token(); !errors.Is(err, io.EOF) {
		if err != nil {
			return err
		}
		return ErrTrailingData
	}
	return nil
}

func scanValue(decoder *json.Decoder) error {
	token, err := decoder.Token()
	if err != nil {
		return err
	}
	delimiter, composite := token.(json.Delim)
	if !composite {
		return nil
	}
	switch delimiter {
	case '{':
		seen := make(map[string]struct{})
		for decoder.More() {
			keyToken, err := decoder.Token()
			if err != nil {
				return err
			}
			key, ok := keyToken.(string)
			if !ok {
				return ErrInvalid
			}
			if _, duplicate := seen[key]; duplicate {
				return ErrDuplicateKey
			}
			seen[key] = struct{}{}
			if err := scanValue(decoder); err != nil {
				return err
			}
		}
		if closing, err := decoder.Token(); err != nil || closing != json.Delim('}') {
			return ErrInvalid
		}
	case '[':
		for decoder.More() {
			if err := scanValue(decoder); err != nil {
				return err
			}
		}
		if closing, err := decoder.Token(); err != nil || closing != json.Delim(']') {
			return ErrInvalid
		}
	default:
		return ErrInvalid
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
