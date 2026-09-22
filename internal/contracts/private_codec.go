package contracts

// Strict decoding and RFC 8785 canonical encoding for private protocol
// DTOs. Mirrors the Python runtime's contracts/codec.py.

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"

	"github.com/ucarion/jcs"
)

var privateVersionError = errors.New("private protocol version mismatch")

// DecodePrivateStrict rejects duplicate keys, unknown fields and trailing
// JSON before returning a semantically validated private DTO.
func DecodePrivateStrict[T Validatable](data []byte) (T, error) {
	var value T
	if err := rejectDuplicateJSONKeys(data); err != nil {
		class := PrivateProtocolErrorSchema
		if errors.Is(err, errDuplicateJSONKey) {
			class = PrivateProtocolErrorDuplicate
		}
		return value, &PrivateProtocolError{Class: class}
	}
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&value); err != nil {
		return value, &PrivateProtocolError{Class: PrivateProtocolErrorSchema}
	}
	if err := ensureJSONEOF(decoder); err != nil {
		return value, &PrivateProtocolError{Class: PrivateProtocolErrorSchema}
	}
	if err := value.Validate(); err != nil {
		class := PrivateProtocolErrorInvariant
		if errors.Is(err, privateVersionError) {
			class = PrivateProtocolErrorVersion
		}
		return value, &PrivateProtocolError{Class: class}
	}
	return value, nil
}

// MarshalPrivateCanonical uses RFC 8785 JCS for cross-language fixtures and
// fingerprints. It remains a private-wire encoder and therefore includes
// SecretString values; callers must never log its result.
func MarshalPrivateCanonical(value any) ([]byte, error) {
	encoded, err := json.Marshal(value)
	if err != nil {
		return nil, fmt.Errorf("encode private protocol value: %w", err)
	}
	var jsonValue any
	if err := json.Unmarshal(encoded, &jsonValue); err != nil {
		return nil, fmt.Errorf("normalize private protocol value: %w", err)
	}
	canonical, err := jcs.Format(jsonValue)
	if err != nil {
		return nil, fmt.Errorf("canonicalize private protocol value: %w", err)
	}
	return []byte(canonical), nil
}

var errDuplicateJSONKey = errors.New("duplicate JSON key")

func rejectDuplicateJSONKeys(data []byte) error {
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.UseNumber()
	if err := scanJSONValue(decoder); err != nil {
		return err
	}
	if token, err := decoder.Token(); !errors.Is(err, io.EOF) {
		if err != nil {
			return err
		}
		return fmt.Errorf("unexpected trailing JSON token %v", token)
	}
	return nil
}

func scanJSONValue(decoder *json.Decoder) error {
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
				return errors.New("JSON object key is not a string")
			}
			if _, duplicate := seen[key]; duplicate {
				return errDuplicateJSONKey
			}
			seen[key] = struct{}{}
			if err := scanJSONValue(decoder); err != nil {
				return err
			}
		}
		closing, err := decoder.Token()
		if err != nil || closing != json.Delim('}') {
			return errors.New("invalid JSON object")
		}
	case '[':
		for decoder.More() {
			if err := scanJSONValue(decoder); err != nil {
				return err
			}
		}
		closing, err := decoder.Token()
		if err != nil || closing != json.Delim(']') {
			return errors.New("invalid JSON array")
		}
	default:
		return errors.New("invalid JSON delimiter")
	}
	return nil
}
