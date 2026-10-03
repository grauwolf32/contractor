package contracts

// Strict decoding and RFC 8785 canonical encoding for private protocol
// DTOs. Mirrors the Python runtime's contracts/codec.py.

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"

	"github.com/grauwolf32/contractor/internal/strictjson"
)

var privateVersionError = errors.New("private protocol version mismatch")

// DecodePrivateStrict rejects duplicate keys, unknown fields and trailing
// JSON before returning a semantically validated private DTO.
func DecodePrivateStrict[T Validatable](data []byte) (T, error) {
	var value T
	if err := strictjson.RejectDuplicateKeys(data); err != nil {
		class := PrivateProtocolErrorSchema
		if errors.Is(err, strictjson.ErrDuplicateKey) {
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
	canonical, err := strictjson.Canonical(value)
	if err != nil {
		return nil, fmt.Errorf("canonicalize private protocol value: %w", err)
	}
	return canonical, nil
}
