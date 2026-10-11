package contracts

// Strict decoding and RFC 8785 canonical encoding for private protocol
// DTOs. Mirrors the Python runtime's contracts/codec.py.

import (
	"errors"
	"fmt"

	"github.com/grauwolf32/contractor/internal/strictjson"
)

var privateVersionError = errors.New("private protocol version mismatch")

// DecodePrivateStrict decodes one private DTO with strictjson.Decode, so
// duplicate keys, case aliases, unknown members and trailing data are
// rejected, before semantic validation. Errors carry only their class.
func DecodePrivateStrict[T Validatable](data []byte) (T, error) {
	var value T
	if err := strictjson.Decode(data, &value); err != nil {
		class := PrivateProtocolErrorSchema
		if errors.Is(err, strictjson.ErrDuplicateKey) {
			class = PrivateProtocolErrorDuplicate
		}
		return value, &PrivateProtocolError{Class: class}
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
// fingerprints. It encodes SecretString redacted, like any default JSON
// encoding (SEC-R88); a fingerprint that must identify a secret passes its
// plaintext explicitly, and such a result must never be logged.
func MarshalPrivateCanonical(value any) ([]byte, error) {
	canonical, err := strictjson.Canonical(value)
	if err != nil {
		return nil, fmt.Errorf("canonicalize private protocol value: %w", err)
	}
	return canonical, nil
}
