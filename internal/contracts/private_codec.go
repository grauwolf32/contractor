package contracts

// Strict decoding and RFC 8785 canonical encoding for private protocol
// DTOs. Mirrors the Python runtime's contracts/codec.py.

import (
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"strings"

	"github.com/grauwolf32/contractor/internal/strictjson"
)

var privateVersionError = errors.New("private protocol version mismatch")

// DecodePrivateStrict is the decoder of every private message the Server
// receives from a Runtime Agent or a Worker: DecodePrivate followed by the
// message's semantic validation. Errors carry only their class, assigned in
// the Runtime codec's order: duplicate key, malformed JSON, API version, then
// schema or invariant.
func DecodePrivateStrict[T Validatable](data []byte) (T, error) {
	var value T
	if err := DecodePrivate(data, &value); err != nil {
		return value, err
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

// DecodePrivate decodes one private message into target, a pointer, with
// strictjson.Decode: duplicate keys, case aliases, unknown members, omitted or
// null required members (the strictjson tags that a test pins to the
// api/v1alpha1 schemas) and trailing data are rejected. It leaves semantic
// validation to the caller, which DecodePrivateStrict runs at once and a
// lifecycle response runs after discarding isolated telemetry. Errors carry
// only their class.
func DecodePrivate(data []byte, target any) error {
	err := strictjson.Decode(data, target)
	if err == nil {
		return nil
	}
	class := PrivateProtocolErrorSchema
	switch typ := reflect.TypeOf(target); {
	case errors.Is(err, strictjson.ErrDuplicateKey):
		class = PrivateProtocolErrorDuplicate
	case typ != nil && typ.Kind() == reflect.Pointer && carriesAPIVersion(typ.Elem()) && !currentAPIVersion(data):
		// The Runtime codec checks the version before the members.
		class = PrivateProtocolErrorVersion
	}
	return &PrivateProtocolError{Class: class}
}

// carriesAPIVersion reports whether struct t has the apiVersion member of a
// standalone private message, like the Runtime's VersionedWireModel.
func carriesAPIVersion(t reflect.Type) bool {
	if t.Kind() != reflect.Struct {
		return false
	}
	for index := range t.NumField() {
		if name, _, _ := strings.Cut(t.Field(index).Tag.Get("json"), ","); name == "apiVersion" {
			return true
		}
	}
	return false
}

// currentAPIVersion reports whether data is not a JSON object or carries the
// current apiVersion, so only an object with a missing or wrong version is a
// version error.
func currentAPIVersion(data []byte) bool {
	var members map[string]json.RawMessage
	if json.Unmarshal(data, &members) != nil || members == nil {
		return true
	}
	var version string
	return json.Unmarshal(members["apiVersion"], &version) == nil && version == APIVersion
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
