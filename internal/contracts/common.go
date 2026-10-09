// Package contracts contains strict, versioned transport DTOs shared with the
// Python Runtime Agent. It deliberately imports no HTTP, persistence, scheduler,
// or agent-framework package.
package contracts

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"net/url"
	"regexp"
	"strings"

	"github.com/grauwolf32/contractor/internal/contentdigest"
	"github.com/grauwolf32/contractor/internal/strictjson"
)

const APIVersion = "contractor/v1alpha1"

// MaxConfigIDLength and MaxConfigVersionLength bound the two parts of a
// configuration <id>@<version> selector, as the public ConfigurationName,
// ConfigVersion and Selector schemas do.
const (
	MaxConfigIDLength      = 128
	MaxConfigVersionLength = 64
)

var (
	// ErrValidation identifies a syntactically valid JSON value that violates a
	// Contractor wire invariant.
	ErrValidation = errors.New("contract validation failed")
	// idPattern and versionPattern are the only Go definitions of the
	// identifier and opaque version grammars. The shared cases in
	// api/testdata/v1alpha1/config-identity-cases.json hold Go, the schemas,
	// the public OpenAPI, the Runtime and the UI to them.
	idPattern      = regexp.MustCompile(`^[a-z][a-z0-9_-]*$`)
	versionPattern = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._+-]*$`)
	// runtimeAgentIDPattern is the only Go definition of a Runtime Agent ID:
	// the lowercase hexadecimal SHA-256 fingerprint of the agent's SPKI.
	runtimeAgentIDPattern = regexp.MustCompile(`^[0-9a-f]{64}$`)
)

// ValidIdentifier reports whether value matches the identifier grammar
// [a-z][a-z0-9_-]*. Callers apply the length bound of their field.
func ValidIdentifier(value string) bool { return idPattern.MatchString(value) }

// ValidVersion reports whether value matches the opaque version grammar
// [A-Za-z0-9][A-Za-z0-9._+-]*. Callers apply the length bound of their field.
func ValidVersion(value string) bool { return versionPattern.MatchString(value) }

// ValidSelector reports whether value is an exact <id>@<version> selector in
// those grammars. Callers apply the length bounds of their field.
func ValidSelector(value string) bool {
	id, version, found := strings.Cut(value, "@")
	return found && ValidIdentifier(id) && ValidVersion(version)
}

// ValidRuntimeAgentID reports whether value is a Runtime Agent ID.
func ValidRuntimeAgentID(value string) bool { return runtimeAgentIDPattern.MatchString(value) }

// ValidConfigID reports whether value is a bounded configuration identifier.
func ValidConfigID(value string) bool {
	return len(value) <= MaxConfigIDLength && ValidIdentifier(value)
}

// ValidConfigVersion reports whether value is a bounded configuration version.
func ValidConfigVersion(value string) bool {
	return len(value) <= MaxConfigVersionLength && ValidVersion(value)
}

// Validatable is implemented by every top-level wire DTO.
type Validatable interface {
	Validate() error
}

// DecodeStrict decodes one JSON object, rejects unknown/trailing fields, and
// runs its semantic validation.
func DecodeStrict[T Validatable](data []byte) (T, error) {
	var value T
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&value); err != nil {
		return value, fmt.Errorf("decode JSON: %w", err)
	}
	if err := EnsureJSONEOF(decoder); err != nil {
		return value, err
	}
	if err := value.Validate(); err != nil {
		return value, err
	}
	return value, nil
}

// DecodeStrictObject decodes one JSON object into target, rejecting unknown
// fields, and returns its raw top-level members so callers can tell an
// explicit null from an omitted field.
func DecodeStrictObject(data []byte, target any) (map[string]json.RawMessage, error) {
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(target); err != nil {
		return nil, err
	}
	var fields map[string]json.RawMessage
	err := json.Unmarshal(data, &fields)
	return fields, err
}

// EnsureJSONEOF rejects any JSON value or data after the one decoder read.
func EnsureJSONEOF(decoder *json.Decoder) error {
	if err := strictjson.RequireEOF(decoder); errors.Is(err, strictjson.ErrTrailingData) {
		return Invalidf("multiple JSON values are not allowed")
	} else if err != nil {
		return fmt.Errorf("decode trailing JSON: %w", err)
	}
	return nil
}

// ValidateAPIVersion requires the exact contractor/v1alpha1 wire version.
func ValidateAPIVersion(value string) error {
	if value != APIVersion {
		return fmt.Errorf("%w: %w", ErrValidation, privateVersionError)
	}
	return nil
}

// ValidateOpaqueID requires a non-blank opaque identifier.
func ValidateOpaqueID(field, value string) error {
	if strings.TrimSpace(value) == "" {
		return Invalidf("%s must not be empty", field)
	}
	return nil
}

// ValidateSelector requires an exact <id>@<version> selector.
func ValidateSelector(field, value string) error {
	if strings.Count(value, "@") != 1 {
		return Invalidf("%s must use exact <id>@<version> syntax", field)
	}
	if !ValidSelector(value) {
		return Invalidf("%s has an invalid exact selector", field)
	}
	return nil
}

// ValidateDigest requires a sha256:<64 lowercase hex> content digest.
func ValidateDigest(field, value string) error {
	if !contentdigest.Valid(value) {
		return Invalidf("%s must be sha256 followed by 64 lowercase hex characters", field)
	}
	return nil
}

// ValidateURL requires an absolute HTTP(S) URL without user information.
func ValidateURL(field, value string) error {
	parsed, err := url.Parse(value)
	if err != nil || parsed.Host == "" || (parsed.Scheme != "http" && parsed.Scheme != "https") {
		return Invalidf("%s must be an absolute HTTP(S) URL", field)
	}
	if parsed.User != nil {
		return Invalidf("%s must not contain URL user information", field)
	}
	return nil
}

// Invalidf returns an ErrValidation error with a formatted, input-free detail.
func Invalidf(format string, args ...any) error {
	return fmt.Errorf("%w: %s", ErrValidation, fmt.Sprintf(format, args...))
}

// SecretString is wire-serializable but redacts String and GoString output.
// Reveal must be used only at the point where a private client is configured.
type SecretString struct {
	value string
}

func NewSecretString(value string) SecretString { return SecretString{value: value} }

func (s SecretString) Reveal() string { return s.value }

func (s SecretString) String() string { return "[REDACTED]" }

func (s SecretString) GoString() string { return "contracts.SecretString([REDACTED])" }

func (s SecretString) MarshalJSON() ([]byte, error) { return json.Marshal(s.value) }

func (s *SecretString) UnmarshalJSON(data []byte) error {
	var value string
	if err := json.Unmarshal(data, &value); err != nil {
		return fmt.Errorf("secret must be a string: %w", err)
	}
	s.value = value
	return nil
}
