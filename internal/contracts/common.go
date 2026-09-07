// Package contracts contains strict, versioned transport DTOs shared with the
// Python Runtime Agent. It deliberately imports no HTTP, persistence, scheduler,
// or agent-framework package.
package contracts

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/url"
	"regexp"
	"strings"
)

const APIVersion = "contractor/v1alpha1"

var (
	// ErrValidation identifies a syntactically valid JSON value that violates a
	// Contractor wire invariant.
	ErrValidation  = errors.New("contract validation failed")
	idPattern      = regexp.MustCompile(`^[a-z][a-z0-9_-]*$`)
	versionPattern = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._+-]*$`)
	digestPattern  = regexp.MustCompile(`^sha256:[0-9a-f]{64}$`)
)

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
	if err := ensureJSONEOF(decoder); err != nil {
		return value, err
	}
	if err := value.Validate(); err != nil {
		return value, err
	}
	return value, nil
}

func ensureJSONEOF(decoder *json.Decoder) error {
	var trailing json.RawMessage
	if err := decoder.Decode(&trailing); errors.Is(err, io.EOF) {
		return nil
	} else if err != nil {
		return fmt.Errorf("decode trailing JSON: %w", err)
	}
	return invalidf("multiple JSON values are not allowed")
}

func validateAPIVersion(value string) error {
	if value != APIVersion {
		return fmt.Errorf("%w: %w", ErrValidation, privateVersionError)
	}
	return nil
}

func validateOpaqueID(field, value string) error {
	if strings.TrimSpace(value) == "" {
		return invalidf("%s must not be empty", field)
	}
	return nil
}

func validateSelector(field, value string) error {
	if strings.Count(value, "@") != 1 {
		return invalidf("%s must use exact <id>@<version> syntax", field)
	}
	id, version, _ := strings.Cut(value, "@")
	if !idPattern.MatchString(id) || !versionPattern.MatchString(version) {
		return invalidf("%s has an invalid exact selector", field)
	}
	return nil
}

func validateDigest(field, value string) error {
	if !digestPattern.MatchString(value) {
		return invalidf("%s must be sha256 followed by 64 lowercase hex characters", field)
	}
	return nil
}

func validateURL(field, value string) error {
	parsed, err := url.Parse(value)
	if err != nil || parsed.Host == "" || (parsed.Scheme != "http" && parsed.Scheme != "https") {
		return invalidf("%s must be an absolute HTTP(S) URL", field)
	}
	if parsed.User != nil {
		return invalidf("%s must not contain URL user information", field)
	}
	return nil
}

func invalidf(format string, args ...any) error {
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
