package public

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"regexp"
	"strings"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/httpapi/httpx"
)

const maxJSONRequestSize = 1 << 20

var (
	publicArtifactNamePattern = regexp.MustCompile(contracts.ArtifactNamePattern)
	publicRevisionPattern     = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}$`)
)

func decodeJSON(w http.ResponseWriter, r *http.Request, target any) error {
	return decodeJSONBounded(w, r, target, maxJSONRequestSize)
}

func decodeJSONBounded(w http.ResponseWriter, r *http.Request, target any, maximum int64) error {
	if maximum <= 0 || r.ContentLength > maximum {
		return fmt.Errorf("%w: JSON body is too large", errInvalidRequest)
	}
	body := http.MaxBytesReader(w, r.Body, maximum)
	decoder := json.NewDecoder(body)
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(target); err != nil {
		var maximum *http.MaxBytesError
		if errors.As(err, &maximum) {
			return fmt.Errorf("%w: JSON body is too large", errInvalidRequest)
		}
		return fmt.Errorf("%w: decode JSON body: %v", errInvalidRequest, err)
	}
	var trailing json.RawMessage
	if err := decoder.Decode(&trailing); !errors.Is(err, io.EOF) {
		if err == nil {
			return fmt.Errorf("%w: multiple JSON values", errInvalidRequest)
		}
		return fmt.Errorf("%w: decode trailing JSON: %v", errInvalidRequest, err)
	}
	return nil
}

func decodeStrictPublicJSON(data []byte, target any) error {
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(target); err != nil {
		return err
	}
	var trailing json.RawMessage
	if err := decoder.Decode(&trailing); !errors.Is(err, io.EOF) {
		return errors.New("JSON value has trailing data")
	}
	return nil
}

// jsonMembers holds the raw top-level members of a JSON object so that an
// absent member can be told apart from an explicit null.
type jsonMembers map[string]json.RawMessage

// null reports whether name is present with a JSON null value.
func (m jsonMembers) null(name string) bool {
	value, present := m[name]
	return present && isJSONNull(value)
}

// missingOrNull reports whether name is absent or a JSON null.
func (m jsonMembers) missingOrNull(name string) bool {
	value, present := m[name]
	return !present || isJSONNull(value)
}

func isJSONNull(raw []byte) bool {
	return bytes.Equal(bytes.TrimSpace(raw), []byte("null"))
}

// decodeStrictWithPresence strictly decodes one JSON value into target and
// also returns its top-level members for presence and null checks. A JSON
// null decodes to nil members.
func decodeStrictWithPresence(data []byte, target any) (jsonMembers, error) {
	if err := decodeStrictPublicJSON(data, target); err != nil {
		return nil, err
	}
	var members jsonMembers
	if err := json.Unmarshal(data, &members); err != nil {
		return nil, err
	}
	return members, nil
}

func requestMediaType(r *http.Request) (string, error) {
	return httpx.RequestMediaType(errInvalidRequest, r)
}

// readArtifactBody deliberately differs from the private adapter: it keeps
// the read cause and does not special-case negative lengths.
func readArtifactBody(w http.ResponseWriter, r *http.Request) ([]byte, error) {
	if r.ContentLength > artifacts.MaxPayloadSize {
		return nil, artifacts.ErrPayloadTooLarge
	}
	body := http.MaxBytesReader(w, r.Body, artifacts.MaxPayloadSize)
	data, err := io.ReadAll(body)
	if err != nil {
		var maximum *http.MaxBytesError
		if errors.As(err, &maximum) {
			return nil, artifacts.ErrPayloadTooLarge
		}
		return nil, fmt.Errorf("read artifact body: %w", err)
	}
	return data, nil
}

func artifactWritePrecondition(r *http.Request) (*string, error) {
	ifMatch := r.Header.Values("If-Match")
	ifNoneMatch := r.Header.Values("If-None-Match")
	if len(ifMatch) > 0 && len(ifNoneMatch) > 0 {
		return nil, fmt.Errorf("%w: If-Match and If-None-Match are mutually exclusive", errInvalidRequest)
	}
	if len(ifNoneMatch) > 0 {
		if len(ifNoneMatch) != 1 || strings.TrimSpace(ifNoneMatch[0]) != "*" {
			return nil, fmt.Errorf("%w: If-None-Match only supports *", errInvalidRequest)
		}
		return nil, nil
	}
	if len(ifMatch) == 0 {
		return nil, nil
	}
	if len(ifMatch) != 1 {
		return nil, fmt.Errorf("%w: exactly one If-Match value is allowed", errInvalidRequest)
	}
	revision, err := httpx.ParseStrongETag(ifMatch[0])
	switch {
	case errors.Is(err, httpx.ErrWeakETag):
		return nil, fmt.Errorf("%w: If-Match requires one strong revision ETag", errInvalidRequest)
	case err != nil:
		return nil, fmt.Errorf("%w: If-Match requires one quoted revision", errInvalidRequest)
	}
	return &revision, nil
}

func exactQuery(raw string, allowed ...string) (url.Values, error) {
	return httpx.ExactQuery(errInvalidRequest, raw, allowed...)
}

func exactQueryWithRepeated(
	raw string, repeatedKey string, maximum int, allowed ...string,
) (url.Values, error) {
	return httpx.ExactQueryWithRepeated(errInvalidRequest, raw, repeatedKey, maximum, allowed...)
}

func validatePublicArtifactName(value string) error {
	if !publicArtifactNamePattern.MatchString(value) {
		return fmt.Errorf("%w: invalid Artifact namespace or name", errInvalidRequest)
	}
	return nil
}

func validatePublicRevision(value string) error {
	if !publicRevisionPattern.MatchString(value) {
		return fmt.Errorf("%w: invalid Artifact revision", errInvalidRequest)
	}
	return nil
}
