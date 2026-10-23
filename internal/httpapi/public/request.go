package public

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strconv"
	"strings"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/httpapi/artifacttransfer"
	"github.com/grauwolf32/contractor/internal/httpapi/httpx"
	"github.com/grauwolf32/contractor/internal/strictjson"
)

const maxJSONRequestSize = 1 << 20

const idempotencyKeyHeader = "Idempotency-Key"

func requireIdempotencyKey(r *http.Request) (string, error) {
	values := r.Header.Values(idempotencyKeyHeader)
	if len(values) != 1 || len(values[0]) == 0 || len(values[0]) > 128 {
		return "", fmt.Errorf("%w: exactly one bounded Idempotency-Key is required", errInvalidRequest)
	}
	for index, character := range values[0] {
		valid := character >= 'a' && character <= 'z' || character >= 'A' && character <= 'Z' ||
			character >= '0' && character <= '9' || index > 0 && strings.ContainsRune("._:-", character)
		if !valid {
			return "", fmt.Errorf("%w: Idempotency-Key contains an invalid character", errInvalidRequest)
		}
	}
	return values[0], nil
}

func decodeJSON(w http.ResponseWriter, r *http.Request, target any) error {
	return decodeJSONBounded(w, r, target, maxJSONRequestSize)
}

// decodeJSONBounded reads at most maximum bytes and decodes them strictly:
// case aliases, duplicate keys, unknown members and trailing data are
// rejected rather than guessed (API-R71).
func decodeJSONBounded(w http.ResponseWriter, r *http.Request, target any, maximum int64) error {
	if maximum <= 0 || r.ContentLength > maximum {
		return fmt.Errorf("%w: JSON body is too large", errRequestTooLarge)
	}
	data, err := io.ReadAll(http.MaxBytesReader(w, r.Body, maximum))
	if err != nil {
		if exceedsBodyLimit(err) {
			return fmt.Errorf("%w: JSON body is too large", errRequestTooLarge)
		}
		return fmt.Errorf("%w: read JSON body: %v", errInvalidRequest, err)
	}
	switch err := strictjson.Decode(data, target); {
	case errors.Is(err, strictjson.ErrTrailingData):
		return fmt.Errorf("%w: multiple JSON values", errInvalidRequest)
	case err != nil:
		return fmt.Errorf("%w: decode JSON body: %v", errInvalidRequest, err)
	}
	return nil
}

// exceedsBodyLimit reports a read stopped by http.MaxBytesReader.
func exceedsBodyLimit(err error) bool {
	var maximum *http.MaxBytesError
	return errors.As(err, &maximum)
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
	if err := strictjson.Decode(data, target); err != nil {
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
func readArtifactBody(r *http.Request, transfer *artifacttransfer.Transfer) ([]byte, error) {
	data, err := transfer.ReadBody(r, artifacts.MaxPayloadSize)
	if err != nil && !errors.Is(err, artifacts.ErrPayloadTooLarge) {
		return nil, fmt.Errorf("read artifact body: %w", err)
	}
	return data, err
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
	// One plain quoted revision in the public Revision grammar, as for
	// ?revision (API-R70): a malformed value is a bad request, never a CAS
	// conflict with the current binding (ART-R70).
	if validatePublicRevision(revision) != nil {
		return nil, fmt.Errorf("%w: If-Match requires one quoted Artifact revision", errInvalidRequest)
	}
	return &revision, nil
}

// parseRuntimeRevisionETag reads the one strong If-Match of a resource whose
// revision is a positive decimal counter.
func parseRuntimeRevisionETag(raw string) (uint64, error) {
	revision, err := httpx.ParseStrongETag(raw)
	switch {
	case errors.Is(err, httpx.ErrWeakETag):
		return 0, fmt.Errorf("%w: binding revision requires one strong ETag", errInvalidRequest)
	case err != nil:
		return 0, fmt.Errorf("%w: binding revision must be quoted", errInvalidRequest)
	}
	parsed, err := strconv.ParseUint(revision, 10, 64)
	if err != nil || parsed == 0 || strconv.FormatUint(parsed, 10) != revision {
		return 0, fmt.Errorf("%w: binding revision is invalid", errInvalidRequest)
	}
	return parsed, nil
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
	if !contracts.ValidName(value) {
		return fmt.Errorf("%w: invalid Artifact namespace or name", errInvalidRequest)
	}
	return nil
}

func validatePublicRevision(value string) error {
	if !contracts.ValidResourceID(value) {
		return fmt.Errorf("%w: invalid Artifact revision", errInvalidRequest)
	}
	return nil
}
