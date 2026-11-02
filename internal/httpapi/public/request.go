package public

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/helpers/strictjson"
	"github.com/grauwolf32/contractor/internal/httpapi/artifacttransfer"
	"github.com/grauwolf32/contractor/internal/httpapi/httpx"
)

const maxJSONRequestSize = 1 << 20

const idempotencyKeyHeader = "Idempotency-Key"

// validateOptionalIdempotencyKey checks the OptionalIdempotencyKey of an
// operation that is idempotent by resource identity and binds no key
// (API-R28): it may be absent, and a supplied one is one header in the
// Idempotency-Key grammar.
func validateOptionalIdempotencyKey(r *http.Request) error {
	if len(r.Header.Values(idempotencyKeyHeader)) == 0 {
		return nil
	}
	_, err := requireIdempotencyKey(r)
	return err
}

// requireIdempotencyKey returns the request's one Idempotency-Key header in
// the public IdempotencyKey grammar (API-R69).
func requireIdempotencyKey(r *http.Request) (string, error) {
	values := r.Header.Values(idempotencyKeyHeader)
	if len(values) != 1 {
		return "", fmt.Errorf("%w: exactly one Idempotency-Key is required", errInvalidRequest)
	}
	if !contracts.ValidIdempotencyKey(values[0]) {
		return "", fmt.Errorf("%w: Idempotency-Key must be 1 to %d characters [A-Za-z0-9][A-Za-z0-9._:-]*",
			errInvalidRequest, contracts.MaxIdempotencyKeyLength)
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
	data, err := readJSONBody(w, r, maximum)
	if err != nil {
		return err
	}
	return decodeRequestJSON(data, target)
}

// decodeOpaqueJSONBounded preserves arbitrary secret material, which is hashed
// or encrypted as bytes rather than stored as jsonb. Login, Git SSH keys and
// Runtime credential material use this exception; their identifiers are validated by
// their owning contracts before storage.
func decodeOpaqueJSONBounded(w http.ResponseWriter, r *http.Request, target any, maximum int64) error {
	data, err := readBoundedBody(w, r, maximum)
	if err != nil {
		return err
	}
	defer clear(data)
	return decodeRequestJSON(data, target)
}

func decodeRequestJSON(data []byte, target any) error {
	switch err := strictjson.Decode(data, target); {
	case errors.Is(err, strictjson.ErrTrailingData):
		return fmt.Errorf("%w: multiple JSON values", errInvalidRequest)
	case err != nil:
		return fmt.Errorf("%w: decode JSON body: %v", errInvalidRequest, err)
	}
	return nil
}

// readJSONBody reads a JSON request body of at most maximum bytes. A larger
// body is 413 request_too_large, unread when its declared length is already
// too large (FAIL-API-1). Any other read failure, such as a body that ends
// before its declared length, is the client's 400 invalid_request, never a
// retryable internal error (API-R112); the bytes read so far are cleared.
func readJSONBody(w http.ResponseWriter, r *http.Request, maximum int64) ([]byte, error) {
	data, err := readBoundedBody(w, r, maximum)
	if err != nil {
		return nil, err
	}
	// PostgreSQL cannot represent U+0000 in jsonb or text. Refuse it before
	// a service can claim a mutation key, including JSON kept as bytes and
	// cast by a later reader (API-R71). A literal backslash is still data.
	if strictjson.HasNULEscape(data) {
		clear(data)
		return nil, fmt.Errorf("%w: JSON body contains a NUL escape", errInvalidRequest)
	}
	return data, nil
}

func readBoundedBody(w http.ResponseWriter, r *http.Request, maximum int64) ([]byte, error) {
	if maximum <= 0 || r.ContentLength > maximum {
		return nil, fmt.Errorf("%w: JSON body is too large", errRequestTooLarge)
	}
	data, err := io.ReadAll(http.MaxBytesReader(w, r.Body, maximum))
	if err != nil {
		clear(data)
		if exceedsBodyLimit(err) {
			return nil, fmt.Errorf("%w: JSON body is too large", errRequestTooLarge)
		}
		return nil, fmt.Errorf("%w: read JSON body: %v", errInvalidRequest, err)
	}
	return data, nil
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

// requireJSONMediaType is the media-type check of every operation whose
// request body is JSON: exactly one Content-Type, application/json without
// parameters. It runs with the query and path checks, before the mutation
// headers, any store read and the body (API-R112). The caller reports a
// failure with its operation's media-type answer: 400 invalid_request, the
// 415 of the Operations settings or the Managed Eval 422 eval_invalid.
func requireJSONMediaType(r *http.Request) error {
	if mediaType, err := requestMediaType(r); err != nil || mediaType != "application/json" {
		return fmt.Errorf("%w: Content-Type must be application/json", errInvalidRequest)
	}
	return nil
}

// requireOptionalJSONMediaType is requireJSONMediaType for an operation whose
// JSON body is optional: a request whose framing announces no body, with no
// positive Content-Length and no Transfer-Encoding, needs no Content-Type.
func requireOptionalJSONMediaType(r *http.Request) error {
	if r.ContentLength == 0 && len(r.TransferEncoding) == 0 {
		return nil
	}
	return requireJSONMediaType(r)
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

// parseRuntimeRevisionETag reads the If-Match of a resource whose revision is
// a positive decimal counter through the one shared reader
// (httpx.ParseRevisionETag); a malformed value is 400 invalid_request, never
// a revision conflict (API-R70).
func parseRuntimeRevisionETag(raw string) (uint64, error) {
	revision, err := httpx.ParseRevisionETag(raw)
	switch {
	case errors.Is(err, httpx.ErrWeakETag):
		return 0, fmt.Errorf("%w: revision precondition requires one strong ETag", errInvalidRequest)
	case errors.Is(err, httpx.ErrUnquotedETag):
		return 0, fmt.Errorf("%w: revision precondition must be quoted", errInvalidRequest)
	case err != nil:
		return 0, fmt.Errorf("%w: revision precondition must be a positive decimal revision", errInvalidRequest)
	}
	return revision, nil
}

// requiredRevision returns the revision of the one If-Match, without an
// If-None-Match, that operation requires (API-R70).
func requiredRevision(r *http.Request, operation string) (uint64, error) {
	raw, err := httpx.RequiredIfMatch(r.Header)
	if err != nil {
		return 0, fmt.Errorf("%w: %s requires one If-Match", errInvalidRequest, operation)
	}
	return parseRuntimeRevisionETag(raw)
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

func requireEmptyBody(w http.ResponseWriter, r *http.Request) error {
	data, err := io.ReadAll(http.MaxBytesReader(w, r.Body, 1))
	if err != nil || len(data) != 0 {
		return fmt.Errorf("%w: request body must be empty", errInvalidRequest)
	}
	return nil
}
