// Package scan holds the scan contracts of api/scan/v1: the HTTPRequestSet
// artifact of neutral prepared HTTP requests, and the ScanPlanPolicy that
// bounds how scan plans select those requests and scanner jobs.
package scan

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/url"
	"reflect"
	"regexp"
	"strconv"
	"strings"
	"unicode/utf8"

	"github.com/grauwolf32/contractor/internal/contentdigest"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/strictjson"
)

const (
	HTTPRequestSetMediaType        = "application/vnd.contractor.http-requests+json"
	HTTPRequestSetSchemaVersion    = 1
	MaxHTTPRequestSetBytes         = 4 * 1024 * 1024
	MaxHTTPRequestSetRequests      = 1000
	MaxHTTPRequestSetOrigins       = 1000
	MaxHTTPRequestSetGaps          = 4096
	MaxHTTPRequestURLBytes         = 8192
	MaxHTTPRequestBodyBytes        = 64 * 1024
	MaxHTTPRequestHeaders          = 64
	MaxHTTPRequestHeaderNameBytes  = 128
	MaxHTTPRequestHeaderValueBytes = 8192
	MaxHTTPRequestHeaderBytes      = 32 * 1024
)

// HTTPRequestSet is neutral preparation output, not a scanner invocation. A
// consumer must select scan parameters and apply its adapter's validation.
type HTTPRequestSet struct {
	SchemaVersion     int                `json:"schemaVersion"`
	Source            RequestSetSource   `json:"source"`
	PreparationDigest string             `json:"preparationDigest"`
	Requests          []RequestSetEntry  `json:"requests"`
	Gaps              []PreparationGap   `json:"gaps"`
	Coverage          RequestSetCoverage `json:"coverage"`
}

type RequestSetSource struct {
	Artifact      contracts.ArtifactRef `json:"artifact"`
	ContentDigest string                `json:"contentDigest"`
}

type RequestSetEntry struct {
	ID            string              `json:"id"`
	ContentDigest string              `json:"contentDigest"`
	Request       PreparedHTTPRequest `json:"request"`
	Origins       []RequestOrigin     `json:"origins"`
}

type PreparedHTTPRequest struct {
	Method  string              `json:"method"`
	URL     string              `json:"url"`
	Headers []HTTPRequestHeader `json:"headers"`
	Body    string              `json:"body"`
}

type HTTPRequestHeader struct {
	Name  string `json:"name"`
	Value string `json:"value"`
}

type RequestOrigin struct {
	Pointer string `json:"pointer"`
}

type PreparationGap struct {
	Pointer string `json:"pointer"`
	Code    string `json:"code"`
}

type RequestSetCoverage struct {
	Operations int  `json:"operations"`
	Prepared   int  `json:"prepared"`
	Skipped    int  `json:"skipped"`
	Complete   bool `json:"complete"`
}

var (
	requestHeaderNamePattern = regexp.MustCompile("^[!#$%&'*+.^_`|~0-9a-z-]+$")
	requestPointerPattern    = regexp.MustCompile(`^#(?:/(?:[^~]|~[01])*)*$`)
	requestGapCodePattern    = regexp.MustCompile(`^[a-z][a-z0-9_]*$`)
)

// RequestContentDigest hashes the exact RFC 8785 canonical request object. The
// caller must supply normalized headers; validation never rewrites content.
func RequestContentDigest(request PreparedHTTPRequest) (string, error) {
	if err := request.Validate(); err != nil {
		return "", err
	}
	data, err := contracts.MarshalPrivateCanonical(request)
	if err != nil {
		return "", err
	}
	return contentdigest.Bytes(data), nil
}

func (r PreparedHTTPRequest) Validate() error {
	switch r.Method {
	case "GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS":
	default:
		return contracts.Invalidf("request method is not supported")
	}
	if len(r.URL) == 0 || len(r.URL) > MaxHTTPRequestURLBytes {
		return contracts.Invalidf("request URL exceeds its byte bound")
	}
	for _, c := range []byte(r.URL) {
		if c <= 32 || c >= 127 || c == '\\' {
			return contracts.Invalidf("request URL must be printable ASCII")
		}
	}
	u, err := url.Parse(r.URL)
	if err != nil || (u.Scheme != "http" && u.Scheme != "https") || u.Hostname() == "" || u.User != nil || strings.Contains(r.URL, "#") || u.Opaque != "" {
		return contracts.Invalidf("request URL must be absolute HTTP(S), without credentials or fragment")
	}
	if strings.HasSuffix(u.Host, ":") {
		return contracts.Invalidf("request URL port must be between 1 and 65535")
	}
	if port := u.Port(); port != "" {
		number, err := strconv.Atoi(port)
		if err != nil || number < 1 || number > 65535 {
			return contracts.Invalidf("request URL port must be between 1 and 65535")
		}
	}
	// url.Parse validates path escapes but deliberately leaves RawQuery opaque.
	if _, err := url.QueryUnescape(u.RawQuery); err != nil {
		return contracts.Invalidf("request URL query contains malformed percent encoding")
	}
	if r.Headers == nil || len(r.Headers) > MaxHTTPRequestHeaders {
		return contracts.Invalidf("request headers must be a non-null bounded array")
	}
	previous, total := "", 0
	for _, header := range r.Headers {
		if len(header.Name) > MaxHTTPRequestHeaderNameBytes || !requestHeaderNamePattern.MatchString(header.Name) || header.Name <= previous {
			return contracts.Invalidf("request headers must have unique, sorted lowercase token names")
		}
		if !utf8.ValidString(header.Value) || len(header.Value) > MaxHTTPRequestHeaderValueBytes {
			return contracts.Invalidf("request header value exceeds its UTF-8 byte bound")
		}
		for _, c := range []byte(header.Value) {
			if (c < 32 && c != '\t') || c == 127 {
				return contracts.Invalidf("request header value contains a control character")
			}
		}
		total += len(header.Name) + len(header.Value)
		previous = header.Name
	}
	if total > MaxHTTPRequestHeaderBytes {
		return contracts.Invalidf("request headers exceed their aggregate byte bound")
	}
	if !utf8.ValidString(r.Body) || len(r.Body) > MaxHTTPRequestBodyBytes {
		return contracts.Invalidf("request body exceeds its UTF-8 byte bound")
	}
	return nil
}

func (s HTTPRequestSet) Validate() error {
	if s.SchemaVersion != HTTPRequestSetSchemaVersion {
		return contracts.Invalidf("unsupported HTTPRequestSet schemaVersion")
	}
	if err := s.Source.Artifact.ValidateExact(); err != nil {
		return err
	}
	if !utf8.ValidString(*s.Source.Artifact.Revision) {
		return contracts.Invalidf("source revision must be valid UTF-8")
	}
	if err := contracts.ValidateDigest("source contentDigest", s.Source.ContentDigest); err != nil {
		return err
	}
	if err := contracts.ValidateDigest("preparationDigest", s.PreparationDigest); err != nil {
		return err
	}
	if s.Requests == nil || len(s.Requests) > MaxHTTPRequestSetRequests {
		return contracts.Invalidf("requests must be a non-null bounded array")
	}
	if s.Gaps == nil || len(s.Gaps) > MaxHTTPRequestSetGaps {
		return contracts.Invalidf("gaps must be a non-null bounded array")
	}
	origins := make(map[string]struct{})
	previous := ""
	for _, entry := range s.Requests {
		digest, err := RequestContentDigest(entry.Request)
		if err != nil {
			return err
		}
		if entry.ContentDigest != digest || entry.ID != "request-"+strings.TrimPrefix(digest, "sha256:") {
			return contracts.Invalidf("request content does not match its digest and ID")
		}
		if entry.ID <= previous {
			return contracts.Invalidf("requests must be sorted by unique ID")
		}
		previous = entry.ID
		if len(entry.Origins) == 0 || len(entry.Origins) > MaxHTTPRequestSetOrigins {
			return contracts.Invalidf("request origins must be a non-empty bounded array")
		}
		previousPointer := ""
		for _, origin := range entry.Origins {
			if !validRequestPointer(origin.Pointer) || origin.Pointer == "#" || origin.Pointer <= previousPointer {
				return contracts.Invalidf("request origins must be sorted unique JSON pointers")
			}
			if _, exists := origins[origin.Pointer]; exists {
				return contracts.Invalidf("operation origin occurs in multiple requests")
			}
			origins[origin.Pointer] = struct{}{}
			if len(origins) > MaxHTTPRequestSetOrigins {
				return contracts.Invalidf("request origins exceed aggregate bound")
			}
			previousPointer = origin.Pointer
		}
	}
	previousPointer, previousCode := "", ""
	for _, gap := range s.Gaps {
		if !validRequestPointer(gap.Pointer) || len(gap.Code) > 64 || !requestGapCodePattern.MatchString(gap.Code) {
			return contracts.Invalidf("preparation gap must have a bounded JSON pointer and code")
		}
		if gap.Pointer < previousPointer || (gap.Pointer == previousPointer && gap.Code <= previousCode) {
			return contracts.Invalidf("preparation gaps must be sorted and unique by pointer and code")
		}
		previousPointer, previousCode = gap.Pointer, gap.Code
	}
	c := s.Coverage
	if c.Operations < 0 || c.Operations > MaxHTTPRequestSetOrigins || c.Prepared != len(origins) || c.Skipped < 0 || c.Skipped != c.Operations-c.Prepared || c.Complete != (c.Skipped == 0 && len(s.Gaps) == 0) {
		return contracts.Invalidf("coverage does not match prepared origins, skipped operations and gaps")
	}
	return nil
}

func validRequestPointer(pointer string) bool {
	return utf8.ValidString(pointer) && len(pointer) <= 8192 && requestPointerPattern.MatchString(pointer)
}

// DecodeHTTPRequestSet rejects missing, duplicate, unknown or null fields,
// noncanonical identities, incomplete coverage, and oversized artifacts.
func DecodeHTTPRequestSet(data []byte) (HTTPRequestSet, error) {
	var value HTTPRequestSet
	if len(data) > MaxHTTPRequestSetBytes || !utf8.Valid(data) {
		return value, contracts.Invalidf("HTTPRequestSet exceeds byte bound or is not UTF-8")
	}
	if err := strictjson.RejectDuplicateKeys(data); err != nil {
		return value, contracts.Invalidf("HTTPRequestSet is not strict JSON")
	}
	if err := requestSetJSONShape(data, reflect.TypeOf(value)); err != nil {
		return value, err
	}
	if err := json.Unmarshal(data, &value); err != nil {
		return value, contracts.Invalidf("HTTPRequestSet field type is invalid")
	}
	if err := value.Validate(); err != nil {
		return value, err
	}
	return value, nil
}

// MarshalHTTPRequestSet validates and serializes the complete artifact as RFC
// 8785 canonical JSON. Provenance digests describe source bytes and preparation
// settings; consumers cannot recompute them without those independent inputs.
func MarshalHTTPRequestSet(value HTTPRequestSet) ([]byte, error) {
	if err := value.Validate(); err != nil {
		return nil, err
	}
	data, err := contracts.MarshalPrivateCanonical(value)
	if err != nil {
		return nil, err
	}
	if len(data) > MaxHTTPRequestSetBytes {
		return nil, contracts.Invalidf("HTTPRequestSet exceeds byte bound")
	}
	return data, nil
}

// All fields in this artifact are mandatory, including zero-valued strings,
// booleans and arrays. Checking the raw shape prevents encoding/json's lenient
// missing/null handling and case-insensitive matching from weakening that rule.
func requestSetJSONShape(data json.RawMessage, kind reflect.Type) error {
	if bytes.Equal(bytes.TrimSpace(data), []byte("null")) {
		return contracts.Invalidf("HTTPRequestSet fields must not be null")
	}
	if kind.Kind() == reflect.Pointer {
		return requestSetJSONShape(data, kind.Elem())
	}
	switch kind.Kind() {
	case reflect.Struct:
		var fields map[string]json.RawMessage
		if err := json.Unmarshal(data, &fields); err != nil || len(fields) != kind.NumField() {
			return contracts.Invalidf("HTTPRequestSet object has missing or unknown fields")
		}
		for i := 0; i < kind.NumField(); i++ {
			field := kind.Field(i)
			name := strings.Split(field.Tag.Get("json"), ",")[0]
			raw, exists := fields[name]
			if !exists {
				return contracts.Invalidf("HTTPRequestSet object is missing field %s", name)
			}
			if err := requestSetJSONShape(raw, field.Type); err != nil {
				return err
			}
		}
	case reflect.Slice:
		var items []json.RawMessage
		if err := json.Unmarshal(data, &items); err != nil {
			return contracts.Invalidf("HTTPRequestSet array has invalid type")
		}
		for _, item := range items {
			if err := requestSetJSONShape(item, kind.Elem()); err != nil {
				return err
			}
		}
	case reflect.String:
		if !strictjson.ValidUnicodeEscapes(data) {
			return contracts.Invalidf("HTTPRequestSet string contains invalid Unicode")
		}
		if err := json.Unmarshal(data, reflect.New(kind).Interface()); err != nil {
			return contracts.Invalidf("HTTPRequestSet scalar has invalid type")
		}
	case reflect.Int, reflect.Bool:
		if err := json.Unmarshal(data, reflect.New(kind).Interface()); err != nil {
			return contracts.Invalidf("HTTPRequestSet scalar has invalid type")
		}
	default:
		return fmt.Errorf("unsupported HTTPRequestSet contract type %s", kind)
	}
	return nil
}
