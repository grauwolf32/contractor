// Package httpx holds request-parsing helpers shared by the public and
// private Server HTTP adapters. Helpers that classify a request as invalid
// wrap the caller's own sentinel, so each adapter keeps its error identity
// and its error-to-status mapping.
package httpx

import (
	"errors"
	"fmt"
	"mime"
	"net/http"
	"net/url"
	"strconv"
	"strings"
)

var (
	// ErrWeakETag reports an entity tag that is weak or a list of tags.
	ErrWeakETag = errors.New("entity tag is not one strong ETag")
	// ErrUnquotedETag reports an entity tag that is not one non-empty quoted value.
	ErrUnquotedETag = errors.New("entity tag is not one quoted value")
)

// ExactQuery parses raw and rejects parameters outside allowed as well as
// repeated parameters. Failures wrap invalid.
func ExactQuery(invalid error, raw string, allowed ...string) (url.Values, error) {
	return ExactQueryWithRepeated(invalid, raw, "", 0, allowed...)
}

// ExactQueryWithRepeated is ExactQuery, except that repeatedKey (when set)
// may occur between one and maximum times.
func ExactQueryWithRepeated(
	invalid error, raw string, repeatedKey string, maximum int, allowed ...string,
) (url.Values, error) {
	values, err := url.ParseQuery(raw)
	if err != nil {
		return nil, fmt.Errorf("%w: invalid query string", invalid)
	}
	accepted := make(map[string]struct{}, len(allowed))
	for _, key := range allowed {
		accepted[key] = struct{}{}
	}
	for key, entries := range values {
		if repeatedKey != "" && key == repeatedKey {
			if len(entries) == 0 || len(entries) > maximum {
				return nil, fmt.Errorf("%w: repeated query parameter exceeds its bound", invalid)
			}
			continue
		}
		if _, ok := accepted[key]; !ok || len(entries) != 1 {
			return nil, fmt.Errorf("%w: unsupported or repeated query parameter", invalid)
		}
	}
	return values, nil
}

// RequestMediaType returns the single, lowercase, parameterless Content-Type
// of r. Failures wrap invalid.
func RequestMediaType(invalid error, r *http.Request) (string, error) {
	values := r.Header.Values("Content-Type")
	if len(values) != 1 {
		return "", fmt.Errorf("%w: exactly one Content-Type is required", invalid)
	}
	mediaType, parameters, err := mime.ParseMediaType(values[0])
	if err != nil || len(parameters) != 0 || mediaType != strings.ToLower(mediaType) {
		return "", fmt.Errorf("%w: Content-Type must be lowercase type/subtype without parameters", invalid)
	}
	return mediaType, nil
}

// ParseStrongETag returns the value of one strong, quoted, non-empty entity
// tag. It returns ErrWeakETag for a weak tag or a tag list and
// ErrUnquotedETag for anything else that is not one quoted value; callers
// attach their own contract message.
func ParseStrongETag(raw string) (string, error) {
	value := strings.TrimSpace(raw)
	if strings.HasPrefix(value, "W/") || strings.Contains(value, ",") {
		return "", ErrWeakETag
	}
	unquoted, err := strconv.Unquote(value)
	if err != nil || unquoted == "" {
		return "", ErrUnquotedETag
	}
	return unquoted, nil
}

// QuotedETag renders an optional revision as an ETag header value; a nil
// revision renders as the empty string.
func QuotedETag(revision *string) string {
	if revision == nil {
		return ""
	}
	return strconv.Quote(*revision)
}
