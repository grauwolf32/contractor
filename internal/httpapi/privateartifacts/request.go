package privateartifacts

import (
	"errors"
	"fmt"
	"io"
	"mime"
	"net/http"
	"net/url"
	"strconv"
	"strings"

	"github.com/grauwolf32/contractor/internal/artifacts"
)

func exactQuery(raw string, allowed ...string) (url.Values, error) {
	values, err := url.ParseQuery(raw)
	if err != nil {
		return nil, fmt.Errorf("%w: invalid query string", errInvalidRequest)
	}
	accepted := make(map[string]struct{}, len(allowed))
	for _, key := range allowed {
		accepted[key] = struct{}{}
	}
	for key, entries := range values {
		if _, ok := accepted[key]; !ok || len(entries) != 1 {
			return nil, fmt.Errorf("%w: unsupported or repeated query parameter", errInvalidRequest)
		}
	}
	return values, nil
}

func requestMediaType(r *http.Request) (string, error) {
	values := r.Header.Values("Content-Type")
	if len(values) != 1 {
		return "", fmt.Errorf("%w: exactly one Content-Type is required", errInvalidRequest)
	}
	mediaType, parameters, err := mime.ParseMediaType(values[0])
	if err != nil || len(parameters) != 0 || mediaType != strings.ToLower(mediaType) {
		return "", fmt.Errorf("%w: Content-Type must be lowercase type/subtype without parameters", errInvalidRequest)
	}
	return mediaType, nil
}

func readArtifactBody(w http.ResponseWriter, r *http.Request) ([]byte, error) {
	if r.ContentLength < -1 || r.ContentLength > artifacts.MaxPayloadSize {
		return nil, artifacts.ErrPayloadTooLarge
	}
	body := http.MaxBytesReader(w, r.Body, artifacts.MaxPayloadSize)
	data, err := io.ReadAll(body)
	if err != nil {
		var maximum *http.MaxBytesError
		if errors.As(err, &maximum) {
			return nil, artifacts.ErrPayloadTooLarge
		}
		return nil, errors.New("read private artifact body")
	}
	return data, nil
}

func artifactWritePrecondition(r *http.Request) (expectedRevision *string, create bool, err error) {
	ifMatch := r.Header.Values("If-Match")
	ifNoneMatch := r.Header.Values("If-None-Match")
	if len(ifMatch) > 0 && len(ifNoneMatch) > 0 {
		return nil, false, fmt.Errorf("%w: If-Match and If-None-Match are mutually exclusive", errInvalidRequest)
	}
	if len(ifNoneMatch) > 0 {
		if len(ifNoneMatch) != 1 || strings.TrimSpace(ifNoneMatch[0]) != "*" {
			return nil, false, fmt.Errorf("%w: If-None-Match only supports *", errInvalidRequest)
		}
		return nil, true, nil
	}
	if len(ifMatch) != 1 {
		return nil, false, fmt.Errorf("%w: exactly one create/update precondition is required", errInvalidRequest)
	}
	value := strings.TrimSpace(ifMatch[0])
	if strings.HasPrefix(value, "W/") || strings.Contains(value, ",") {
		return nil, false, fmt.Errorf("%w: If-Match requires one strong revision ETag", errInvalidRequest)
	}
	revision, unquoteErr := strconv.Unquote(value)
	if unquoteErr != nil || revision == "" {
		return nil, false, fmt.Errorf("%w: If-Match requires one quoted revision", errInvalidRequest)
	}
	return &revision, false, nil
}

func quotedETag(revision *string) string {
	if revision == nil {
		return ""
	}
	return strconv.Quote(*revision)
}
