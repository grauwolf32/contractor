package public

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"mime"
	"net/http"
	"net/url"
	"regexp"
	"strconv"
	"strings"

	"github.com/grauwolf32/contractor/internal/artifacts"
)

const maxJSONRequestSize = 1 << 20

var (
	publicArtifactNamePattern = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$`)
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
	value := strings.TrimSpace(ifMatch[0])
	if strings.HasPrefix(value, "W/") || strings.Contains(value, ",") {
		return nil, fmt.Errorf("%w: If-Match requires one strong revision ETag", errInvalidRequest)
	}
	revision, err := strconv.Unquote(value)
	if err != nil || revision == "" {
		return nil, fmt.Errorf("%w: If-Match requires one quoted revision", errInvalidRequest)
	}
	return &revision, nil
}

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

func quotedETag(revision *string) string {
	if revision == nil {
		return ""
	}
	return strconv.Quote(*revision)
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
