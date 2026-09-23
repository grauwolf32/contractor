package privateartifacts

import (
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/httpapi/httpx"
)

func exactQuery(raw string, allowed ...string) (url.Values, error) {
	return httpx.ExactQuery(errInvalidRequest, raw, allowed...)
}

func requestMediaType(r *http.Request) (string, error) {
	return httpx.RequestMediaType(errInvalidRequest, r)
}

// readArtifactBody deliberately differs from the public adapter: it also
// rejects negative lengths other than "unknown" and hides the read cause.
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
	revision, err := httpx.ParseStrongETag(ifMatch[0])
	switch {
	case errors.Is(err, httpx.ErrWeakETag):
		return nil, false, fmt.Errorf("%w: If-Match requires one strong revision ETag", errInvalidRequest)
	case err != nil:
		return nil, false, fmt.Errorf("%w: If-Match requires one quoted revision", errInvalidRequest)
	}
	return &revision, false, nil
}
