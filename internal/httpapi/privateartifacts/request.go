package privateartifacts

import (
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/httpapi/artifacttransfer"
	"github.com/grauwolf32/contractor/internal/httpapi/httpx"
	"github.com/grauwolf32/contractor/internal/strictjson"
)

func exactQuery(raw string, allowed ...string) (url.Values, error) {
	return httpx.ExactQuery(errInvalidRequest, raw, allowed...)
}

func requestMediaType(r *http.Request) (string, error) {
	return httpx.RequestMediaType(errInvalidRequest, r)
}

// decodeBoundedJSON strictly decodes exactly one JSON value of at most maximum
// bytes. Errors carry no body content; callers map them to their own codes.
func decodeBoundedJSON(w http.ResponseWriter, r *http.Request, maximum int64, target any) error {
	if r.ContentLength < -1 || r.ContentLength > maximum {
		return fmt.Errorf("%w: JSON body is too large", errInvalidRequest)
	}
	data, err := io.ReadAll(http.MaxBytesReader(w, r.Body, maximum))
	if err != nil {
		return fmt.Errorf("%w: read JSON body", errInvalidRequest)
	}
	if err := strictjson.Decode(data, target); errors.Is(err, strictjson.ErrTrailingData) {
		return fmt.Errorf("%w: JSON body has trailing data", errInvalidRequest)
	} else if err != nil {
		return fmt.Errorf("%w: decode JSON body", errInvalidRequest)
	}
	return nil
}

// readArtifactBody deliberately differs from the public adapter: it also
// rejects negative lengths other than "unknown" and hides the read cause.
func readArtifactBody(r *http.Request, transfer *artifacttransfer.Transfer) ([]byte, error) {
	if r.ContentLength < -1 {
		return nil, artifacts.ErrPayloadTooLarge
	}
	data, err := transfer.ReadBody(r, artifacts.MaxPayloadSize)
	if err != nil && !errors.Is(err, artifacts.ErrPayloadTooLarge) {
		return nil, errors.New("read private artifact body")
	}
	return data, err
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
