package public

import (
	"crypto/hmac"
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"net/url"
	"strconv"
)

const (
	defaultPageLimit = 50
	maxPageLimit     = 200
	maxCursorBytes   = 512
	pageCursorV1     = 1
)

type pageInfoResponse struct {
	HasMore    bool    `json:"hasMore"`
	NextCursor *string `json:"nextCursor,omitempty"`
}

type pageCursor struct {
	Version int      `json:"v"`
	Kind    string   `json:"k"`
	Values  []string `json:"p"`
}

func pageQuery(raw string, extra ...string) (url.Values, int, string, error) {
	allowed := append([]string{"limit", "cursor"}, extra...)
	values, err := exactQuery(raw, allowed...)
	if err != nil {
		return nil, 0, "", err
	}
	limit := defaultPageLimit
	if entries, present := values["limit"]; present {
		parsed, parseErr := strconv.Atoi(entries[0])
		if parseErr != nil || parsed < 1 || parsed > maxPageLimit {
			return nil, 0, "", fmt.Errorf("%w: limit must be between 1 and %d", errInvalidRequest, maxPageLimit)
		}
		limit = parsed
	}
	var cursor string
	if entries, present := values["cursor"]; present {
		cursor = entries[0]
		if cursor == "" || len(cursor) > maxCursorBytes {
			return nil, 0, "", fmt.Errorf("%w: cursor is empty or too large", errInvalidRequest)
		}
	}
	return values, limit, cursor, nil
}

func (h *handler) encodePageCursor(kind string, values ...string) (string, error) {
	payload, err := json.Marshal(pageCursor{
		Version: pageCursorV1, Kind: pageCursorKind(kind), Values: values,
	})
	if err != nil {
		return "", fmt.Errorf("encode page cursor: %w", err)
	}
	mac := hmac.New(sha256.New, h.tokenDigest[:])
	_, _ = mac.Write(payload)
	signed := append(payload, mac.Sum(nil)...)
	encoded := base64.RawURLEncoding.EncodeToString(signed)
	if len(encoded) > maxCursorBytes {
		return "", fmt.Errorf("encode page cursor: result is too large")
	}
	return encoded, nil
}

func (h *handler) decodePageCursor(encoded, kind string, valueCount int) ([]string, error) {
	if encoded == "" {
		return nil, nil
	}
	signed, err := base64.RawURLEncoding.DecodeString(encoded)
	if err != nil || len(signed) <= sha256.Size {
		return nil, fmt.Errorf("%w: invalid cursor", errInvalidRequest)
	}
	payload, signature := signed[:len(signed)-sha256.Size], signed[len(signed)-sha256.Size:]
	mac := hmac.New(sha256.New, h.tokenDigest[:])
	_, _ = mac.Write(payload)
	if !hmac.Equal(signature, mac.Sum(nil)) {
		return nil, fmt.Errorf("%w: invalid cursor", errInvalidRequest)
	}
	var cursor pageCursor
	if err := json.Unmarshal(payload, &cursor); err != nil || cursor.Version != pageCursorV1 ||
		cursor.Kind != pageCursorKind(kind) || len(cursor.Values) != valueCount {
		return nil, fmt.Errorf("%w: invalid cursor", errInvalidRequest)
	}
	for _, value := range cursor.Values {
		if value == "" {
			return nil, fmt.Errorf("%w: invalid cursor", errInvalidRequest)
		}
	}
	return cursor.Values, nil
}

func pageCursorKind(value string) string {
	digest := sha256.Sum256([]byte(value))
	return base64.RawURLEncoding.EncodeToString(digest[:])
}
