package public

import (
	"crypto/hmac"
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"net/url"
	"strconv"

	"github.com/grauwolf32/contractor/internal/pagelimit"
)

const (
	defaultPageLimit = 50
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
	return parsePageQuery(values)
}

func pageQueryWithRepeated(
	raw string, repeatedKey string, maximum int, extra ...string,
) (url.Values, int, string, error) {
	allowed := append([]string{"limit", "cursor"}, extra...)
	values, err := exactQueryWithRepeated(raw, repeatedKey, maximum, allowed...)
	if err != nil {
		return nil, 0, "", err
	}
	return parsePageQuery(values)
}

func parsePageQuery(values url.Values) (url.Values, int, string, error) {
	limit := defaultPageLimit
	if entries, present := values["limit"]; present {
		parsed, parseErr := strconv.Atoi(entries[0])
		if parseErr != nil || !pagelimit.ValidPage(parsed) {
			return nil, 0, "", fmt.Errorf("%w: limit must be between 1 and %d", errInvalidRequest, pagelimit.MaxRows)
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

// pageCursors seals and opens the continuation tokens of every paged
// collection. Each context that pages holds one; the key is the bearer token
// digest, so a token minted by one Server instance verifies on another with
// the same bearer token.
type pageCursors struct {
	key [sha256.Size]byte
}

// paginate trims rows fetched with pagelimit.FetchRows(limit) to one page. When the fetch
// returned a row beyond limit, the page reports hasMore with a cursor of the
// given kind positioned at the last kept row.
func paginate[T any](
	cursors pageCursors, rows []T, limit int, kind string, position func(last T) []string,
) ([]T, pageInfoResponse, error) {
	page := pageInfoResponse{}
	if len(rows) <= limit {
		return rows, page, nil
	}
	rows = rows[:limit]
	next, err := cursors.encodePageCursor(kind, position(rows[len(rows)-1])...)
	if err != nil {
		return nil, page, err
	}
	page.HasMore = true
	page.NextCursor = &next
	return rows, page, nil
}

func (c pageCursors) encodePageCursor(kind string, values ...string) (string, error) {
	payload, err := json.Marshal(pageCursor{
		Version: pageCursorV1, Kind: pageCursorKind(kind), Values: values,
	})
	if err != nil {
		return "", fmt.Errorf("encode page cursor: %w", err)
	}
	encoded := c.seal("", payload)
	if len(encoded) > maxCursorBytes {
		return "", fmt.Errorf("encode page cursor: result is too large")
	}
	return encoded, nil
}

func (c pageCursors) decodePageCursor(encoded, kind string, valueCount int) ([]string, error) {
	if encoded == "" {
		return nil, nil
	}
	payload, ok := c.open("", encoded)
	if !ok {
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

// seal renders payload followed by its MAC as unpadded base64url. Page
// cursors, eval cursors and eval bin tokens share this framing; domain
// separates token families (page and eval cursors use the empty domain).
func (c pageCursors) seal(domain string, payload []byte) string {
	signed := append(payload, c.mac(domain, payload)...)
	return base64.RawURLEncoding.EncodeToString(signed)
}

// open reverses seal and reports whether the MAC verified.
func (c pageCursors) open(domain, encoded string) ([]byte, bool) {
	signed, err := base64.RawURLEncoding.DecodeString(encoded)
	if err != nil || len(signed) <= sha256.Size {
		return nil, false
	}
	payload, signature := signed[:len(signed)-sha256.Size], signed[len(signed)-sha256.Size:]
	return payload, hmac.Equal(signature, c.mac(domain, payload))
}

// mac is HMAC-SHA256 over domain||payload keyed by the bearer token digest;
// an empty domain adds no bytes.
func (c pageCursors) mac(domain string, payload []byte) []byte {
	mac := hmac.New(sha256.New, c.key[:])
	_, _ = mac.Write([]byte(domain))
	_, _ = mac.Write(payload)
	return mac.Sum(nil)
}

func pageCursorKind(value string) string {
	digest := sha256.Sum256([]byte(value))
	return base64.RawURLEncoding.EncodeToString(digest[:])
}
