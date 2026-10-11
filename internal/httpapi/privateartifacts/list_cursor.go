package privateartifacts

import (
	"encoding/base64"
	"encoding/json"
	"fmt"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/strictjson"
)

const listCursorVersion = 1

// listCursor is the position behind an opaque private list cursor: the last
// binding of the previous page. It carries no authority; the allocation path
// still selects the RunScope and the store validates the position. The
// encoding is unpadded base64url of strict JSON so the Runtime treats it as an
// opaque token of the shared grammar (LIM-ART-5).
type listCursor struct {
	Version   int    `json:"v"`
	Namespace string `json:"namespace"`
	Name      string `json:"name"`
}

func encodeListCursor(last artifacts.ArtifactRef) (string, error) {
	payload, err := json.Marshal(listCursor{Version: listCursorVersion, Namespace: last.Namespace, Name: last.Name})
	if err != nil {
		return "", fmt.Errorf("encode artifact list cursor: %w", err)
	}
	encoded := base64.RawURLEncoding.EncodeToString(payload)
	if err := contracts.ValidateArtifactListCursor(encoded); err != nil {
		return "", fmt.Errorf("encode artifact list cursor: %w", err)
	}
	return encoded, nil
}

func decodeListCursor(encoded string) (listCursor, error) {
	invalid := fmt.Errorf("%w: invalid artifact list cursor", errInvalidRequest)
	if contracts.ValidateArtifactListCursor(encoded) != nil {
		return listCursor{}, invalid
	}
	payload, err := base64.RawURLEncoding.Strict().DecodeString(encoded)
	if err != nil {
		return listCursor{}, invalid
	}
	var cursor listCursor
	if strictjson.Decode(payload, &cursor) != nil || cursor.Version != listCursorVersion ||
		cursor.Namespace == "" || cursor.Name == "" {
		return listCursor{}, invalid
	}
	// Only the exact encoding this Server issues is accepted, which also
	// rejects duplicate keys, reordered fields and alternative escapes.
	if canonical, err := encodeListCursor(artifacts.ArtifactRef{Namespace: cursor.Namespace, Name: cursor.Name}); err != nil || canonical != encoded {
		return listCursor{}, invalid
	}
	return cursor, nil
}
