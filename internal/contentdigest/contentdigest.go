// Package contentdigest formats the "sha256:<hex>" digests that Contractor
// persists for content and idempotent request identities.
package contentdigest

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/grauwolf32/contractor/internal/strictjson"
)

// Bytes returns the "sha256:<lowercase hex>" digest of data.
func Bytes(data []byte) string {
	sum := sha256.Sum256(data)
	return "sha256:" + hex.EncodeToString(sum[:])
}

// JSON returns the Bytes digest of the encoding/json encoding of value, so the
// shape of the value a caller passes is part of the identity it digests.
func JSON(value any) (string, error) {
	encoded, err := json.Marshal(value)
	if err != nil {
		return "", err
	}
	return Bytes(encoded), nil
}

// CanonicalJSON returns the Bytes digest of the RFC 8785 canonical JSON of
// value, after encoding/json has reduced it to the I-JSON data model, so two
// values with the same JSON meaning have one digest. Configuration manifest
// digests use it.
func CanonicalJSON(value any) (string, error) {
	canonical, err := strictjson.Canonical(value)
	if err != nil {
		return "", fmt.Errorf("canonicalize digest input: %w", err)
	}
	return Bytes(canonical), nil
}

// Valid reports whether value is a "sha256:<64 lowercase hex>" digest, the
// only spelling Bytes produces.
func Valid(value string) bool {
	digits, ok := strings.CutPrefix(value, "sha256:")
	if !ok || len(digits) != sha256.Size*2 {
		return false
	}
	for _, digit := range []byte(digits) {
		if (digit < '0' || digit > '9') && (digit < 'a' || digit > 'f') {
			return false
		}
	}
	return true
}
