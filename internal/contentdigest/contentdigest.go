// Package contentdigest formats the "sha256:<hex>" digests that Contractor
// persists for content and idempotent request identities.
package contentdigest

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"strings"
)

// Bytes returns the "sha256:<lowercase hex>" digest of data.
func Bytes(data []byte) string {
	sum := sha256.Sum256(data)
	return "sha256:" + hex.EncodeToString(sum[:])
}

// JSON returns the Bytes digest of the encoding/json encoding of value. The
// encoding is part of every stored request digest, so callers must not change
// the shape of the values they pass.
func JSON(value any) (string, error) {
	encoded, err := json.Marshal(value)
	if err != nil {
		return "", err
	}
	return Bytes(encoded), nil
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
