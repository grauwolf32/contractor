// Package randomid generates opaque identifiers from 128 bits of crypto/rand
// output.
package randomid

import (
	"crypto/rand"
	"encoding/hex"
)

// New returns prefix followed by 32 lowercase hex characters.
func New(prefix string) (string, error) {
	var raw [16]byte
	if _, err := rand.Read(raw[:]); err != nil {
		return "", err
	}
	return prefix + hex.EncodeToString(raw[:]), nil
}
