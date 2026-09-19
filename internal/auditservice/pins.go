package auditservice

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
)

// ErrPinnedSelectionChanged is a safe, stable precondition failure for trusted
// callers that prepared exact selections. Ordinary public creation leaves these
// optional expectations empty. An idempotent replay precedes these checks.
var ErrPinnedSelectionChanged = errors.New("prepared execution selections changed")

func checkExpectedDigest(expected string, value any) error {
	if expected == "" {
		return nil
	}
	raw, err := json.Marshal(value)
	if err != nil {
		return err
	}
	sum := sha256.Sum256(raw)
	if expected != "sha256:"+hex.EncodeToString(sum[:]) {
		return ErrPinnedSelectionChanged
	}
	return nil
}
