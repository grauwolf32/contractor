package runservice

import (
	"errors"

	"github.com/grauwolf32/contractor/internal/contentdigest"
)

// ErrPinnedSelectionChanged is a safe, stable precondition failure for trusted
// callers that prepared exact selections. Ordinary public creation leaves these
// optional expectations empty. An idempotent replay precedes these checks.
var ErrPinnedSelectionChanged = errors.New("prepared execution selections changed")

func checkExpectedDigest(expected string, value any) error {
	if expected == "" {
		return nil
	}
	digest, err := contentdigest.JSON(value)
	if err != nil {
		return err
	}
	if expected != digest {
		return ErrPinnedSelectionChanged
	}
	return nil
}
