package contracts

import (
	"errors"

	"github.com/grauwolf32/contractor/internal/contentdigest"
)

// ErrPinnedSelectionChanged is a safe, stable precondition failure for trusted
// callers that prepared exact selections. Ordinary public creation leaves these
// optional expectations empty. An idempotent replay precedes these checks.
var ErrPinnedSelectionChanged = errors.New("prepared execution selections changed")

// CheckPinnedSelection returns ErrPinnedSelectionChanged unless expected is
// empty or equals the contentdigest.JSON digest of the current selection.
func CheckPinnedSelection(expected string, current any) error {
	if expected == "" {
		return nil
	}
	digest, err := contentdigest.JSON(current)
	if err != nil {
		return err
	}
	if expected != digest {
		return ErrPinnedSelectionChanged
	}
	return nil
}
