package contentdigest

import "errors"

// ErrPinnedSelectionChanged is a safe, stable precondition failure for trusted
// callers that prepared exact selections, such as the Workflow, Runtime
// configuration, Skills or Audit standards an Eval member was prepared with.
// Ordinary public creation leaves these optional expectations empty. An
// idempotent replay precedes these checks.
var ErrPinnedSelectionChanged = errors.New("prepared execution selections changed")

// CheckPinnedSelection returns ErrPinnedSelectionChanged unless expected is
// empty or equals the JSON digest of the current selection.
func CheckPinnedSelection(expected string, current any) error {
	if expected == "" {
		return nil
	}
	digest, err := JSON(current)
	if err != nil {
		return err
	}
	if expected != digest {
		return ErrPinnedSelectionChanged
	}
	return nil
}
