package auditservice

import "errors"

// ErrRoundPreparationInconsistent classifies a Round preparation state that
// retained Audit data cannot legitimately reach, such as a drifted proposal
// descriptor or an undecodable pinned snapshot. Retrying cannot repair it.
var ErrRoundPreparationInconsistent = errors.New("Audit Round preparation reached an inconsistent state")

// RoundPreparationError reports one inconsistent Round preparation state.
// Diagnostic is a fixed phrase naming the violated invariant and never holds
// stored or caller-controlled text. Cause, when present, only adds detail to
// the error text for operators; the error unwraps to
// ErrRoundPreparationInconsistent alone.
type RoundPreparationError struct {
	Diagnostic string
	Cause      error
}

func inconsistentRound(diagnostic string, cause error) error {
	return &RoundPreparationError{Diagnostic: diagnostic, Cause: cause}
}

func (e *RoundPreparationError) Error() string {
	message := ErrRoundPreparationInconsistent.Error() + ": " + e.Diagnostic
	if e.Cause != nil {
		message += ": " + e.Cause.Error()
	}
	return message
}

func (e *RoundPreparationError) Unwrap() error { return ErrRoundPreparationInconsistent }
