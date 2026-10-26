package auditservice

import "errors"

// ErrNextRoundAdmissionPending means a bounded admission step committed and
// its durable generation needs another reconciliation before acceptance.
// It is progress, not an empty candidate set or a target execution failure.
var ErrNextRoundAdmissionPending = errors.New("next Audit Round admission is pending")
