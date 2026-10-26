package auditservice

import "errors"

// ErrNextRoundAdmissionPending means a bounded admission step committed and
// its durable generation needs another reconciliation before acceptance.
// It is progress, not an empty candidate set or a target execution failure.
var ErrNextRoundAdmissionPending = errors.New("next Audit Round admission is pending")

// ErrNextRoundAdmissionWaiting keeps an unchanged blocked generation pending
// until its exact readiness condition changes, without reporting another write.
var ErrNextRoundAdmissionWaiting = errors.New("next Audit Round admission awaits exact source readiness")
