package postgres

// IsTransientFailure reports a database wait or connection failure that a
// later attempt of the same operation may outlast: a serialization or
// deadlock abort, a lock, statement or pool-acquire timeout, or an
// unavailable connection or server (Classify). It does not prove that the
// failed attempt had no effect, because a timeout or lost connection can
// follow a commit. Only an idempotent callee, which accepts its own already
// committed outcome, may be retried on it.
func IsTransientFailure(err error) bool {
	switch Classify(err) {
	case ClassTransactionConflict, ClassTimeout, ClassUnavailable:
		return true
	default:
		return false
	}
}
