package postgres

// SQLSTATE codes that Contractor classifies.
const (
	SQLStateStringDataRightTruncation = "22001"
	SQLStateInvalidTextRepresentation = "22P02"
	SQLStateNotNullViolation          = "23502"
	SQLStateForeignKeyViolation       = "23503"
	SQLStateUniqueViolation           = "23505"
	SQLStateCheckViolation            = "23514"
	SQLStateExclusionViolation        = "23P01"
	SQLStateSerializationFailure      = "40001"
	SQLStateDeadlockDetected          = "40P01"
	SQLStateInsufficientPrivilege     = "42501"
	SQLStateTooManyConnections        = "53300"
	SQLStateLockNotAvailable          = "55P03"
	SQLStateQueryCanceled             = "57014"
)

// ConstraintError maps a unique violation to conflict and a foreign-key,
// check, length or text-representation violation to invalid. It returns nil
// for every other error so callers keep their own fallback.
func ConstraintError(err error, conflict, invalid error) error {
	switch SQLState(err) {
	case SQLStateUniqueViolation:
		return conflict
	case SQLStateForeignKeyViolation, SQLStateCheckViolation,
		SQLStateStringDataRightTruncation, SQLStateInvalidTextRepresentation:
		return invalid
	default:
		return nil
	}
}
