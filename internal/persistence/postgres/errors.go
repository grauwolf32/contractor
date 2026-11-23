package postgres

// safeError retains database/context identity for internal recovery without
// including SQL, connection credentials or provider payloads in Error().
type safeError struct {
	message string
	cause   error
}

func (e *safeError) Error() string { return e.message }
func (e *safeError) Unwrap() error { return e.cause }

// WrapError requires a static safe message. The cause remains discoverable
// through errors.Is/As and SQLState, but is never rendered implicitly.
func WrapError(message string, cause error) error {
	return &safeError{message: message, cause: cause}
}
