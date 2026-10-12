package reporting

// WorkerStateReadResult is one conditional read of a live Worker's State: a
// Snapshot and its strong ETag, or NotModified with the caller's ETag when
// the State has not changed.
type WorkerStateReadResult struct {
	Snapshot    *AgentStateSnapshot
	ETag        string
	NotModified bool
}

// WorkerStateReadError is a failed Worker State read. It deliberately retains
// no wrapped transport error, so it is safe to normalize into a model-facing
// projection failure.
type WorkerStateReadError struct {
	StatusCode int
	Code       string
	Retryable  bool
}

func (e *WorkerStateReadError) Error() string {
	return "Runtime Agent Worker State read failed (" + e.Code + ")"
}
