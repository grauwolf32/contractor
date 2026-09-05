package contracts

// WorkerSessionMode selects the allocation-local ADK conversation lifecycle.
// It is immutable Stage configuration and never a Planner or A2A task input.
type WorkerSessionMode string

const (
	WorkerSessionIsolated WorkerSessionMode = "isolated"
	WorkerSessionShared   WorkerSessionMode = "shared"
)

func (m WorkerSessionMode) Validate() error {
	switch m {
	case WorkerSessionIsolated, WorkerSessionShared:
		return nil
	default:
		return invalidf("workerSessionMode must be isolated or shared")
	}
}
