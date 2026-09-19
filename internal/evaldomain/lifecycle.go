package evaldomain

import "errors"

// State is the experiment lifecycle, independent of execution and assessment
// states. Only this policy defines commands and observed transitions.
type State string

const (
	StateDraft       State = "draft"
	StatePreparing   State = "preparing"
	StateReady       State = "ready"
	StateRunning     State = "running"
	StatePausing     State = "pausing"
	StatePaused      State = "paused"
	StateSettling    State = "settling"
	StateFinished    State = "finished"
	StateCancelling  State = "cancelling"
	StateCancelled   State = "cancelled"
	StateInterrupted State = "interrupted"
)

type ControlMode string

const (
	ControlServer   ControlMode = "server"
	ControlExternal ControlMode = "external"
)

type CommandKind string

const (
	CommandPrepare   CommandKind = "prepare"
	CommandStart     CommandKind = "start"
	CommandPause     CommandKind = "pause"
	CommandResume    CommandKind = "resume"
	CommandCancel    CommandKind = "cancel"
	CommandFinalize  CommandKind = "finalize"
	CommandDuplicate CommandKind = "duplicate"
)

type Lifecycle struct {
	State             State
	ControlMode       ControlMode
	Outstanding       int
	HasPlan           bool
	HasDraft          bool
	DeletionRequested bool
}

func (state State) Terminal() bool {
	return state == StateFinished || state == StateCancelled
}

func (mode ControlMode) Allows(kind CommandKind) bool {
	switch kind {
	case CommandPrepare, CommandStart, CommandPause, CommandResume, CommandDuplicate:
		return mode == ControlServer
	case CommandFinalize:
		return mode == ControlExternal
	case CommandCancel:
		return mode == ControlServer || mode == ControlExternal
	default:
		return false
	}
}

// CommandTarget returns the same policy used to advertise available UI actions.
// Duplicate produces a separate draft and leaves this lifecycle unchanged.
func (l Lifecycle) CommandTarget(kind CommandKind) (State, error) {
	if l.DeletionRequested {
		return "", Failure("eval_project_deleting")
	}
	if !l.ControlMode.Allows(kind) {
		return "", Failure("eval_external_control")
	}
	switch kind {
	case CommandDuplicate:
		if l.HasDraft {
			return StateDraft, nil
		}
	case CommandPrepare:
		if l.State == StateDraft {
			return StatePreparing, nil
		}
	case CommandStart:
		if l.State == StateReady && l.HasPlan {
			return StateRunning, nil
		}
	case CommandPause:
		if l.State == StateRunning {
			return StatePausing, nil
		}
	case CommandResume:
		if l.State == StatePaused || l.State == StateInterrupted {
			return StateRunning, nil
		}
	case CommandCancel:
		if l.HasPlan && !l.State.Terminal() && l.State != StateCancelling {
			return StateCancelling, nil
		}
	case CommandFinalize:
		if l.State == StateReady || l.State == StateRunning {
			return StateSettling, nil
		}
	}
	return "", Failure("eval_not_ready")
}

func (l Lifecycle) AllowedCommands() []CommandKind {
	commands := []CommandKind{}
	for _, kind := range []CommandKind{
		CommandPrepare, CommandStart, CommandPause, CommandResume,
		CommandFinalize, CommandCancel, CommandDuplicate,
	} {
		if _, err := l.CommandTarget(kind); err == nil {
			commands = append(commands, kind)
		}
	}
	return commands
}

// BudgetStop applies until accepted executions drain, including settling and
// recovery after an interruption. Cancelling already carries the stop fence.
func (l Lifecycle) BudgetStop(exhausted bool) (State, bool) {
	if !exhausted || l.State.Terminal() || l.State == StateCancelling {
		return "", false
	}
	if l.Outstanding > 0 {
		return StateCancelling, true
	}
	switch l.State {
	case StateRunning, StatePaused, StatePausing:
		return StateSettling, true
	case StateInterrupted:
		return StateCancelling, true
	default:
		return "", false
	}
}

// Observed transitions belong to the coordinator. Command transitions and plan
// freeze use their own admission checks, but share these named states.
var ErrOutstandingExecutions = errors.New("accepted evaluation executions have not drained")

func (l Lifecycle) ValidateObservation(target State) error {
	if l.DeletionRequested && target != StateCancelled && target != StateCancelling && target != StateInterrupted {
		return Failure("eval_project_deleting")
	}
	if (target.Terminal() || target == StatePaused) && l.Outstanding != 0 {
		return ErrOutstandingExecutions
	}
	if !l.allowsObservation(target) {
		return Failure("eval_invalid")
	}
	return nil
}

func (l Lifecycle) allowsObservation(target State) bool {
	switch l.State {
	case StatePreparing:
		return target == StatePreparing || target == StateDraft || target == StateInterrupted
	case StateRunning:
		return target == StateRunning || target == StateSettling || target == StateCancelling || target == StateInterrupted
	case StatePausing:
		return target == StatePaused || target == StateSettling || target == StateCancelling || target == StateInterrupted
	case StatePaused:
		return target == StateCancelling || target == StateSettling
	case StateSettling:
		return target == StateFinished || target == StateCancelling || target == StateInterrupted
	case StateCancelling:
		return target == StateCancelled || target == StateInterrupted
	case StateInterrupted:
		return target == StateCancelling
	default:
		return false
	}
}

type CommandCompletion int

const (
	CommandPending CommandCompletion = iota
	CommandSucceeded
	CommandFailed
)

func (l Lifecycle) Completion(kind CommandKind) CommandCompletion {
	var pending, succeeded bool
	switch kind {
	case CommandStart, CommandResume:
		return CommandSucceeded // Their accepted transaction committed the transition.
	case CommandPrepare:
		pending, succeeded = l.State == StatePreparing, l.HasPlan
	case CommandPause:
		pending, succeeded = l.State == StatePausing, l.State == StatePaused
	case CommandCancel:
		pending, succeeded = l.State == StateCancelling, l.State == StateCancelled
	case CommandFinalize:
		pending, succeeded = l.State == StateSettling, l.State == StateFinished
	default:
		return CommandFailed
	}
	if pending {
		return CommandPending
	}
	if succeeded {
		return CommandSucceeded
	}
	return CommandFailed
}

// Admission never reopens settling. Existing accepted operations are recovered
// separately and must remain recoverable after the stop or deletion fence.
func (l Lifecycle) ValidateAdmission(exhausted bool, maxInFlight int) error {
	if l.DeletionRequested {
		return Failure("eval_project_deleting")
	}
	if l.State != StateRunning && !(l.ControlMode == ControlExternal && l.State == StateReady) {
		return Failure("eval_not_ready")
	}
	if exhausted {
		return Failure("eval_budget_exhausted")
	}
	if l.Outstanding >= maxInFlight {
		return Failure("eval_not_ready")
	}
	return nil
}
