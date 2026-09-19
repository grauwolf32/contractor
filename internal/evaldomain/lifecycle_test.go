package evaldomain

import (
	"errors"
	"slices"
	"testing"
)

func TestLifecycleCommandsFollowControlAndState(t *testing.T) {
	tests := []struct {
		mode     ControlMode
		state    State
		commands []CommandKind
	}{
		{ControlServer, StateDraft, []CommandKind{CommandPrepare, CommandDuplicate}},
		{ControlServer, StateReady, []CommandKind{CommandStart, CommandCancel, CommandDuplicate}},
		{ControlServer, StateRunning, []CommandKind{CommandPause, CommandCancel, CommandDuplicate}},
		{ControlServer, StatePaused, []CommandKind{CommandResume, CommandCancel, CommandDuplicate}},
		{ControlServer, StateSettling, []CommandKind{CommandCancel, CommandDuplicate}},
		{ControlServer, StateFinished, []CommandKind{CommandDuplicate}},
		{ControlExternal, StateReady, []CommandKind{CommandFinalize, CommandCancel}},
		{ControlExternal, StateRunning, []CommandKind{CommandFinalize, CommandCancel}},
		{ControlExternal, StateSettling, []CommandKind{CommandCancel}},
		{ControlExternal, StateFinished, []CommandKind{}},
	}
	for _, tt := range tests {
		t.Run(string(tt.mode)+"/"+string(tt.state), func(t *testing.T) {
			l := Lifecycle{State: tt.state, ControlMode: tt.mode, HasPlan: tt.state != StateDraft, HasDraft: tt.mode == ControlServer}
			if got := l.AllowedCommands(); !slices.Equal(got, tt.commands) {
				t.Fatalf("commands=%v want=%v", got, tt.commands)
			}
			l.DeletionRequested = true
			if len(l.AllowedCommands()) != 0 {
				t.Fatal("deletion advertised commands")
			}
		})
	}
}

func TestLifecycleBudgetStopsAllAcceptedWork(t *testing.T) {
	for _, state := range []State{StateRunning, StatePausing, StatePaused, StateSettling, StateInterrupted} {
		l := Lifecycle{State: state, ControlMode: ControlServer, Outstanding: 8, HasPlan: true}
		target, stop := l.BudgetStop(true)
		if !stop || target != StateCancelling || l.ValidateObservation(target) != nil {
			t.Fatalf("budget left accepted work in %s: %s %v", state, target, stop)
		}
		if _, stop = l.BudgetStop(false); stop {
			t.Fatalf("unused budget stopped %s", state)
		}
	}
	l := Lifecycle{State: StateSettling, Outstanding: 8}
	if err := l.ValidateObservation(StateFinished); !errors.Is(err, ErrOutstandingExecutions) {
		t.Fatalf("premature terminal transition: %v", err)
	}
	l.Outstanding = 0
	if err := l.ValidateObservation(StateFinished); err != nil {
		t.Fatal(err)
	}
	if _, stop := l.BudgetStop(true); stop {
		t.Fatal("drained settling cannot start another cancellation")
	}
}

func TestLifecycleRecoversCommandsFromCommittedFacts(t *testing.T) {
	for _, tt := range []struct {
		lifecycle Lifecycle
		kind      CommandKind
		want      CommandCompletion
	}{
		{Lifecycle{State: StatePreparing}, CommandPrepare, CommandPending},
		{Lifecycle{State: StateDraft}, CommandPrepare, CommandFailed},
		{Lifecycle{State: StateRunning, HasPlan: true}, CommandPrepare, CommandSucceeded},
		{Lifecycle{State: StateSettling, Outstanding: 8}, CommandFinalize, CommandPending},
		{Lifecycle{State: StateFinished}, CommandFinalize, CommandSucceeded},
		{Lifecycle{State: StateCancelled}, CommandPause, CommandFailed},
		{Lifecycle{State: StateCancelled}, CommandStart, CommandSucceeded},
	} {
		if got := tt.lifecycle.Completion(tt.kind); got != tt.want {
			t.Fatalf("%s/%s completion=%v want=%v", tt.lifecycle.State, tt.kind, got, tt.want)
		}
	}
}

func TestLifecycleAdmissionKeepsClosedStatesAndBudgetsFenced(t *testing.T) {
	for _, state := range []State{StatePausing, StatePaused, StateSettling, StateCancelling, StateCancelled, StateFinished, StateInterrupted} {
		l := Lifecycle{State: state, ControlMode: ControlServer}
		if l.ValidateAdmission(false, 8) == nil {
			t.Fatalf("admission reopened %s", state)
		}
	}
	l := Lifecycle{State: StateRunning, ControlMode: ControlServer, Outstanding: 7}
	if err := l.ValidateAdmission(false, 8); err != nil {
		t.Fatal(err)
	}
	if l.ValidateAdmission(true, 8) == nil {
		t.Fatal("admitted beyond budget")
	}
	l.Outstanding = 8
	if l.ValidateAdmission(false, 8) == nil {
		t.Fatal("admitted beyond concurrency")
	}
	l = Lifecycle{State: StateReady, ControlMode: ControlExternal}
	if err := l.ValidateAdmission(false, 8); err != nil {
		t.Fatal(err)
	}
	l.DeletionRequested = true
	if l.ValidateAdmission(false, 8) == nil {
		t.Fatal("admitted after deletion")
	}
}
