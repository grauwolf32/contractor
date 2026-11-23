package runstore

import (
	"encoding/json"
	"fmt"
	"slices"
	"strings"
	"time"

	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/stageresult"
)

type StageExecutionState string

const (
	StagePreparing   StageExecutionState = "preparing"
	StageRunning     StageExecutionState = "running"
	StageFinalizing  StageExecutionState = "finalizing"
	StageAborting    StageExecutionState = "aborting"
	StageSucceeded   StageExecutionState = "succeeded"
	StageFailed      StageExecutionState = "failed"
	StageInterrupted StageExecutionState = "interrupted"
	StageCancelled   StageExecutionState = "cancelled"
)

// activeStageStates and terminalStageStates partition the StageExecution
// states (SCH-R42). They are the one definition of both sets: Go code asks
// StageExecutionState.Terminal, and the StageStateSets tests hold the OpenAPI
// enum, every StageExecution-state list in Go SQL and the stage_executions
// CHECKs and partial-index predicates to them. SQL spells the lists as
// literals for the reason the Run state lists do: PostgreSQL proves a
// partial-index predicate only from constants written in the query.
var (
	activeStageStates   = [...]StageExecutionState{StagePreparing, StageRunning, StageFinalizing, StageAborting}
	terminalStageStates = [...]StageExecutionState{StageSucceeded, StageFailed, StageInterrupted, StageCancelled}
)

// Terminal reports whether s is a terminal StageExecution state (SCH-R42).
func (s StageExecutionState) Terminal() bool {
	return slices.Contains(terminalStageStates[:], s)
}

type PinnedContextArtifact struct {
	Required bool                   `json:"required"`
	Artifact *contracts.ArtifactRef `json:"artifact,omitempty"`
}

// StageContextSnapshot explicitly retains optional absence as an entry whose
// Artifact is nil. Present refs must always carry an exact revision.
type StageContextSnapshot struct {
	Parameters   map[string]string                `json:"parameters"`
	Artifacts    map[string]PinnedContextArtifact `json:"artifacts"`
	Continuation *stageresult.SummaryContinuation `json:"continuation,omitempty"`
}

func (s StageContextSnapshot) Clone() StageContextSnapshot {
	result := StageContextSnapshot{}
	if s.Parameters != nil {
		result.Parameters = make(map[string]string, len(s.Parameters))
	}
	if s.Artifacts != nil {
		result.Artifacts = make(map[string]PinnedContextArtifact, len(s.Artifacts))
	}
	for key, value := range s.Parameters {
		result.Parameters[key] = value
	}
	for key, value := range s.Artifacts {
		if value.Artifact != nil {
			ref := value.Artifact.Clone()
			value.Artifact = &ref
		}
		result.Artifacts[key] = value
	}
	if s.Continuation != nil {
		continuation := s.Continuation.Clone()
		result.Continuation = &continuation
	}
	return result
}

type CreateStageExecutionParams struct {
	StageExecutionID          string
	RunID                     string
	StageName                 string
	Attempt                   int
	PreviousExecutionID       *string
	ResumeSourceExecutionID   *string
	ExecutionConfigVariant    StageExecutionConfigVariant
	EscalationOrdinal         *int
	StageSpecSchemaVersion    string
	StageSpecSnapshot         json.RawMessage
	StageContextSchemaVersion string
	StageContext              StageContextSnapshot
	// LaneClass is resolved.StageLaneClass of the effective Stage spec. The
	// statement that creates the StageExecution stores it on the Run, which
	// the pool of that class then claims (SCH-R175).
	LaneClass resolved.LaneClass
}

// StageExecutionConfigVariant identifies which already-pinned Stage
// executionConfig was copied into the immutable Stage spec snapshot.
type StageExecutionConfigVariant string

const (
	StageExecutionConfigBase                  StageExecutionConfigVariant = "base"
	StageExecutionConfigFailedEscalation      StageExecutionConfigVariant = "failed_escalation"
	StageExecutionConfigInterruptedEscalation StageExecutionConfigVariant = "interrupted_escalation"
	StageExecutionConfigSummaryEscalation     StageExecutionConfigVariant = "summary_escalation"
)

type TerminationOutcome string

const (
	TerminationCancelled   TerminationOutcome = "cancelled"
	TerminationInterrupted TerminationOutcome = "interrupted"
)

type TerminationPhase string

const (
	TerminationPreparing TerminationPhase = "preparing"
	TerminationRunning   TerminationPhase = "running"
)

type StageTermination struct {
	Outcome    TerminationOutcome `json:"outcome"`
	Code       string             `json:"code"`
	Message    string             `json:"message"`
	Retryable  bool               `json:"retryable"`
	Phase      TerminationPhase   `json:"phase"`
	OccurredAt time.Time          `json:"occurredAt"`
}

func (t StageTermination) Validate() error {
	if t.Outcome != TerminationCancelled && t.Outcome != TerminationInterrupted {
		return invalidf("unknown StageTermination outcome %q", t.Outcome)
	}
	if strings.TrimSpace(t.Code) == "" || strings.TrimSpace(t.Message) == "" {
		return invalidf("StageTermination code and message are required")
	}
	if t.Phase != TerminationPreparing && t.Phase != TerminationRunning {
		return invalidf("unknown StageTermination phase %q", t.Phase)
	}
	if t.OccurredAt.IsZero() {
		return invalidf("StageTermination occurredAt is required")
	}
	return nil
}

type StageExecution struct {
	AdmittedAt                   *time.Time
	StageExecutionID             string
	RunID                        string
	StageName                    string
	Attempt                      int
	PreviousExecutionID          *string
	ResumeSourceExecutionID      *string
	ExecutionConfigVariant       StageExecutionConfigVariant
	EscalationOrdinal            *int
	StageSpecSchemaVersion       string
	StageSpecSnapshot            json.RawMessage
	StageContextSchemaVersion    string
	StageContext                 StageContextSnapshot
	State                        StageExecutionState
	StateReason                  Reason
	PlannerSessionID             *string
	PlannerInvocationID          *string
	CandidateResultSchemaVersion *string
	CandidateResult              *stageresult.StageContentResult
	AcceptedResultSchemaVersion  *string
	AcceptedResult               *stageresult.StageContentResult
	TerminationSchemaVersion     *string
	Termination                  *StageTermination
	FinalizationID               *string
	FinalizationDeadline         *time.Time
	AbortID                      *string
	AbortDeadline                *time.Time
	CreatedAt                    time.Time
	UpdatedAt                    time.Time
	PlannerStartedAt             *time.Time
	TerminalAt                   *time.Time
}

// StageLineage is the identity, state, execution configuration and lineage
// of one StageExecution without any of its snapshots: what the Scheduler
// reads of a Run's Stage history to count escalation attempts (SCH-R164).
type StageLineage struct {
	StageExecutionID        string
	RunID                   string
	StageName               string
	Attempt                 int
	State                   StageExecutionState
	ExecutionConfigVariant  StageExecutionConfigVariant
	EscalationOrdinal       *int
	PreviousExecutionID     *string
	ResumeSourceExecutionID *string
	PlannerSessionID        *string
	PlannerInvocationID     *string
}

// StageTransitionAction is the durable Scheduler decision made after one
// StageExecution reaches a semantic terminal outcome. A next/retry/escalate
// decision identifies the newly-created execution in the same transaction.
type StageTransitionAction string

const (
	StageTransitionNext     StageTransitionAction = "next"
	StageTransitionRetry    StageTransitionAction = "retry"
	StageTransitionEscalate StageTransitionAction = "escalate"
	StageTransitionSucceed  StageTransitionAction = "succeed"
	StageTransitionFail     StageTransitionAction = "fail"
)

type StageTransitionDecision struct {
	SourceExecutionID   string
	RunID               string
	Action              StageTransitionAction
	TargetStageName     *string
	TargetExecutionID   *string
	EscalationOrdinal   *int
	EscalationExhausted bool
	DecidedAt           time.Time
}

type RecordStageTransitionDecisionParams struct {
	SourceExecutionID   string
	RunID               string
	Action              StageTransitionAction
	TargetStageName     *string
	TargetExecutionID   *string
	EscalationOrdinal   *int
	EscalationExhausted bool
}

type EnterFinalizingParams struct {
	StageExecutionID    string
	ResultSchemaVersion string
	Candidate           stageresult.StageContentResult
	FinalizationID      string
	Deadline            time.Time
	Reason              Reason
}

type EnterAbortingParams struct {
	StageExecutionID         string
	ExpectedState            StageExecutionState
	TerminationSchemaVersion string
	Termination              StageTermination
	AbortID                  string
	Deadline                 time.Time
	Reason                   Reason
}

func (s StageContextSnapshot) Validate() error {
	if s.Continuation != nil {
		if err := s.Continuation.Validate(); err != nil {
			return fmt.Errorf("%w: invalid summary continuation: %v", ErrInvalid, err)
		}
	}
	for name := range s.Parameters {
		if strings.TrimSpace(name) == "" {
			return invalidf("StageContext parameter name is required")
		}
	}
	for name, pinned := range s.Artifacts {
		if strings.TrimSpace(name) == "" {
			return invalidf("StageContext artifact name is required")
		}
		if pinned.Artifact != nil {
			if err := pinned.Artifact.ValidateExact(); err != nil {
				return fmt.Errorf("%w: StageContext artifact %q: %v", ErrInvalid, name, err)
			}
		}
	}
	return nil
}
