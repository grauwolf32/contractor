package runstore

import (
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/grauwolf32/contractor/internal/contracts"
)

type WorkflowRunState string

const (
	RunInitializing WorkflowRunState = "initializing"
	RunRunning      WorkflowRunState = "running"
	RunCancelling   WorkflowRunState = "cancelling"
	RunSucceeded    WorkflowRunState = "succeeded"
	RunFailed       WorkflowRunState = "failed"
	RunCancelled    WorkflowRunState = "cancelled"
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

type Reason struct {
	Code    string
	Message string
}

type SchedulerClaim struct {
	ClaimID   string
	ClaimedAt time.Time
	ExpiresAt time.Time
}

type WorkflowRun struct {
	RunID                     string
	OwnerID                   string
	WorkflowName              string
	WorkflowVersion           string
	WorkflowSchemaVersion     string
	WorkflowSnapshot          json.RawMessage
	Parameters                map[string]string
	State                     WorkflowRunState
	StateReason               Reason
	CancellationSchemaVersion *string
	Cancellation              *WorkflowRunCancellation
	SchedulerClaim            *SchedulerClaim
	CreatedAt                 time.Time
	UpdatedAt                 time.Time
	StartedAt                 *time.Time
	FinishedAt                *time.Time
}

const CancellationUserRequested = "user_cancelled"

type WorkflowRunCancellation struct {
	Code        string    `json:"code"`
	RequestedAt time.Time `json:"requestedAt"`
	RequestedBy *string   `json:"requestedBy,omitempty"`
	Reason      *string   `json:"reason,omitempty"`
}

func (c WorkflowRunCancellation) Validate() error {
	if c.Code != CancellationUserRequested {
		return invalidf("WorkflowRunCancellation code must be %q", CancellationUserRequested)
	}
	if c.RequestedAt.IsZero() {
		return invalidf("WorkflowRunCancellation requestedAt is required")
	}
	for field, value := range map[string]*string{
		"requestedBy": c.RequestedBy,
		"reason":      c.Reason,
	} {
		if value != nil && strings.TrimSpace(*value) == "" {
			return invalidf("WorkflowRunCancellation %s must not be empty", field)
		}
	}
	return nil
}

type CreateRunParams struct {
	RunID                 string
	OwnerID               string
	WorkflowName          string
	WorkflowVersion       string
	WorkflowSchemaVersion string
	WorkflowSnapshot      json.RawMessage
	Parameters            map[string]string
}

type CreateRunIdempotentParams struct {
	CreateRunParams
	IdempotencyKey string
	RequestDigest  string
}

type PinnedContextArtifact struct {
	Required bool                   `json:"required"`
	Artifact *contracts.ArtifactRef `json:"artifact,omitempty"`
}

// StageContextSnapshot explicitly retains optional absence as an entry whose
// Artifact is nil. Present refs must always carry an exact revision.
type StageContextSnapshot struct {
	Parameters map[string]string                `json:"parameters"`
	Artifacts  map[string]PinnedContextArtifact `json:"artifacts"`
}

type CreateStageExecutionParams struct {
	StageExecutionID          string
	RunID                     string
	StageName                 string
	Attempt                   int
	PreviousExecutionID       *string
	ExecutionConfigVariant    StageExecutionConfigVariant
	EscalationOrdinal         *int
	StageSpecSchemaVersion    string
	StageSpecSnapshot         json.RawMessage
	StageContextSchemaVersion string
	StageContext              StageContextSnapshot
}

// StageExecutionConfigVariant identifies which already-pinned Stage
// executionConfig was copied into the immutable Stage spec snapshot.
type StageExecutionConfigVariant string

const (
	StageExecutionConfigBase                  StageExecutionConfigVariant = "base"
	StageExecutionConfigFailedEscalation      StageExecutionConfigVariant = "failed_escalation"
	StageExecutionConfigInterruptedEscalation StageExecutionConfigVariant = "interrupted_escalation"
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
	StageExecutionID             string
	RunID                        string
	StageName                    string
	Attempt                      int
	PreviousExecutionID          *string
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
	CandidateResult              *contracts.StageContentResult
	AcceptedResultSchemaVersion  *string
	AcceptedResult               *contracts.StageContentResult
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

type StartPlannerParams struct {
	StageExecutionID   string
	SessionID          string
	InvocationID       string
	StateSchemaVersion string
	InitialState       json.RawMessage
	EventID            string
	EventSchemaVersion string
	Event              json.RawMessage
	Reason             Reason
	RunEvent           RunEventAppend
}

type EnterFinalizingParams struct {
	StageExecutionID    string
	ResultSchemaVersion string
	Candidate           contracts.StageContentResult
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

type PlannerSession struct {
	SessionID          string
	StageExecutionID   string
	InvocationID       string
	StateSchemaVersion string
	State              json.RawMessage
	NextEventSequence  int64
	CreatedAt          time.Time
	UpdatedAt          time.Time
}

type PlannerEvent struct {
	EventID            string
	SessionID          string
	SequenceNumber     int64
	EventSchemaVersion string
	Event              json.RawMessage
	RunID              *string
	RunEventSequence   *int64
	CreatedAt          time.Time
}

type AppendPlannerEventParams struct {
	EventID               string
	SessionID             string
	StageExecutionID      string
	InvocationID          string
	SequenceNumber        int64
	EventSchemaVersion    string
	Event                 json.RawMessage
	NewStateSchemaVersion string
	NewState              json.RawMessage
	RunEvent              RunEventAppend
}

type RunEventKind string

const (
	RunEventPlannerStarted           RunEventKind = "planner.started"
	RunEventPlannerRequestRecorded   RunEventKind = "planner.request_recorded"
	RunEventPlannerActivity          RunEventKind = "planner.activity"
	RunEventPlannerPlanChanged       RunEventKind = "planner.plan_changed"
	RunEventPlannerCurrentChanged    RunEventKind = "planner.current_changed"
	RunEventPlannerDispatchSelected  RunEventKind = "planner.dispatch_selected"
	RunEventPlannerDispatchStarted   RunEventKind = "planner.dispatch_started"
	RunEventPlannerDispatchCompleted RunEventKind = "planner.dispatch_completed"
	RunEventPlannerFinishRequested   RunEventKind = "planner.finish_requested"
	RunEventPlannerCompleted         RunEventKind = "planner.completed"
	RunEventPlannerFailed            RunEventKind = "planner.failed"
)

// RunEventAppend is the validated event half of a Planner/session mutation.
// The store allocates its per-Run sequence in the same SQL statement.
type RunEventAppend struct {
	EventID            string
	EventSchemaVersion string
	Kind               RunEventKind
	Data               json.RawMessage
}

type WorkflowRunEvent struct {
	RunID              string
	SequenceNumber     int64
	EventID            string
	EventSchemaVersion string
	Kind               RunEventKind
	Data               json.RawMessage
	OccurredAt         time.Time
}

type WorkflowRunEventCursor struct {
	Generation string
	Sequence   int64
}

type StageAllocation struct {
	AllocationID           string
	StageExecutionID       string
	LogicalAgentName       string
	Namespace              string
	AgentTemplateRef       contracts.AgentTemplateRef
	WorkerRuntimeRef       contracts.WorkerRuntimeRef
	RuntimeAgentInstanceID string
	CreatedAt              time.Time
}

// StageExecutionReport is the trusted Server envelope around one bounded
// Runtime Agent report. Identity comes from the allocation record, never from
// model-visible data.
type StageExecutionReport struct {
	StageExecutionID    string
	AllocationID        string
	LogicalAgentName    string
	ReportSchemaVersion string
	Report              contracts.AllocationFinalReport
	ReceivedAt          time.Time
	ExpiresAt           time.Time
}

type RecordStageExecutionReportParams struct {
	StageExecutionID    string
	AllocationID        string
	LogicalAgentName    string
	ReportSchemaVersion string
	Report              contracts.AllocationFinalReport
	Secrets             []string
}

type PlannerExecutionReport struct {
	StageExecutionID    string
	SessionID           string
	InvocationID        string
	StartedAt           time.Time
	FinishedAt          time.Time
	ReportSchemaVersion string
	Report              contracts.ExecutionReport
	ReceivedAt          time.Time
	ExpiresAt           time.Time
}

type RecordPlannerExecutionReportParams struct {
	StageExecutionID    string
	SessionID           string
	InvocationID        string
	StartedAt           time.Time
	FinishedAt          time.Time
	ReportSchemaVersion string
	Report              contracts.ExecutionReport
	Secrets             []string
}

func validateReason(reason Reason) error {
	if strings.TrimSpace(reason.Code) == "" {
		return invalidf("state reason code is required")
	}
	return nil
}

func validateOpaque(field, value string) error {
	if strings.TrimSpace(value) == "" {
		return invalidf("%s is required", field)
	}
	return nil
}

func validateJSONObject(field string, value json.RawMessage) error {
	var object map[string]json.RawMessage
	if len(value) == 0 || json.Unmarshal(value, &object) != nil || object == nil {
		return invalidf("%s must be one JSON object", field)
	}
	return nil
}

func (s StageContextSnapshot) Validate() error {
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
