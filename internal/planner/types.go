package planner

import (
	"context"
	"time"

	workflowconfig "github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/telemetry"
)

const (
	PassthroughRef = "passthrough@1"
	StreamlineRef  = "streamline@1"
	RouterRef      = "router@1"
)

// StageContext is the immutable input snapshot visible to one Planner
// invocation. A nil artifact records an explicitly absent optional binding.
type StageContext struct {
	Parameters map[string]string
	Artifacts  map[string]*contracts.ArtifactRef
}

// Invocation contains only Stage-local semantic input and already prepared
// Workers. It deliberately has no capacity or Runtime Agent selection API.
type Invocation struct {
	StageExecutionID string
	RunID            string
	Stage            workflowconfig.ResolvedStage
	Context          StageContext
	Workers          map[string]contracts.WorkerHandle
	ModelAccess      *ModelAccess
	Instrumentation  telemetry.PlannerInstrumentation
	Deadline         time.Time
}

func InvocationInstrumentation(invocation Invocation) telemetry.PlannerInstrumentation {
	if invocation.Instrumentation == nil {
		return telemetry.NoopPlannerInstrumentation()
	}
	return invocation.Instrumentation
}

// ModelAccess is resolved from the immutable Run snapshot. Token is held only
// for construction of this invocation's model client and is never persisted in
// Planner state or events.
type ModelAccess struct {
	ModelPolicy contracts.ResolvedModelPolicy
	LLMGateway  contracts.ResolvedLLMGatewayConfig
	Credential  *contracts.LLMCredentialRef
	Token       contracts.SecretString
}

type ArtifactMetadata struct {
	MediaType string
}

// ArtifactInspector resolves metadata for an exact ref inside one RunScope.
// The Planner never receives artifact bytes through this interface.
type ArtifactInspector interface {
	Inspect(context.Context, string, contracts.ArtifactRef) (ArtifactMetadata, error)
}

// WorkerInvoker is the framework-neutral A2A boundary used by Planner code.
type WorkerInvoker interface {
	Invoke(
		context.Context,
		string,
		contracts.WorkerHandle,
		contracts.StageContentRequest,
	) (contracts.WorkerCompletion, error)
}

// WorkerStateReader is the framework-neutral live-State boundary used only by
// modeled Planner projections. Implementations own physical Runtime routing;
// callers supply a previously validated WorkerHandle and an optional strong
// ETag, never a model-selected URL or allocation identity.
type WorkerStateReader interface {
	ReadWorkerState(context.Context, contracts.WorkerHandle, string) (WorkerStateReadResult, error)
}

type WorkerStateReadResult struct {
	Snapshot    *contracts.AgentStateSnapshot
	ETag        string
	NotModified bool
}

// WorkerStateReadError deliberately retains no wrapped transport error. It is
// safe to normalize into a model-facing projection failure.
type WorkerStateReadError struct {
	StatusCode int
	Code       string
	Retryable  bool
}

func (e *WorkerStateReadError) Error() string {
	return "Runtime Agent Worker State read failed (" + e.Code + ")"
}

type SessionIdentity struct {
	SessionID        string
	StageExecutionID string
	InvocationID     string
}

type Completion struct {
	Result  *contracts.StageContentResult `json:"result,omitempty"`
	Failure *Failure                      `json:"failure,omitempty"`
}

// SessionStart grants invocation ownership only for a newly created durable
// session. A completed session instead returns its recorded completion.
type SessionStart struct {
	Identity   SessionIdentity
	Invoke     bool
	Completion *Completion
}

// RequestFacts is the bounded audit representation of one Worker request.
// Parameter values, objective text, instruction text, and artifact bytes are
// intentionally absent.
type RequestFacts struct {
	Bindings           []string
	ObjectiveDigest    string
	InstructionsDigest string
	ParameterNames     []string
	Artifacts          map[string]contracts.ArtifactRef
}

type SessionService interface {
	Begin(context.Context, string) (SessionStart, error)
	RecordRequest(context.Context, SessionIdentity, RequestFacts) error
	Complete(context.Context, SessionIdentity, Completion) error
}

// PlannerEventKind is the closed public fact vocabulary emitted by modeled
// Planners. Generic ADK payloads and model text are not event kinds.
type PlannerEventKind string

const (
	PlannerEventStarted           PlannerEventKind = "planner.started"
	PlannerEventRequestRecorded   PlannerEventKind = "planner.request_recorded"
	PlannerEventActivity          PlannerEventKind = "planner.activity"
	PlannerEventPlanChanged       PlannerEventKind = "planner.plan_changed"
	PlannerEventCurrentChanged    PlannerEventKind = "planner.current_changed"
	PlannerEventDispatchSelected  PlannerEventKind = "planner.dispatch_selected"
	PlannerEventDispatchStarted   PlannerEventKind = "planner.dispatch_started"
	PlannerEventDispatchCompleted PlannerEventKind = "planner.dispatch_completed"
	PlannerEventFinishRequested   PlannerEventKind = "planner.finish_requested"
	PlannerEventCompleted         PlannerEventKind = "planner.completed"
	PlannerEventFailed            PlannerEventKind = "planner.failed"
)

// PlannerPlanTransition compares one authoritative durable plan revision with
// its proposed successor. ExpectedRevision is the caller's compare value.
type PlannerPlanTransition struct {
	Kind             PlannerEventKind
	ExpectedRevision uint64
	Plan             PlannerPlanProjection
}

// PlannerFact is a bounded event that does not itself change the plan. Key is
// a stable adapter-generated idempotency identity, never model-provided text.
type PlannerFact struct {
	Kind         PlannerEventKind
	Key          string
	PlanRevision uint64
	SubtaskID    string
	CallID       string
	WorkerName   string
	Outcome      string
	Code         string
}

// PlanSessionService is the modeled-Planner extension of SessionService.
// Passthrough does not create a subtask plan and only needs SessionService.
type PlanSessionService interface {
	SessionService
	RecordPlan(context.Context, SessionIdentity, PlannerPlanTransition) error
	RecordFact(context.Context, SessionIdentity, PlannerFact) error
	LoadPlan(context.Context, SessionIdentity) (PlannerPlanProjection, bool, error)
}

type Planner interface {
	Run(context.Context) (contracts.StageContentResult, error)
}

// ReportProvider exposes telemetry collected by a Planner implementation after
// Run returns. Scheduler treats it as best-effort and never as semantic input.
type ReportProvider interface {
	ExecutionReport() (contracts.ExecutionReport, bool)
}

type Factory interface {
	Ref() string
	Create(Invocation) (Planner, error)
}
