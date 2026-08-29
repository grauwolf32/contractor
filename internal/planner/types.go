package planner

import (
	"context"
	"time"

	workflowconfig "github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
)

const (
	PassthroughRef = "passthrough@1"
	StreamlineRef  = "streamline@1"
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
	Deadline         time.Time
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
	) (contracts.StageContentResult, error)
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
