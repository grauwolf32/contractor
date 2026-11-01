package runstore

import (
	"encoding/json"
	"slices"
	"strings"
	"time"

	"github.com/grauwolf32/contractor/internal/agentskills"
	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contracts/runtimesettings"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
)

type WorkflowRunState string

const (
	RunInitializing WorkflowRunState = "initializing"
	RunPending      WorkflowRunState = "pending"
	RunWaiting      WorkflowRunState = "waiting"
	RunRunning      WorkflowRunState = "running"
	RunCancelling   WorkflowRunState = "cancelling"
	RunSucceeded    WorkflowRunState = "succeeded"
	RunFailed       WorkflowRunState = "failed"
	RunCancelled    WorkflowRunState = "cancelled"
)

// activeRunStates and terminalRunStates partition the WorkflowRun states
// (RUN-R30, RUN-R39). They are the one definition of both sets: Go code asks
// RunLifecycleActive and RunLifecycleTerminal, and the RunStateSets tests hold
// the OpenAPI enums, every Run-state list in Go SQL and the workflow_runs
// CHECKs and partial-index predicates to them. SQL spells the lists as
// literals rather than binding them as an array parameter: PostgreSQL proves
// a partial-index predicate only from constants written in the query.
var (
	activeRunStates   = [...]WorkflowRunState{RunInitializing, RunPending, RunRunning, RunWaiting, RunCancelling}
	terminalRunStates = [...]WorkflowRunState{RunSucceeded, RunFailed, RunCancelled}
)

// Valid reports whether state is one of the WorkflowRun states.
func (s WorkflowRunState) Valid() bool {
	return RunLifecycleActive.Includes(s) || RunLifecycleTerminal.Includes(s)
}

// WorkflowRunLifecycle is a read-only grouping used by owner list queries. It
// does not add a durable WorkflowRun state.
type WorkflowRunLifecycle string

const (
	RunLifecycleActive   WorkflowRunLifecycle = "active"
	RunLifecycleTerminal WorkflowRunLifecycle = "terminal"
)

func (l WorkflowRunLifecycle) Valid() bool {
	return l == RunLifecycleActive || l == RunLifecycleTerminal
}

func (l WorkflowRunLifecycle) Includes(state WorkflowRunState) bool {
	switch l {
	case RunLifecycleActive:
		return slices.Contains(activeRunStates[:], state)
	case RunLifecycleTerminal:
		return slices.Contains(terminalRunStates[:], state)
	default:
		return false
	}
}

// OutputPublicationMode is immutable WorkflowRun authority. Ordinary Runs
// publish accepted outputs into their owning Project, while Audit-managed Runs
// leave frozen outputs in RunScope for the Audit collector to import under
// collision-free protected names.
type OutputPublicationMode string

const (
	PublicationOrdinary     OutputPublicationMode = "ordinary"
	PublicationAuditManaged OutputPublicationMode = "audit-managed"
)

func (m OutputPublicationMode) Valid() bool {
	return m == PublicationOrdinary || m == PublicationAuditManaged
}

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
	RequiredPlacementLabels   []string
	RunID                     string
	OwnerID                   string
	ProjectID                 *string
	WorkflowName              string
	WorkflowVersion           string
	WorkflowSchemaVersion     string
	WorkflowSnapshot          json.RawMessage
	Parameters                map[string]string
	MetadataLabels            RunMetadataLabels
	RuntimeLabels             []string
	RuntimeConfig             runtimeconfig.RunSnapshot
	ProjectHTTPTarget         *runtimesettings.HTTPOriginTargetRef
	SkillSnapshot             []agentskills.RunSkillSnapshot
	PublicationMode           OutputPublicationMode
	AuditExecutionID          *string
	AuditSubmissionKey        *string
	AuditCompletion           *AuditCompletionSnapshot
	State                     WorkflowRunState
	StateReason               Reason
	CancellationSchemaVersion *string
	Cancellation              *WorkflowRunCancellation
	SchedulerClaim            *SchedulerClaim
	// SchedulerLaneClass is the lane class of the StageExecution the
	// Scheduler progresses next: the active one, otherwise the entry Stage.
	SchedulerLaneClass resolved.LaneClass
	CreatedAt          time.Time
	UpdatedAt          time.Time
	StartedAt          *time.Time
	FinishedAt         *time.Time
}

// WorkflowRunSummary is the bounded owner-list projection. In particular it
// excludes the immutable Workflow snapshot and user parameter values.
type WorkflowRunSummary struct {
	RunID           string
	ProjectID       *string
	WorkflowName    string
	WorkflowVersion string
	MetadataLabels  RunMetadataLabels
	State           WorkflowRunState
	CreatedAt       time.Time
	UpdatedAt       time.Time
	FinishedAt      *time.Time
}

// RunQueueMembership is a display filter over immutable Project membership.
// It does not affect Scheduler eligibility or claim order.
type RunQueueMembership string

const (
	RunQueueStandalone RunQueueMembership = "standalone"
	RunQueueProject    RunQueueMembership = "project"
	RunQueueEvaluation RunQueueMembership = "evaluation"
)

func (m RunQueueMembership) Valid() bool {
	return m == RunQueueStandalone || m == RunQueueProject || m == RunQueueEvaluation
}

// WorkflowRunQueueItem is the bounded owner-facing projection of one
// non-terminal Run. Project metadata is safe display data and event cursor is
// the exact starting point for an optional read-only UI subscription.
type WorkflowRunQueueItem struct {
	RunID           string
	ProjectID       *string
	ProjectName     string
	ProjectKind     string
	WorkflowName    string
	WorkflowVersion string
	MetadataLabels  RunMetadataLabels
	State           WorkflowRunState
	EventCursor     WorkflowRunEventCursor
	CreatedAt       time.Time
	UpdatedAt       time.Time
}

// OwnerQueueControl is the durable admission gate for one authenticated
// owner. Revision zero is the effective running representation when no row has
// been materialized yet; persisted revisions begin at one.
type OwnerQueueControl struct {
	OwnerID   string
	Paused    bool
	Revision  uint64
	UpdatedAt time.Time
}

type UpdateOwnerQueueControlParams struct {
	OwnerID          string
	ExpectedRevision uint64
	Paused           bool
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
	RequiredPlacementLabels []string
	RunID                   string
	OwnerID                 string
	ProjectID               *string
	WorkflowName            string
	WorkflowVersion         string
	WorkflowSchemaVersion   string
	WorkflowSnapshot        json.RawMessage
	Parameters              map[string]string
	MetadataLabels          RunMetadataLabels
	RuntimeConfig           runtimeconfig.RunSnapshot
	ProjectHTTPTarget       *runtimesettings.HTTPOriginTargetRef
	// LaneClass is the lane class of the Workflow's entry Stage
	// (resolved.EntryLaneClass), which the Run carries until the Scheduler
	// creates its first StageExecution (SCH-R175).
	LaneClass resolved.LaneClass
}

type CreateRunIdempotentParams struct {
	CreateRunParams
	IdempotencyKey string
	RequestDigest  string
}

// CreateAuditRunParams is intentionally separate from public idempotent Run
// creation. Only the trusted Run Service may supply the immutable Audit
// association and its already-durable submission key.
type CreateAuditRunParams struct {
	CreateRunParams
	AuditExecutionID   string
	AuditSubmissionKey string
}

// ListRunsParams is a stable newest-first owner query. BeforeCreatedAt and
// BeforeRunID are either both set or both absent and form the keyset cursor.
type ListRunsParams struct {
	OwnerID                string
	ProjectID              *string
	State                  *WorkflowRunState
	Lifecycle              *WorkflowRunLifecycle
	MetadataLabelSelectors []RunMetadataLabelSelector
	BeforeCreatedAt        *time.Time
	BeforeRunID            string
	Limit                  int
}

// ListRunQueueParams is an immutable-created-at, oldest-first keyset query.
// AfterCreatedAt and AfterRunID are either both set or both absent.
type ListRunQueueParams struct {
	OwnerID        string
	State          *WorkflowRunState
	Membership     *RunQueueMembership
	AfterCreatedAt *time.Time
	AfterRunID     string
	Limit          int
}
