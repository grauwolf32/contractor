// Package public implements the authenticated single-user HTTP API. Scope IDs
// are derived from authentication and route-owned Run records, never accepted
// as arbitrary request fields.
package public

import (
	"context"
	"errors"
	"log/slog"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditservice"
	"github.com/grauwolf32/contractor/internal/auditstandards"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/auth"
	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/controlplane"
	"github.com/grauwolf32/contractor/internal/credentials"
	"github.com/grauwolf32/contractor/internal/findingintake"
	"github.com/grauwolf32/contractor/internal/gatewayrecovery"
	publicevents "github.com/grauwolf32/contractor/internal/httpapi/public/events"
	"github.com/grauwolf32/contractor/internal/performance"
	"github.com/grauwolf32/contractor/internal/planner"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/grauwolf32/contractor/internal/runservice"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
	"github.com/grauwolf32/contractor/internal/settingsstore"
	"github.com/grauwolf32/contractor/internal/telemetry"
)

type RunReader interface {
	ResumableStage(context.Context, string, string) (*string, error)
	GetRun(context.Context, string) (runstore.WorkflowRun, error)
	ListRuns(context.Context, runstore.ListRunsParams) ([]runstore.WorkflowRunSummary, error)
	RunDeletionBlocker(context.Context, string, string) (*runstore.RunNotDeletableReason, error)
	ListRunOutputPublications(context.Context, string) ([]runstore.RunOutputPublication, error)
	ListStageExecutions(context.Context, string) ([]runstore.StageExecution, error)
	ListStageAllocationsBatch(context.Context, []string) (map[string][]runstore.StageAllocation, error)
	ListStageTransitionDecisions(context.Context, string) ([]runstore.StageTransitionDecision, error)
	GetRunEventCursor(context.Context, string) (runstore.WorkflowRunEventCursor, error)
}

type RunQueue interface {
	ListRunQueue(context.Context, runstore.ListRunQueueParams) ([]runstore.WorkflowRunQueueItem, error)
	GetOwnerQueueControl(context.Context, string) (runstore.OwnerQueueControl, error)
	UpdateOwnerQueueControl(context.Context, runstore.UpdateOwnerQueueControlParams) (runstore.OwnerQueueControl, error)
}

type RunLifecycle interface {
	ResumeFailedRun(context.Context, string, string, string, string) (runstore.ResumeRunResult, error)
	DeleteReleasedTerminalRun(context.Context, string, string) error
	RequestRunCancellation(context.Context, string, runstore.WorkflowRunCancellation) (runstore.WorkflowRun, error)
}

type PlannerPlanReader interface {
	LoadPlans(context.Context, []planner.SessionIdentity) (map[string]planner.PlannerPlanProjection, error)
}

type MetricsReader interface {
	GetStageMetricsBatch(context.Context, []string) (map[string]telemetry.StageMetricsRecord, error)
}

type OperationsReader interface {
	SnapshotOperations() controlplane.OperationsSnapshot
}

type PerformanceReader interface {
	Snapshot() performance.SnapshotResponse
	History(context.Context, time.Time, time.Time, string) (performance.HistoryResponse, error)
}

type AllocationResourceReader interface {
	ListAllocationResourceHistory(
		context.Context, telemetry.AllocationResourceHistoryParams,
	) ([]telemetry.AllocationResourceSummary, error)
	ListStageAllocationResources(
		context.Context, string, []string,
	) (map[string][]telemetry.AllocationResourceSummary, error)
}

type OperationsInvalidator interface {
	InvalidateOperations(controlplane.OperationsResource, string) error
}

type SchedulerSettingsManagement interface {
	GetSchedulerSettings(context.Context) (settingsstore.SchedulerSettings, error)
	UpdateSchedulerSettings(
		context.Context, settingsstore.UpdateSchedulerSettingsParams,
	) (settingsstore.SchedulerSettings, error)
}

type RunNotifier interface {
	Wake()
}

type RunCreationService interface {
	CreatePublic(context.Context, runservice.PublicCreateParams) (runservice.CreateResult, error)
}

type RunCancellationNotifier interface {
	Cancel(string)
}

// ConfigurationCatalog is the atomically swappable read view consumed by the
// public API. Both a bootstrap Snapshot and the managed configuration Manager
// implement it.
type ConfigurationCatalog interface {
	AgentTemplateWorkflowBindings(string) (config.AgentTemplateWorkflowBindings, error)
	AgentInstructions(string) (config.AgentInstructions, error)
	Workflow(string) (config.ResolvedWorkflow, error)
	Workflows() []config.ResolvedWorkflow
	ResolveRunWorkflow(
		context.Context,
		string,
		config.ExecutionConfigPatch,
		config.CredentialLookup,
	) (config.ResolvedWorkflow, error)
	Configurations(config.ConfigurationKind) ([]config.ConfigurationResource, error)
	Configuration(config.ConfigurationKind, string) (config.ConfigurationResource, error)
}

type ConfigurationPublisher interface {
	Publish(context.Context, config.PublicationRequest) (config.PublicationResult, error)
}

type ManagedCredentialLifecycle interface {
	ListCredentials(context.Context, string, int) ([]credentials.Record, error)
	GetCredential(context.Context, string) (credentials.Record, error)
	Create(context.Context, credentials.CreateRequest) (credentials.CreateResult, error)
	Delete(context.Context, credentials.DeleteRequest) (credentials.DeleteResult, error)
	WithRunCreation(context.Context, func() error) error
}

type RuntimeConfigManagement interface {
	Publish(context.Context, []byte, string, string) (runtimeconfig.PublishResult, error)
	ListVersions(context.Context, string, string, int) ([]runtimeconfig.Version, error)
	GetVersion(context.Context, string, string) (runtimeconfig.Version, error)
	ListBindings(context.Context, string, int) ([]runtimeconfig.Binding, error)
	GetBinding(context.Context, string) (runtimeconfig.Binding, error)
	CreateBinding(context.Context, string, runtimeconfig.Ref, string, string, time.Time) (runtimeconfig.BindingMutationResult, error)
	Rebind(context.Context, string, uint64, runtimeconfig.Ref, string, string, time.Time) (runtimeconfig.BindingMutationResult, error)
	DeleteBinding(context.Context, string, uint64, string, string, time.Time) (runtimeconfig.BindingMutationResult, error)
}

type RuntimeCredentialManagement interface {
	List(context.Context, string, int) ([]credentials.RuntimeCredentialMetadata, error)
	Get(context.Context, string) (credentials.RuntimeCredentialMetadata, error)
	Create(context.Context, credentials.RuntimeCredentialCreateRequest) (credentials.RuntimeCredentialCreateResult, error)
	Delete(context.Context, string, string) (credentials.RuntimeCredentialDeleteResult, error)
	ValidateRuntimeCredentialUse(context.Context, credentials.RuntimeCredentialUser, string, ...string) error
	WithCredentialReferences(context.Context, func() error) error
}

type RuntimeAgentPrincipalManagement interface {
	List(context.Context, string, int) ([]controlplane.RuntimeAgentPrincipalProjection, bool, error)
	Get(context.Context, string) (controlplane.RuntimeAgentPrincipalProjection, error)
	ReplaceLabels(context.Context, string, uint64, []string, string, string, time.Time) (controlplane.RuntimeAgentPrincipalProjection, bool, error)
	Delete(context.Context, string, uint64, string, string, time.Time) (bool, error)
}

type ProjectManagement interface {
	Create(context.Context, projectstore.CreateParams) (projectstore.Project, bool, error)
	Get(context.Context, string, string) (projectstore.Project, error)
	List(context.Context, projectstore.ListParams) ([]projectstore.Project, error)
	Update(context.Context, projectstore.UpdateParams) (projectstore.Project, error)
	BeginDeletion(context.Context, projectstore.BeginDeletionParams) (projectstore.Project, bool, error)
}

type AuditManagement interface {
	ListOwnerFindings(context.Context, auditservice.OwnerFindingListParams) ([]auditservice.Finding, error)
	ListOwnerReviews(context.Context, auditservice.OwnerReviewListParams) ([]auditservice.ReviewRequest, error)
	GetWorkspace(context.Context, string, string) (auditservice.WorkspaceSummary, error)
	ListFindingsPage(context.Context, auditservice.FindingListParams) (auditservice.FindingPage, error)
	ListReviewsPage(context.Context, auditservice.ReviewListParams) (auditservice.ReviewPage, error)
	Profiles() []auditservice.ProfileProjection
	Profile(auditservice.ProfileSelector) (auditservice.ProfileProjection, error)
	Standards(context.Context, string) ([]auditstandards.PackageProjection, error)
	Standard(context.Context, string, auditstandards.Reference) (auditstandards.PackageProjection, error)
	CreateDraft(context.Context, auditservice.CreateDraftParams) (auditstore.Audit, bool, error)
	Start(context.Context, auditservice.StartParams) (auditservice.StartedAudit, error)
	Pause(context.Context, auditservice.MutationParams) (auditservice.MutationResult, error)
	Resume(context.Context, auditservice.MutationParams) (auditservice.MutationResult, error)
	Cancel(context.Context, auditservice.MutationParams) (auditservice.MutationResult, error)
	Delete(context.Context, auditservice.MutationParams) (auditservice.MutationResult, error)
	Get(context.Context, string, string) (auditstore.Audit, error)
	List(context.Context, auditstore.ListParams) ([]auditstore.Audit, error)
	ListItems(context.Context, auditstore.ListItemsParams) ([]auditstore.Item, error)
	ListItemAttempts(context.Context, string, string, []string) (map[string][]auditstore.ItemAttempt, error)
	GetRound(context.Context, string, string, string) (auditstore.Round, error)
	ListCoverage(context.Context, string, string, string, int, int) ([]auditstore.CoverageRow, error)
	GetReport(context.Context, string, string) (auditservice.ReportProjection, error)
	GetFinding(context.Context, string, string, string) (auditservice.Finding, error)
	CreateFindingReview(context.Context, auditservice.CreateFindingReviewParams) (auditservice.FindingReviewResult, error)
	DecideFinding(context.Context, auditservice.DecideFindingParams) (auditservice.FindingDecisionResult, error)
	DecideActionReview(context.Context, auditservice.DecideActionReviewParams) (auditservice.ActionReviewDecisionResult, error)
	GetReview(context.Context, string, string, string) (auditservice.ReviewRequest, error)
	ListFindingProvenance(context.Context, auditservice.ProvenanceListParams) ([]auditservice.FindingProvenance, error)
}

type FindingProposalManagement interface {
	ListRun(context.Context, string, string, findingintake.ListQuery) ([]findingintake.Receipt, error)
	ListAuditInbox(context.Context, string, string, findingintake.ListQuery) ([]findingintake.Receipt, error)
	ImportIntoAudit(context.Context, findingintake.ImportRequest) (findingintake.AuditHold, bool, error)
}

type Dependencies struct {
	GatewayRecovery        *gatewayrecovery.Service
	Evals                  EvalManagement
	EvalNotifier           interface{ Wake() }
	GitImports             GitImportService
	GitKeys                GitKeySettings
	Authentication         *auth.Service
	BrowserOrigins         auth.OriginPolicy
	InsecureLoopbackCookie bool
	// TrustedPeers resolves the login rate-limit client behind reverse proxies;
	// its zero value attributes every failure to the socket peer.
	TrustedPeers            auth.PeerPolicy
	Config                  ConfigurationCatalog
	ConfigurationPublisher  ConfigurationPublisher
	Credentials             config.CredentialLookup
	ManagedCredentials      ManagedCredentialLifecycle
	RuntimeConfigs          RuntimeConfigManagement
	RuntimeCredentials      RuntimeCredentialManagement
	RuntimeAgentPrincipals  RuntimeAgentPrincipalManagement
	Projects                ProjectManagement
	Audits                  AuditManagement
	FindingProposals        FindingProposalManagement
	FindingCollections      FindingCollectionManagement
	Runs                    RunReader
	RunQueue                RunQueue
	RunLifecycle            RunLifecycle
	RunCreator              RunCreationService
	PlannerPlans            PlannerPlanReader
	Metrics                 MetricsReader
	Operations              OperationsReader
	Performance             PerformanceReader
	AllocationResources     AllocationResourceReader
	OperationsInvalidator   OperationsInvalidator
	SchedulerSettings       SchedulerSettingsManagement
	Events                  *publicevents.Hub
	Artifacts               *artifacts.Service
	BearerToken             contracts.SecretString
	NewID                   func(string) (string, error)
	NewRequestID            func() (string, error)
	RunNotifier             RunNotifier
	ProjectDeletionNotifier RunNotifier
	Now                     func() time.Time
	Logger                  *slog.Logger
}

var (
	errInvalidRequest = errors.New("invalid public API request")
	// errRequestTooLarge reports a JSON request body beyond its declared
	// bound. It is 413 like an oversized artifact, not a malformed request.
	errRequestTooLarge = errors.New("public API request body is too large")
)
