// Package public implements the authenticated single-user HTTP API. Scope IDs
// are derived from authentication and route-owned Run records, never accepted
// as arbitrary request fields.
package public

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"time"
	"unicode/utf8"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auth"
	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/controlplane"
	"github.com/grauwolf32/contractor/internal/credentials"
	publicevents "github.com/grauwolf32/contractor/internal/httpapi/public/events"
	"github.com/grauwolf32/contractor/internal/planner"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
	"github.com/grauwolf32/contractor/internal/telemetry"
)

type RunReader interface {
	GetRun(context.Context, string) (runstore.WorkflowRun, error)
	LookupRunIdempotency(context.Context, string, string, string) (runstore.WorkflowRun, bool, error)
	ListRuns(context.Context, runstore.ListRunsParams) ([]runstore.WorkflowRunSummary, error)
	ListRunQueue(context.Context, runstore.ListRunQueueParams) ([]runstore.WorkflowRunQueueItem, error)
	ListRunOutputPublications(context.Context, string) ([]runstore.RunOutputPublication, error)
	ListStageExecutions(context.Context, string) ([]runstore.StageExecution, error)
	ListStageAllocations(context.Context, string) ([]runstore.StageAllocation, error)
	ListStageTransitionDecisions(context.Context, string) ([]runstore.StageTransitionDecision, error)
	GetRunEventCursor(context.Context, string) (runstore.WorkflowRunEventCursor, error)
	ListRunEvents(context.Context, string, int64, int) ([]runstore.WorkflowRunEvent, error)
	RequestRunCancellation(context.Context, string, runstore.WorkflowRunCancellation) (runstore.WorkflowRun, error)
}

type PlannerPlanReader interface {
	LoadPlan(context.Context, planner.SessionIdentity) (planner.PlannerPlanProjection, bool, error)
}

type RunWriter interface {
	PinRuntimeLabels(
		context.Context, []string, config.CredentialLookup,
	) (runtimeconfig.RunSnapshot, error)
	CreateRun(context.Context, runstore.CreateRunParams) (runstore.WorkflowRun, error)
	CreateRunIdempotent(
		context.Context,
		runstore.CreateRunIdempotentParams,
	) (runstore.WorkflowRun, bool, error)
	SetRunSkillSelections(context.Context, string, []contracts.RunSkillSnapshot) error
	TransitionRun(
		context.Context,
		string,
		runstore.WorkflowRunState,
		runstore.WorkflowRunState,
		runstore.Reason,
	) (runstore.WorkflowRun, error)
}

type MetricsReader interface {
	GetStageMetrics(context.Context, string) (telemetry.StageMetricsRecord, error)
}

type OperationsReader interface {
	SnapshotOperations() controlplane.OperationsSnapshot
}

type OperationsInvalidator interface {
	InvalidateOperations(controlplane.OperationsResource, string) error
}

// UnitOfWork supplies transaction-bound Run and Artifact stores. The callback
// commits only when it returns nil.
type UnitOfWork interface {
	Do(context.Context, func(RunWriter, *artifacts.Service) error) error
}

type RunNotifier interface {
	Wake()
}

type RunSkillInitializer interface {
	InitializeRunSkills(context.Context, string) (runstore.WorkflowRun, error)
}

type RunCancellationNotifier interface {
	Cancel(string)
}

// ConfigurationCatalog is the atomically swappable read view consumed by the
// public API. Both a bootstrap Snapshot and the managed configuration Manager
// implement it.
type ConfigurationCatalog interface {
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
}

type RuntimeAgentPrincipalManagement interface {
	List(context.Context, string, int) ([]controlplane.RuntimeAgentPrincipalProjection, error)
	Get(context.Context, string) (controlplane.RuntimeAgentPrincipalProjection, error)
	ReplaceLabels(context.Context, string, uint64, []string, string, string, time.Time) (controlplane.RuntimeAgentPrincipalProjection, bool, error)
	Delete(context.Context, string, uint64, string, string, time.Time) (bool, error)
}

type ProjectManagement interface {
	Create(context.Context, projectstore.CreateParams) (projectstore.Project, bool, error)
	Get(context.Context, string, string) (projectstore.Project, error)
	List(context.Context, projectstore.ListParams) ([]projectstore.Project, error)
	Update(context.Context, projectstore.UpdateParams) (projectstore.Project, error)
}

type Dependencies struct {
	Authentication         *auth.Service
	BrowserOrigins         auth.OriginPolicy
	InsecureLoopbackCookie bool
	Config                 ConfigurationCatalog
	ConfigurationPublisher ConfigurationPublisher
	Credentials            config.CredentialLookup
	ManagedCredentials     ManagedCredentialLifecycle
	RuntimeConfigs         RuntimeConfigManagement
	RuntimeCredentials     RuntimeCredentialManagement
	RuntimeAgentPrincipals RuntimeAgentPrincipalManagement
	Projects               ProjectManagement
	Runs                   RunReader
	PlannerPlans           PlannerPlanReader
	Metrics                MetricsReader
	Operations             OperationsReader
	OperationsInvalidator  OperationsInvalidator
	Events                 *publicevents.Hub
	Artifacts              *artifacts.Service
	Transactions           UnitOfWork
	BearerToken            contracts.SecretString
	NewID                  func(string) (string, error)
	NewRequestID           func() (string, error)
	RunNotifier            RunNotifier
	RunSkills              RunSkillInitializer
	Now                    func() time.Time
	Logger                 *slog.Logger
}

var errInvalidRequest = errors.New("invalid public API request")

type createRunRequest struct {
	Workflow        string                           `json:"workflow"`
	RuntimeLabels   runRuntimeLabels                 `json:"runtimeLabels,omitempty"`
	Labels          runMetadataLabels                `json:"labels,omitempty"`
	Parameters      map[string]string                `json:"parameters"`
	Artifacts       map[string]contracts.ArtifactRef `json:"artifacts"`
	ExecutionConfig config.ExecutionConfigPatch      `json:"executionConfig"`
}

type runRuntimeLabels []string

func (l *runRuntimeLabels) UnmarshalJSON(data []byte) error {
	if bytes.Equal(bytes.TrimSpace(data), []byte("null")) {
		return errors.New("runtimeLabels must be an array")
	}
	var value []string
	if err := json.Unmarshal(data, &value); err != nil || value == nil {
		return errors.New("runtimeLabels must be an array")
	}
	*l = value
	return nil
}

type runMetadataLabels map[string]string

func (l *runMetadataLabels) UnmarshalJSON(data []byte) error {
	if !utf8.Valid(data) {
		return errors.New("labels must contain valid UTF-8")
	}
	decoder := json.NewDecoder(bytes.NewReader(data))
	opening, err := decoder.Token()
	if err != nil || opening != json.Delim('{') {
		return errors.New("labels must be an object of strings")
	}
	result := make(map[string]string)
	for decoder.More() {
		token, err := decoder.Token()
		if err != nil {
			return errors.New("labels must be an object of strings")
		}
		key, ok := token.(string)
		if !ok {
			return errors.New("labels must have string keys")
		}
		if _, duplicate := result[key]; duplicate {
			return fmt.Errorf("labels contains duplicate key %q", key)
		}
		var value string
		if err := decoder.Decode(&value); err != nil {
			return fmt.Errorf("label %q must have a string value", key)
		}
		result[key] = value
	}
	closing, err := decoder.Token()
	if err != nil || closing != json.Delim('}') {
		return errors.New("labels must be an object of strings")
	}
	if _, err := decoder.Token(); !errors.Is(err, io.EOF) {
		return errors.New("labels must contain one object")
	}
	normalized, err := runstore.NormalizeRunMetadataLabels(result)
	if err != nil {
		return err
	}
	*l = runMetadataLabels(normalized)
	return nil
}

type createRunResponse struct {
	RunID                string                     `json:"runId"`
	ProjectID            *string                    `json:"projectId,omitempty"`
	State                runstore.WorkflowRunState  `json:"state"`
	RuntimeLabels        []string                   `json:"runtimeLabels"`
	Labels               runstore.RunMetadataLabels `json:"labels"`
	RuntimeConfiguration runRuntimeConfigResponse   `json:"runtimeConfiguration"`
}

type createProjectRequest struct {
	Kind        projectstore.Kind `json:"kind"`
	Name        string            `json:"name"`
	Description string            `json:"description,omitempty"`
}

func (r *createProjectRequest) UnmarshalJSON(data []byte) error {
	type wire createProjectRequest
	var decoded wire
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&decoded); err != nil {
		return err
	}
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(data, &fields); err != nil {
		return err
	}
	for _, required := range []string{"kind", "name"} {
		value, present := fields[required]
		if !present || bytes.Equal(bytes.TrimSpace(value), []byte("null")) {
			return fmt.Errorf("%s is required", required)
		}
	}
	if value, present := fields["description"]; present && bytes.Equal(bytes.TrimSpace(value), []byte("null")) {
		return errors.New("description cannot be null")
	}
	*r = createProjectRequest(decoded)
	return nil
}

type updateProjectRequest struct {
	Name        *string `json:"name,omitempty"`
	Description *string `json:"description,omitempty"`
}

func (r *updateProjectRequest) UnmarshalJSON(data []byte) error {
	type wire updateProjectRequest
	var decoded wire
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&decoded); err != nil {
		return err
	}
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(data, &fields); err != nil {
		return err
	}
	if len(fields) == 0 {
		return errors.New("at least one Project field is required")
	}
	for name, value := range fields {
		if bytes.Equal(bytes.TrimSpace(value), []byte("null")) {
			return fmt.Errorf("%s cannot be null", name)
		}
	}
	*r = updateProjectRequest(decoded)
	return nil
}

type projectResponse struct {
	ProjectID   string            `json:"projectId"`
	Kind        projectstore.Kind `json:"kind"`
	Name        string            `json:"name"`
	Description string            `json:"description"`
	Revision    string            `json:"revision"`
	CreatedAt   time.Time         `json:"createdAt"`
	UpdatedAt   time.Time         `json:"updatedAt"`
}

type projectPageResponse struct {
	Items []projectResponse `json:"items"`
	Page  pageInfoResponse  `json:"page"`
}

type cancelRunRequest struct {
	Reason *string `json:"reason"`
}

type cancelRunResponse struct {
	RunID        string                            `json:"runId"`
	State        runstore.WorkflowRunState         `json:"state"`
	Cancellation *runstore.WorkflowRunCancellation `json:"cancellation,omitempty"`
}

type artifactWriteResponse struct {
	Artifact  contracts.ArtifactRef `json:"artifact"`
	MediaType string                `json:"mediaType"`
	Size      int64                 `json:"size"`
}

type artifactPageResponse struct {
	Items []artifacts.Metadata `json:"items"`
	Page  pageInfoResponse     `json:"page"`
}

type artifactLineagePageResponse struct {
	Items []artifacts.LineageEdge `json:"items"`
	Page  pageInfoResponse        `json:"page"`
}

type runStatusResponse struct {
	RunID                  string                            `json:"runId"`
	ProjectID              *string                           `json:"projectId,omitempty"`
	Workflow               string                            `json:"workflow"`
	State                  runstore.WorkflowRunState         `json:"state"`
	RuntimeLabels          []string                          `json:"runtimeLabels"`
	Labels                 runstore.RunMetadataLabels        `json:"labels"`
	RuntimeConfiguration   runRuntimeConfigResponse          `json:"runtimeConfiguration"`
	Cancellation           *runstore.WorkflowRunCancellation `json:"cancellation,omitempty"`
	Parameters             map[string]string                 `json:"parameters,omitempty"`
	Inputs                 map[string]contracts.ArtifactRef  `json:"inputs,omitempty"`
	Attempts               []stageAttemptResponse            `json:"attempts"`
	Transitions            []stageTransitionResponse         `json:"transitions"`
	Outputs                map[string]contracts.ArtifactRef  `json:"outputs"`
	OutputPublications     []outputPublicationResponse       `json:"outputPublications"`
	EventCursor            *eventCursorResponse              `json:"eventCursor,omitempty"`
	ActiveStageExecutionID *string                           `json:"activeStageExecutionId,omitempty"`
	CreatedAt              time.Time                         `json:"createdAt,omitempty"`
	UpdatedAt              time.Time                         `json:"updatedAt,omitempty"`
	StartedAt              *time.Time                        `json:"startedAt,omitempty"`
	FinishedAt             *time.Time                        `json:"finishedAt,omitempty"`
}

type outputPublicationResponse struct {
	Output       string                           `json:"output"`
	Status       runstore.OutputPublicationStatus `json:"status"`
	Source       contracts.ArtifactRef            `json:"source"`
	Target       *contracts.ArtifactRef           `json:"target,omitempty"`
	ErrorCode    string                           `json:"errorCode,omitempty"`
	ErrorMessage string                           `json:"errorMessage,omitempty"`
	CreatedAt    time.Time                        `json:"createdAt"`
}

type pinnedRuntimeConfigResponse struct {
	Label           string            `json:"label"`
	BindingRevision string            `json:"bindingRevision"`
	Config          runtimeconfig.Ref `json:"config"`
}

type runRuntimeConfigResponse struct {
	Default pinnedRuntimeConfigResponse   `json:"default"`
	Labels  []pinnedRuntimeConfigResponse `json:"labels"`
}

type stageAttemptResponse struct {
	StageExecutionID     string                             `json:"stageExecutionId"`
	Stage                string                             `json:"stage"`
	Objective            string                             `json:"objective,omitempty"`
	Attempt              int                                `json:"attempt"`
	PreviousExecutionID  *string                            `json:"previousExecutionId,omitempty"`
	ExecutionConfig      stageExecutionConfigResponse       `json:"executionConfig"`
	State                runstore.StageExecutionState       `json:"state"`
	Result               *contracts.StageContentResult      `json:"result,omitempty"`
	Termination          *runstore.StageTermination         `json:"termination,omitempty"`
	Metrics              *telemetry.Summary                 `json:"metrics,omitempty"`
	Diagnostics          *telemetry.AttemptDiagnostics      `json:"diagnostics,omitempty"`
	Plan                 *planner.PlannerPlanProjection     `json:"plan,omitempty"`
	RuntimeConfiguration *stageRuntimeConfigurationResponse `json:"runtimeConfiguration,omitempty"`
	CreatedAt            time.Time                          `json:"createdAt,omitempty"`
	UpdatedAt            time.Time                          `json:"updatedAt,omitempty"`
	PlannerStartedAt     *time.Time                         `json:"plannerStartedAt,omitempty"`
	TerminalAt           *time.Time                         `json:"terminalAt,omitempty"`
}

type stageRuntimeConfigurationResponse struct {
	Allocations []stageRuntimeAllocationResponse `json:"allocations"`
}

type stageRuntimeAllocationResponse struct {
	LogicalAgent    string                                     `json:"logicalAgent"`
	AgentLabels     []pinnedRuntimeConfigResponse              `json:"agentLabels"`
	RuntimeAdapters []contracts.RuntimeAdapterRef              `json:"runtimeAdapters"`
	Origins         runtimeconfig.ResolvedRuntimeConfigOrigins `json:"origins"`
	Status          string                                     `json:"status"`
}

type eventCursorResponse struct {
	Generation string `json:"generation"`
	Sequence   string `json:"sequence"`
}

type runSummaryResponse struct {
	RunID      string                     `json:"runId"`
	ProjectID  *string                    `json:"projectId,omitempty"`
	Workflow   string                     `json:"workflow"`
	State      runstore.WorkflowRunState  `json:"state"`
	Labels     runstore.RunMetadataLabels `json:"labels"`
	CreatedAt  time.Time                  `json:"createdAt"`
	UpdatedAt  time.Time                  `json:"updatedAt"`
	FinishedAt *time.Time                 `json:"finishedAt,omitempty"`
}

type runPageResponse struct {
	Items []runSummaryResponse `json:"items"`
	Page  pageInfoResponse     `json:"page"`
}

type queueProjectResponse struct {
	ProjectID string            `json:"projectId"`
	Name      string            `json:"name"`
	Kind      projectstore.Kind `json:"kind"`
}

type queueItemResponse struct {
	RunID       string                     `json:"runId"`
	Project     *queueProjectResponse      `json:"project,omitempty"`
	Workflow    string                     `json:"workflow"`
	State       runstore.WorkflowRunState  `json:"state"`
	Labels      runstore.RunMetadataLabels `json:"labels"`
	EventCursor eventCursorResponse        `json:"eventCursor"`
	CreatedAt   time.Time                  `json:"createdAt"`
	UpdatedAt   time.Time                  `json:"updatedAt"`
}

type queuePageResponse struct {
	Items []queueItemResponse `json:"items"`
	Page  pageInfoResponse    `json:"page"`
}

type stageExecutionConfigResponse struct {
	Variant           runstore.StageExecutionConfigVariant           `json:"variant"`
	EscalationOrdinal *int                                           `json:"escalationOrdinal,omitempty"`
	Ref               *config.ExecutionConfigRef                     `json:"ref,omitempty"`
	Planner           *consumerExecutionConfigRefsResponse           `json:"planner,omitempty"`
	Agents            map[string]consumerExecutionConfigRefsResponse `json:"agents"`
}

type consumerExecutionConfigRefsResponse struct {
	ModelPolicy contracts.ModelPolicyRef       `json:"modelPolicy"`
	LLMGateway  *contracts.LLMGatewayConfigRef `json:"llmGateway,omitempty"`
	Credential  *contracts.LLMCredentialRef    `json:"credential,omitempty"`
	Origins     config.ExecutionConfigOrigins  `json:"origins"`
}

type workflowPageResponse struct {
	Items []workflowSummaryResponse `json:"items"`
	Page  pageInfoResponse          `json:"page"`
}

type configurationPageResponse struct {
	Items []config.ConfigurationResource `json:"items"`
	Page  pageInfoResponse               `json:"page"`
}

type publishConfigurationRequest struct {
	Name        string                         `json:"name"`
	Version     string                         `json:"version"`
	ModelPolicy *config.ModelPolicyPublication `json:"modelPolicy,omitempty"`
	LLMGateway  *config.LLMGatewayPublication  `json:"llmGateway,omitempty"`
}

type createCredentialRequest struct {
	CredentialID  string                        `json:"credentialId"`
	LLMGateway    contracts.LLMGatewayConfigRef `json:"llmGateway"`
	Label         *string                       `json:"label,omitempty"`
	GatewayPolicy credentials.GatewayPolicy     `json:"gatewayPolicy"`
}

func (r *createCredentialRequest) UnmarshalJSON(data []byte) error {
	type wire createCredentialRequest
	var decoded wire
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&decoded); err != nil {
		return err
	}
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(data, &fields); err != nil {
		return err
	}
	if raw, present := fields["label"]; present && bytes.Equal(bytes.TrimSpace(raw), []byte("null")) {
		return errors.New("credential label cannot be null")
	}
	if raw, present := fields["gatewayPolicy"]; present && !bytes.Equal(bytes.TrimSpace(raw), []byte("null")) {
		var policyFields map[string]json.RawMessage
		if err := json.Unmarshal(raw, &policyFields); err != nil {
			return err
		}
		for _, name := range []string{
			"maxBudget", "budgetDuration", "tpmLimit", "rpmLimit", "maxParallelRequests",
		} {
			if value, exists := policyFields[name]; exists && bytes.Equal(bytes.TrimSpace(value), []byte("null")) {
				return errors.New("Gateway policy optional fields cannot be null")
			}
		}
		if _, exists := policyFields["budgetDuration"]; exists && decoded.GatewayPolicy.BudgetDuration == "" {
			return errors.New("Gateway policy budget duration cannot be empty")
		}
	}
	*r = createCredentialRequest(decoded)
	return nil
}

type credentialPageResponse struct {
	Items []credentials.Record `json:"items"`
	Page  pageInfoResponse     `json:"page"`
}

type runtimeConfigResourceResponse struct {
	Ref       runtimeconfig.Ref `json:"ref"`
	Document  json.RawMessage   `json:"document"`
	BuiltIn   bool              `json:"builtIn"`
	CreatedBy string            `json:"createdBy"`
	CreatedAt time.Time         `json:"createdAt"`
}

type runtimeConfigPageResponse struct {
	Items []runtimeConfigResourceResponse `json:"items"`
	Page  pageInfoResponse                `json:"page"`
}

type runtimeLabelPageResponse struct {
	Items []runtimeLabelResponse `json:"items"`
	Page  pageInfoResponse       `json:"page"`
}

type runtimeLabelResponse struct {
	Label     string            `json:"label"`
	Config    runtimeconfig.Ref `json:"config"`
	Revision  string            `json:"revision"`
	CreatedBy string            `json:"createdBy"`
	CreatedAt time.Time         `json:"createdAt"`
	UpdatedBy string            `json:"updatedBy"`
	UpdatedAt time.Time         `json:"updatedAt"`
}

type runtimeLabelMutationRequest struct {
	Config runtimeconfig.Ref `json:"config"`
}

type createRuntimeCredentialRequest struct {
	CredentialID string
	Material     credentials.RuntimeCredentialMaterial
}

func (r *createRuntimeCredentialRequest) UnmarshalJSON(data []byte) error {
	var envelope struct {
		CredentialID string                            `json:"credentialId"`
		Kind         credentials.RuntimeCredentialKind `json:"kind"`
		Material     json.RawMessage                   `json:"material"`
	}
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&envelope); err != nil {
		return err
	}
	if len(envelope.Material) == 0 || bytes.Equal(bytes.TrimSpace(envelope.Material), []byte("null")) {
		return errors.New("Runtime credential material is required")
	}
	var material credentials.RuntimeCredentialMaterial
	var err error
	switch envelope.Kind {
	case credentials.RuntimeCredentialOTLPHeaders:
		var value struct {
			Headers map[string]string `json:"headers"`
		}
		if err = decodeStrictPublicJSON(envelope.Material, &value); err == nil {
			material, err = credentials.NewOTLPHeadersCredential(value.Headers)
		}
	case credentials.RuntimeCredentialProxyBasic:
		var value struct {
			Username string `json:"username"`
			Password string `json:"password"`
		}
		if err = decodeStrictPublicJSON(envelope.Material, &value); err == nil {
			material, err = credentials.NewHTTPProxyBasicCredential(value.Username, value.Password)
		}
	case credentials.RuntimeCredentialProxyBearer:
		var value struct {
			Token string `json:"token"`
		}
		if err = decodeStrictPublicJSON(envelope.Material, &value); err == nil {
			material, err = credentials.NewHTTPProxyBearerCredential(value.Token)
		}
	case credentials.RuntimeCredentialCaidoBearer:
		var value struct {
			Token string `json:"token"`
		}
		if err = decodeStrictPublicJSON(envelope.Material, &value); err == nil {
			material, err = credentials.NewCaidoBearerCredential(value.Token)
		}
	default:
		err = credentials.ErrRuntimeCredentialInvalid
	}
	if err != nil {
		material.Destroy()
		return err
	}
	r.CredentialID = envelope.CredentialID
	r.Material = material
	return nil
}

type runtimeCredentialPageResponse struct {
	Items []credentials.RuntimeCredentialMetadata `json:"items"`
	Page  pageInfoResponse                        `json:"page"`
}

type runtimeAgentPrincipalResponse struct {
	RuntimeAgentID          string                                         `json:"runtimeAgentId"`
	Labels                  []string                                       `json:"labels"`
	Revision                string                                         `json:"revision"`
	Availability            controlplane.RuntimeAgentPrincipalAvailability `json:"availability"`
	RequiredRuntimeAdapters []string                                       `json:"requiredRuntimeAdapters"`
	MissingRuntimeAdapters  []string                                       `json:"missingRuntimeAdapters"`
	Live                    *controlplane.RuntimeAgentObservation          `json:"live,omitempty"`
	CreatedBy               string                                         `json:"createdBy"`
	CreatedAt               time.Time                                      `json:"createdAt"`
	UpdatedBy               string                                         `json:"updatedBy"`
	UpdatedAt               time.Time                                      `json:"updatedAt"`
}

type runtimeAgentPrincipalPageResponse struct {
	Items []runtimeAgentPrincipalResponse `json:"items"`
	Page  pageInfoResponse                `json:"page"`
}

type runtimeAgentLabelsMutationRequest struct {
	Labels []string `json:"labels"`
}

func (r *runtimeAgentLabelsMutationRequest) UnmarshalJSON(data []byte) error {
	var envelope struct {
		Labels json.RawMessage `json:"labels"`
	}
	if err := decodeStrictPublicJSON(data, &envelope); err != nil {
		return err
	}
	if len(envelope.Labels) == 0 || bytes.Equal(bytes.TrimSpace(envelope.Labels), []byte("null")) {
		return errors.New("Runtime Agent labels must be an array")
	}
	if err := json.Unmarshal(envelope.Labels, &r.Labels); err != nil || r.Labels == nil {
		return errors.New("Runtime Agent labels must be an array")
	}
	return nil
}

type operationsCursorResponse struct {
	Generation string `json:"generation"`
	Revision   string `json:"revision"`
}

type loginRequest struct {
	Username string `json:"username"`
	Password string `json:"password"`
}

type authSessionResponse struct {
	Principal         auth.Principal `json:"principal"`
	CSRFToken         string         `json:"csrfToken"`
	IdleExpiresAt     time.Time      `json:"idleExpiresAt"`
	AbsoluteExpiresAt time.Time      `json:"absoluteExpiresAt"`
}

type operationsSnapshotResponse struct {
	Cursor        operationsCursorResponse               `json:"cursor"`
	RuntimeAgents []controlplane.RuntimeAgentObservation `json:"runtimeAgents"`
	Allocations   []controlplane.AllocationObservation   `json:"allocations"`
}

type runtimeAgentPageResponse struct {
	Cursor operationsCursorResponse               `json:"cursor"`
	Items  []controlplane.RuntimeAgentObservation `json:"items"`
	Page   pageInfoResponse                       `json:"page"`
}

type allocationPageResponse struct {
	Cursor operationsCursorResponse             `json:"cursor"`
	Items  []controlplane.AllocationObservation `json:"items"`
	Page   pageInfoResponse                     `json:"page"`
}

type workflowSummaryResponse struct {
	Ref        config.WorkflowRef              `json:"ref"`
	EntryStage string                          `json:"entryStage"`
	Parameters map[string]config.ParameterSlot `json:"parameters"`
	Inputs     map[string]config.ArtifactSlot  `json:"inputs"`
	Outputs    map[string]config.ArtifactSlot  `json:"outputs"`
}

type workflowResourceResponse struct {
	workflowSummaryResponse
	Stages map[string]workflowStageResponse `json:"stages"`
}

type instructionsRefResponse struct {
	Ref    string `json:"ref"`
	Digest string `json:"digest"`
}

type workflowAgentBindingResponse struct {
	Template  contracts.AgentTemplateRef `json:"template"`
	Namespace string                     `json:"namespace"`
	Skills    []contracts.ArtifactRef    `json:"skills"`
}

type workflowStageResponse struct {
	Objective        string                                  `json:"objective"`
	Instructions     instructionsRefResponse                 `json:"instructions"`
	Planner          config.PlannerRef                       `json:"planner"`
	Agents           map[string]workflowAgentBindingResponse `json:"agents"`
	ExecutionConfig  resolvedStageExecutionConfigResponse    `json:"executionConfig"`
	ContextArtifacts map[string]config.ContextArtifact       `json:"contextArtifacts"`
	ResultArtifacts  map[string]config.ArtifactSlot          `json:"resultArtifacts"`
	WorkflowOutputs  map[string]string                       `json:"workflowOutputs"`
	On               workflowTransitionsResponse             `json:"on"`
}

type resolvedStageExecutionConfigResponse struct {
	Planner *consumerExecutionConfigRefsResponse           `json:"planner,omitempty"`
	Agents  map[string]consumerExecutionConfigRefsResponse `json:"agents"`
}

type workflowTransitionsResponse struct {
	Succeeded   workflowTransitionResponse `json:"succeeded"`
	Failed      workflowTransitionResponse `json:"failed"`
	Interrupted workflowTransitionResponse `json:"interrupted"`
}

type workflowTransitionResponse struct {
	Kind            config.TransitionKind             `json:"kind"`
	NextStage       string                            `json:"nextStage,omitempty"`
	MaxAttempts     int                               `json:"maxAttempts,omitempty"`
	ExecutionConfig *workflowEscalationConfigResponse `json:"executionConfig,omitempty"`
	Then            *workflowTransitionResponse       `json:"then,omitempty"`
}

type workflowEscalationConfigResponse struct {
	Ref       *config.ExecutionConfigRef           `json:"ref,omitempty"`
	Effective resolvedStageExecutionConfigResponse `json:"effective"`
}

type stageTransitionResponse struct {
	SourceExecutionID   string                         `json:"sourceExecutionId"`
	Action              runstore.StageTransitionAction `json:"action"`
	TargetStage         *string                        `json:"targetStage,omitempty"`
	TargetExecutionID   *string                        `json:"targetExecutionId,omitempty"`
	EscalationOrdinal   *int                           `json:"escalationOrdinal,omitempty"`
	EscalationExhausted bool                           `json:"escalationExhausted"`
	DecidedAt           time.Time                      `json:"decidedAt"`
}

type errorResponse struct {
	Code      string `json:"code"`
	Message   string `json:"message"`
	Retryable bool   `json:"retryable"`
	RequestID string `json:"requestId"`
	Details   any    `json:"details,omitempty"`
}

type runtimeCredentialInUseDetailsResponse struct {
	Kind          string   `json:"kind"`
	BindingLabels []string `json:"bindingLabels"`
	RunIDs        []string `json:"runIds"`
	AllocationIDs []string `json:"allocationIds"`
}

type runtimeLabelInUseDetailsResponse struct {
	Kind            string   `json:"kind"`
	RuntimeAgentIDs []string `json:"runtimeAgentIds"`
}

type credentialInUseDetailsResponse struct {
	Kind   string   `json:"kind"`
	RunIDs []string `json:"runIds"`
}
