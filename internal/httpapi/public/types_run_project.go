package public

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"time"
	"unicode/utf8"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/gatewayrecovery"
	"github.com/grauwolf32/contractor/internal/planner"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
	"github.com/grauwolf32/contractor/internal/telemetry"
)

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
	if isJSONNull(data) {
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
	RunID                string                         `json:"runId"`
	ProjectID            *string                        `json:"projectId,omitempty"`
	State                runstore.WorkflowRunState      `json:"state"`
	RuntimeLabels        []string                       `json:"runtimeLabels"`
	Labels               runstore.RunMetadataLabels     `json:"labels"`
	RuntimeConfiguration runRuntimeConfigResponse       `json:"runtimeConfiguration"`
	ProjectHTTPTarget    *contracts.HTTPOriginTargetRef `json:"projectHttpTarget,omitempty"`
}

type createProjectRequest struct {
	Kind        projectstore.Kind `json:"kind"`
	Name        string            `json:"name"`
	Description string            `json:"description,omitempty"`
}

func (r *createProjectRequest) UnmarshalJSON(data []byte) error {
	type wire createProjectRequest
	var decoded wire
	fields, err := decodeStrictWithPresence(data, &decoded)
	if err != nil {
		return err
	}
	for _, required := range []string{"kind", "name"} {
		if fields.missingOrNull(required) {
			return fmt.Errorf("%s is required", required)
		}
	}
	if fields.null("description") {
		return errors.New("description cannot be null")
	}
	*r = createProjectRequest(decoded)
	return nil
}

type updateProjectRequest struct {
	Name        *string                   `json:"name,omitempty"`
	Description *string                   `json:"description,omitempty"`
	HTTPTarget  *projectHTTPTargetRequest `json:"httpTarget,omitempty"`
	targetSet   bool
}

type projectHTTPTargetRequest struct {
	URL        string                          `json:"url"`
	Credential *contracts.RuntimeCredentialRef `json:"credential,omitempty"`
}

func (r *updateProjectRequest) UnmarshalJSON(data []byte) error {
	type wire updateProjectRequest
	var decoded wire
	fields, err := decodeStrictWithPresence(data, &decoded)
	if err != nil {
		return err
	}
	if len(fields) == 0 {
		return errors.New("at least one Project field is required")
	}
	for name, value := range fields {
		if name != "httpTarget" && isJSONNull(value) {
			return fmt.Errorf("%s cannot be null", name)
		}
	}
	if raw, present := fields["httpTarget"]; present {
		decoded.targetSet = true
		if !isJSONNull(raw) {
			var target projectHTTPTargetRequest
			targetFields, err := decodeStrictWithPresence(raw, &target)
			if err != nil {
				return err
			}
			if targetFields.missingOrNull("url") {
				return errors.New("httpTarget.url is required")
			}
			candidate := contracts.HTTPOriginTargetRef{URL: target.URL, Credential: target.Credential}
			if err := candidate.Validate(); err != nil {
				return err
			}
			decoded.HTTPTarget = &target
		}
	}
	*r = updateProjectRequest(decoded)
	return nil
}

type projectResponse struct {
	ProjectID   string                         `json:"projectId"`
	Kind        projectstore.Kind              `json:"kind"`
	Name        string                         `json:"name"`
	Description string                         `json:"description"`
	HTTPTarget  *contracts.HTTPOriginTargetRef `json:"httpTarget,omitempty"`
	Lifecycle   projectstore.Lifecycle         `json:"lifecycle"`
	Deletion    *projectDeletionResponse       `json:"deletion,omitempty"`
	Revision    string                         `json:"revision"`
	CreatedAt   time.Time                      `json:"createdAt"`
	UpdatedAt   time.Time                      `json:"updatedAt"`
}

type projectDeletionResponse struct {
	Phase       projectstore.DeletionPhase `json:"phase"`
	RequestedAt time.Time                  `json:"requestedAt"`
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
	Recovery               *gatewayrecovery.Status           `json:"recovery,omitempty"`
	RunID                  string                            `json:"runId"`
	ProjectID              *string                           `json:"projectId,omitempty"`
	Workflow               string                            `json:"workflow"`
	State                  runstore.WorkflowRunState         `json:"state"`
	Deletable              bool                              `json:"deletable"`
	ResumeStageExecutionID *string                           `json:"resumeStageExecutionId,omitempty"`
	RuntimeLabels          []string                          `json:"runtimeLabels"`
	Labels                 runstore.RunMetadataLabels        `json:"labels"`
	RuntimeConfiguration   runRuntimeConfigResponse          `json:"runtimeConfiguration"`
	ProjectHTTPTarget      *contracts.HTTPOriginTargetRef    `json:"projectHttpTarget,omitempty"`
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

type runRepeatDraftResponse struct {
	SourceRunID string                 `json:"sourceRunId"`
	Authority   string                 `json:"authority"`
	Workflow    config.WorkflowRef     `json:"workflow"`
	ProjectID   *string                `json:"projectId,omitempty"`
	AuditID     *string                `json:"auditId,omitempty"`
	Draft       *runRepeatDraft        `json:"draft,omitempty"`
	Notices     []runRepeatDraftNotice `json:"notices"`
}

type runRepeatDraft struct {
	Parameters      map[string]string                  `json:"parameters"`
	RuntimeLabels   []string                           `json:"runtimeLabels"`
	Labels          runstore.RunMetadataLabels         `json:"labels"`
	ExecutionConfig runRepeatExecutionConfig           `json:"executionConfig"`
	Inputs          map[string]runRepeatInputSelection `json:"inputs"`
}

type runRepeatExecutionConfig struct {
	Status string                      `json:"status"`
	Value  config.ExecutionConfigPatch `json:"value"`
}

type runRepeatInputSelection struct {
	Status      string                 `json:"status"`
	SourceScope artifacts.ScopeKind    `json:"sourceScope,omitempty"`
	Artifact    *contracts.ArtifactRef `json:"artifact,omitempty"`
	Metadata    *artifacts.Metadata    `json:"metadata,omitempty"`
	Code        string                 `json:"code,omitempty"`
	Message     string                 `json:"message,omitempty"`
}

type runRepeatDraftNotice struct {
	Code     string `json:"code"`
	Severity string `json:"severity"`
	Field    string `json:"field,omitempty"`
	Message  string `json:"message"`
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
	StageExecutionID     string                                `json:"stageExecutionId"`
	Stage                string                                `json:"stage"`
	Objective            string                                `json:"objective,omitempty"`
	Attempt              int                                   `json:"attempt"`
	PreviousExecutionID  *string                               `json:"previousExecutionId,omitempty"`
	ExecutionConfig      stageExecutionConfigResponse          `json:"executionConfig"`
	State                runstore.StageExecutionState          `json:"state"`
	Result               *contracts.StageContentResult         `json:"result,omitempty"`
	Termination          *runstore.StageTermination            `json:"termination,omitempty"`
	Metrics              *telemetry.Summary                    `json:"metrics,omitempty"`
	Diagnostics          *telemetry.AttemptDiagnostics         `json:"diagnostics,omitempty"`
	Plan                 *planner.PlannerPlanProjection        `json:"plan,omitempty"`
	RuntimeConfiguration *stageRuntimeConfigurationResponse    `json:"runtimeConfiguration,omitempty"`
	Resources            []telemetry.AllocationResourceSummary `json:"resources"`
	CreatedAt            time.Time                             `json:"createdAt,omitempty"`
	UpdatedAt            time.Time                             `json:"updatedAt,omitempty"`
	PlannerStartedAt     *time.Time                            `json:"plannerStartedAt,omitempty"`
	TerminalAt           *time.Time                            `json:"terminalAt,omitempty"`
}

type allocationResourcePageResponse struct {
	Items []telemetry.AllocationResourceSummary `json:"items"`
	Page  pageInfoResponse                      `json:"page"`
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
	Deletable  bool                       `json:"deletable"`
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

type ownerQueueControlResponse struct {
	Paused    bool       `json:"paused"`
	Revision  string     `json:"revision"`
	UpdatedAt *time.Time `json:"updatedAt,omitempty"`
}

type updateOwnerQueueControlRequest struct {
	Paused *bool `json:"paused"`
}

type stageExecutionConfigResponse struct {
	Variant           runstore.StageExecutionConfigVariant           `json:"variant"`
	EscalationOrdinal *int                                           `json:"escalationOrdinal,omitempty"`
	Ref               *config.ExecutionConfigRef                     `json:"ref,omitempty"`
	Planner           *consumerExecutionConfigRefsResponse           `json:"planner,omitempty"`
	Agents            map[string]consumerExecutionConfigRefsResponse `json:"agents"`
}

type consumerExecutionConfigRefsResponse struct {
	ModelPolicy contracts.ModelPolicyRef       `json:"modelPolicy,omitzero"`
	LLMGateway  *contracts.LLMGatewayConfigRef `json:"llmGateway,omitempty"`
	Credential  *contracts.LLMCredentialRef    `json:"credential,omitempty"`
	Origins     config.ExecutionConfigOrigins  `json:"origins"`
}
