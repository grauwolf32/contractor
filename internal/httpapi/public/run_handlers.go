package public

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contracts/runtimesettings"
	"github.com/grauwolf32/contractor/internal/pagelimit"
	"github.com/grauwolf32/contractor/internal/planner"
	"github.com/grauwolf32/contractor/internal/runservice"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
	"github.com/grauwolf32/contractor/internal/telemetry"
)

const maxCancellationReasonBytes = 4096

type RunReader interface {
	GetRun(context.Context, string) (runstore.WorkflowRun, error)
	ListRuns(context.Context, runstore.ListRunsParams) ([]runstore.WorkflowRunSummary, error)
}

// RunDetailReader reads a Run detail, its event cursor included, from one
// snapshot (API-R48).
type RunDetailReader interface {
	Read(context.Context, string, string) (runservice.RunDetail, error)
}

type RunLifecycle interface {
	ResumeFailedRun(context.Context, string, string, string, string) (runstore.ResumeRunResult, error)
	RequestRunCancellation(context.Context, string, runstore.WorkflowRunCancellation) (runstore.WorkflowRun, error)
}

type RunDeletion interface {
	DeletableRuns(context.Context, string, []string) (map[string]bool, error)
	DeleteReleasedTerminalRun(context.Context, string, string) error
}

type RunCreationService interface {
	CreatePublic(context.Context, runservice.PublicCreateParams) (runservice.CreateResult, error)
}

// RunNotifier is the Scheduler: Wake makes it look for admissible work and
// Cancel makes it stop the active work of one cancelling Run now.
type RunNotifier interface {
	Wake()
	Cancel(string)
}

// GatewayRecovery retries the gateway wait of one owner's Run.
type GatewayRecovery interface {
	Retry(context.Context, string, string) error
}

// RunDependencies serve Run creation, listing, detail, cancellation, resume,
// gateway retry and deletion.
type RunDependencies struct {
	Runs            RunReader
	RunDetails      RunDetailReader
	RunLifecycle    RunLifecycle
	RunDeletion     RunDeletion
	RunCreator      RunCreationService
	RunNotifier     RunNotifier
	GatewayRecovery GatewayRecovery
	Projects        ProjectReader
	NewID           func(string) (string, error)
	Now             func() time.Time
}

type runHandlers struct {
	RunDependencies
	cursors pageCursors
}

func (h *runHandlers) listRuns(w http.ResponseWriter, r *http.Request) {
	if rejectHead(w, r) {
		return
	}
	h.listRunsFromProject(w, r, nil)
}

func (h *runHandlers) listProjectRuns(w http.ResponseWriter, r *http.Request) {
	if rejectHead(w, r) {
		return
	}
	projectID := r.PathValue("projectId")
	if _, err := h.Projects.Get(
		r.Context(), principalUserID(r.Context()), projectID,
	); err != nil {
		handleError(w, err)
		return
	}
	h.listRunsFromProject(w, r, &projectID)
}

func (h *runHandlers) listRunsFromProject(w http.ResponseWriter, r *http.Request, projectID *string) {
	query, limit, encodedCursor, err := pageQueryWithRepeated(
		r.URL.RawQuery, "label", runstore.MaxRunMetadataLabels, "state", "lifecycle",
	)
	if err != nil {
		handleError(w, err)
		return
	}
	var state *runstore.WorkflowRunState
	if values, present := query["state"]; present {
		candidate := runstore.WorkflowRunState(values[0])
		if !candidate.Valid() {
			handleError(w, fmt.Errorf("%w: unknown Run state", errInvalidRequest))
			return
		}
		state = &candidate
	}
	var lifecycle *runstore.WorkflowRunLifecycle
	if values, present := query["lifecycle"]; present {
		candidate := runstore.WorkflowRunLifecycle(values[0])
		if !candidate.Valid() {
			handleError(w, fmt.Errorf("%w: unknown Run lifecycle", errInvalidRequest))
			return
		}
		lifecycle = &candidate
	}
	selectors, err := parseRunMetadataLabelSelectors(query["label"])
	if err != nil {
		handleError(w, err)
		return
	}
	cursorKind := runListCursorKindForProject(state, lifecycle, selectors, projectID)
	cursor, err := h.cursors.decodePageCursor(encodedCursor, cursorKind, 2)
	if err != nil {
		handleError(w, err)
		return
	}
	params := runstore.ListRunsParams{
		OwnerID: principalUserID(r.Context()), State: state, Lifecycle: lifecycle,
		ProjectID: projectID, MetadataLabelSelectors: selectors, Limit: pagelimit.FetchRows(limit),
	}
	if len(cursor) != 0 {
		before, parseErr := time.Parse(time.RFC3339Nano, cursor[0])
		if parseErr != nil {
			handleError(w, fmt.Errorf("%w: invalid Run cursor", errInvalidRequest))
			return
		}
		params.BeforeCreatedAt = &before
		params.BeforeRunID = cursor[1]
	}
	runs, err := h.Runs.ListRuns(r.Context(), params)
	if err != nil {
		handleError(w, err)
		return
	}
	runs, page, err := paginate(h.cursors, runs, limit, cursorKind, func(last runstore.WorkflowRunSummary) []string {
		return []string{last.CreatedAt.UTC().Format(time.RFC3339Nano), last.RunID}
	})
	if err != nil {
		handleError(w, err)
		return
	}
	runIDs := make([]string, 0, len(runs))
	for _, run := range runs {
		runIDs = append(runIDs, run.RunID)
	}
	deletable, err := h.RunDeletion.DeletableRuns(r.Context(), params.OwnerID, runIDs)
	if err != nil {
		handleError(w, err)
		return
	}
	items := make([]runSummaryResponse, 0, len(runs))
	for _, run := range runs {
		items = append(items, runSummaryResponse{
			RunID: run.RunID, ProjectID: run.ProjectID,
			Workflow: run.WorkflowName + "@" + run.WorkflowVersion,
			State:    run.State, Deletable: deletable[run.RunID],
			CreatedAt: run.CreatedAt, UpdatedAt: run.UpdatedAt,
			Labels: run.MetadataLabels.Clone(), FinishedAt: run.FinishedAt,
		})
	}
	writeJSON(w, http.StatusOK, runPageResponse{Items: items, Page: page})
}

func parseRunMetadataLabelSelectors(values []string) ([]runstore.RunMetadataLabelSelector, error) {
	selectors := make([]runstore.RunMetadataLabelSelector, 0, len(values))
	for _, value := range values {
		parts := strings.SplitN(value, "=", 2)
		if len(parts) != 2 {
			return nil, fmt.Errorf("%w: Run label selector must contain key=value", errInvalidRequest)
		}
		selectors = append(selectors, runstore.RunMetadataLabelSelector{Key: parts[0], Value: parts[1]})
	}
	normalized, err := runstore.NormalizeRunMetadataLabelSelectors(selectors)
	if err != nil {
		return nil, fmt.Errorf("%w: invalid Run label selector", errInvalidRequest)
	}
	return normalized, nil
}

func runListCursorKindForProject(
	state *runstore.WorkflowRunState,
	lifecycle *runstore.WorkflowRunLifecycle,
	selectors []runstore.RunMetadataLabelSelector,
	projectID *string,
) string {
	kind := "runs"
	if projectID != nil {
		kind += ":project:" + *projectID
	}
	if state != nil {
		kind += ":" + string(*state)
	}
	if lifecycle != nil {
		kind += ":lifecycle:" + string(*lifecycle)
	}
	if len(selectors) == 0 {
		return kind
	}
	encoded, _ := json.Marshal(selectors)
	return kind + ":labels:" + string(encoded)
}

func (h *runHandlers) createRun(w http.ResponseWriter, r *http.Request) {
	h.createRunFromProject(w, r, nil)
}

func (h *runHandlers) createProjectRun(w http.ResponseWriter, r *http.Request) {
	projectID := r.PathValue("projectId")
	if _, err := h.Projects.Get(
		r.Context(), principalUserID(r.Context()), projectID,
	); err != nil {
		handleError(w, err)
		return
	}
	h.createRunFromProject(w, r, &projectID)
}

func (h *runHandlers) createRunFromProject(w http.ResponseWriter, r *http.Request, projectID *string) {
	if _, err := exactQuery(r.URL.RawQuery); err != nil {
		handleError(w, err)
		return
	}
	var request createRunRequest
	if err := decodeJSON(w, r, &request); err != nil {
		handleError(w, err)
		return
	}
	normalizedLabels, err := runtimeconfig.NormalizeRunLabels([]string(request.RuntimeLabels))
	if err != nil {
		handleError(w, err)
		return
	}
	metadataLabels, err := runstore.NormalizeRunMetadataLabels(request.Labels)
	if err != nil {
		handleError(w, err)
		return
	}
	idempotencyKey, err := requireIdempotencyKey(r)
	if err != nil {
		handleError(w, err)
		return
	}
	credentialUser := runtimeCredentialUser(r.Context())
	params := runservice.PublicCreateParams{
		OwnerID: credentialUser.UserID, OperationsPrincipal: credentialUser.Operations,
		ProjectID: projectID, Workflow: request.Workflow,
		ExecutionConfig: request.ExecutionConfig, RuntimeLabels: normalizedLabels,
		MetadataLabels: metadataLabels, Parameters: request.Parameters, Inputs: request.Artifacts,
		IdempotencyKey: idempotencyKey,
		NewRunID:       func() (string, error) { return h.NewID("run_") },
	}
	params.RequestDigest, err = runservice.PublicRequestDigest(params)
	if err != nil {
		handleError(w, fmt.Errorf("digest Run request: %w", err))
		return
	}
	result, err := h.RunCreator.CreatePublic(r.Context(), params)
	if err != nil {
		handleError(w, err)
		return
	}
	// Skill package validation and RunScope forking are intentionally deferred
	// to Scheduler recovery. The creation transaction above has already pinned
	// one complete exact owner-source outcome, so returning the initializing Run
	// neither holds the HTTP request on bounded archive work nor permits a later
	// owner binding update to change what recovery will read.
	if result.Created {
		h.RunNotifier.Wake()
	}
	if result.Replayed {
		w.Header().Set("Idempotency-Replayed", "true")
	}
	writeJSON(w, http.StatusAccepted, createRunReadModel(result.Run))
}

func createRunReadModel(run runstore.WorkflowRun) createRunResponse {
	return createRunResponse{
		RunID: run.RunID, ProjectID: run.ProjectID, State: run.State,
		RuntimeLabels:        append([]string{}, run.RuntimeLabels...),
		Labels:               run.MetadataLabels.Clone(),
		RuntimeConfiguration: runtimeConfigReadModel(run.RuntimeConfig),
		ProjectHTTPTarget:    run.ProjectHTTPTarget.Clone(),
	}
}

func (h *runHandlers) cancelRun(w http.ResponseWriter, r *http.Request) {
	if _, err := exactQuery(r.URL.RawQuery); err != nil {
		handleError(w, err)
		return
	}
	var request *cancelRunRequest
	if err := decodeJSON(w, r, &request); err != nil {
		handleError(w, err)
		return
	}
	if request == nil {
		handleError(w, fmt.Errorf("%w: cancellation request must be an object", errInvalidRequest))
		return
	}
	if request.Reason != nil {
		trimmed := strings.TrimSpace(*request.Reason)
		if trimmed == "" || len([]byte(trimmed)) > maxCancellationReasonBytes {
			handleError(w, fmt.Errorf("%w: cancellation reason is empty or too large", errInvalidRequest))
			return
		}
		request.Reason = &trimmed
	}
	run, err := ownedRun(r, h.Runs)
	if err != nil {
		handleError(w, err)
		return
	}
	requestedBy := principalUserID(r.Context())
	run, err = h.RunLifecycle.RequestRunCancellation(r.Context(), run.RunID, runstore.WorkflowRunCancellation{
		Code:        runstore.CancellationUserRequested,
		RequestedAt: h.Now().UTC().Round(0),
		RequestedBy: &requestedBy,
		Reason:      request.Reason,
	})
	if err != nil {
		handleError(w, err)
		return
	}
	status := http.StatusOK
	if run.State == runstore.RunCancelling {
		status = http.StatusAccepted
		h.RunNotifier.Cancel(run.RunID)
	}
	writeJSON(w, status, cancelRunResponse{
		RunID: run.RunID, State: run.State, Cancellation: run.Cancellation,
	})
}

func (h *runHandlers) resumeRun(w http.ResponseWriter, r *http.Request) {
	if _, err := exactQuery(r.URL.RawQuery); err != nil {
		handleError(w, err)
		return
	}
	var request struct {
		StageExecutionID string `json:"stageExecutionId"`
	}
	if err := decodeJSON(w, r, &request); err != nil {
		handleError(w, err)
		return
	}
	if request.StageExecutionID == "" {
		handleError(w, fmt.Errorf("%w: stageExecutionId is required", errInvalidRequest))
		return
	}
	run, err := ownedRun(r, h.Runs)
	if err != nil {
		handleError(w, err)
		return
	}
	targetID, err := h.NewID("stage_execution_")
	if err != nil {
		handleError(w, err)
		return
	}
	result, err := h.RunLifecycle.ResumeFailedRun(r.Context(), principalUserID(r.Context()), run.RunID, request.StageExecutionID, targetID)
	if err != nil {
		handleError(w, err)
		return
	}
	h.RunNotifier.Wake()
	writeJSON(w, http.StatusAccepted, result)
}

func (h *runHandlers) deleteRun(w http.ResponseWriter, r *http.Request) {
	if _, err := exactQuery(r.URL.RawQuery); err != nil ||
		r.ContentLength > 0 || len(r.TransferEncoding) != 0 {
		handleError(w, errInvalidRequest)
		return
	}
	if err := h.RunDeletion.DeleteReleasedTerminalRun(
		r.Context(), principalUserID(r.Context()), r.PathValue("runId"),
	); err != nil {
		handleError(w, err)
		return
	}
	w.WriteHeader(http.StatusNoContent)
}

func (h *runHandlers) getRun(w http.ResponseWriter, r *http.Request) {
	if r.Method == http.MethodHead {
		methodNotAllowed(w, r)
		return
	}
	if _, err := exactQuery(r.URL.RawQuery); err != nil {
		handleError(w, err)
		return
	}
	detail, err := h.RunDetails.Read(
		r.Context(), principalUserID(r.Context()), r.PathValue("runId"),
	)
	if err != nil {
		handleError(w, err)
		return
	}
	attempts, activeExecutionID, err := stageAttemptsReadModel(detail)
	if err != nil {
		handleError(w, err)
		return
	}
	run := detail.Run
	writeJSON(w, http.StatusOK, runStatusResponse{
		Recovery:               detail.Recovery,
		ResumeStageExecutionID: detail.ResumeStageExecutionID,
		RunID:                  run.RunID, ProjectID: run.ProjectID,
		Workflow: run.WorkflowName + "@" + run.WorkflowVersion,
		State:    run.State, Deletable: detail.DeletionBlocker == nil,
		RuntimeLabels:        append([]string{}, run.RuntimeLabels...),
		Labels:               run.MetadataLabels.Clone(),
		RuntimeConfiguration: runtimeConfigReadModel(run.RuntimeConfig),
		ProjectHTTPTarget:    run.ProjectHTTPTarget.Clone(),
		Cancellation:         run.Cancellation, Parameters: run.Parameters,
		Inputs: detail.Inputs, Attempts: attempts,
		Transitions: stageTransitionsReadModel(detail.Transitions), Outputs: detail.Outputs,
		OutputPublications: outputPublicationsReadModel(detail.OutputPublications),
		EventCursor: &eventCursorResponse{
			Generation: detail.EventCursor.Generation,
			Sequence:   strconv.FormatInt(detail.EventCursor.Sequence, 10),
		},
		ActiveStageExecutionID: activeExecutionID,
		CreatedAt:              run.CreatedAt, UpdatedAt: run.UpdatedAt,
		StartedAt: run.StartedAt, FinishedAt: run.FinishedAt,
	})
}

// stageAttemptsReadModel projects StageExecutions in order and reports the
// last non-terminal one as the active execution.
func stageAttemptsReadModel(detail runservice.RunDetail) ([]stageAttemptResponse, *string, error) {
	attempts := make([]stageAttemptResponse, 0, len(detail.Executions))
	var activeExecutionID *string
	for _, execution := range detail.Executions {
		stage, err := resolved.DecodeResolvedStageSnapshot(execution.StageSpecSnapshot)
		if err != nil {
			return nil, nil, fmt.Errorf("decode StageExecution read model: %w", err)
		}
		executionConfig, err := stageExecutionConfigReadModel(execution)
		if err != nil {
			return nil, nil, err
		}
		var metrics *telemetry.Summary
		var diagnostics *telemetry.AttemptDiagnostics
		if record, ok := detail.Metrics[execution.StageExecutionID]; ok {
			value := record.Summary
			metrics = &value
			diagnosticValue := telemetry.ProjectAttemptDiagnostics(record.Metrics)
			diagnostics = &diagnosticValue
		}
		var plan *planner.PlannerPlanProjection
		if loaded, ok := detail.Plans[execution.StageExecutionID]; ok {
			plan = &loaded
		}
		runtimeConfiguration := stageRuntimeConfigurationReadModel(detail.Allocations[execution.StageExecutionID])
		resources := detail.Resources[execution.StageExecutionID]
		if resources == nil {
			resources = []telemetry.AllocationResourceSummary{}
		}
		if !terminalStageState(execution.State) {
			value := execution.StageExecutionID
			activeExecutionID = &value
		}
		attempts = append(attempts, stageAttemptResponse{
			StageExecutionID:     execution.StageExecutionID,
			Stage:                execution.StageName,
			Objective:            stage.Objective,
			Attempt:              execution.Attempt,
			PreviousExecutionID:  execution.PreviousExecutionID,
			ExecutionConfig:      executionConfig,
			State:                execution.State,
			Result:               execution.AcceptedResult,
			Termination:          execution.Termination,
			Metrics:              metrics,
			Diagnostics:          diagnostics,
			Plan:                 plan,
			RuntimeConfiguration: runtimeConfiguration,
			Resources:            resources,
			CreatedAt:            execution.CreatedAt,
			UpdatedAt:            execution.UpdatedAt,
			PlannerStartedAt:     execution.PlannerStartedAt,
			TerminalAt:           execution.TerminalAt,
		})
	}
	return attempts, activeExecutionID, nil
}

func stageTransitionsReadModel(decisions []runstore.StageTransitionDecision) []stageTransitionResponse {
	transitions := make([]stageTransitionResponse, 0, len(decisions))
	for _, decision := range decisions {
		transitions = append(transitions, stageTransitionResponse{
			SourceExecutionID: decision.SourceExecutionID,
			Action:            decision.Action, TargetStage: decision.TargetStageName,
			TargetExecutionID:   decision.TargetExecutionID,
			EscalationOrdinal:   decision.EscalationOrdinal,
			EscalationExhausted: decision.EscalationExhausted,
			DecidedAt:           decision.DecidedAt,
		})
	}
	return transitions
}

func outputPublicationsReadModel(records []runstore.RunOutputPublication) []outputPublicationResponse {
	publications := make([]outputPublicationResponse, len(records))
	for index, record := range records {
		publications[index] = outputPublicationResponse{
			Output: record.OutputName, Status: record.Status,
			Source: record.Source, Target: record.Target,
			ErrorCode: record.ErrorCode, ErrorMessage: record.ErrorMessage,
			CreatedAt: record.CreatedAt,
		}
	}
	return publications
}

func stageRuntimeConfigurationReadModel(
	allocations []runstore.StageAllocation,
) *stageRuntimeConfigurationResponse {
	projected := make([]stageRuntimeAllocationResponse, 0, len(allocations))
	for _, allocation := range allocations {
		configuration := allocation.RuntimeConfiguration
		if configuration == nil {
			continue
		}
		agentLabels := make([]pinnedRuntimeConfigResponse, len(configuration.Provenance.AgentLabels))
		for index, pin := range configuration.Provenance.AgentLabels {
			agentLabels[index] = pinnedRuntimeConfigResponse{
				Label:           pin.Label,
				BindingRevision: strconv.FormatUint(pin.BindingRevision, 10),
				Config: runtimeconfig.Ref{
					Name: pin.Config.Name, Version: pin.Config.Version, Digest: pin.Config.Digest,
				},
			}
		}
		status := "pinned"
		if allocation.ReleaseCompletedAt != nil {
			status = "released"
		} else if allocation.ReleaseAttemptedAt != nil {
			status = "release_pending"
		}
		projected = append(projected, stageRuntimeAllocationResponse{
			LogicalAgent: allocation.LogicalAgentName,
			AgentLabels:  agentLabels,
			RuntimeAdapters: append([]runtimesettings.RuntimeAdapterRef{},
				configuration.Provenance.RuntimeAdapters...),
			Origins: configuration.Origins,
			Status:  status,
		})
	}
	if len(projected) == 0 {
		return nil
	}
	sort.Slice(projected, func(left, right int) bool {
		return projected[left].LogicalAgent < projected[right].LogicalAgent
	})
	return &stageRuntimeConfigurationResponse{Allocations: projected}
}

func runtimeConfigReadModel(snapshot runtimeconfig.RunSnapshot) runRuntimeConfigResponse {
	project := func(pin runtimeconfig.PinnedLabel) pinnedRuntimeConfigResponse {
		return pinnedRuntimeConfigResponse{
			Label: pin.Label, BindingRevision: strconv.FormatUint(pin.BindingRevision, 10), Config: pin.Config,
		}
	}
	result := runRuntimeConfigResponse{
		Default: project(snapshot.Default),
		Labels:  make([]pinnedRuntimeConfigResponse, len(snapshot.Labels)),
	}
	for index := range snapshot.Labels {
		result.Labels[index] = project(snapshot.Labels[index])
	}
	return result
}

func terminalStageState(state runstore.StageExecutionState) bool {
	switch state {
	case runstore.StageSucceeded, runstore.StageFailed, runstore.StageInterrupted, runstore.StageCancelled:
		return true
	default:
		return false
	}
}

func stageExecutionConfigReadModel(
	execution runstore.StageExecution,
) (stageExecutionConfigResponse, error) {
	variant := execution.ExecutionConfigVariant
	if variant == "" {
		variant = runstore.StageExecutionConfigBase
	}
	stage, err := resolved.DecodeResolvedStageSnapshot(execution.StageSpecSnapshot)
	if err != nil {
		return stageExecutionConfigResponse{}, fmt.Errorf(
			"decode StageExecution executionConfig read model: %w", err,
		)
	}
	result := stageExecutionConfigResponse{
		Variant: variant, EscalationOrdinal: execution.EscalationOrdinal,
		Agents: make(map[string]consumerExecutionConfigRefsResponse, len(stage.ExecutionConfig.Agents)),
	}
	switch variant {
	case runstore.StageExecutionConfigBase:
	case runstore.StageExecutionConfigFailedEscalation:
		if stage.On.Failed.Escalate == nil {
			return stageExecutionConfigResponse{}, fmt.Errorf("failed escalation Stage spec has no profile")
		}
		if stage.On.Failed.Escalate.ExecutionConfig.Ref != nil {
			ref := *stage.On.Failed.Escalate.ExecutionConfig.Ref
			result.Ref = &ref
		}
	case runstore.StageExecutionConfigInterruptedEscalation:
		if stage.On.Interrupted.Escalate == nil {
			return stageExecutionConfigResponse{}, fmt.Errorf("interrupted escalation Stage spec has no profile")
		}
		if stage.On.Interrupted.Escalate.ExecutionConfig.Ref != nil {
			ref := *stage.On.Interrupted.Escalate.ExecutionConfig.Ref
			result.Ref = &ref
		}
	case runstore.StageExecutionConfigSummaryEscalation:
		if stage.On.Summarized == nil || stage.On.Summarized.Escalate == nil || execution.StageContext.Continuation == nil {
			return stageExecutionConfigResponse{}, fmt.Errorf("summary escalation Stage spec has no saved policy/context")
		}
		if stage.On.Summarized.Escalate.ExecutionConfig.Ref != nil {
			ref := *stage.On.Summarized.Escalate.ExecutionConfig.Ref
			result.Ref = &ref
		}
	default:
		return stageExecutionConfigResponse{}, fmt.Errorf("unknown Stage executionConfig variant %q", variant)
	}
	if stage.ExecutionConfig.Planner != nil {
		planner := consumerExecutionConfigRefs(*stage.ExecutionConfig.Planner)
		result.Planner = &planner
	}
	for logicalName, selection := range stage.ExecutionConfig.Agents {
		result.Agents[logicalName] = consumerExecutionConfigRefs(selection)
	}
	return result, nil
}
