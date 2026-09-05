package public

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/grauwolf32/contractor/internal/agentskills"
	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/planner"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
	"github.com/grauwolf32/contractor/internal/telemetry"
)

const maxCancellationReasonBytes = 4096

const idempotencyKeyHeader = "Idempotency-Key"

func (h *handler) listRuns(w http.ResponseWriter, r *http.Request) {
	if h.rejectHead(w, r) {
		return
	}
	h.listRunsFromProject(w, r, nil)
}

func (h *handler) listProjectRuns(w http.ResponseWriter, r *http.Request) {
	if h.rejectHead(w, r) {
		return
	}
	projectID := r.PathValue("projectId")
	if _, err := h.dependencies.Projects.Get(
		r.Context(), principalUserID(r.Context()), projectID,
	); err != nil {
		h.handleError(w, err)
		return
	}
	h.listRunsFromProject(w, r, &projectID)
}

func (h *handler) listRunsFromProject(w http.ResponseWriter, r *http.Request, projectID *string) {
	query, limit, encodedCursor, err := pageQueryWithRepeated(
		r.URL.RawQuery, "label", runstore.MaxRunMetadataLabels, "state", "lifecycle",
	)
	if err != nil {
		h.handleError(w, err)
		return
	}
	var state *runstore.WorkflowRunState
	if values, present := query["state"]; present {
		candidate := runstore.WorkflowRunState(values[0])
		if !publicRunState(candidate) {
			h.handleError(w, fmt.Errorf("%w: unknown Run state", errInvalidRequest))
			return
		}
		state = &candidate
	}
	var lifecycle *runstore.WorkflowRunLifecycle
	if values, present := query["lifecycle"]; present {
		candidate := runstore.WorkflowRunLifecycle(values[0])
		if !candidate.Valid() {
			h.handleError(w, fmt.Errorf("%w: unknown Run lifecycle", errInvalidRequest))
			return
		}
		lifecycle = &candidate
	}
	selectors, err := parseRunMetadataLabelSelectors(query["label"])
	if err != nil {
		h.handleError(w, err)
		return
	}
	cursorKind := runListCursorKindForProject(state, lifecycle, selectors, projectID)
	cursor, err := h.decodePageCursor(encodedCursor, cursorKind, 2)
	if err != nil {
		h.handleError(w, err)
		return
	}
	params := runstore.ListRunsParams{
		OwnerID: principalUserID(r.Context()), State: state, Lifecycle: lifecycle,
		ProjectID: projectID, MetadataLabelSelectors: selectors, Limit: limit + 1,
	}
	if len(cursor) != 0 {
		before, parseErr := time.Parse(time.RFC3339Nano, cursor[0])
		if parseErr != nil {
			h.handleError(w, fmt.Errorf("%w: invalid Run cursor", errInvalidRequest))
			return
		}
		params.BeforeCreatedAt = &before
		params.BeforeRunID = cursor[1]
	}
	runs, err := h.dependencies.Runs.ListRuns(r.Context(), params)
	if err != nil {
		h.handleError(w, err)
		return
	}
	page := pageInfoResponse{}
	if len(runs) > limit {
		runs = runs[:limit]
		last := runs[len(runs)-1]
		next, cursorErr := h.encodePageCursor(
			cursorKind, last.CreatedAt.UTC().Format(time.RFC3339Nano), last.RunID,
		)
		if cursorErr != nil {
			h.handleError(w, cursorErr)
			return
		}
		page.HasMore = true
		page.NextCursor = &next
	}
	items := make([]runSummaryResponse, 0, len(runs))
	for _, run := range runs {
		items = append(items, runSummaryResponse{
			RunID: run.RunID, ProjectID: run.ProjectID,
			Workflow: run.WorkflowName + "@" + run.WorkflowVersion,
			State:    run.State, Deletable: run.Deletable,
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

func runListCursorKind(
	state *runstore.WorkflowRunState,
	lifecycle *runstore.WorkflowRunLifecycle,
	selectors []runstore.RunMetadataLabelSelector,
) string {
	return runListCursorKindForProject(state, lifecycle, selectors, nil)
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

func publicRunState(state runstore.WorkflowRunState) bool {
	switch state {
	case runstore.RunInitializing, runstore.RunRunning, runstore.RunCancelling,
		runstore.RunSucceeded, runstore.RunFailed, runstore.RunCancelled:
		return true
	default:
		return false
	}
}

func (h *handler) createRun(w http.ResponseWriter, r *http.Request) {
	h.createRunFromProject(w, r, nil)
}

func (h *handler) createProjectRun(w http.ResponseWriter, r *http.Request) {
	projectID := r.PathValue("projectId")
	if _, err := h.dependencies.Projects.Get(
		r.Context(), principalUserID(r.Context()), projectID,
	); err != nil {
		h.handleError(w, err)
		return
	}
	h.createRunFromProject(w, r, &projectID)
}

func (h *handler) createRunFromProject(w http.ResponseWriter, r *http.Request, projectID *string) {
	if _, err := exactQuery(r.URL.RawQuery); err != nil {
		h.handleError(w, err)
		return
	}
	var request createRunRequest
	if err := decodeJSON(w, r, &request); err != nil {
		h.handleError(w, err)
		return
	}
	normalizedLabels, err := runtimeconfig.NormalizeRunLabels([]string(request.RuntimeLabels))
	if err != nil {
		h.handleError(w, err)
		return
	}
	request.RuntimeLabels = runRuntimeLabels(normalizedLabels)
	metadataLabels, err := runstore.NormalizeRunMetadataLabels(request.Labels)
	if err != nil {
		h.handleError(w, err)
		return
	}
	request.Labels = runMetadataLabels(metadataLabels)
	idempotencyKey, err := requireIdempotencyKey(r)
	if err != nil {
		h.handleError(w, err)
		return
	}
	requestDigest, err := createRunRequestDigestForProject(request, projectID)
	if err != nil {
		h.handleError(w, fmt.Errorf("digest Run request: %w", err))
		return
	}
	ownerID := principalUserID(r.Context())
	storedRun, replayed, err := h.dependencies.Runs.LookupRunIdempotency(
		r.Context(), ownerID, idempotencyKey, requestDigest,
	)
	if err != nil {
		h.handleError(w, err)
		return
	}
	if replayed {
		w.Header().Set("Idempotency-Replayed", "true")
		writeJSON(w, http.StatusAccepted, createRunReadModel(storedRun))
		return
	}
	runID, err := h.dependencies.NewID("run_")
	if err != nil {
		h.handleError(w, fmt.Errorf("generate Run ID: %w", err))
		return
	}

	created := false
	storedRun = runstore.WorkflowRun{}
	var skillRefs []contracts.ArtifactRef
	var templateSkillSets [][]string
	var projectHTTPTarget *contracts.HTTPOriginTargetRef
	createPinnedRun := func() error {
		return h.dependencies.ManagedCredentials.WithRunCreation(r.Context(), func() error {
			workflow, resolveErr := h.dependencies.Config.ResolveRunWorkflow(
				r.Context(), request.Workflow, request.ExecutionConfig, h.dependencies.Credentials,
			)
			if resolveErr != nil {
				return fmt.Errorf("%w: invalid Workflow or executionConfig selection: %v", errInvalidRequest, resolveErr)
			}
			if err := validateRunInputs(workflow, request); err != nil {
				return err
			}
			skillRefs, resolveErr = config.WorkflowSkillRefs(workflow)
			if resolveErr != nil {
				return fmt.Errorf("%w: invalid Workflow Skill selection: %v", errInvalidRequest, resolveErr)
			}
			templateSkillSets = config.WorkflowSkillSets(workflow)
			if len(skillRefs) > 0 && h.dependencies.RunSkills == nil {
				return fmt.Errorf("Run Skill initializer is not configured")
			}
			workflowSnapshot, err := json.Marshal(workflow)
			if err != nil {
				return fmt.Errorf("encode resolved Workflow: %w", err)
			}
			return h.dependencies.Transactions.Do(r.Context(), func(runs RunWriter, artifactService *artifacts.Service) error {
				skillCatalog, catalogErr := agentskills.NewCatalog(artifactService)
				if catalogErr != nil {
					return catalogErr
				}
				runtimeConfig, pinErr := runs.PinRuntimeLabels(
					r.Context(), []string(request.RuntimeLabels), h.dependencies.Credentials,
				)
				if pinErr != nil {
					return pinErr
				}
				var selectedSkills []contracts.RunSkillSnapshot
				if len(skillRefs) > 0 {
					selectedSkills, catalogErr = skillCatalog.SelectRunSources(r.Context(), ownerID, skillRefs)
					if catalogErr != nil {
						return catalogErr
					}
					if catalogErr = agentskills.ValidateSelectedLimits(selectedSkills, templateSkillSets); catalogErr != nil {
						return catalogErr
					}
				}
				var createErr error
				storedRun, created, createErr = runs.CreateRunIdempotent(
					r.Context(), runstore.CreateRunIdempotentParams{
						CreateRunParams: runstore.CreateRunParams{
							RunID: runID, OwnerID: ownerID,
							ProjectID:    projectID,
							WorkflowName: workflow.Ref.Name, WorkflowVersion: workflow.Ref.Version,
							WorkflowSchemaVersion: contracts.APIVersion,
							WorkflowSnapshot:      workflowSnapshot,
							Parameters:            cloneParameters(request.Parameters),
							MetadataLabels:        metadataLabels,
							RuntimeConfig:         runtimeConfig,
							ProjectHTTPTarget:     projectHTTPTarget,
						},
						IdempotencyKey: idempotencyKey,
						RequestDigest:  requestDigest,
					},
				)
				if createErr != nil {
					return createErr
				}
				if !created {
					return nil
				}
				if len(selectedSkills) > 0 {
					if err := runs.SetRunSkillSelections(r.Context(), runID, selectedSkills); err != nil {
						return err
					}
					if err := skillCatalog.PinRunSources(r.Context(), ownerID, runID, selectedSkills); err != nil {
						return err
					}
				}

				slots := sortedArtifactSlots(request.Artifacts)
				for _, slot := range slots {
					var forked artifacts.ForkResult
					var err error
					if projectID == nil {
						forked, err = artifactService.ForkInput(
							r.Context(), ownerID, request.Artifacts[slot], runID, slot,
						)
					} else {
						forked, err = artifactService.ForkProjectInput(
							r.Context(), *projectID, request.Artifacts[slot], runID, slot,
						)
					}
					if err != nil {
						return err
					}
					if !acceptsMediaType(workflow.Inputs[slot].MediaTypes, forked.MediaType) {
						return fmt.Errorf("%w: input %q has unsupported media type", errInvalidRequest, slot)
					}
				}
				if len(selectedSkills) == 0 {
					storedRun, err = runs.TransitionRun(
						r.Context(), runID, runstore.RunInitializing, runstore.RunRunning,
						runstore.Reason{Code: "initialized"},
					)
					return err
				}
				storedRun.SkillSnapshot = append([]contracts.RunSkillSnapshot(nil), selectedSkills...)
				storedRun.StateReason = runstore.Reason{Code: runstore.SkillInitializationPendingReason}
				return nil
			})
		})
	}
	if projectID == nil {
		err = createPinnedRun()
	} else if h.dependencies.RuntimeCredentials == nil {
		err = errors.New("Runtime credential service is not configured")
	} else {
		err = h.dependencies.RuntimeCredentials.WithCredentialReferences(r.Context(), func() error {
			project, projectErr := h.dependencies.Projects.Get(r.Context(), ownerID, *projectID)
			if projectErr != nil {
				return projectErr
			}
			if project.Lifecycle == projectstore.LifecycleDeleting {
				return projectstore.ErrDeleting
			}
			projectHTTPTarget = cloneHTTPOriginTarget(project.HTTPTarget)
			if projectHTTPTarget != nil && projectHTTPTarget.Credential != nil {
				if validateErr := h.dependencies.RuntimeCredentials.ValidateRuntimeCredential(
					r.Context(), projectHTTPTarget.Credential.CredentialID,
					string(projectHTTPTarget.Credential.Kind),
				); validateErr != nil {
					return validateErr
				}
			}
			return createPinnedRun()
		})
	}
	if err != nil {
		if !errors.Is(err, runstore.ErrConflict) && persistencepostgres.SQLState(err) != "40001" {
			h.handleError(w, err)
			return
		}
		// A concurrent idempotency winner may commit after the REPEATABLE READ
		// snapshot began. Re-read after rollback before returning a conflict.
		var recovered bool
		storedRun, recovered, _ = h.dependencies.Runs.LookupRunIdempotency(
			r.Context(), ownerID, idempotencyKey, requestDigest,
		)
		if !recovered {
			h.handleError(w, err)
			return
		}
		created = false
	}
	// Skill package validation and RunScope forking are intentionally deferred
	// to Scheduler recovery. The creation transaction above has already pinned
	// one complete exact owner-source outcome, so returning the initializing Run
	// neither holds the HTTP request on bounded archive work nor permits a later
	// owner binding update to change what recovery will read.
	if created && h.dependencies.RunNotifier != nil {
		h.dependencies.RunNotifier.Wake()
	}
	if !created {
		w.Header().Set("Idempotency-Replayed", "true")
	}
	writeJSON(w, http.StatusAccepted, createRunReadModel(storedRun))
}

func createRunReadModel(run runstore.WorkflowRun) createRunResponse {
	return createRunResponse{
		RunID: run.RunID, ProjectID: run.ProjectID, State: run.State,
		RuntimeLabels:        append([]string{}, run.RuntimeLabels...),
		Labels:               run.MetadataLabels.Clone(),
		RuntimeConfiguration: runtimeConfigReadModel(run.RuntimeConfig),
		ProjectHTTPTarget:    cloneHTTPOriginTarget(run.ProjectHTTPTarget),
	}
}

func requireIdempotencyKey(r *http.Request) (string, error) {
	values := r.Header.Values(idempotencyKeyHeader)
	if len(values) != 1 || len(values[0]) == 0 || len(values[0]) > 128 {
		return "", fmt.Errorf("%w: exactly one bounded Idempotency-Key is required", errInvalidRequest)
	}
	for index, character := range values[0] {
		valid := character >= 'a' && character <= 'z' || character >= 'A' && character <= 'Z' ||
			character >= '0' && character <= '9' || index > 0 && strings.ContainsRune("._:-", character)
		if !valid {
			return "", fmt.Errorf("%w: Idempotency-Key contains an invalid character", errInvalidRequest)
		}
	}
	return values[0], nil
}

func createRunRequestDigest(request createRunRequest) (string, error) {
	labels, err := runtimeconfig.NormalizeRunLabels([]string(request.RuntimeLabels))
	if err != nil {
		return "", err
	}
	parameters := request.Parameters
	if parameters == nil {
		parameters = map[string]string{}
	}
	artifactRefs := request.Artifacts
	if artifactRefs == nil {
		artifactRefs = map[string]contracts.ArtifactRef{}
	}
	canonical := map[string]any{
		"workflow": request.Workflow, "parameters": parameters, "artifacts": artifactRefs,
		"executionConfig": request.ExecutionConfig.CanonicalValue(),
	}
	// Preserve the pre-Runtime-label digest for the empty set so response-loss replay of
	// Runs created before migration 000018 remains exact after upgrade.
	if len(labels) != 0 {
		canonical["runtimeLabels"] = labels
	}
	metadataLabels, err := runstore.NormalizeRunMetadataLabels(request.Labels)
	if err != nil {
		return "", err
	}
	if len(metadataLabels) != 0 {
		canonical["labels"] = metadataLabels
	}
	encoded, err := json.Marshal(canonical)
	if err != nil {
		return "", err
	}
	digest := sha256.Sum256(encoded)
	return "sha256:" + hex.EncodeToString(digest[:]), nil
}

func createRunRequestDigestForProject(request createRunRequest, projectID *string) (string, error) {
	requestDigest, err := createRunRequestDigest(request)
	if err != nil || projectID == nil {
		return requestDigest, err
	}
	encoded, err := json.Marshal(map[string]string{
		"sourceScope":      "project",
		"projectId":        *projectID,
		"runRequestDigest": requestDigest,
	})
	if err != nil {
		return "", err
	}
	digest := sha256.Sum256(encoded)
	return "sha256:" + hex.EncodeToString(digest[:]), nil
}

func (h *handler) cancelRun(w http.ResponseWriter, r *http.Request) {
	if _, err := exactQuery(r.URL.RawQuery); err != nil {
		h.handleError(w, err)
		return
	}
	var request *cancelRunRequest
	if err := decodeJSON(w, r, &request); err != nil {
		h.handleError(w, err)
		return
	}
	if request == nil {
		h.handleError(w, fmt.Errorf("%w: cancellation request must be an object", errInvalidRequest))
		return
	}
	if request.Reason != nil {
		trimmed := strings.TrimSpace(*request.Reason)
		if trimmed == "" || len([]byte(trimmed)) > maxCancellationReasonBytes {
			h.handleError(w, fmt.Errorf("%w: cancellation reason is empty or too large", errInvalidRequest))
			return
		}
		request.Reason = &trimmed
	}
	run, err := h.ownedRun(r)
	if err != nil {
		h.handleError(w, err)
		return
	}
	requestedBy := principalUserID(r.Context())
	run, err = h.dependencies.Runs.RequestRunCancellation(r.Context(), run.RunID, runstore.WorkflowRunCancellation{
		Code:        runstore.CancellationUserRequested,
		RequestedAt: h.dependencies.Now().UTC().Round(0),
		RequestedBy: &requestedBy,
		Reason:      request.Reason,
	})
	if err != nil {
		h.handleError(w, err)
		return
	}
	status := http.StatusOK
	if run.State == runstore.RunCancelling {
		status = http.StatusAccepted
		if notifier, ok := h.dependencies.RunNotifier.(RunCancellationNotifier); ok {
			notifier.Cancel(run.RunID)
		} else if h.dependencies.RunNotifier != nil {
			h.dependencies.RunNotifier.Wake()
		}
	}
	writeJSON(w, status, cancelRunResponse{
		RunID: run.RunID, State: run.State, Cancellation: run.Cancellation,
	})
}

func (h *handler) deleteRun(w http.ResponseWriter, r *http.Request) {
	if _, err := exactQuery(r.URL.RawQuery); err != nil ||
		r.ContentLength > 0 || len(r.TransferEncoding) != 0 {
		h.handleError(w, errInvalidRequest)
		return
	}
	if err := h.dependencies.Runs.DeleteReleasedTerminalRun(
		r.Context(), principalUserID(r.Context()), r.PathValue("runID"),
	); err != nil {
		h.handleError(w, err)
		return
	}
	w.WriteHeader(http.StatusNoContent)
}

func validateRunInputs(workflow config.ResolvedWorkflow, request createRunRequest) error {
	for name := range request.Parameters {
		if _, ok := workflow.Parameters[name]; !ok {
			return fmt.Errorf("%w: unknown parameter %q", errInvalidRequest, name)
		}
	}
	for name, slot := range workflow.Parameters {
		if _, ok := request.Parameters[name]; slot.Required && !ok {
			return fmt.Errorf("%w: required parameter %q is missing", errInvalidRequest, name)
		}
	}
	for name, ref := range request.Artifacts {
		if _, ok := workflow.Inputs[name]; !ok {
			return fmt.Errorf("%w: unknown input artifact %q", errInvalidRequest, name)
		}
		if err := ref.Validate(); err != nil {
			return fmt.Errorf("%w: invalid input artifact %q", errInvalidRequest, name)
		}
	}
	for name, slot := range workflow.Inputs {
		if _, ok := request.Artifacts[name]; slot.Required && !ok {
			return fmt.Errorf("%w: required input artifact %q is missing", errInvalidRequest, name)
		}
	}
	return nil
}

func cloneParameters(source map[string]string) map[string]string {
	result := make(map[string]string, len(source))
	for key, value := range source {
		result[key] = value
	}
	return result
}

func sortedArtifactSlots(source map[string]contracts.ArtifactRef) []string {
	result := make([]string, 0, len(source))
	for slot := range source {
		result = append(result, slot)
	}
	sort.Strings(result)
	return result
}

func acceptsMediaType(accepted []string, actual string) bool {
	for _, mediaType := range accepted {
		if mediaType == "*/*" || mediaType == actual {
			return true
		}
	}
	return false
}

func (h *handler) getRun(w http.ResponseWriter, r *http.Request) {
	if r.Method == http.MethodHead {
		h.methodNotAllowed(w, r)
		return
	}
	if _, err := exactQuery(r.URL.RawQuery); err != nil {
		h.handleError(w, err)
		return
	}
	run, err := h.ownedRun(r)
	if err != nil {
		h.handleError(w, err)
		return
	}
	executions, err := h.dependencies.Runs.ListStageExecutions(r.Context(), run.RunID)
	if err != nil {
		h.handleError(w, err)
		return
	}
	decisions, err := h.dependencies.Runs.ListStageTransitionDecisions(r.Context(), run.RunID)
	if err != nil {
		h.handleError(w, err)
		return
	}
	outputs, err := h.runOutputs(r, run.RunID)
	if err != nil {
		h.handleError(w, err)
		return
	}
	publicationRecords, err := h.dependencies.Runs.ListRunOutputPublications(r.Context(), run.RunID)
	if err != nil {
		h.handleError(w, err)
		return
	}
	outputPublications := make([]outputPublicationResponse, len(publicationRecords))
	for index, record := range publicationRecords {
		outputPublications[index] = outputPublicationResponse{
			Output: record.OutputName, Status: record.Status,
			Source: record.Source, Target: record.Target,
			ErrorCode: record.ErrorCode, ErrorMessage: record.ErrorMessage,
			CreatedAt: record.CreatedAt,
		}
	}
	inputs, err := h.runArtifactsByNamespace(r, run.RunID, "inputs")
	if err != nil {
		h.handleError(w, err)
		return
	}
	eventCursor, err := h.dependencies.Runs.GetRunEventCursor(r.Context(), run.RunID)
	if err != nil {
		h.handleError(w, err)
		return
	}
	attempts := make([]stageAttemptResponse, 0, len(executions))
	var activeExecutionID *string
	deletable := runstore.RunLifecycleTerminal.Includes(run.State)
	for _, execution := range executions {
		stage, err := config.DecodeResolvedStageSnapshot(execution.StageSpecSnapshot)
		if err != nil {
			h.handleError(w, fmt.Errorf("decode StageExecution read model: %w", err))
			return
		}
		executionConfig, configErr := stageExecutionConfigReadModel(execution)
		if configErr != nil {
			h.handleError(w, configErr)
			return
		}
		var metrics *telemetry.Summary
		var diagnostics *telemetry.AttemptDiagnostics
		if h.dependencies.Metrics != nil {
			if record, metricsErr := h.dependencies.Metrics.GetStageMetrics(
				r.Context(), execution.StageExecutionID,
			); metricsErr == nil {
				value := record.Summary
				metrics = &value
				diagnosticValue := telemetry.ProjectAttemptDiagnostics(record.Metrics)
				diagnostics = &diagnosticValue
			}
		}
		var plan *planner.PlannerPlanProjection
		if h.dependencies.PlannerPlans != nil && execution.PlannerSessionID != nil &&
			execution.PlannerInvocationID != nil {
			loaded, present, planErr := h.dependencies.PlannerPlans.LoadPlan(
				r.Context(), planner.SessionIdentity{
					SessionID:        *execution.PlannerSessionID,
					StageExecutionID: execution.StageExecutionID,
					InvocationID:     *execution.PlannerInvocationID,
				},
			)
			if planErr != nil {
				h.handleError(w, planErr)
				return
			}
			if present {
				value := loaded
				plan = &value
			}
		}
		allocations, allocationErr := h.dependencies.Runs.ListStageAllocations(
			r.Context(), execution.StageExecutionID,
		)
		if allocationErr != nil {
			h.handleError(w, allocationErr)
			return
		}
		for _, allocation := range allocations {
			if allocation.ReleaseCompletedAt == nil {
				deletable = false
			}
		}
		runtimeConfiguration := stageRuntimeConfigurationReadModel(allocations)
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
			CreatedAt:            execution.CreatedAt,
			UpdatedAt:            execution.UpdatedAt,
			PlannerStartedAt:     execution.PlannerStartedAt,
			TerminalAt:           execution.TerminalAt,
		})
	}
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
	writeJSON(w, http.StatusOK, runStatusResponse{
		RunID: run.RunID, ProjectID: run.ProjectID,
		Workflow: run.WorkflowName + "@" + run.WorkflowVersion,
		State:    run.State, Deletable: deletable,
		RuntimeLabels:        append([]string{}, run.RuntimeLabels...),
		Labels:               run.MetadataLabels.Clone(),
		RuntimeConfiguration: runtimeConfigReadModel(run.RuntimeConfig),
		ProjectHTTPTarget:    cloneHTTPOriginTarget(run.ProjectHTTPTarget),
		Cancellation:         run.Cancellation, Parameters: run.Parameters,
		Inputs: inputs, Attempts: attempts, Transitions: transitions, Outputs: outputs,
		OutputPublications: outputPublications,
		EventCursor: &eventCursorResponse{
			Generation: eventCursor.Generation, Sequence: strconv.FormatInt(eventCursor.Sequence, 10),
		},
		ActiveStageExecutionID: activeExecutionID,
		CreatedAt:              run.CreatedAt, UpdatedAt: run.UpdatedAt,
		StartedAt: run.StartedAt, FinishedAt: run.FinishedAt,
	})
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
			RuntimeAdapters: append([]contracts.RuntimeAdapterRef{},
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
	stage, err := config.DecodeResolvedStageSnapshot(execution.StageSpecSnapshot)
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

func consumerExecutionConfigRefs(
	selection config.ResolvedConsumerExecutionConfig,
) consumerExecutionConfigRefsResponse {
	result := consumerExecutionConfigRefsResponse{
		ModelPolicy: selection.ModelPolicy.Ref,
		Origins:     selection.Origins,
	}
	if selection.LLMGateway != nil {
		gateway := selection.LLMGateway.Ref
		result.LLMGateway = &gateway
	}
	if selection.Credential != nil {
		credential := *selection.Credential
		result.Credential = &credential
	}
	return result
}

func (h *handler) ownedRun(r *http.Request) (runstore.WorkflowRun, error) {
	run, err := h.dependencies.Runs.GetRun(r.Context(), r.PathValue("runID"))
	if err != nil {
		return runstore.WorkflowRun{}, err
	}
	if run.OwnerID != principalUserID(r.Context()) {
		return runstore.WorkflowRun{}, runstore.ErrNotFound
	}
	return run, nil
}

func (h *handler) runOutputs(r *http.Request, runID string) (map[string]contracts.ArtifactRef, error) {
	return h.runArtifactsByNamespace(r, runID, "outputs")
}

func (h *handler) runArtifactsByNamespace(
	r *http.Request, runID string, namespace string,
) (map[string]contracts.ArtifactRef, error) {
	store, err := h.dependencies.Artifacts.Run(runID)
	if err != nil {
		return nil, err
	}
	refs, err := store.List(r.Context(), &namespace)
	if err != nil {
		return nil, err
	}
	result := make(map[string]contracts.ArtifactRef, len(refs))
	for _, ref := range refs {
		read, err := store.Read(r.Context(), ref)
		if err != nil {
			return nil, err
		}
		result[ref.Name] = read.Ref
	}
	return result, nil
}

func (h *handler) getRunOutput(w http.ResponseWriter, r *http.Request) {
	if r.Method == http.MethodHead {
		h.methodNotAllowed(w, r)
		return
	}
	if _, err := exactQuery(r.URL.RawQuery); err != nil {
		h.handleError(w, err)
		return
	}
	run, err := h.ownedRun(r)
	if err != nil {
		h.handleError(w, err)
		return
	}
	store, err := h.dependencies.Artifacts.Run(run.RunID)
	if err != nil {
		h.handleError(w, err)
		return
	}
	result, err := store.Read(r.Context(), contracts.ArtifactRef{
		Namespace: "outputs", Name: r.PathValue("slot"),
	})
	if err != nil {
		h.handleError(w, err)
		return
	}
	w.Header().Set("Content-Type", result.Payload.MediaType)
	w.Header().Set("ETag", quotedETag(result.Ref.Revision))
	w.Header().Set("Content-Length", fmt.Sprintf("%d", len(result.Payload.Data)))
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write(result.Payload.Data)
}
