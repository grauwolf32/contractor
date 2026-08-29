package public

import (
	"encoding/json"
	"fmt"
	"net/http"
	"sort"
	"strings"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/runstore"
)

const maxCancellationReasonBytes = 4096

func (h *handler) createRun(w http.ResponseWriter, r *http.Request) {
	if _, err := exactQuery(r.URL.RawQuery); err != nil {
		h.handleError(w, err)
		return
	}
	var request createRunRequest
	if err := decodeJSON(w, r, &request); err != nil {
		h.handleError(w, err)
		return
	}
	workflow, err := h.dependencies.Config.Workflow(request.Workflow)
	if err != nil {
		h.handleError(w, fmt.Errorf("%w: unknown Workflow selector", errInvalidRequest))
		return
	}
	if err := validateRunInputs(workflow, request); err != nil {
		h.handleError(w, err)
		return
	}
	workflowSnapshot, err := json.Marshal(workflow)
	if err != nil {
		h.handleError(w, fmt.Errorf("encode resolved Workflow: %w", err))
		return
	}
	runID, err := h.dependencies.NewID("run_")
	if err != nil {
		h.handleError(w, fmt.Errorf("generate Run ID: %w", err))
		return
	}

	err = h.dependencies.Transactions.Do(r.Context(), func(runs RunWriter, artifactService *artifacts.Service) error {
		if _, err := runs.CreateRun(r.Context(), runstore.CreateRunParams{
			RunID: runID, OwnerID: h.dependencies.UserID,
			WorkflowName: workflow.Ref.Name, WorkflowVersion: workflow.Ref.Version,
			WorkflowSchemaVersion: contracts.APIVersion,
			WorkflowSnapshot:      workflowSnapshot,
			Parameters:            cloneParameters(request.Parameters),
		}); err != nil {
			return err
		}

		slots := sortedArtifactSlots(request.Artifacts)
		for _, slot := range slots {
			forked, err := artifactService.ForkInput(
				r.Context(), h.dependencies.UserID, request.Artifacts[slot], runID, slot,
			)
			if err != nil {
				return err
			}
			if !acceptsMediaType(workflow.Inputs[slot].MediaTypes, forked.MediaType) {
				return fmt.Errorf("%w: input %q has unsupported media type", errInvalidRequest, slot)
			}
		}
		_, err := runs.TransitionRun(
			r.Context(), runID, runstore.RunInitializing, runstore.RunRunning,
			runstore.Reason{Code: "initialized"},
		)
		return err
	})
	if err != nil {
		h.handleError(w, err)
		return
	}
	if h.dependencies.RunNotifier != nil {
		h.dependencies.RunNotifier.Wake()
	}
	writeJSON(w, http.StatusAccepted, createRunResponse{RunID: runID, State: runstore.RunRunning})
}

func (h *handler) cancelRun(w http.ResponseWriter, r *http.Request) {
	if _, err := exactQuery(r.URL.RawQuery); err != nil {
		h.handleError(w, err)
		return
	}
	var request cancelRunRequest
	if err := decodeJSON(w, r, &request); err != nil {
		h.handleError(w, err)
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
	requestedBy := h.dependencies.UserID
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
	outputs, err := h.runOutputs(r, run.RunID)
	if err != nil {
		h.handleError(w, err)
		return
	}
	attempts := make([]stageAttemptResponse, 0, len(executions))
	for _, execution := range executions {
		attempts = append(attempts, stageAttemptResponse{
			StageExecutionID: execution.StageExecutionID,
			Stage:            execution.StageName,
			Attempt:          execution.Attempt,
			State:            execution.State,
			Result:           execution.AcceptedResult,
			Termination:      execution.Termination,
		})
	}
	writeJSON(w, http.StatusOK, runStatusResponse{
		RunID: run.RunID, Workflow: run.WorkflowName + "@" + run.WorkflowVersion,
		State: run.State, Cancellation: run.Cancellation, Attempts: attempts, Outputs: outputs,
	})
}

func (h *handler) ownedRun(r *http.Request) (runstore.WorkflowRun, error) {
	run, err := h.dependencies.Runs.GetRun(r.Context(), r.PathValue("runID"))
	if err != nil {
		return runstore.WorkflowRun{}, err
	}
	if run.OwnerID != h.dependencies.UserID {
		return runstore.WorkflowRun{}, runstore.ErrNotFound
	}
	return run, nil
}

func (h *handler) runOutputs(r *http.Request, runID string) (map[string]contracts.ArtifactRef, error) {
	store, err := h.dependencies.Artifacts.Run(runID)
	if err != nil {
		return nil, err
	}
	namespace := "outputs"
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
