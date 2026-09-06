package public

import (
	"fmt"
	"net/http"
	"sort"

	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/runstore"
)

const workflowPageCursorKind = "workflows"

func (h *handler) listWorkflows(w http.ResponseWriter, r *http.Request) {
	if h.rejectHead(w, r) {
		return
	}
	values, limit, encodedCursor, err := pageQuery(r.URL.RawQuery, "q", "name")
	if err != nil {
		h.handleError(w, err)
		return
	}
	_, namePresent := values["name"]
	query, err := parseCatalogQuery(values.Get("q"), values.Get("name"), namePresent)
	if err != nil {
		h.handleError(w, err)
		return
	}
	workflows := h.dependencies.Config.Workflows()
	sourceFingerprint, err := catalogSourceFingerprint(workflows)
	if err != nil {
		h.handleError(w, err)
		return
	}
	cursorKind := catalogPageCursorKind(workflowPageCursorKind, query, sourceFingerprint)
	cursor, err := h.decodePageCursor(encodedCursor, cursorKind, 1)
	if err != nil {
		h.handleError(w, err)
		return
	}
	after := ""
	if len(cursor) != 0 {
		after = cursor[0]
	}

	items := make([]workflowSummaryResponse, 0, min(limit, len(workflows)))
	for _, workflow := range workflows {
		authored := []string(nil)
		if workflow.Presentation != nil {
			authored = []string{workflow.Presentation.DisplayName, workflow.Presentation.Description}
		}
		if !catalogMatches(query, workflow.Ref.Name, workflow.Ref.Version, authored...) {
			continue
		}
		selector := workflow.Ref.Name + "@" + workflow.Ref.Version
		if selector <= after {
			continue
		}
		items = append(items, workflowSummaryReadModel(workflow))
		if len(items) == limit+1 {
			break
		}
	}
	page := pageInfoResponse{}
	if len(items) > limit {
		items = items[:limit]
		last := items[len(items)-1].Ref
		next, cursorErr := h.encodePageCursor(cursorKind, last.Name+"@"+last.Version)
		if cursorErr != nil {
			h.handleError(w, cursorErr)
			return
		}
		page.HasMore = true
		page.NextCursor = &next
	}
	writeJSON(w, http.StatusOK, workflowPageResponse{Items: items, Page: page})
}

func (h *handler) getWorkflow(w http.ResponseWriter, r *http.Request) {
	if h.rejectHead(w, r) {
		return
	}
	if _, err := exactQuery(r.URL.RawQuery); err != nil {
		h.handleError(w, err)
		return
	}
	selector := r.PathValue("name") + "@" + r.PathValue("version")
	if _, err := config.ParseSelector(selector); err != nil {
		h.handleError(w, fmt.Errorf("%w: invalid Workflow selector", errInvalidRequest))
		return
	}
	workflow, err := h.dependencies.Config.Workflow(selector)
	if err != nil {
		h.handleError(w, fmt.Errorf("%w: Workflow was not found", runstore.ErrNotFound))
		return
	}
	writeJSON(w, http.StatusOK, workflowResourceReadModel(workflow))
}

func workflowSummaryReadModel(workflow config.ResolvedWorkflow) workflowSummaryResponse {
	return workflowSummaryResponse{
		Ref: workflow.Ref, Presentation: workflow.Presentation, EntryStage: workflow.EntryStage,
		Parameters: workflow.Parameters, Inputs: workflow.Inputs, Outputs: workflow.Outputs,
	}
}

func workflowResourceReadModel(workflow config.ResolvedWorkflow) workflowResourceResponse {
	result := workflowResourceResponse{
		workflowSummaryResponse: workflowSummaryReadModel(workflow),
		Stages:                  make(map[string]workflowStageResponse, len(workflow.Stages)),
	}
	stageNames := make([]string, 0, len(workflow.Stages))
	for name := range workflow.Stages {
		stageNames = append(stageNames, name)
	}
	sort.Strings(stageNames)
	for _, name := range stageNames {
		stage := workflow.Stages[name]
		agents := make(map[string]workflowAgentBindingResponse, len(stage.Agents))
		for logicalName, binding := range stage.Agents {
			agents[logicalName] = workflowAgentBindingResponse{
				Template: binding.Template.Ref, Namespace: binding.Namespace,
				Skills: append(
					[]contracts.ArtifactRef{}, binding.Template.Skills...,
				),
			}
		}
		result.Stages[name] = workflowStageResponse{
			Objective: stage.Objective,
			Instructions: instructionsRefResponse{
				Ref: stage.Instructions.Ref, Digest: stage.Instructions.Digest,
			},
			Planner: stage.Planner, Session: stage.Session, Agents: agents,
			ExecutionConfig:  resolvedStageExecutionConfigReadModel(stage.ExecutionConfig),
			ContextArtifacts: stage.Context.Artifacts,
			ResultArtifacts:  stage.Result.Artifacts,
			WorkflowOutputs:  stage.WorkflowOutputs,
			On: workflowTransitionsResponse{
				Succeeded:   workflowTransitionReadModel(stage.On.Succeeded),
				Failed:      workflowTransitionReadModel(stage.On.Failed),
				Interrupted: workflowTransitionReadModel(stage.On.Interrupted),
			},
		}
	}
	return result
}

func resolvedStageExecutionConfigReadModel(
	selection config.ResolvedStageExecutionConfig,
) resolvedStageExecutionConfigResponse {
	result := resolvedStageExecutionConfigResponse{
		Agents: make(map[string]consumerExecutionConfigRefsResponse, len(selection.Agents)),
	}
	if selection.Planner != nil {
		planner := consumerExecutionConfigRefs(*selection.Planner)
		result.Planner = &planner
	}
	for name, agent := range selection.Agents {
		result.Agents[name] = consumerExecutionConfigRefs(agent)
	}
	return result
}

func workflowTransitionReadModel(action config.TransitionAction) workflowTransitionResponse {
	result := workflowTransitionResponse{Kind: action.Kind, NextStage: action.NextStage}
	if action.Retry != nil {
		result.MaxAttempts = action.Retry.MaxAttempts
		then := workflowTransitionReadModel(action.Retry.Then)
		result.Then = &then
	}
	if action.Escalate != nil {
		result.MaxAttempts = action.Escalate.MaxAttempts
		result.ExecutionConfig = &workflowEscalationConfigResponse{
			Ref: action.Escalate.ExecutionConfig.Ref,
			Effective: resolvedStageExecutionConfigReadModel(
				action.Escalate.ExecutionConfig.Effective,
			),
		}
		then := workflowTransitionReadModel(action.Escalate.Then)
		result.Then = &then
	}
	return result
}
