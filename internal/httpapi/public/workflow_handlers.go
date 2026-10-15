package public

import (
	"context"
	"fmt"
	"net/http"
	"sort"

	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/pagelimit"
	"github.com/grauwolf32/contractor/internal/runstore"
)

const workflowPageCursorKind = "workflows"

// CatalogReader is the atomically swappable read view of published
// configuration. Both a bootstrap Snapshot and the managed configuration
// Manager implement it.
type CatalogReader interface {
	WorkflowDiscovery() (config.WorkflowDiscovery, error)
	Workflow(string) (resolved.ResolvedWorkflow, error)
	ConfigurationDiscovery(config.ConfigurationKind) (config.ConfigurationDiscovery, error)
	Configuration(config.ConfigurationKind, string) (config.ConfigurationResource, error)
	AgentInstructions(string) (config.AgentInstructions, error)
	AgentTemplateWorkflowBindings(string) (config.AgentTemplateWorkflowBindings, error)
}

type ConfigurationPublisher interface {
	Publish(context.Context, config.PublicationRequest) (config.PublicationResult, error)
}

// CatalogDependencies serve Workflow and configuration discovery and
// managed configuration publication.
type CatalogDependencies struct {
	Config                 CatalogReader
	ConfigurationPublisher ConfigurationPublisher
	OperationsInvalidator  OperationsInvalidator
}

type catalogHandlers struct {
	CatalogDependencies
	cursors pageCursors
}

func (h *catalogHandlers) listWorkflows(w http.ResponseWriter, r *http.Request) {
	if rejectHead(w, r) {
		return
	}
	values, limit, encodedCursor, err := pageQuery(r.URL.RawQuery, "q", "name")
	if err != nil {
		handleError(w, err)
		return
	}
	_, namePresent := values["name"]
	query, err := parseCatalogQuery(values.Get("q"), values.Get("name"), namePresent)
	if err != nil {
		handleError(w, err)
		return
	}
	// The snapshot fingerprinted its Workflows once; a page copies only the
	// Workflows it returns (CAT-R67).
	discovery, err := h.Config.WorkflowDiscovery()
	if err != nil {
		handleError(w, err)
		return
	}
	cursorKind := catalogPageCursorKind(workflowPageCursorKind, query, discovery.SourceFingerprint)
	cursor, err := h.cursors.decodePageCursor(encodedCursor, cursorKind, 1)
	if err != nil {
		handleError(w, err)
		return
	}
	after := ""
	if len(cursor) != 0 {
		after = cursor[0]
	}

	workflows := discovery.Page(after, pagelimit.FetchRows(limit), func(
		ref resolved.WorkflowRef, presentation resolved.WorkflowPresentation,
	) bool {
		return catalogMatches(query, ref.Name, ref.Version, presentation.DisplayName, presentation.Description)
	})
	items := make([]workflowSummaryResponse, 0, len(workflows))
	for _, workflow := range workflows {
		items = append(items, workflowSummaryReadModel(workflow))
	}
	items, page, err := paginate(h.cursors, items, limit, cursorKind, func(last workflowSummaryResponse) []string {
		return []string{last.Ref.Name + "@" + last.Ref.Version}
	})
	if err != nil {
		handleError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, workflowPageResponse{Items: items, Page: page})
}

func (h *catalogHandlers) getWorkflow(w http.ResponseWriter, r *http.Request) {
	if rejectHead(w, r) {
		return
	}
	if _, err := exactQuery(r.URL.RawQuery); err != nil {
		handleError(w, err)
		return
	}
	selector := r.PathValue("name") + "@" + r.PathValue("version")
	if _, err := resolved.ParseSelector(selector); err != nil {
		handleError(w, fmt.Errorf("%w: invalid Workflow selector", errInvalidRequest))
		return
	}
	workflow, err := h.Config.Workflow(selector)
	if err != nil {
		handleError(w, fmt.Errorf("%w: Workflow was not found", runstore.ErrNotFound))
		return
	}
	writeJSON(w, http.StatusOK, workflowResourceReadModel(workflow))
}

func workflowSummaryReadModel(workflow resolved.ResolvedWorkflow) workflowSummaryResponse {
	return workflowSummaryResponse{
		MaxSubtasks: workflow.MaxSubtasks,
		Ref:         workflow.Ref, Presentation: workflow.Presentation, EntryStage: workflow.EntryStage,
		Parameters: workflow.Parameters, Inputs: workflow.Inputs, Outputs: workflow.Outputs,
	}
}

func workflowResourceReadModel(workflow resolved.ResolvedWorkflow) workflowResourceResponse {
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
		if stage.On.Summarized != nil {
			response := result.Stages[name]
			action := workflowTransitionReadModel(*stage.On.Summarized)
			response.On.Summarized = &action
			result.Stages[name] = response
		}
	}
	return result
}

func resolvedStageExecutionConfigReadModel(
	selection resolved.ResolvedStageExecutionConfig,
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

func workflowTransitionReadModel(action resolved.TransitionAction) workflowTransitionResponse {
	result := workflowTransitionResponse{Kind: action.Kind, NextStage: action.NextStage}
	if action.Retry != nil {
		result.MaxAttempts = action.Retry.MaxAttempts
		then := workflowTransitionReadModel(action.Retry.Then)
		result.Then = &then
	}
	if action.Escalate != nil {
		result.MaxAttempts = action.Escalate.MaxAttempts
		result.Planner = action.Escalate.Planner
		result.DeadlineSeconds = action.Escalate.DeadlineSeconds
		result.ExecutionConfig = &workflowEscalationConfigResponse{
			Ref: action.Escalate.ExecutionConfig.Ref,
			Effective: resolvedStageExecutionConfigReadModel(
				action.Escalate.ExecutionConfig.Effective,
			),
		}
		then := workflowTransitionReadModel(action.Escalate.Then)
		if action.Escalate.Planner == nil {
			result.Then = &then
		}
	}
	return result
}
