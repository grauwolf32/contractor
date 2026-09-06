package public

import (
	"fmt"
	"net/http"
	"sort"

	"github.com/grauwolf32/contractor/internal/config"
)

const agentTemplateWorkflowBindingsCursorKind = "agent-template-workflow-bindings"

type agentTemplateWorkflowBindingResponse struct {
	Workflow      config.WorkflowRef `json:"workflow"`
	Stage         string             `json:"stage"`
	LogicalWorker string             `json:"logicalWorker"`
}

type agentTemplateWorkflowBindingPageResponse struct {
	Items []agentTemplateWorkflowBindingResponse `json:"items"`
	Page  pageInfoResponse                       `json:"page"`
}

func (h *handler) listAgentTemplateWorkflowBindings(w http.ResponseWriter, r *http.Request) {
	if h.rejectHead(w, r) {
		return
	}
	_, limit, encodedCursor, err := pageQuery(r.URL.RawQuery)
	if err != nil {
		h.handleError(w, err)
		return
	}
	selector := r.PathValue("name") + "@" + r.PathValue("version")
	if _, err := config.ParseSelector(selector); err != nil {
		h.handleError(w, fmt.Errorf("%w: invalid AgentTemplate selector", errInvalidRequest))
		return
	}
	template, err := h.dependencies.Config.Configuration(config.ConfigurationAgentTemplates, selector)
	if err != nil {
		h.handleError(w, err)
		return
	}

	workflows := h.dependencies.Config.Workflows()
	sourceFingerprint, err := catalogSourceFingerprint(struct {
		Template  config.ConfigurationRef
		Workflows []config.ResolvedWorkflow
	}{Template: template.Ref, Workflows: workflows})
	if err != nil {
		h.handleError(w, err)
		return
	}
	cursorKind := catalogPageCursorKind(
		agentTemplateWorkflowBindingsCursorKind+":"+selector,
		catalogQuery{}, sourceFingerprint,
	)
	cursor, err := h.decodePageCursor(encodedCursor, cursorKind, 3)
	if err != nil {
		h.handleError(w, err)
		return
	}

	items := make([]agentTemplateWorkflowBindingResponse, 0)
	for _, workflow := range workflows {
		for stageName, stage := range workflow.Stages {
			for logicalWorker, binding := range stage.Agents {
				if binding.Template.Ref.TemplateID == template.Ref.Name &&
					binding.Template.Ref.Version == template.Ref.Version &&
					binding.Template.Ref.Digest == template.Ref.Digest {
					items = append(items, agentTemplateWorkflowBindingResponse{
						Workflow: workflow.Ref, Stage: stageName, LogicalWorker: logicalWorker,
					})
				}
			}
		}
	}
	sort.Slice(items, func(left, right int) bool {
		return workflowBindingLess(items[left], items[right])
	})

	after := agentTemplateWorkflowBindingResponse{}
	if len(cursor) != 0 {
		name, version, ok := splitWorkflowSelector(cursor[0])
		if !ok {
			h.handleError(w, fmt.Errorf("%w: invalid cursor", errInvalidRequest))
			return
		}
		after = agentTemplateWorkflowBindingResponse{
			Workflow: config.WorkflowRef{Name: name, Version: version},
			Stage:    cursor[1], LogicalWorker: cursor[2],
		}
	}
	pageItems := make([]agentTemplateWorkflowBindingResponse, 0, min(limit+1, len(items)))
	for _, item := range items {
		if len(cursor) != 0 && !workflowBindingLess(after, item) {
			continue
		}
		pageItems = append(pageItems, item)
		if len(pageItems) == limit+1 {
			break
		}
	}
	page := pageInfoResponse{}
	if len(pageItems) > limit {
		pageItems = pageItems[:limit]
		last := pageItems[len(pageItems)-1]
		next, cursorErr := h.encodePageCursor(
			cursorKind,
			last.Workflow.Name+"@"+last.Workflow.Version,
			last.Stage,
			last.LogicalWorker,
		)
		if cursorErr != nil {
			h.handleError(w, cursorErr)
			return
		}
		page.HasMore = true
		page.NextCursor = &next
	}
	writeJSON(w, http.StatusOK, agentTemplateWorkflowBindingPageResponse{
		Items: pageItems, Page: page,
	})
}

func workflowBindingLess(left, right agentTemplateWorkflowBindingResponse) bool {
	if left.Workflow.Name != right.Workflow.Name {
		return left.Workflow.Name < right.Workflow.Name
	}
	if left.Workflow.Version != right.Workflow.Version {
		return left.Workflow.Version < right.Workflow.Version
	}
	if left.Stage != right.Stage {
		return left.Stage < right.Stage
	}
	return left.LogicalWorker < right.LogicalWorker
}

func splitWorkflowSelector(selector string) (string, string, bool) {
	parsed, err := config.ParseSelector(selector)
	if err != nil {
		return "", "", false
	}
	return parsed.ID, parsed.Version, true
}
