package public

import (
	"fmt"
	"net/http"

	"github.com/grauwolf32/contractor/internal/config"
)

const agentTemplateWorkflowBindingsCursorKind = "agent-template-workflow-bindings"

type agentTemplateWorkflowBindingResponse = config.AgentTemplateWorkflowBinding

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
	index, err := h.dependencies.Config.AgentTemplateWorkflowBindings(selector)
	if err != nil {
		h.handleError(w, err)
		return
	}
	cursorKind := catalogPageCursorKind(
		agentTemplateWorkflowBindingsCursorKind+":"+selector,
		catalogQuery{}, index.SourceFingerprint,
	)
	cursor, err := h.decodePageCursor(encodedCursor, cursorKind, 3)
	if err != nil {
		h.handleError(w, err)
		return
	}

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
	var afterBinding *agentTemplateWorkflowBindingResponse
	if len(cursor) != 0 {
		afterBinding = &after
	}
	pageItems := index.Page(afterBinding, limit+1)
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

func splitWorkflowSelector(selector string) (string, string, bool) {
	parsed, err := config.ParseSelector(selector)
	if err != nil {
		return "", "", false
	}
	return parsed.ID, parsed.Version, true
}
