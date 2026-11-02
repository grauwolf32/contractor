package public

import (
	"fmt"
	"net/http"

	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/helpers/pagelimit"
)

const agentTemplateWorkflowBindingsCursorKind = "agent-template-workflow-bindings"

type agentTemplateWorkflowBindingResponse = config.AgentTemplateWorkflowBinding

type agentTemplateWorkflowBindingPageResponse struct {
	Items []agentTemplateWorkflowBindingResponse `json:"items"`
	Page  pageInfoResponse                       `json:"page"`
}

func (h *catalogHandlers) listAgentTemplateWorkflowBindings(w http.ResponseWriter, r *http.Request) error {
	_, limit, encodedCursor, err := pageQuery(r.URL.RawQuery)
	if err != nil {
		return err
	}
	selector := r.PathValue("name") + "@" + r.PathValue("version")
	if _, err := resolved.ParseSelector(selector); err != nil {
		return fmt.Errorf("%w: invalid AgentTemplate selector", errInvalidRequest)
	}
	index, err := h.Config.AgentTemplateWorkflowBindings(selector)
	if err != nil {
		return err
	}
	cursorKind := catalogPageCursorKind(
		agentTemplateWorkflowBindingsCursorKind+":"+selector,
		catalogQuery{}, index.SourceFingerprint,
	)
	cursor, err := h.cursors.decodePageCursor(encodedCursor, cursorKind, 3)
	if err != nil {
		return err
	}

	after := agentTemplateWorkflowBindingResponse{}
	if len(cursor) != 0 {
		name, version, ok := splitWorkflowSelector(cursor[0])
		if !ok {
			return fmt.Errorf("%w: invalid cursor", errInvalidRequest)
		}
		after = agentTemplateWorkflowBindingResponse{
			Workflow: resolved.WorkflowRef{Name: name, Version: version},
			Stage:    cursor[1], LogicalWorker: cursor[2],
		}
	}
	var afterBinding *agentTemplateWorkflowBindingResponse
	if len(cursor) != 0 {
		afterBinding = &after
	}
	pageItems := index.Page(afterBinding, pagelimit.FetchRows(limit))
	pageItems, page, err := paginate(h.cursors, pageItems, limit, cursorKind, func(last config.AgentTemplateWorkflowBinding) []string {
		return []string{
			last.Workflow.Name + "@" + last.Workflow.Version,
			last.Stage,
			last.LogicalWorker,
		}
	})
	if err != nil {
		return err
	}
	writeJSON(w, http.StatusOK, agentTemplateWorkflowBindingPageResponse{
		Items: pageItems, Page: page,
	})
	return nil
}

func splitWorkflowSelector(selector string) (string, string, bool) {
	parsed, err := resolved.ParseSelector(selector)
	if err != nil {
		return "", "", false
	}
	return parsed.ID, parsed.Version, true
}
