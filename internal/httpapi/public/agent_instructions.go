package public

import (
	"fmt"
	"net/http"
	"strconv"

	"github.com/grauwolf32/contractor/internal/config"
)

func (h *catalogHandlers) getAgentInstructions(w http.ResponseWriter, r *http.Request) {
	if rejectHead(w, r) {
		return
	}
	if _, err := exactQuery(r.URL.RawQuery); err != nil {
		handleError(w, err)
		return
	}
	selector := r.PathValue("name") + "@" + r.PathValue("version")
	if _, err := config.ParseSelector(selector); err != nil {
		handleError(w, fmt.Errorf("%w: invalid AgentTemplate selector", errInvalidRequest))
		return
	}
	resource, err := h.Config.AgentInstructions(selector)
	if err != nil {
		handleError(w, err)
		return
	}
	w.Header().Set("ETag", strconv.Quote(resource.Template.Digest))
	w.Header().Set("Cache-Control", "private, no-cache")
	writeJSON(w, http.StatusOK, resource)
}
