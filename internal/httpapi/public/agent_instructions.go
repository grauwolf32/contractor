package public

import (
	"fmt"
	"net/http"
	"strconv"

	"github.com/grauwolf32/contractor/internal/config"
)

func (h *handler) getAgentInstructions(w http.ResponseWriter, r *http.Request) {
	if h.rejectHead(w, r) {
		return
	}
	if _, err := exactQuery(r.URL.RawQuery); err != nil {
		h.handleError(w, err)
		return
	}
	selector := r.PathValue("name") + "@" + r.PathValue("version")
	if _, err := config.ParseSelector(selector); err != nil {
		h.handleError(w, fmt.Errorf("%w: invalid AgentTemplate selector", errInvalidRequest))
		return
	}
	resource, err := h.dependencies.Config.AgentInstructions(selector)
	if err != nil {
		h.handleError(w, err)
		return
	}
	w.Header().Set("ETag", strconv.Quote(resource.Template.Digest))
	w.Header().Set("Cache-Control", "private, no-cache")
	writeJSON(w, http.StatusOK, resource)
}
