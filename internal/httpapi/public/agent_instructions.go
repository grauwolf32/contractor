package public

import (
	"fmt"
	"net/http"
	"strconv"

	"github.com/grauwolf32/contractor/internal/config/resolved"
)

func (h *catalogHandlers) getAgentInstructions(w http.ResponseWriter, r *http.Request) error {
	if _, err := exactQuery(r.URL.RawQuery); err != nil {
		return err
	}
	selector := r.PathValue("name") + "@" + r.PathValue("version")
	if _, err := resolved.ParseSelector(selector); err != nil {
		return fmt.Errorf("%w: invalid AgentTemplate selector", errInvalidRequest)
	}
	resource, err := h.Config.AgentInstructions(selector)
	if err != nil {
		return err
	}
	w.Header().Set("ETag", strconv.Quote(resource.Template.Digest))
	w.Header().Set("Cache-Control", "private, no-cache")
	writeJSON(w, http.StatusOK, resource)
	return nil
}
