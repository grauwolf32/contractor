package public

import (
	"fmt"
	"net/http"

	"github.com/grauwolf32/contractor/internal/gatewayrecovery"
	"github.com/grauwolf32/contractor/internal/runstore"
)

func (h *handler) retryRunGateway(w http.ResponseWriter, r *http.Request) {
	if _, err := exactQuery(r.URL.RawQuery); err != nil {
		h.handleError(w, err)
		return
	}
	var request struct{}
	if err := decodeJSON(w, r, &request); err != nil {
		h.handleError(w, err)
		return
	}
	run, err := h.ownedRun(r)
	if err != nil {
		h.handleError(w, err)
		return
	}
	if h.dependencies.GatewayRecovery == nil {
		h.handleError(w, fmt.Errorf("gateway recovery is unavailable"))
		return
	}
	if err := h.dependencies.GatewayRecovery.Retry(r.Context(), run.OwnerID, run.RunID); err != nil {
		if gatewayrecovery.IsUnavailable(err) {
			err = fmt.Errorf("%w: Run has no retryable gateway wait", runstore.ErrConflict)
		}
		h.handleError(w, err)
		return
	}
	if h.dependencies.RunNotifier != nil {
		h.dependencies.RunNotifier.Wake()
	}
	writeJSON(w, http.StatusAccepted, struct {
		RunID string `json:"runId"`
	}{run.RunID})
}
