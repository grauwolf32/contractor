package public

import (
	"fmt"
	"net/http"

	"github.com/grauwolf32/contractor/internal/gatewayrecovery"
	"github.com/grauwolf32/contractor/internal/runstore"
)

func (h *runHandlers) retryRunGateway(w http.ResponseWriter, r *http.Request) {
	if _, err := exactQuery(r.URL.RawQuery); err != nil {
		handleError(w, err)
		return
	}
	var request struct{}
	if err := decodeJSON(w, r, &request); err != nil {
		handleError(w, err)
		return
	}
	run, err := ownedRun(r, h.Runs)
	if err != nil {
		handleError(w, err)
		return
	}
	if err := h.GatewayRecovery.Retry(r.Context(), run.OwnerID, run.RunID); err != nil {
		if gatewayrecovery.IsUnavailable(err) {
			err = fmt.Errorf("%w: Run has no retryable gateway wait", runstore.ErrConflict)
		}
		handleError(w, err)
		return
	}
	h.RunNotifier.Wake()
	writeJSON(w, http.StatusAccepted, struct {
		RunID string `json:"runId"`
	}{run.RunID})
}
