package public

import (
	"fmt"
	"net/http"

	"github.com/grauwolf32/contractor/internal/gatewayrecovery"
	"github.com/grauwolf32/contractor/internal/runstore"
)

func (h *runHandlers) retryRunGateway(w http.ResponseWriter, r *http.Request) error {
	if _, err := exactQuery(r.URL.RawQuery); err != nil {
		return err
	}
	if err := requireJSONMediaType(r); err != nil {
		return err
	}
	var request struct{}
	if err := decodeJSON(w, r, &request); err != nil {
		return err
	}
	run, err := ownedRun(r, h.Runs)
	if err != nil {
		return err
	}
	if err := h.GatewayRecovery.Retry(r.Context(), run.OwnerID, run.RunID); err != nil {
		if gatewayrecovery.IsUnavailable(err) {
			err = fmt.Errorf("%w: Run has no retryable gateway wait", runstore.ErrConflict)
		}
		return err
	}
	h.RunNotifier.Wake()
	writeJSON(w, http.StatusAccepted, struct {
		RunID string `json:"runId"`
	}{run.RunID})
	return nil
}
