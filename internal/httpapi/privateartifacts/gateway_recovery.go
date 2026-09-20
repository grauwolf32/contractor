package privateartifacts

import (
	"encoding/json"
	"errors"
	"io"
	"net/http"

	"github.com/grauwolf32/contractor/internal/controlplane"
	"github.com/grauwolf32/contractor/internal/gatewayrecovery"
)

// Control metadata only: identifiers, action, normalized code and retry timing.
const maximumRecoveryRequestBytes = 4096

func (h *handler) gatewayRecovery(w http.ResponseWriter, r *http.Request) {
	grant, identity, err := h.allocationGrant(r)
	if err != nil || grant.Lost || grant.WriteFenced {
		h.writeError(w, http.StatusConflict, "allocation_unavailable", "Allocation is not active", false)
		return
	}
	if h.dependencies.GatewayRecovery == nil {
		h.writeError(w, http.StatusServiceUnavailable, "recovery_unavailable", "Model recovery is unavailable", true)
		return
	}
	var request gatewayrecovery.Request
	decoder := json.NewDecoder(http.MaxBytesReader(w, r.Body, maximumRecoveryRequestBytes))
	decoder.DisallowUnknownFields()
	if decoder.Decode(&request) != nil || decoder.Decode(&struct{}{}) != io.EOF || request.Validate() != nil {
		h.writeError(w, http.StatusBadRequest, "invalid_recovery_request", "Invalid recovery request", false)
		return
	}
	var result gatewayrecovery.Decision
	err = h.dependencies.Registry.WithWriteGrant(grant.AllocationID, func(current controlplane.AllocationGrant) error {
		if _, err := h.runStoreForGrant(grant.AllocationID, current, identity, true); err != nil {
			return gatewayrecovery.ErrUnavailable
		}
		var updateErr error
		result, updateErr = h.dependencies.GatewayRecovery.Update(r.Context(), current.AllocationID, request)
		return updateErr
	})
	if err != nil {
		status := http.StatusServiceUnavailable
		retryable := true
		if gatewayrecovery.IsUnavailable(err) || errors.Is(err, controlplane.ErrAllocationNotFound) {
			status = http.StatusConflict
			retryable = false
		}
		h.writeError(w, status, "recovery_unavailable", "Model recovery request could not be applied", retryable)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(result)
}
