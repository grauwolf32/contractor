package privateartifacts

import (
	"context"
	"errors"
	"net/http"

	"github.com/grauwolf32/contractor/internal/controlplane"
	"github.com/grauwolf32/contractor/internal/gatewayrecovery"
)

// Control metadata only: identifiers, action, normalized code and retry timing.
const maximumRecoveryRequestBytes = 4096

// gatewayRecoveryUpdater is the part of gatewayrecovery.Service this API uses.
type gatewayRecoveryUpdater interface {
	Update(context.Context, string, gatewayrecovery.Request) (gatewayrecovery.Decision, error)
}

func (h *handler) gatewayRecovery(w http.ResponseWriter, r *http.Request) {
	grant, identity, err := h.allocationGrant(r)
	if err != nil || grant.Lost || grant.WriteFenced {
		h.writeError(w, http.StatusConflict, "allocation_unavailable", "Allocation is not active", false)
		return
	}
	if h.recovery == nil {
		h.writeError(w, http.StatusServiceUnavailable, "recovery_unavailable", "Model recovery is unavailable", true)
		return
	}
	var request gatewayrecovery.Request
	if mediaType, err := requestMediaType(r); err != nil || mediaType != "application/json" ||
		decodeBoundedJSON(w, r, maximumRecoveryRequestBytes, &request) != nil || request.Validate() != nil {
		h.writeError(w, http.StatusBadRequest, "invalid_recovery_request", "Invalid recovery request", false)
		return
	}
	var result gatewayrecovery.Decision
	err = h.dependencies.Registry.WithWriteGrant(grant.AllocationID, func(current controlplane.AllocationGrant) error {
		if _, err := h.runStoreForGrant(grant.AllocationID, current, identity, true); err != nil {
			return gatewayrecovery.ErrUnavailable
		}
		var updateErr error
		result, updateErr = h.recovery.Update(r.Context(), current.AllocationID, request)
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
	writeJSON(w, http.StatusOK, result)
}
