package privateartifacts

import (
	"context"
	"encoding/json"
	"net/http"
	"strconv"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/controlplane"
	"github.com/grauwolf32/contractor/internal/helpers/strictjson"
)

type PlannerOperations interface {
	ExecuteOperation(context.Context, string, string, contracts.PlannerOperation) (contracts.PlannerOperationResult, error)
}

func (h *handler) plannerOperation(w http.ResponseWriter, r *http.Request) {
	grant, _, err := h.allocationGrant(r)
	if err != nil || grant.Lost || grant.WriteFenced {
		h.writeError(w, http.StatusConflict, "allocation_unavailable", "Allocation is not active", false)
		return
	}
	var raw json.RawMessage
	mediaType, err := requestMediaType(r)
	_, queryErr := exactQuery(r.URL.RawQuery)
	if err != nil || queryErr != nil || mediaType != "application/json" || decodeBoundedJSON(w, r, contracts.MaxPlannerOperationBytes, &raw) != nil {
		h.writeError(w, http.StatusBadRequest, "invalid_request", "Planner operation is invalid", false)
		return
	}
	request, err := contracts.DecodePrivateStrict[contracts.PlannerOperation](raw)
	if err != nil {
		h.writeError(w, http.StatusBadRequest, "invalid_request", "Planner operation is invalid", false)
		return
	}
	if h.dependencies.PlannerOperations == nil {
		h.writeError(w, http.StatusServiceUnavailable, "planner_execution_failed", "Planner execution is unavailable", true)
		return
	}
	// Admission is checked under the existing grant fence. The operation itself
	// performs network/store work outside the Registry lock, under its Run claim.
	err = h.dependencies.Registry.WithWriteGrant(grant.AllocationID, func(current controlplane.AllocationGrant) error {
		if current.Lost || current.WriteFenced || current.StageExecutionID != grant.StageExecutionID || current.RuntimeInstanceID != grant.RuntimeInstanceID {
			return controlplane.ErrAllocationLost
		}
		return nil
	})
	if err != nil {
		h.writeError(w, http.StatusConflict, "allocation_unavailable", "Allocation is not active", false)
		return
	}
	result, err := h.dependencies.PlannerOperations.ExecuteOperation(r.Context(), grant.StageExecutionID, grant.AllocationID, request)
	if err != nil {
		h.writeError(w, http.StatusConflict, "planner_operation_invalid", "Planner operation could not be applied", false)
		return
	}
	// Canonical UTF-8 avoids HTML/JavaScript escaping or a trailing newline
	// changing the operation's shared byte budget at the transport boundary.
	body, err := strictjson.Canonical(result)
	if err != nil {
		h.writeError(w, http.StatusServiceUnavailable, "planner_execution_failed", "Planner reply could not be encoded", true)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Content-Length", strconv.Itoa(len(body)))
	w.Header().Set("Cache-Control", "no-store")
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write(body)
}
