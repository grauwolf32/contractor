package privateartifacts

import (
	"context"
	"net/http"
	"strconv"

	"github.com/grauwolf32/contractor/internal/controlplane"
	"github.com/grauwolf32/contractor/internal/findingintake"
	"github.com/grauwolf32/contractor/internal/httpapi/artifacttransfer"
)

// withFindingReadGrant joins fence/release through the existing Registry gate.
// The domain owner also proves the durable allocation and Project in its transaction.
func (h *handler) withFindingReadGrant(r *http.Request, operation func(context.Context, controlplane.AllocationGrant) (any, error)) (any, error) {
	_, identity, err := h.allocationGrant(r)
	if err != nil {
		return nil, err
	}
	if h.dependencies.ProjectFindings == nil {
		return nil, errArtifactAccessDenied
	}
	ctx, cancel := artifacttransfer.StorageContext(r.Context())
	defer cancel()
	var result any
	err = h.dependencies.Registry.WithWriteGrant(r.PathValue("allocationID"), func(grant controlplane.AllocationGrant) error {
		if _, err := h.runStoreForGrant(r.PathValue("allocationID"), grant, identity, false); err != nil {
			return err
		}
		var err error
		result, err = operation(ctx, grant)
		return err
	})
	return result, err
}

func (h *handler) listProjectFindings(w http.ResponseWriter, r *http.Request) {
	query, err := exactQuery(r.URL.RawQuery, "limit", "cursor")
	if err != nil {
		h.handleError(w, err)
		return
	}
	limit := 20
	if value, present := query["limit"]; present {
		limit, err = strconv.Atoi(value[0])
		if err != nil || limit < 1 || limit > 100 {
			h.handleError(w, errInvalidRequest)
			return
		}
	}
	result, err := h.withFindingReadGrant(r, func(ctx context.Context, grant controlplane.AllocationGrant) (any, error) {
		return h.dependencies.ProjectFindings.ListAllocationFindings(ctx, grant, limit, query.Get("cursor"))
	})
	if err != nil {
		h.handleError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, result)
}
func (h *handler) readProjectFinding(w http.ResponseWriter, r *http.Request) {
	query, err := exactQuery(r.URL.RawQuery, "revision")
	if err != nil {
		h.handleError(w, err)
		return
	}
	revision, err := strconv.ParseUint(query.Get("revision"), 10, 64)
	if err != nil {
		h.handleError(w, errInvalidRequest)
		return
	}
	result, err := h.withFindingReadGrant(r, func(ctx context.Context, grant controlplane.AllocationGrant) (any, error) {
		return h.dependencies.ProjectFindings.ReadAllocationFinding(ctx, grant, r.PathValue("findingID"), revision)
	})
	if err != nil {
		h.handleError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, result)
}
func (h *handler) selectProjectFindings(w http.ResponseWriter, r *http.Request) {
	if _, err := exactQuery(r.URL.RawQuery); err != nil {
		h.handleError(w, err)
		return
	}
	media, err := requestMediaType(r)
	if err != nil || media != "application/json" {
		h.handleError(w, errInvalidRequest)
		return
	}
	var input findingintake.FindingSelection
	if err := decodeBoundedJSON(w, r, 1<<20, &input); err != nil {
		h.handleError(w, err)
		return
	}
	result, err := h.withFindingReadGrant(r, func(ctx context.Context, grant controlplane.AllocationGrant) (any, error) {
		if grant.WritePolicy != controlplane.WriteInputsAndIntermediates {
			return nil, errArtifactAccessDenied
		}
		return h.dependencies.ProjectFindings.SelectAllocationFindings(ctx, grant, input)
	})
	if err != nil {
		h.handleError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, result)
}
