package public

import (
	"errors"
	"net/http"
	"strconv"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/controlplane"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
)

func (h *runtimeConfigurationHandlers) listRuntimeAgentPrincipals(w http.ResponseWriter, r *http.Request) error {
	w.Header().Set("Cache-Control", "no-store")
	_, limit, encodedCursor, err := pageQuery(r.URL.RawQuery)
	if err != nil {
		return err
	}
	cursor, err := h.cursors.decodePageCursor(encodedCursor, "runtime-agent-principals", 1)
	if err != nil {
		return err
	}
	after := ""
	if len(cursor) != 0 {
		after = cursor[0]
	}
	principals, hasMore, err := h.RuntimeAgentPrincipals.List(r.Context(), after, limit)
	if err != nil {
		return err
	}
	page := pageInfoResponse{HasMore: hasMore}
	if hasMore {
		if len(principals) == 0 {
			return errors.New("Runtime Agent principal page is empty despite more results")
		}
		next, cursorErr := h.cursors.encodePageCursor("runtime-agent-principals", principals[len(principals)-1].Principal.RuntimeAgentID)
		if cursorErr != nil {
			return cursorErr
		}
		page.NextCursor = &next
	}
	items := make([]runtimeAgentPrincipalResponse, len(principals))
	for index := range principals {
		if err := principals[index].Validate(); err != nil {
			return err
		}
		items[index] = runtimeAgentPrincipalResource(principals[index])
	}
	writeJSON(w, http.StatusOK, runtimeAgentPrincipalPageResponse{Items: items, Page: page})
	return nil
}

func (h *runtimeConfigurationHandlers) getRuntimeAgentPrincipal(w http.ResponseWriter, r *http.Request) error {
	w.Header().Set("Cache-Control", "no-store")
	if _, err := exactQuery(r.URL.RawQuery); err != nil {
		return err
	}
	id := r.PathValue("runtimeAgentId")
	if !contracts.ValidRuntimeAgentID(id) {
		return errInvalidRequest
	}
	principal, err := h.RuntimeAgentPrincipals.Get(r.Context(), id)
	if err != nil {
		return err
	}
	if err := principal.Validate(); err != nil {
		return err
	}
	w.Header().Set("ETag", strconv.Quote(strconv.FormatUint(principal.Principal.LabelRevision, 10)))
	writeJSON(w, http.StatusOK, runtimeAgentPrincipalResource(principal))
	return nil
}

func (h *runtimeConfigurationHandlers) putRuntimeAgentLabels(w http.ResponseWriter, r *http.Request) error {
	w.Header().Set("Cache-Control", "no-store")
	if _, err := exactQuery(r.URL.RawQuery); err != nil {
		return err
	}
	if err := requireJSONMediaType(r); err != nil {
		return err
	}
	id := r.PathValue("runtimeAgentId")
	if !contracts.ValidRuntimeAgentID(id) {
		return errInvalidRequest
	}
	key, err := requireIdempotencyKey(r)
	if err != nil {
		return err
	}
	revision, err := runtimeLabelDeletePrecondition(r)
	if err != nil {
		return err
	}
	var request runtimeAgentLabelsMutationRequest
	if err := decodeJSON(w, r, &request); err != nil {
		return err
	}
	// The request decoder admits only a JSON array.
	labels, err := runtimeconfig.NormalizeLabels(request.Labels)
	if err != nil {
		return errInvalidRequest
	}
	principal, replayed, err := h.RuntimeAgentPrincipals.ReplaceLabels(
		r.Context(), id, revision, labels, key, principalUserID(r.Context()), h.Now(),
	)
	if err != nil {
		return err
	}
	if err := principal.Validate(); err != nil {
		return err
	}
	if !replayed {
		if err := h.OperationsInvalidator.InvalidateOperations(
			controlplane.OperationsRuntimeAgent, id,
		); err != nil {
			return err
		}
	}
	h.auditRuntimeMutation(r, "runtime_agent.labels_replace", id, replayed)
	w.Header().Set("ETag", strconv.Quote(strconv.FormatUint(principal.Principal.LabelRevision, 10)))
	if replayed {
		w.Header().Set("Idempotency-Replayed", "true")
	}
	writeJSON(w, http.StatusOK, runtimeAgentPrincipalResource(principal))
	return nil
}

func (h *runtimeConfigurationHandlers) deleteRuntimeAgentPrincipal(w http.ResponseWriter, r *http.Request) error {
	w.Header().Set("Cache-Control", "no-store")
	if _, err := exactQuery(r.URL.RawQuery); err != nil {
		return err
	}
	id := r.PathValue("runtimeAgentId")
	if !contracts.ValidRuntimeAgentID(id) {
		return errInvalidRequest
	}
	key, err := requireIdempotencyKey(r)
	if err != nil {
		return err
	}
	revision, err := runtimeLabelDeletePrecondition(r)
	if err != nil {
		return err
	}
	replayed, err := h.RuntimeAgentPrincipals.Delete(
		r.Context(), id, revision, key, principalUserID(r.Context()), h.Now(),
	)
	if err != nil {
		return err
	}
	if !replayed {
		if err := h.OperationsInvalidator.InvalidateOperations(
			controlplane.OperationsRuntimeAgent, id,
		); err != nil {
			return err
		}
	}
	h.auditRuntimeMutation(r, "runtime_agent.delete", id, replayed)
	if replayed {
		w.Header().Set("Idempotency-Replayed", "true")
	}
	w.WriteHeader(http.StatusNoContent)
	return nil
}

func runtimeAgentPrincipalResource(
	projection controlplane.RuntimeAgentPrincipalProjection,
) runtimeAgentPrincipalResponse {
	principal := projection.Principal
	return runtimeAgentPrincipalResponse{
		RuntimeAgentID:          principal.RuntimeAgentID,
		Labels:                  append([]string{}, principal.Labels...),
		Revision:                strconv.FormatUint(principal.LabelRevision, 10),
		Availability:            projection.Availability,
		RequiredRuntimeAdapters: append([]string{}, projection.RequiredRuntimeAdapters...),
		MissingRuntimeAdapters:  append([]string{}, projection.MissingRuntimeAdapters...),
		Live:                    projection.Live,
		CreatedBy:               principal.CreatedBy, CreatedAt: principal.CreatedAt,
		UpdatedBy: principal.UpdatedBy, UpdatedAt: principal.UpdatedAt,
	}
}
