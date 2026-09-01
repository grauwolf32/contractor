package public

import (
	"errors"
	"net/http"
	"sort"
	"strconv"

	"github.com/grauwolf32/contractor/internal/controlplane"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
)

func (h *handler) listRuntimeAgentPrincipals(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Cache-Control", "no-store")
	if h.rejectHead(w, r) {
		return
	}
	_, limit, encodedCursor, err := pageQuery(r.URL.RawQuery)
	if err != nil {
		h.handleError(w, err)
		return
	}
	cursor, err := h.decodePageCursor(encodedCursor, "runtime-agent-principals", 1)
	if err != nil {
		h.handleError(w, err)
		return
	}
	after := ""
	if len(cursor) != 0 {
		after = cursor[0]
	}
	principals, err := h.dependencies.RuntimeAgentPrincipals.List(r.Context(), after, limit+1)
	if err != nil {
		h.handleError(w, err)
		return
	}
	page := pageInfoResponse{}
	if len(principals) > limit {
		principals = principals[:limit]
		next, cursorErr := h.encodePageCursor(
			"runtime-agent-principals", principals[len(principals)-1].Principal.RuntimeAgentID,
		)
		if cursorErr != nil {
			h.handleError(w, cursorErr)
			return
		}
		page.HasMore = true
		page.NextCursor = &next
	}
	items := make([]runtimeAgentPrincipalResponse, len(principals))
	for index := range principals {
		if err := principals[index].Validate(); err != nil {
			h.handleError(w, err)
			return
		}
		items[index] = runtimeAgentPrincipalResource(principals[index])
	}
	writeJSON(w, http.StatusOK, runtimeAgentPrincipalPageResponse{Items: items, Page: page})
}

func (h *handler) getRuntimeAgentPrincipal(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Cache-Control", "no-store")
	if h.rejectHead(w, r) {
		return
	}
	if _, err := exactQuery(r.URL.RawQuery); err != nil {
		h.handleError(w, err)
		return
	}
	id := r.PathValue("runtimeAgentId")
	if !runtimeAgentPrincipalIDPattern.MatchString(id) {
		h.handleError(w, errInvalidRequest)
		return
	}
	principal, err := h.dependencies.RuntimeAgentPrincipals.Get(r.Context(), id)
	if errors.Is(err, runtimeconfig.ErrNotFound) {
		h.writeError(w, http.StatusNotFound, "not_found", "resource was not found", false)
		return
	}
	if err != nil {
		h.handleError(w, err)
		return
	}
	if err := principal.Validate(); err != nil {
		h.handleError(w, err)
		return
	}
	w.Header().Set("ETag", strconv.Quote(strconv.FormatUint(principal.Principal.LabelRevision, 10)))
	writeJSON(w, http.StatusOK, runtimeAgentPrincipalResource(principal))
}

func (h *handler) putRuntimeAgentLabels(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Cache-Control", "no-store")
	if _, err := exactQuery(r.URL.RawQuery); err != nil {
		h.handleError(w, err)
		return
	}
	if mediaType, err := requestMediaType(r); err != nil || mediaType != "application/json" {
		h.handleError(w, errInvalidRequest)
		return
	}
	id := r.PathValue("runtimeAgentId")
	if !runtimeAgentPrincipalIDPattern.MatchString(id) {
		h.handleError(w, errInvalidRequest)
		return
	}
	key, err := requireIdempotencyKey(r)
	if err != nil {
		h.handleError(w, err)
		return
	}
	revision, err := runtimeLabelDeletePrecondition(r)
	if err != nil {
		h.handleError(w, err)
		return
	}
	var request runtimeAgentLabelsMutationRequest
	if err := decodeJSON(w, r, &request); err != nil {
		h.handleError(w, err)
		return
	}
	labels, err := normalizeRuntimeAgentLabels(request.Labels)
	if err != nil {
		h.handleError(w, err)
		return
	}
	principal, replayed, err := h.dependencies.RuntimeAgentPrincipals.ReplaceLabels(
		r.Context(), id, revision, labels, key, principalUserID(r.Context()), h.dependencies.Now(),
	)
	if errors.Is(err, runtimeconfig.ErrNotFound) {
		h.writeError(w, http.StatusNotFound, "not_found", "resource was not found", false)
		return
	}
	if err != nil {
		h.handleError(w, err)
		return
	}
	if err := principal.Validate(); err != nil {
		h.handleError(w, err)
		return
	}
	if !replayed {
		if err := h.dependencies.OperationsInvalidator.InvalidateOperations(
			controlplane.OperationsRuntimeAgent, id,
		); err != nil {
			h.handleError(w, err)
			return
		}
	}
	h.auditRuntimeMutation(r, "runtime_agent.labels_replace", id, replayed)
	w.Header().Set("ETag", strconv.Quote(strconv.FormatUint(principal.Principal.LabelRevision, 10)))
	if replayed {
		w.Header().Set("Idempotency-Replayed", "true")
	}
	writeJSON(w, http.StatusOK, runtimeAgentPrincipalResource(principal))
}

func (h *handler) deleteRuntimeAgentPrincipal(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Cache-Control", "no-store")
	if _, err := exactQuery(r.URL.RawQuery); err != nil {
		h.handleError(w, err)
		return
	}
	id := r.PathValue("runtimeAgentId")
	if !runtimeAgentPrincipalIDPattern.MatchString(id) {
		h.handleError(w, errInvalidRequest)
		return
	}
	key, err := requireIdempotencyKey(r)
	if err != nil {
		h.handleError(w, err)
		return
	}
	revision, err := runtimeLabelDeletePrecondition(r)
	if err != nil {
		h.handleError(w, err)
		return
	}
	replayed, err := h.dependencies.RuntimeAgentPrincipals.Delete(
		r.Context(), id, revision, key, principalUserID(r.Context()), h.dependencies.Now(),
	)
	if errors.Is(err, runtimeconfig.ErrNotFound) {
		h.writeError(w, http.StatusNotFound, "not_found", "resource was not found", false)
		return
	}
	if err != nil {
		h.handleError(w, err)
		return
	}
	if !replayed {
		if err := h.dependencies.OperationsInvalidator.InvalidateOperations(
			controlplane.OperationsRuntimeAgent, id,
		); err != nil {
			h.handleError(w, err)
			return
		}
	}
	h.auditRuntimeMutation(r, "runtime_agent.delete", id, replayed)
	if replayed {
		w.Header().Set("Idempotency-Replayed", "true")
	}
	w.WriteHeader(http.StatusNoContent)
}

func normalizeRuntimeAgentLabels(labels []string) ([]string, error) {
	if labels == nil || len(labels) > 32 {
		return nil, errInvalidRequest
	}
	result := append([]string{}, labels...)
	sort.Strings(result)
	for index, label := range result {
		if !runtimeConfigIDPattern.MatchString(label) || label == runtimeconfig.DefaultLabel ||
			index > 0 && result[index-1] == label {
			return nil, errInvalidRequest
		}
	}
	return result, nil
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
