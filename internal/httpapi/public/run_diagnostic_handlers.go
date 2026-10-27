package public

import (
	"errors"
	"fmt"
	"net/http"

	"github.com/grauwolf32/contractor/internal/apierror"
	"github.com/grauwolf32/contractor/internal/telemetry"
)

type runtimeDiagnosticPageResponse struct {
	Status string                        `json:"status"`
	Reason string                        `json:"reason,omitempty"`
	Items  []telemetry.RuntimeDiagnostic `json:"items"`
	Page   pageInfoResponse              `json:"page"`
}

func (h *runHandlers) runtimeDiagnostics(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Cache-Control", "no-store")
	query, limit, cursor, err := pageQuery(r.URL.RawQuery)
	if err != nil {
		handleError(w, err)
		return
	}
	if _, specified := query["limit"]; !specified {
		limit = telemetry.DefaultRuntimeDiagnosticLimit
	}
	if limit > telemetry.MaxRuntimeDiagnosticLimit {
		handleError(w, fmt.Errorf("%w: Runtime diagnostic limit exceeds maximum", errInvalidRequest))
		return
	}
	params := telemetry.RuntimeDiagnosticParams{
		OwnerID: principalUserID(r.Context()), RunID: r.PathValue("runId"),
		StageExecutionID: r.PathValue("stageExecutionId"), Limit: limit,
	}
	kind := "runtime-diagnostics:" + params.OwnerID + ":" + params.RunID + ":" + params.StageExecutionID
	position, err := h.cursors.decodePageCursor(cursor, kind, 2)
	if err != nil {
		handleError(w, err)
		return
	}
	if position != nil {
		params.Generation, params.After = position[0], position[1]
	}
	result, err := h.Diagnostics.ReadRuntimeDiagnostics(r.Context(), params)
	if err != nil {
		handleRunDiagnosticError(w, err)
		return
	}
	page := pageInfoResponse{HasMore: result.More}
	if result.More {
		if result.Generation == "" || result.After == "" || len(result.Items) == 0 {
			handleError(w, errors.New("nonadvancing Runtime diagnostic page"))
			return
		}
		next, err := h.cursors.encodePageCursor(kind, result.Generation, result.After)
		if err != nil {
			handleError(w, err)
			return
		}
		page.NextCursor = &next
	}
	writeJSON(w, http.StatusOK, runtimeDiagnosticPageResponse{
		Status: result.Status, Reason: result.Reason, Items: result.Items, Page: page,
	})
}

func handleRunDiagnosticError(w http.ResponseWriter, err error) {
	switch {
	case errors.Is(err, telemetry.ErrInvalid):
		handleError(w, fmt.Errorf("%w: diagnostic request", errInvalidRequest))
	case errors.Is(err, telemetry.ErrNotFound):
		writeError(w, apierror.NotFound, "")
	case errors.Is(err, telemetry.ErrConflict):
		writeError(w, apierror.Conflict, "Diagnostic data changed; refresh the first page")
	default:
		handleError(w, err)
	}
}
