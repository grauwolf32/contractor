package public

import (
	"errors"
	"fmt"
	"net/http"
	"strconv"
	"strings"

	"github.com/grauwolf32/contractor/internal/apierror"
	"github.com/grauwolf32/contractor/internal/telemetry"
)

type runtimeDiagnosticPageResponse struct {
	Status string                        `json:"status"`
	Reason string                        `json:"reason,omitempty"`
	Items  []telemetry.RuntimeDiagnostic `json:"items"`
	Page   pageInfoResponse              `json:"page"`
}

type toolDiagnosticPageResponse struct {
	telemetry.ToolDiagnosticPage
	Page pageInfoResponse `json:"page"`
}

func (h *runHandlers) toolCalls(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Cache-Control", "no-store")
	query, limit, cursor, err := pageQuery(r.URL.RawQuery, "participant", "logicalAgent")
	if err != nil {
		handleError(w, err)
		return
	}
	if _, specified := query["limit"]; !specified {
		limit = telemetry.DefaultToolDiagnosticLimit
	}
	if limit > telemetry.MaxToolDiagnosticLimit {
		handleError(w, fmt.Errorf("%w: tool diagnostic limit exceeds maximum", errInvalidRequest))
		return
	}
	p := telemetry.ToolDiagnosticParams{OwnerID: principalUserID(r.Context()), RunID: r.PathValue("runId"), StageExecutionID: r.PathValue("stageExecutionId"), Participant: query.Get("participant"), LogicalAgent: query.Get("logicalAgent"), Limit: limit}
	if _, present := query["logicalAgent"]; present && p.Participant == "planner" {
		handleError(w, fmt.Errorf("%w: planner has no logicalAgent selector", errInvalidRequest))
		return
	}
	kind := diagnosticCursorKind("tool-diagnostics", p.OwnerID, p.RunID, p.StageExecutionID, p.Participant, p.LogicalAgent)
	position, err := h.cursors.decodePageCursor(cursor, kind, 2)
	if err != nil {
		handleError(w, err)
		return
	}
	if position != nil {
		p.Generation = position[0]
		p.Offset, err = strconv.Atoi(position[1])
		if err != nil || p.Offset < 1 || strconv.Itoa(p.Offset) != position[1] {
			handleError(w, fmt.Errorf("%w: invalid diagnostic offset", errInvalidRequest))
			return
		}
	}
	result, err := h.Diagnostics.ReadToolCallDiagnostics(r.Context(), p)
	if err != nil {
		handleRunDiagnosticError(w, err)
		return
	}
	page := pageInfoResponse{HasMore: result.More}
	if result.More {
		if result.Generation == "" || result.NextOffset <= p.Offset || len(result.Items) == 0 {
			handleError(w, errors.New("nonadvancing tool diagnostic page"))
			return
		}
		next, err := h.cursors.encodePageCursor(kind, result.Generation, strconv.Itoa(result.NextOffset))
		if err != nil {
			handleError(w, err)
			return
		}
		page.NextCursor = &next
	}
	writeJSON(w, http.StatusOK, toolDiagnosticPageResponse{ToolDiagnosticPage: result, Page: page})
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
	kind := diagnosticCursorKind("runtime-diagnostics", params.OwnerID, params.RunID, params.StageExecutionID)
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

// Resource identifiers can contain colons. Quote every scope member before
// hashing the kind so different owner/Run/Stage tuples never share a context.
func diagnosticCursorKind(family string, parts ...string) string {
	var result strings.Builder
	result.WriteString(family)
	for _, part := range parts {
		result.WriteByte(':')
		result.WriteString(strconv.Quote(part))
	}
	return result.String()
}
