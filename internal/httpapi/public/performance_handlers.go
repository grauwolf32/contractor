package public

import (
	"errors"
	"fmt"
	"net/http"
	"time"

	"github.com/grauwolf32/contractor/internal/performance"
	"github.com/grauwolf32/contractor/internal/telemetry"
)

func (h *handler) getPerformance(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Cache-Control", "no-store")
	if h.rejectHead(w, r) || !h.requireOperationsCapability(w, r) {
		return
	}
	if _, err := exactQuery(r.URL.RawQuery); err != nil {
		h.handleError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, h.dependencies.Performance.Snapshot())
}

func (h *handler) listAllocationResourceHistory(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Cache-Control", "no-store")
	if h.rejectHead(w, r) || !h.requireOperationsCapability(w, r) {
		return
	}
	query, limit, encodedCursor, err := pageQuery(r.URL.RawQuery, "runId")
	if err != nil {
		h.handleError(w, err)
		return
	}
	if limit > 100 {
		h.handleError(w, fmt.Errorf("%w: allocation history limit must be at most 100", errInvalidRequest))
		return
	}
	runID := query.Get("runId")
	if _, present := query["runId"]; present && runID == "" {
		h.handleError(w, fmt.Errorf("%w: runId cannot be empty", errInvalidRequest))
		return
	}
	params := telemetry.AllocationResourceHistoryParams{
		OwnerID: principalUserID(r.Context()), RunID: runID, Limit: limit + 1,
	}
	var upperAt time.Time
	var upperAllocationID string
	if encodedCursor != "" {
		values, decodeErr := h.decodePageCursor(encodedCursor, "operations:allocation-history", 5)
		if decodeErr != nil {
			h.handleError(w, decodeErr)
			return
		}
		expectedFilter := "all"
		if runID != "" {
			expectedFilter = "run:" + runID
		}
		if values[0] != expectedFilter {
			h.handleError(w, fmt.Errorf("%w: cursor does not match the Run filter", errInvalidRequest))
			return
		}
		upperAt, err = time.Parse(time.RFC3339Nano, values[1])
		if err != nil {
			h.handleError(w, fmt.Errorf("%w: invalid allocation history cursor", errInvalidRequest))
			return
		}
		afterAt, parseErr := time.Parse(time.RFC3339Nano, values[3])
		if parseErr != nil {
			h.handleError(w, fmt.Errorf("%w: invalid allocation history cursor", errInvalidRequest))
			return
		}
		upperAllocationID = values[2]
		if afterAt.After(upperAt) ||
			(afterAt.Equal(upperAt) && values[4] > upperAllocationID) {
			h.handleError(w, fmt.Errorf("%w: invalid allocation history cursor bounds", errInvalidRequest))
			return
		}
		params.UpperFinishedAt, params.UpperAllocationID = &upperAt, upperAllocationID
		params.AfterFinishedAt, params.AfterAllocationID = &afterAt, values[4]
	}
	items, err := h.dependencies.AllocationResources.ListAllocationResourceHistory(r.Context(), params)
	if err != nil {
		h.handleError(w, err)
		return
	}
	page := pageInfoResponse{}
	if len(items) > limit {
		items = items[:limit]
		if encodedCursor == "" {
			upperAt, upperAllocationID = items[0].FinishedAt, items[0].AllocationID
		}
		filter := "all"
		if runID != "" {
			filter = "run:" + runID
		}
		last := items[len(items)-1]
		next, cursorErr := h.encodePageCursor(
			"operations:allocation-history", filter,
			upperAt.UTC().Format(time.RFC3339Nano), upperAllocationID,
			last.FinishedAt.UTC().Format(time.RFC3339Nano), last.AllocationID,
		)
		if cursorErr != nil {
			h.handleError(w, cursorErr)
			return
		}
		page.HasMore, page.NextCursor = true, &next
	}
	writeJSON(w, http.StatusOK, allocationResourcePageResponse{Items: items, Page: page})
}

func (h *handler) getPerformanceHistory(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Cache-Control", "no-store")
	if h.rejectHead(w, r) || !h.requireOperationsCapability(w, r) {
		return
	}
	query, err := exactQuery(r.URL.RawQuery, "from", "to", "step")
	if err != nil {
		h.handleError(w, err)
		return
	}
	if query.Get("from") == "" || query.Get("to") == "" || query.Get("step") == "" {
		h.handleError(w, fmt.Errorf("%w: from, to and step are required", errInvalidRequest))
		return
	}
	from, fromErr := time.Parse(time.RFC3339Nano, query.Get("from"))
	to, toErr := time.Parse(time.RFC3339Nano, query.Get("to"))
	if fromErr != nil || toErr != nil {
		h.handleError(w, fmt.Errorf("%w: from and to must be RFC3339 timestamps", errInvalidRequest))
		return
	}
	result, err := h.dependencies.Performance.History(r.Context(), from, to, query.Get("step"))
	if errors.Is(err, performance.ErrHistoryRange) || errors.Is(err, performance.ErrHistoryStep) {
		err = fmt.Errorf("%w: performance history range or step is invalid", errInvalidRequest)
	}
	if err != nil {
		h.handleError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, result)
}
