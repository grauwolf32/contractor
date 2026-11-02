package public

import (
	"errors"
	"fmt"
	"net/http"
	"time"

	"github.com/grauwolf32/contractor/internal/helpers/pagelimit"
	"github.com/grauwolf32/contractor/internal/performance"
	"github.com/grauwolf32/contractor/internal/telemetry"
)

// The Operations routes below are authorized by requireOperationsRoutes.

func (h *operationsHandlers) getPerformance(w http.ResponseWriter, r *http.Request) error {
	w.Header().Set("Cache-Control", "no-store")
	if _, err := exactQuery(r.URL.RawQuery); err != nil {
		return err
	}
	writeJSON(w, http.StatusOK, h.Performance.Snapshot())
	return nil
}

func (h *operationsHandlers) listAllocationResourceHistory(w http.ResponseWriter, r *http.Request) error {
	w.Header().Set("Cache-Control", "no-store")
	query, limit, encodedCursor, err := pageQuery(r.URL.RawQuery, "runId")
	if err != nil {
		return err
	}
	if limit > 100 {
		return fmt.Errorf("%w: allocation history limit must be at most 100", errInvalidRequest)
	}
	runID := query.Get("runId")
	if _, present := query["runId"]; present && runID == "" {
		return fmt.Errorf("%w: runId cannot be empty", errInvalidRequest)
	}
	params := telemetry.AllocationResourceHistoryParams{
		OwnerID: principalUserID(r.Context()), RunID: runID, Limit: pagelimit.FetchRows(limit),
	}
	filter := "all"
	if runID != "" {
		filter = "run:" + runID
	}
	var upperAt time.Time
	var upperAllocationID string
	if encodedCursor != "" {
		values, decodeErr := h.cursors.decodePageCursor(encodedCursor, "operations:allocation-history", 5)
		if decodeErr != nil {
			return decodeErr
		}
		if values[0] != filter {
			return fmt.Errorf("%w: cursor does not match the Run filter", errInvalidRequest)
		}
		upperAt, err = time.Parse(time.RFC3339Nano, values[1])
		if err != nil {
			return fmt.Errorf("%w: invalid allocation history cursor", errInvalidRequest)
		}
		afterAt, parseErr := time.Parse(time.RFC3339Nano, values[3])
		if parseErr != nil {
			return fmt.Errorf("%w: invalid allocation history cursor", errInvalidRequest)
		}
		upperAllocationID = values[2]
		if afterAt.After(upperAt) ||
			(afterAt.Equal(upperAt) && values[4] > upperAllocationID) {
			return fmt.Errorf("%w: invalid allocation history cursor bounds", errInvalidRequest)
		}
		params.UpperFinishedAt, params.UpperAllocationID = &upperAt, upperAllocationID
		params.AfterFinishedAt, params.AfterAllocationID = &afterAt, values[4]
	}
	items, err := h.AllocationResources.ListAllocationResourceHistory(r.Context(), params)
	if err != nil {
		return err
	}
	if encodedCursor == "" && len(items) != 0 {
		// The first page pins the upper boundary that later pages keep.
		upperAt, upperAllocationID = items[0].FinishedAt, items[0].AllocationID
	}
	items, page, err := paginate(h.cursors, items, limit, "operations:allocation-history",
		func(last telemetry.AllocationResourceSummary) []string {
			return []string{
				filter, upperAt.UTC().Format(time.RFC3339Nano), upperAllocationID,
				last.FinishedAt.UTC().Format(time.RFC3339Nano), last.AllocationID,
			}
		})
	if err != nil {
		return err
	}
	writeJSON(w, http.StatusOK, allocationResourcePageResponse{Items: items, Page: page})
	return nil
}

func (h *operationsHandlers) getPerformanceHistory(w http.ResponseWriter, r *http.Request) error {
	w.Header().Set("Cache-Control", "no-store")
	query, err := exactQuery(r.URL.RawQuery, "from", "to", "step")
	if err != nil {
		return err
	}
	if query.Get("from") == "" || query.Get("to") == "" || query.Get("step") == "" {
		return fmt.Errorf("%w: from, to and step are required", errInvalidRequest)
	}
	from, fromErr := time.Parse(time.RFC3339Nano, query.Get("from"))
	to, toErr := time.Parse(time.RFC3339Nano, query.Get("to"))
	if fromErr != nil || toErr != nil {
		return fmt.Errorf("%w: from and to must be RFC3339 timestamps", errInvalidRequest)
	}
	result, err := h.Performance.History(r.Context(), from, to, query.Get("step"))
	if errors.Is(err, performance.ErrHistoryRange) || errors.Is(err, performance.ErrHistoryStep) {
		err = fmt.Errorf("%w: performance history range or step is invalid", errInvalidRequest)
	}
	if err != nil {
		return err
	}
	writeJSON(w, http.StatusOK, result)
	return nil
}
