package public

import (
	"fmt"
	"net/http"
	"sort"
	"strconv"

	"github.com/grauwolf32/contractor/internal/controlplane"
)

func (h *handler) getOperationsSnapshot(w http.ResponseWriter, r *http.Request) {
	if h.rejectHead(w, r) {
		return
	}
	if _, err := exactQuery(r.URL.RawQuery); err != nil {
		h.handleError(w, err)
		return
	}
	snapshot, err := h.operationsSnapshot()
	if err != nil {
		h.handleError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, operationsSnapshotResponse{
		Cursor:        operationsCursor(snapshot.Cursor),
		RuntimeAgents: snapshot.RuntimeAgents,
		Allocations:   snapshot.Allocations,
	})
}

func (h *handler) listRuntimeAgents(w http.ResponseWriter, r *http.Request) {
	if h.rejectHead(w, r) {
		return
	}
	_, limit, encodedCursor, err := pageQuery(r.URL.RawQuery)
	if err != nil {
		h.handleError(w, err)
		return
	}
	snapshot, err := h.operationsSnapshot()
	if err != nil {
		h.handleError(w, err)
		return
	}
	after, err := h.operationsPageAfter(encodedCursor, "operations:runtime-agents", snapshot.Cursor)
	if err != nil {
		h.handleError(w, err)
		return
	}
	items := make([]controlplane.RuntimeAgentObservation, 0, min(limit+1, len(snapshot.RuntimeAgents)))
	for _, item := range snapshot.RuntimeAgents {
		if item.InstanceID <= after {
			continue
		}
		items = append(items, item)
		if len(items) == limit+1 {
			break
		}
	}
	page := pageInfoResponse{}
	if len(items) > limit {
		items = items[:limit]
		next, cursorErr := h.encodeOperationsPageCursor(
			"operations:runtime-agents", snapshot.Cursor, items[len(items)-1].InstanceID,
		)
		if cursorErr != nil {
			h.handleError(w, cursorErr)
			return
		}
		page.HasMore = true
		page.NextCursor = &next
	}
	writeJSON(w, http.StatusOK, runtimeAgentPageResponse{
		Cursor: operationsCursor(snapshot.Cursor), Items: items, Page: page,
	})
}

func (h *handler) listAllocations(w http.ResponseWriter, r *http.Request) {
	if h.rejectHead(w, r) {
		return
	}
	_, limit, encodedCursor, err := pageQuery(r.URL.RawQuery)
	if err != nil {
		h.handleError(w, err)
		return
	}
	snapshot, err := h.operationsSnapshot()
	if err != nil {
		h.handleError(w, err)
		return
	}
	after, err := h.operationsPageAfter(encodedCursor, "operations:allocations", snapshot.Cursor)
	if err != nil {
		h.handleError(w, err)
		return
	}
	items := make([]controlplane.AllocationObservation, 0, min(limit+1, len(snapshot.Allocations)))
	for _, item := range snapshot.Allocations {
		if item.AllocationID <= after {
			continue
		}
		items = append(items, item)
		if len(items) == limit+1 {
			break
		}
	}
	page := pageInfoResponse{}
	if len(items) > limit {
		items = items[:limit]
		next, cursorErr := h.encodeOperationsPageCursor(
			"operations:allocations", snapshot.Cursor, items[len(items)-1].AllocationID,
		)
		if cursorErr != nil {
			h.handleError(w, cursorErr)
			return
		}
		page.HasMore = true
		page.NextCursor = &next
	}
	writeJSON(w, http.StatusOK, allocationPageResponse{
		Cursor: operationsCursor(snapshot.Cursor), Items: items, Page: page,
	})
}

func (h *handler) operationsSnapshot() (controlplane.OperationsSnapshot, error) {
	snapshot := h.dependencies.Operations.SnapshotOperations()
	if err := snapshot.Validate(); err != nil {
		return controlplane.OperationsSnapshot{}, fmt.Errorf("invalid Operations snapshot: %w", err)
	}
	runtimeAgents := make([]controlplane.RuntimeAgentObservation, len(snapshot.RuntimeAgents))
	copy(runtimeAgents, snapshot.RuntimeAgents)
	snapshot.RuntimeAgents = runtimeAgents
	allocations := make([]controlplane.AllocationObservation, len(snapshot.Allocations))
	copy(allocations, snapshot.Allocations)
	snapshot.Allocations = allocations
	sort.Slice(snapshot.RuntimeAgents, func(left, right int) bool {
		return snapshot.RuntimeAgents[left].InstanceID < snapshot.RuntimeAgents[right].InstanceID
	})
	sort.Slice(snapshot.Allocations, func(left, right int) bool {
		return snapshot.Allocations[left].AllocationID < snapshot.Allocations[right].AllocationID
	})
	return snapshot, nil
}

func (h *handler) operationsPageAfter(
	encoded string,
	kind string,
	cursor controlplane.OperationsCursor,
) (string, error) {
	values, err := h.decodePageCursor(encoded, kind, 3)
	if err != nil || len(values) == 0 {
		return "", err
	}
	revision, parseErr := strconv.ParseUint(values[1], 10, 64)
	if parseErr != nil || strconv.FormatUint(revision, 10) != values[1] ||
		values[0] != cursor.Generation || revision != cursor.Revision {
		return "", fmt.Errorf("%w: Operations page cursor requires resynchronization", errInvalidRequest)
	}
	return values[2], nil
}

func (h *handler) encodeOperationsPageCursor(
	kind string,
	cursor controlplane.OperationsCursor,
	after string,
) (string, error) {
	return h.encodePageCursor(
		kind, cursor.Generation, strconv.FormatUint(cursor.Revision, 10), after,
	)
}

func operationsCursor(source controlplane.OperationsCursor) operationsCursorResponse {
	return operationsCursorResponse{
		Generation: source.Generation,
		Revision:   strconv.FormatUint(source.Revision, 10),
	}
}
