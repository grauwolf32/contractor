package public

import (
	"context"
	"fmt"
	"log/slog"
	"net/http"
	"strconv"
	"time"

	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/controlplane"
	"github.com/grauwolf32/contractor/internal/helpers/pagelimit"
	"github.com/grauwolf32/contractor/internal/performance"
	"github.com/grauwolf32/contractor/internal/settingsstore"
	"github.com/grauwolf32/contractor/internal/telemetry"
)

// OperationsReader returns a caller-owned, detached snapshot, sorted by instance
// and allocation ID. Capability lists (including tool names) are sorted too.
type OperationsReader interface {
	SnapshotOperations() controlplane.OperationsSnapshot
}

type PerformanceReader interface {
	Snapshot() performance.SnapshotResponse
	History(context.Context, time.Time, time.Time, string) (performance.HistoryResponse, error)
}

type AllocationResourceReader interface {
	ListAllocationResourceHistory(
		context.Context, telemetry.AllocationResourceHistoryParams,
	) ([]telemetry.AllocationResourceSummary, error)
}

type SchedulerSettingsManagement interface {
	GetSchedulerSettings(context.Context) (settingsstore.SchedulerSettings, error)
	UpdateSchedulerSettings(
		context.Context, settingsstore.UpdateSchedulerSettingsParams,
	) (settingsstore.SchedulerSettings, error)
}

type ModelSelectionManagement interface {
	GetModelSelection(context.Context) (config.ModelSelectionInfo, error)
	ReplaceModelSelection(context.Context, settingsstore.ModelSelectionValue, uint64, string) (config.ModelSelectionInfo, error)
}

// OperationsDependencies serve the Operations snapshot, Runtime Agent and
// allocation pages, performance and allocation history, and the Scheduler
// settings.
type OperationsDependencies struct {
	ModelSelection        ModelSelectionManagement
	Operations            OperationsReader
	Performance           PerformanceReader
	AllocationResources   AllocationResourceReader
	SchedulerSettings     SchedulerSettingsManagement
	OperationsInvalidator OperationsInvalidator
	// RunNotifier wakes the Scheduler after its settings changed.
	RunNotifier Waker
	Logger      *slog.Logger
}

type operationsHandlers struct {
	OperationsDependencies
	cursors pageCursors
}

func (h *operationsHandlers) getOperationsSnapshot(w http.ResponseWriter, r *http.Request) error {
	if _, err := exactQuery(r.URL.RawQuery); err != nil {
		return err
	}
	snapshot, err := h.operationsSnapshot()
	if err != nil {
		return err
	}
	writeJSON(w, http.StatusOK, operationsSnapshotResponse{
		Cursor:        operationsCursor(snapshot.Cursor),
		RuntimeAgents: snapshot.RuntimeAgents,
		Allocations:   snapshot.Allocations,
	})
	return nil
}

func (h *operationsHandlers) listRuntimeAgents(w http.ResponseWriter, r *http.Request) error {
	_, limit, encodedCursor, err := pageQuery(r.URL.RawQuery)
	if err != nil {
		return err
	}
	snapshot, err := h.operationsSnapshot()
	if err != nil {
		return err
	}
	after, err := h.operationsPageAfter(encodedCursor, "operations:runtime-agents", snapshot.Cursor.Generation)
	if err != nil {
		return err
	}
	items := make([]controlplane.RuntimeAgentObservation, 0, min(pagelimit.FetchRows(limit), len(snapshot.RuntimeAgents)))
	for _, item := range snapshot.RuntimeAgents {
		if item.InstanceID <= after {
			continue
		}
		items = append(items, item)
		if len(items) == pagelimit.FetchRows(limit) {
			break
		}
	}
	items, page, err := paginate(h.cursors, items, limit, "operations:runtime-agents", func(last controlplane.RuntimeAgentObservation) []string {
		return []string{snapshot.Cursor.Generation, last.InstanceID}
	})
	if err != nil {
		return err
	}
	writeJSON(w, http.StatusOK, runtimeAgentPageResponse{
		Cursor: operationsCursor(snapshot.Cursor), Items: items, Page: page,
	})
	return nil
}

func (h *operationsHandlers) listAllocations(w http.ResponseWriter, r *http.Request) error {
	_, limit, encodedCursor, err := pageQuery(r.URL.RawQuery)
	if err != nil {
		return err
	}
	snapshot, err := h.operationsSnapshot()
	if err != nil {
		return err
	}
	after, err := h.operationsPageAfter(encodedCursor, "operations:allocations", snapshot.Cursor.Generation)
	if err != nil {
		return err
	}
	items := make([]controlplane.AllocationObservation, 0, min(pagelimit.FetchRows(limit), len(snapshot.Allocations)))
	for _, item := range snapshot.Allocations {
		if item.AllocationID <= after {
			continue
		}
		items = append(items, item)
		if len(items) == pagelimit.FetchRows(limit) {
			break
		}
	}
	items, page, err := paginate(h.cursors, items, limit, "operations:allocations", func(last controlplane.AllocationObservation) []string {
		return []string{snapshot.Cursor.Generation, last.AllocationID}
	})
	if err != nil {
		return err
	}
	writeJSON(w, http.StatusOK, allocationPageResponse{
		Cursor: operationsCursor(snapshot.Cursor), Items: items, Page: page,
	})
	return nil
}

func (h *operationsHandlers) operationsSnapshot() (controlplane.OperationsSnapshot, error) {
	snapshot := h.Operations.SnapshotOperations()
	if err := snapshot.Validate(); err != nil {
		return controlplane.OperationsSnapshot{}, fmt.Errorf("invalid Operations snapshot: %w", err)
	}

	return snapshot, nil
}

func (h *operationsHandlers) operationsPageAfter(
	encoded string,
	kind string,
	generation string,
) (string, error) {
	values, err := h.cursors.decodePageCursor(encoded, kind, 2)
	if err != nil || len(values) == 0 {
		return "", err
	}
	if values[0] != generation {
		return "", fmt.Errorf("%w: Operations list cursor belongs to a different Server generation", errInvalidRequest)
	}
	return values[1], nil
}

func operationsCursor(source controlplane.OperationsCursor) operationsCursorResponse {
	return operationsCursorResponse{
		Generation: source.Generation,
		Revision:   strconv.FormatUint(source.Revision, 10),
	}
}
