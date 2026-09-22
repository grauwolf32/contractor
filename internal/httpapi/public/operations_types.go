package public

// Operations: the read ports behind the Operations views, plus the
// Runtime Agent, allocation and scheduler settings bodies.

import (
	"context"
	"time"

	"github.com/grauwolf32/contractor/internal/controlplane"
	"github.com/grauwolf32/contractor/internal/performance"
	"github.com/grauwolf32/contractor/internal/settingsstore"
	"github.com/grauwolf32/contractor/internal/telemetry"
)

type MetricsReader interface {
	GetStageMetricsBatch(context.Context, []string) (map[string]telemetry.StageMetricsRecord, error)
}

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
	ListStageAllocationResources(
		context.Context, string, []string,
	) (map[string][]telemetry.AllocationResourceSummary, error)
}

type OperationsInvalidator interface {
	InvalidateOperations(controlplane.OperationsResource, string) error
}

type SchedulerSettingsManagement interface {
	GetSchedulerSettings(context.Context) (settingsstore.SchedulerSettings, error)
	UpdateSchedulerSettings(
		context.Context, settingsstore.UpdateSchedulerSettingsParams,
	) (settingsstore.SchedulerSettings, error)
}

type operationsCursorResponse struct {
	Generation string `json:"generation"`
	Revision   string `json:"revision"`
}

type schedulerSettingsResponse struct {
	MaxConcurrentRuns int       `json:"maxConcurrentRuns"`
	Revision          string    `json:"revision"`
	UpdatedAt         time.Time `json:"updatedAt"`
}

type updateSchedulerSettingsRequest struct {
	MaxConcurrentRuns int `json:"maxConcurrentRuns"`
}

type operationsSnapshotResponse struct {
	Cursor        operationsCursorResponse               `json:"cursor"`
	RuntimeAgents []controlplane.RuntimeAgentObservation `json:"runtimeAgents"`
	Allocations   []controlplane.AllocationObservation   `json:"allocations"`
}

type runtimeAgentPageResponse struct {
	Cursor operationsCursorResponse               `json:"cursor"`
	Items  []controlplane.RuntimeAgentObservation `json:"items"`
	Page   pageInfoResponse                       `json:"page"`
}

type allocationPageResponse struct {
	Cursor operationsCursorResponse             `json:"cursor"`
	Items  []controlplane.AllocationObservation `json:"items"`
	Page   pageInfoResponse                     `json:"page"`
}
