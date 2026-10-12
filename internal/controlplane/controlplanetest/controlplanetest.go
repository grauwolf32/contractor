// Package controlplanetest gives tests of other packages committed Runtime
// Agent reservations on a live Registry without a database. Production code
// never imports it.
package controlplanetest

import (
	"errors"

	"github.com/grauwolf32/contractor/internal/contracts/reporting"
	"github.com/grauwolf32/contractor/internal/contracts/runtimesettings"
	"github.com/grauwolf32/contractor/internal/controlplane"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
	"github.com/grauwolf32/contractor/internal/runtimeconfig/runtimeconfigtest"
)

// Reserve reserves one slot for every binding of request on registry in the
// two steps that controlplane.PlacementAllocator takes: provisional
// reservations that may use every Runtime Agent eligible now, then a commit
// with the model-free RuntimeConfig resolved from the built-in default and
// performance collection disabled. The committed reservations therefore name
// no model or LLM Gateway; a test that reads them places through
// PlacementAllocator instead. A request whose bindings find no eligible
// Runtime Agent is controlplane.ErrInsufficientCapacity.
func Reserve(
	registry *controlplane.InMemoryRegistry, request controlplane.ReservationRequest,
) ([]controlplane.Reservation, error) {
	edges := []controlplane.CandidateEdge{}
	for _, candidate := range registry.PlacementCandidates() {
		for _, binding := range request.Bindings {
			edges = append(edges, controlplane.CandidateEdge{
				LogicalAgentName:          binding.LogicalAgentName,
				RuntimeAgentID:            candidate.Principal.RuntimeAgentID,
				RuntimeAgentInstanceID:    candidate.Registration.InstanceID,
				RuntimeAgentLabelRevision: candidate.Principal.LabelRevision,
				RequiredRuntimeAdapters:   []runtimesettings.RuntimeAdapterRef{},
			})
		}
	}
	builtIn := runtimeconfigtest.BuiltInRunSnapshot().Default
	resolved, err := runtimeconfig.ResolveRuntimeConfig(runtimeconfig.ResolveRuntimeConfigInput{
		ModelFree: true,
		Default: runtimeconfig.PinnedRuntimeConfig{
			Label: builtIn.Label, BindingRevision: builtIn.BindingRevision, Config: builtIn.Config,
		},
	})
	if err != nil {
		return nil, err
	}
	reservations, err := registry.ReserveCandidateEdges(request, edges)
	if err != nil {
		return nil, err
	}
	pinned := make(map[string]controlplane.PinnedReservationConfig, len(reservations))
	for _, reservation := range reservations {
		pinned[reservation.Grant.AllocationID] = controlplane.PinnedReservationConfig{
			RuntimeAgentLabelRevision:   reservation.RuntimeAgentLabelRevision,
			Resolved:                    resolved,
			PerformanceCollectionPolicy: reporting.PerformanceCollectionDisabled,
		}
	}
	committed, err := registry.CommitCandidateReservations(request.StageExecutionID, pinned)
	if err != nil {
		return nil, errors.Join(err, registry.DiscardCandidateReservations(request.StageExecutionID))
	}
	return committed, nil
}
