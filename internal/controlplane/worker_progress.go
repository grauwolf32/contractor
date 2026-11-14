package controlplane

import (
	"context"
	"errors"

	"github.com/grauwolf32/contractor/internal/contracts/reporting"
)

// WorkerProgressReader projects only numeric State from an allocation selected
// by an owner-authorized diagnostic query. It never supplies stop ownership.
type WorkerProgressReader struct {
	registry *InMemoryRegistry
	client   *RuntimeControlClient
}

func NewWorkerProgressReader(registry *InMemoryRegistry, client *RuntimeControlClient) *WorkerProgressReader {
	return &WorkerProgressReader{registry: registry, client: client}
}

func (r *WorkerProgressReader) ReadWorkerProgress(ctx context.Context, run, stage, allocation, name string) (*reporting.WorkerCompactionProgress, *reporting.WorkerModelPhases, error) {
	reservation, err := r.registry.GetReservation(allocation)
	if errors.Is(err, ErrAllocationNotFound) {
		return nil, nil, nil
	}
	if err != nil {
		return nil, nil, err
	}
	grant := reservation.Grant
	if grant.RunID != run || grant.StageExecutionID != stage || grant.LogicalAgentName != name || grant.Lost || grant.WriteFenced {
		return nil, nil, nil
	}
	target, err := r.client.endpoint(reservation.ControlURL, allocation, "agent-state")
	if err != nil {
		return nil, nil, err
	}
	state, err := r.client.readWorkerState(ctx, target, grant.RuntimeAgentID, "")
	if err != nil {
		var readError *reporting.WorkerStateReadError
		// A not-yet-prepared or already-released allocation has no observation.
		if errors.As(err, &readError) && readError.StatusCode == 404 {
			return nil, nil, nil
		}
		return nil, nil, err
	}
	// Release/authority changes during I/O invalidate the live observation.
	current, err := r.registry.GetGrant(allocation)
	if err != nil || current.Lost || current.WriteFenced || current.RuntimeInstanceID != grant.RuntimeInstanceID {
		return nil, nil, nil
	}
	invocation := state.Snapshot.State.CurrentInvocation
	if invocation == nil {
		invocation = state.Snapshot.State.LastCompletedInvocation
	}
	var compaction *reporting.WorkerCompactionProgress
	if invocation != nil && invocation.Compaction != nil {
		value := invocation.Compaction.Clone()
		compaction = &value
	}
	return compaction, state.Snapshot.State.Metrics.ModelPhases, nil
}
