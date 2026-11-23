package controlplane

import (
	"context"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/control"
)

type projectionGrantReader struct{ grant AllocationGrant }

func (r *projectionGrantReader) GetGrant(string) (AllocationGrant, error) { return r.grant, nil }

type allocationPreparationFunc func(context.Context, Reservation, control.AllocationSpec, func(control.AllocationSpec) (contracts.WorkerHandle, error)) (contracts.WorkerHandle, error)

func (f allocationPreparationFunc) WithPreparedSpec(ctx context.Context, reservation Reservation, spec control.AllocationSpec, dispatch func(control.AllocationSpec) (contracts.WorkerHandle, error)) (contracts.WorkerHandle, error) {
	return f(ctx, reservation, spec, dispatch)
}
