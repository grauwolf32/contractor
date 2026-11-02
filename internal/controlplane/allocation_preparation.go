package controlplane

import (
	"context"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/control"
)

// AllocationGrantReader rechecks allocation authority after preparation I/O.
type AllocationGrantReader interface {
	GetGrant(string) (AllocationGrant, error)
}

// AllocationSpecPreparation supplies trusted allocation data around one synchronous
// protocol call. It owns any private material until dispatch returns, after the
// client has checked the Agent Card, and clears it on every exit. The client
// validates the completed spec and rechecks the grant immediately before dispatch.
// Composition supplies the concrete Audit engagement adapter; the core does not
// resolve product data or select credentials.
type AllocationSpecPreparation interface {
	WithPreparedSpec(context.Context, Reservation, control.AllocationSpec, func(control.AllocationSpec) (contracts.WorkerHandle, error)) (contracts.WorkerHandle, error)
}
