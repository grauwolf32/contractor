package scheduler

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/reporting"
	"github.com/grauwolf32/contractor/internal/controlplane"
	"github.com/grauwolf32/contractor/internal/runstore"
)

// The Registry's real lease renewal is covered in controlplane tests. This
// adapter exercises the production batch controller through a durably deferred
// Scheduler claim, with Runtime rejecting a stale prepare lease.
type deferredLeaseRegistry struct {
	now       func() time.Time
	confirmed time.Time
}

func (r *deferredLeaseRegistry) PrepareReservation(reservation controlplane.Reservation) (controlplane.Reservation, error) {
	if !r.now().Before(r.confirmed) {
		return controlplane.Reservation{}, controlplane.ErrAllocationLost
	}
	reservation.LeaseExpiresAt = r.confirmed
	return reservation, nil
}

func (*deferredLeaseRegistry) SetWriteFence(string) error { return nil }
func (*deferredLeaseRegistry) SetAllocationPhase(string, controlplane.AllocationAuthoritativePhase, *controlplane.SafeReason) error {
	return nil
}
func (*deferredLeaseRegistry) RecordAllocationReport(string, reporting.AllocationFinalReport) error {
	return nil
}
func (*deferredLeaseRegistry) Release(string) error             { return nil }
func (*deferredLeaseRegistry) ReleaseLost(string) (bool, error) { return false, nil }

type deferredLeaseRuntime struct {
	now     func() time.Time
	workers *memoryWorkers
}

func (r *deferredLeaseRuntime) Prepare(ctx context.Context, reservation controlplane.Reservation, settings contracts.WorkerExecutionSettings) (contracts.WorkerHandle, error) {
	if !r.now().Before(reservation.LeaseExpiresAt) {
		return contracts.WorkerHandle{}, errors.New("Runtime rejected expired prepare lease")
	}
	handles, err := r.workers.PrepareAll(ctx, []controlplane.Reservation{reservation}, map[string]contracts.WorkerExecutionSettings{
		reservation.Grant.LogicalAgentName: settings,
	})
	return handles[reservation.Grant.LogicalAgentName], err
}
func (*deferredLeaseRuntime) Finalize(context.Context, controlplane.Reservation, string, time.Time) (reporting.AllocationFinalReport, error) {
	return reporting.AllocationFinalReport{}, errors.New("unexpected Runtime finalize")
}
func (*deferredLeaseRuntime) Abort(context.Context, controlplane.Reservation, string, contracts.TerminationError, time.Time) (reporting.AllocationFinalReport, error) {
	return reporting.AllocationFinalReport{}, errors.New("unexpected Runtime abort")
}
func (*deferredLeaseRuntime) Release(context.Context, controlplane.Reservation) error {
	return errors.New("unexpected Runtime release")
}

type deferredLeaseWorkers struct {
	WorkerController
	batch *controlplane.RuntimeBatchController
}

func (w deferredLeaseWorkers) PrepareAll(ctx context.Context, reservations []controlplane.Reservation, settings map[string]contracts.WorkerExecutionSettings) (map[string]contracts.WorkerHandle, error) {
	return w.batch.PrepareAll(ctx, reservations, settings)
}

func TestPostgresDeferredAdmissionRefreshesPrepareLease(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	failing := &failOnceAdmitPersistence{}
	h := newDeferredPlacementHarness(t, ctx, func(persistence AtomicPersistence) AtomicPersistence {
		failing.AtomicPersistence = persistence
		return failing
	})
	if worked, err := h.scheduler.RunOnce(ctx); !worked || err == nil || !failing.failed {
		t.Fatalf("deferred admission = (%v, %v)", worked, err)
	}
	if len(h.workers.allocator.cached) != 1 {
		t.Fatalf("placement was not pinned: %+v", h.workers.allocator.cached)
	}
	originalLease := h.workers.allocator.cached[0].LeaseExpiresAt
	now := originalLease.Add(10 * time.Second)
	registry := &deferredLeaseRegistry{now: func() time.Time { return now }, confirmed: now.Add(time.Minute)}
	batch, err := controlplane.NewRuntimeBatchController(
		&deferredLeaseRuntime{now: registry.now, workers: h.workers}, registry,
		controlplane.RuntimeBatchOptions{Now: registry.now},
	)
	if err != nil {
		t.Fatal(err)
	}
	h.scheduler.workers = deferredLeaseWorkers{WorkerController: h.workers, batch: batch}
	h.scheduler.options.Clock = staticClock{now: now}
	h.scheduler.options.PlannerTimeout = 3 * time.Minute
	h.workers.clock = staticClock{now: now}
	if worked, err := h.scheduler.RunOnce(ctx); !worked || err != nil {
		t.Fatalf("resumed Stage = (%v, %v)", worked, err)
	}
	h.requireSingleSucceededStage(t, ctx)
	if len(h.workers.preparedReservations) != 1 || len(h.workers.preparedReservations[0]) != 1 ||
		!h.workers.preparedReservations[0][0].LeaseExpiresAt.Equal(registry.confirmed) {
		t.Fatalf("Runtime prepared with stale lease: %+v", h.workers.preparedReservations)
	}
}

func TestSchedulerReportsLostPrepareLeaseAsControlLeaseExpired(t *testing.T) {
	h := newSchedulerHarness(t)
	h.workers.prepareError = controlplane.ErrAllocationLost
	if worked, err := h.scheduler.RunOnce(context.Background()); !worked || err != nil {
		t.Fatalf("lost prepare lease RunOnce = (%v, %v)", worked, err)
	}
	execution := h.store.stages[0]
	if execution.State != runstore.StageInterrupted || execution.Termination == nil ||
		execution.Termination.Code != "control_lease_expired" || !execution.Termination.Retryable {
		t.Fatalf("lost prepare lease termination = %+v", execution)
	}
}
