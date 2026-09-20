package scheduler

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/controlplane"
	"github.com/grauwolf32/contractor/internal/runstore"
)

func TestSchedulerRejectsMissingStageCreationTimeBeforeAllocation(t *testing.T) {
	for _, state := range []runstore.StageExecutionState{runstore.StagePreparing, runstore.StageRunning} {
		t.Run(string(state), func(t *testing.T) {
			h := newSchedulerHarness(t)
			execution := h.persistedExecution(t, state)
			execution.CreatedAt = time.Time{}
			h.store.stages = []runstore.StageExecution{execution}
			worked, err := h.scheduler.RunOnce(t.Context())
			if err != nil || !worked || h.store.run.State != runstore.RunFailed || h.store.run.StateReason.Code != "scheduler_state_invalid" {
				t.Fatalf("missing timestamp: worked=%t err=%v state=%s reason=%+v", worked, err, h.store.run.State, h.store.run.StateReason)
			}
			if h.allocator.reserveCalls != 0 || h.workers.prepareCalls != 0 || h.planners.createCalls != 0 || !h.store.stages[0].CreatedAt.IsZero() {
				t.Fatal("missing creation time was replaced or execution started")
			}
		})
	}
}

func TestSchedulerRejectsIncompleteAllocationProvenanceBeforePreparingWorkers(t *testing.T) {
	for _, test := range []struct {
		name   string
		mutate func(*controlplane.Reservation)
	}{
		{"missing Runtime config", func(r *controlplane.Reservation) { r.ResolvedRuntimeConfig = nil }},
		{"missing collection policy", func(r *controlplane.Reservation) { r.PerformanceCollectionPolicy = "" }},
		{"invalid collection policy", func(r *controlplane.Reservation) { r.PerformanceCollectionPolicy = "unknown" }},
		{"missing label revision", func(r *controlplane.Reservation) { r.RuntimeAgentLabelRevision = 0 }},
		{"missing Runtime identity", func(r *controlplane.Reservation) { r.Grant.RuntimeAgentID = "" }},
	} {
		t.Run(test.name, func(t *testing.T) {
			h := newSchedulerHarness(t)
			execution := h.persistedExecution(t, runstore.StagePreparing)
			h.store.stages = []runstore.StageExecution{execution}
			r, err := h.allocator.reservation(execution.StageExecutionID)
			if err != nil {
				t.Fatal(err)
			}
			test.mutate(&r)
			h.allocator.cached = []controlplane.Reservation{r}
			h.allocator.grants[r.Grant.AllocationID] = r.Grant
			worked, err := h.scheduler.RunOnce(t.Context())
			if !worked || err == nil {
				t.Fatalf("incomplete allocation accepted: worked=%t err=%v", worked, err)
			}
			if len(h.store.allocations) != 0 || h.workers.prepareCalls != 0 || h.planners.createCalls != 0 {
				t.Fatal("incomplete provenance was persisted or used for execution")
			}
		})
	}
}

func TestSchedulerRejectsIncompleteDurableAllocationProvenanceOnRecovery(t *testing.T) {
	h := newSchedulerHarness(t)
	execution := h.persistedExecution(t, runstore.StageRunning)
	h.store.stages = []runstore.StageExecution{execution}
	h.installRecordedReservation(execution.StageExecutionID)
	h.store.allocations[0].RuntimeConfiguration = nil
	worked, err := h.scheduler.RunOnce(t.Context())
	if !worked || err == nil || h.workers.prepareCalls != 0 || h.planners.createCalls != 0 {
		t.Fatalf("incomplete durable provenance accepted: worked=%t err=%v", worked, err)
	}
}

func TestWorkerSettingsRequireCompleteReservationSet(t *testing.T) {
	for _, test := range []struct {
		name   string
		mutate func([]controlplane.Reservation) []controlplane.Reservation
	}{
		{"absent", func(_ []controlplane.Reservation) []controlplane.Reservation { return nil }},
		{"incomplete", func(rs []controlplane.Reservation) []controlplane.Reservation { return rs[:1] }},
		{"missing config", func(rs []controlplane.Reservation) []controlplane.Reservation {
			for i := range rs {
				rs[i].ResolvedRuntimeConfig = nil
			}
			return rs
		}},
		{"duplicate Worker", func(rs []controlplane.Reservation) []controlplane.Reservation { rs[1] = rs[0]; return rs }},
	} {
		t.Run(test.name, func(t *testing.T) {
			h := newSchedulerHarness(t)
			stage := h.workflow.Stages[h.workflow.EntryStage]
			stage.Agents["reviewer"] = stage.Agents["builder"]
			stage.ExecutionConfig.Agents["reviewer"] = stage.ExecutionConfig.Agents["builder"]
			credentialCalls := 0
			h.scheduler.options.Credentials = credentialResolverFunc(func(context.Context, contracts.LLMCredentialRef, contracts.LLMGatewayConfigRef) (contracts.SecretString, error) {
				credentialCalls++
				return contracts.SecretString{}, errors.New("unexpected credential access")
			})
			settings, err := h.scheduler.workerExecutionSettingsForRun(t.Context(), h.store.run, stage, test.mutate(schedulerTestReservations(t, stage)))
			if err == nil || settings != nil || credentialCalls != 0 {
				t.Fatalf("incomplete reservations reached materialization: settings=%v err=%v credential calls=%d", settings, err, credentialCalls)
			}
		})
	}
}

type observingContextAllocator struct {
	*memoryAllocator
	observed context.Context
}

func (a *observingContextAllocator) ReserveAllContext(ctx context.Context, request controlplane.ReservationRequest) ([]controlplane.Reservation, error) {
	a.observed = ctx
	return a.memoryAllocator.ReserveAllContext(ctx, request)
}

func TestSchedulerPassesReservationContextAndCancellation(t *testing.T) {
	h := newSchedulerHarness(t)
	allocator := &observingContextAllocator{memoryAllocator: h.allocator}
	h.scheduler.allocator = allocator
	workflow, err := decodeExecutableWorkflow(h.store.run)
	if err != nil {
		t.Fatal(err)
	}
	execution := h.persistedExecution(t, runstore.StagePreparing)
	ctx, cancel := context.WithTimeout(t.Context(), time.Minute)
	defer cancel()
	cancel()
	_, _, err = h.scheduler.liveOrNewReservations(ctx, h.store.run, workflow, execution)
	if !errors.Is(err, context.Canceled) || allocator.observed != ctx || h.allocator.reserveCalls != 0 {
		t.Fatalf("reservation ignored caller context: err=%v observed=%v calls=%d", err, allocator.observed == ctx, h.allocator.reserveCalls)
	}
}
