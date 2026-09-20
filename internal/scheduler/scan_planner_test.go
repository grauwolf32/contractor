package scheduler

import (
	"context"
	"encoding/json"
	"errors"
	"reflect"
	"testing"
	"time"

	workflowconfig "github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/controlplane"
	"github.com/grauwolf32/contractor/internal/planner"
	"github.com/grauwolf32/contractor/internal/runstore"
)

func TestSchedulerDispatchesFixedScanWorkersWithDurableClaimAndNoModelAccess(t *testing.T) {
	snapshot, err := workflowconfig.Load("../../configs/scan", workflowconfig.MVPDescriptors())
	if err != nil {
		t.Fatal(err)
	}
	for _, test := range []struct {
		workflow string
		workers  int
	}{
		{workflow: "request-set-scan@1", workers: 1},
		{workflow: "target-scan-plan@1", workers: 2},
	} {
		t.Run(test.workflow, func(t *testing.T) {
			h := newSchedulerHarness(t)
			workflow, err := snapshot.Workflow(test.workflow)
			if err != nil {
				t.Fatal(err)
			}
			stage := workflow.Stages[workflow.EntryStage]
			h.workflow, h.allocator.workflow, h.workers.workflow = workflow, workflow, workflow
			h.store.run.WorkflowName, h.store.run.WorkflowVersion = workflow.Ref.Name, workflow.Ref.Version
			h.store.run.WorkflowSnapshot, err = json.Marshal(workflow)
			if err != nil {
				t.Fatal(err)
			}
			h.store.run.Parameters = map[string]string{}
			for _, binding := range stage.Context.Artifacts {
				ref := exactRef(binding.Namespace, binding.Name, "scan-input-r1")
				h.artifacts.current["run-1/"+binding.Namespace+"/"+binding.Name] = *ref.Revision
				h.artifacts.values["run-1/"+binding.Namespace+"/"+binding.Name+"/"+*ref.Revision] = ResolvedArtifact{
					Ref: ref, MediaType: workflow.Inputs[binding.Name].MediaTypes[0],
				}
			}
			outputBinding := stage.Result.Artifacts["report"].From
			output := exactRef(outputBinding.Namespace, outputBinding.Name, "scan-report-r1")
			h.artifacts.values["run-1/"+output.Namespace+"/"+output.Name+"/"+*output.Revision] = ResolvedArtifact{Ref: output, MediaType: "application/json"}
			h.planners.result = contracts.StageContentResult{
				APIVersion: contracts.APIVersion, Outcome: contracts.StageSucceeded, Summary: "Scan report persisted",
				Artifacts: map[string]contracts.ArtifactRef{"report": output},
			}
			// The generic harness predates durable claim snapshots and only
			// allocates its first Worker. These adapters expose the real Run
			// claim shape and preserve the entire fixed reservation request.
			claimedStore := &scanClaimStore{memorySchedulerStore: h.store, now: h.clock.now}
			allocator := &scanFixedAllocator{memoryAllocator: h.allocator}
			registry := &scanPlannerRegistry{t: t, inner: h.planners, store: claimedStore}
			h.scheduler.store, h.scheduler.allocator, h.scheduler.planners = claimedStore, allocator, registry
			credentialCalls := 0
			h.scheduler.options.Credentials = credentialResolverFunc(func(context.Context, contracts.LLMCredentialRef, contracts.LLMGatewayConfigRef) (contracts.SecretString, error) {
				credentialCalls++
				return contracts.SecretString{}, errors.New("model-free scan must not resolve a gateway credential")
			})
			worked, err := h.scheduler.RunOnce(t.Context())
			if err != nil || !worked || h.store.run.State != runstore.RunSucceeded {
				t.Fatalf("scan RunOnce = worked:%t err:%v state:%s reason:%+v", worked, err, h.store.run.State, h.store.run.StateReason)
			}
			invocation := h.planners.invocation
			if registry.ref != planner.ScanPlanRef || h.planners.runCalls != 1 || invocation.SchedulerClaimID == "" || invocation.SchedulerClaimID != claimedStore.lastClaim {
				t.Fatalf("Planner identity/claim = ref:%s calls:%d invocation:%q claimed:%q", registry.ref, h.planners.runCalls, invocation.SchedulerClaimID, claimedStore.lastClaim)
			}
			if invocation.ModelAccess != nil || invocation.Stage.ExecutionConfig.Planner != nil || credentialCalls != 0 {
				t.Fatalf("model-free Planner gained gateway access: access:%+v credentials:%d", invocation.ModelAccess, credentialCalls)
			}
			if len(invocation.Workers) != test.workers || len(allocator.request.Bindings) != test.workers || len(h.store.allocations) != test.workers {
				t.Fatalf("fixed Worker set = handles:%d reservations:%d persisted:%d", len(invocation.Workers), len(allocator.request.Bindings), len(h.store.allocations))
			}
			for name, binding := range stage.Agents {
				handle, exists := invocation.Workers[name]
				if !exists || handle.AllocationID == "" || handle.AgentTemplateRef != binding.Template.Ref || handle.WorkerRuntimeRef != binding.Template.Runtime {
					t.Fatalf("Worker %s lost its pinned identity: %+v", name, handle)
				}
				settings := h.workers.preparedSettings[0][name]
				if settings.RuntimeSettings.LLMGatewayURL != "" || settings.RuntimeSettings.LLMGatewayToken != nil || settings.ModelPolicy.Model != "" {
					t.Fatalf("tool Worker %s gained model settings: %+v", name, settings)
				}
			}
			if !reflect.DeepEqual(invocation.Stage.ScanPlan, stage.ScanPlan) {
				t.Fatal("Scheduler changed the immutable scan policy")
			}
			for name, binding := range stage.Context.Artifacts {
				ref := invocation.Context.Artifacts[name]
				if ref == nil || ref.ValidateExact() != nil || ref.Namespace != binding.Namespace || ref.Name != binding.Name || *ref.Revision != "scan-input-r1" {
					t.Fatalf("scan source was not pinned: %+v", invocation.Context.Artifacts)
				}
			}
			assertOrderedEvents(t, h.events.values, "record_allocation", "prepare", "planner", "fence", "enter_finalizing", "accept", "release")
		})
	}
}

type scanClaimStore struct {
	*memorySchedulerStore
	now       time.Time
	lastClaim string
}

func (s *scanClaimStore) ClaimRunnableRun(ctx context.Context, id string, duration time.Duration) (runstore.WorkflowRun, error) {
	run, err := s.memorySchedulerStore.ClaimRunnableRun(ctx, id, duration)
	if err != nil {
		return run, err
	}
	s.lastClaim = id
	run.SchedulerClaim = &runstore.SchedulerClaim{ClaimID: id, ClaimedAt: s.now, ExpiresAt: s.now.Add(duration)}
	s.memorySchedulerStore.run = run
	return run, nil
}

type scanFixedAllocator struct {
	*memoryAllocator
	request controlplane.ReservationRequest
}

func (a *scanFixedAllocator) ReserveAll(request controlplane.ReservationRequest) ([]controlplane.Reservation, error) {
	a.reserveCalls++
	a.events.add("reserve")
	a.request = request
	if len(a.cached) == 0 {
		for _, binding := range request.Bindings {
			one := request
			one.Bindings = []controlplane.BindingRequirement{binding}
			reservation := a.reservationForRequest(one)
			a.cached = append(a.cached, reservation)
			a.grants[reservation.Grant.AllocationID] = reservation.Grant
		}
	}
	return append([]controlplane.Reservation(nil), a.cached...), nil
}

type scanPlannerRegistry struct {
	t     *testing.T
	inner *memoryPlannerRegistry
	store *scanClaimStore
	ref   string
}

func (r *scanPlannerRegistry) Create(ref string, invocation planner.Invocation) (planner.Planner, error) {
	r.ref = ref
	if ref != planner.ScanPlanRef || r.store.run.SchedulerClaim == nil || invocation.SchedulerClaimID != r.store.run.SchedulerClaim.ClaimID {
		r.t.Fatalf("scan Planner did not receive the current durable Run claim: ref=%s invocation=%q run=%+v", ref, invocation.SchedulerClaimID, r.store.run.SchedulerClaim)
	}
	// Reuse the harness lifecycle implementation after checking the selected
	// factory identity; its semantic result is fixed by the test above.
	return r.inner.Create(planner.PassthroughRef, invocation)
}
