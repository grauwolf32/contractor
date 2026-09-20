package session

import (
	"context"
	"encoding/json"
	"errors"
	"reflect"
	"strings"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/planner"
	"github.com/grauwolf32/contractor/internal/runstore"
)

func TestScanConcurrentBeginAndJobClaimHaveSingleWinner(t *testing.T) {
	store, service := newScanTestService(t)
	var won atomic.Int32
	var group sync.WaitGroup
	identities := make(chan planner.ScanSessionIdentity, 16)
	for range 16 {
		group.Go(func() {
			started, err := service.BeginScan(t.Context(), "stage-scan", "claim-1")
			if err != nil {
				if !errors.Is(err, planner.ErrInvocationInProgress) {
					t.Errorf("BeginScan: %v", err)
				}
				return
			}
			if started.Invoke {
				won.Add(1)
				identities <- started.Identity
			}
		})
	}
	group.Wait()
	if won.Load() != 1 {
		t.Fatalf("BeginScan winners = %d", won.Load())
	}
	identity := <-identities
	plan := scanTestPlan("job-1")
	if err := service.InitializeScan(t.Context(), identity, plan); err != nil {
		t.Fatal(err)
	}
	won.Store(0)
	for range 32 {
		group.Go(func() {
			claimed, err := service.ClaimScanJob(t.Context(), identity, "job-1")
			if err != nil {
				t.Errorf("ClaimScanJob: %v", err)
			}
			if claimed {
				won.Add(1)
			}
		})
	}
	group.Wait()
	if won.Load() != 1 {
		t.Fatalf("job claim winners = %d", won.Load())
	}
	events := store.snapshotEvents()
	if len(events) != 4 {
		t.Fatalf("start, owner, plan and intent events = %d", len(events))
	}
	for _, event := range events[1:] {
		if event.SchedulerClaimID != "claim-1" {
			t.Fatal("scan mutation omitted durable claim fence")
		}
	}
}

func TestScanTakeoverPreservesCompletedMarksStartedUnknownAndResumesPending(t *testing.T) {
	store, service := newScanTestService(t)
	started := beginScanTest(t, service)
	plan := scanTestPlan("done", "uncertain", "pending")
	if err := service.InitializeScan(t.Context(), started.Identity, plan); err != nil {
		t.Fatal(err)
	}
	claimScanTest(t, service, started.Identity, "done")
	completed := cloneScanJob(plan.Jobs[0])
	completed.Status, completed.Report = planner.ScanJobCompleted, scanTestRef("report")
	if err := service.FinishScanJob(t.Context(), started.Identity, completed); err != nil {
		t.Fatal(err)
	}
	claimScanTest(t, service, started.Identity, "uncertain")
	store.setClaim("claim-2")
	// A distinct service instance proves recovery uses only the durable store.
	restarted, _ := New(store, Options{})
	recovered, err := restarted.BeginScan(t.Context(), "stage-scan", "claim-2")
	if err != nil || !recovered.Invoke {
		t.Fatalf("takeover = (%+v, %v)", recovered, err)
	}
	if !reflect.DeepEqual(recovered.State.Jobs[0], completed) || recovered.State.Jobs[1].Status != planner.ScanJobUnknown ||
		recovered.State.Jobs[1].Code != "scan_outcome_unknown" || recovered.State.Jobs[2].Status != planner.ScanJobPending {
		t.Fatalf("recovered jobs = %+v", recovered.State.Jobs)
	}
	for _, id := range []string{"done", "uncertain"} {
		if claimed, err := restarted.ClaimScanJob(t.Context(), recovered.Identity, id); claimed || err != nil {
			t.Fatalf("terminal replay %s = (%t, %v)", id, claimed, err)
		}
	}
	claimScanTest(t, restarted, recovered.Identity, "pending")
	if err := service.FinishScanJob(t.Context(), started.Identity, completed); !errors.Is(err, runstore.ErrConflict) {
		t.Fatalf("late old-owner completion = %v", err)
	}
	if claimed, err := service.ClaimScanJob(t.Context(), started.Identity, "pending"); claimed || !errors.Is(err, runstore.ErrConflict) {
		t.Fatalf("old-owner claim = (%t, %v)", claimed, err)
	}
	last := cloneScanJob(recovered.State.Jobs[2])
	last.Status, last.Code = planner.ScanJobIncomplete, "scan_cancelled"
	if err := restarted.FinishScanJob(t.Context(), recovered.Identity, last); err != nil {
		t.Fatal(err)
	}
	completion := scanTestCompletion()
	if err := restarted.CompleteScan(t.Context(), recovered.Identity, completion); err != nil {
		t.Fatal(err)
	}
	store.setClaim("claim-3")
	replayed, err := restarted.BeginScan(t.Context(), "stage-scan", "claim-3")
	if err != nil || replayed.Invoke || replayed.Completion == nil || !reflect.DeepEqual(*replayed.Completion, completion) {
		t.Fatalf("completion replay = (%+v, %v)", replayed, err)
	}
}

func TestScanLostIntentAcknowledgementNeverGrantsDispatch(t *testing.T) {
	store, service := newScanTestService(t)
	started := beginScanTest(t, service)
	if err := service.InitializeScan(t.Context(), started.Identity, scanTestPlan("job")); err != nil {
		t.Fatal(err)
	}
	store.loseAcknowledgement("scan_job_started")
	if claimed, err := service.ClaimScanJob(t.Context(), started.Identity, "job"); claimed || err == nil {
		t.Fatalf("lost intent acknowledgement = (%t, %v)", claimed, err)
	}
	if claimed, err := service.ClaimScanJob(t.Context(), started.Identity, "job"); claimed || err != nil {
		t.Fatalf("repeated persisted intent = (%t, %v)", claimed, err)
	}
	store.setClaim("claim-2")
	recovered, err := service.BeginScan(t.Context(), "stage-scan", "claim-2")
	if err != nil || recovered.State.Jobs[0].Status != planner.ScanJobUnknown {
		t.Fatalf("lost acknowledgement recovery = (%+v, %v)", recovered, err)
	}
}

func TestScanOwnershipPersistedBeforePlanAndLostAcknowledgement(t *testing.T) {
	store, service := newScanTestService(t)
	store.loseAcknowledgement("scan_owner_acquired")
	if started, err := service.BeginScan(t.Context(), "stage-scan", "claim-1"); started.Invoke || err == nil {
		t.Fatalf("lost owner acknowledgement = (%+v, %v)", started, err)
	}
	if _, err := service.BeginScan(t.Context(), "stage-scan", "claim-1"); !errors.Is(err, planner.ErrInvocationInProgress) {
		t.Fatalf("duplicate owner = %v", err)
	}
	store.setClaim("claim-2")
	recovered, err := service.BeginScan(t.Context(), "stage-scan", "claim-2")
	if err != nil || !recovered.Invoke || recovered.State.Plan != nil || len(recovered.State.Jobs) != 0 {
		t.Fatalf("recovery before plan = (%+v, %v)", recovered, err)
	}
}

func TestScanCurrentDatabaseClaimFencesEveryMutation(t *testing.T) {
	for _, operation := range []string{"begin", "initialize", "claim", "finish", "complete"} {
		t.Run(operation, func(t *testing.T) {
			store, service := newScanTestService(t)
			if operation == "begin" {
				store.setClaim("claim-2")
				if start, err := service.BeginScan(t.Context(), "stage-scan", "claim-1"); start.Invoke || !errors.Is(err, runstore.ErrConflict) {
					t.Fatalf("stale BeginScan = (%+v, %v)", start, err)
				}
				return
			}
			started := beginScanTest(t, service)
			plan := scanTestPlan("job")
			if operation != "initialize" {
				if err := service.InitializeScan(t.Context(), started.Identity, plan); err != nil {
					t.Fatal(err)
				}
			}
			terminal := cloneScanJob(plan.Jobs[0])
			terminal.Status, terminal.Code = planner.ScanJobIncomplete, "scan_cancelled"
			if operation == "complete" {
				if err := service.FinishScanJob(t.Context(), started.Identity, terminal); err != nil {
					t.Fatal(err)
				}
			}
			before := len(store.snapshotEvents())
			store.setClaim("claim-2") // state owner is still claim-1; only SQL fence can reject.
			var err error
			switch operation {
			case "initialize":
				err = service.InitializeScan(t.Context(), started.Identity, plan)
			case "claim":
				var claimed bool
				claimed, err = service.ClaimScanJob(t.Context(), started.Identity, "job")
				if claimed {
					t.Fatal("stale claim granted dispatch")
				}
			case "finish":
				err = service.FinishScanJob(t.Context(), started.Identity, terminal)
			case "complete":
				err = service.CompleteScan(t.Context(), started.Identity, scanTestCompletion())
			}
			if !errors.Is(err, runstore.ErrConflict) || len(store.snapshotEvents()) != before {
				t.Fatalf("stale %s mutation = %v", operation, err)
			}
		})
	}
}

func TestScanPlanImmutableAndTransitionsBounded(t *testing.T) {
	_, service := newScanTestService(t)
	started := beginScanTest(t, service)
	if claimed, err := service.ClaimScanJob(t.Context(), started.Identity, "job"); claimed || err == nil {
		t.Fatal("claim before exact plan was accepted")
	}
	plan := scanTestPlan("job")
	if err := service.InitializeScan(t.Context(), started.Identity, plan); err != nil {
		t.Fatal(err)
	}
	if err := service.InitializeScan(t.Context(), started.Identity, plan); err != nil {
		t.Fatal("identical initialization failed", err)
	}
	changed := cloneScanState(plan)
	changed.Jobs[0].Worker = "other"
	if err := service.InitializeScan(t.Context(), started.Identity, changed); !errors.Is(err, runstore.ErrConflict) {
		t.Fatalf("changed plan = %v", err)
	}
	if err := service.CompleteScan(t.Context(), started.Identity, scanTestCompletion()); err == nil {
		t.Fatal("pending job was hidden by completion")
	}
	job := cloneScanJob(plan.Jobs[0])
	job.Status = planner.ScanJobCompleted
	if err := service.FinishScanJob(t.Context(), started.Identity, job); err == nil {
		t.Fatal("completed job without intent was accepted")
	}
	job.Status, job.Code = planner.ScanJobUnavailable, "scan_worker_unavailable"
	if err := service.FinishScanJob(t.Context(), started.Identity, job); err != nil {
		t.Fatal(err)
	}
	job.Status = planner.ScanJobCompleted
	if err := service.FinishScanJob(t.Context(), started.Identity, job); !errors.Is(err, runstore.ErrConflict) {
		t.Fatal("terminal status was overwritten", err)
	}
	for _, mutate := range []func(*planner.ScanState){
		func(p *planner.ScanState) { p.Plan.Revision = nil },
		func(p *planner.ScanState) { p.Jobs = append(p.Jobs, p.Jobs[0]) },
		func(p *planner.ScanState) {
			p.Jobs[0].InputArtifacts["request"] = contracts.ArtifactRef{Namespace: "inputs", Name: "request"}
		},
		func(p *planner.ScanState) { p.Jobs[0].Status = "arbitrary" },
		func(p *planner.ScanState) { p.Jobs[0].Code = "secret value\n" },
		func(p *planner.ScanState) { p.Jobs = make([]planner.ScanJobRecord, planner.MaxScanJobs+1) },
	} {
		invalid := cloneScanState(plan)
		mutate(&invalid)
		if err := validateScanState(invalid); err == nil {
			t.Fatal("invalid state accepted")
		}
	}
}

func TestScanJobWithoutArtifactInputsNormalizesNilMap(t *testing.T) {
	_, service := newScanTestService(t)
	started := beginScanTest(t, service)
	plan := scanTestPlan("nmap-job")
	plan.Jobs[0].InputArtifacts = nil
	for range 2 {
		if err := service.InitializeScan(t.Context(), started.Identity, plan); err != nil {
			t.Fatal(err)
		}
	}
	claimScanTest(t, service, started.Identity, "nmap-job")
	job := plan.Jobs[0]
	job.Status = planner.ScanJobCompleted
	if err := service.FinishScanJob(t.Context(), started.Identity, job); err != nil {
		t.Fatal(err)
	}
}

type fencedScanStore struct {
	*memoryStore
	fenceMu sync.Mutex
	claim   string
	loseAck string
}

func newScanTestService(t *testing.T) (*fencedScanStore, *Service) {
	t.Helper()
	store := &fencedScanStore{memoryStore: &memoryStore{execution: runstore.StageExecution{
		StageExecutionID: "stage-scan", State: runstore.StagePreparing,
	}}, claim: "claim-1"}
	service, err := New(store, Options{})
	if err != nil {
		t.Fatal(err)
	}
	return store, service
}

func (s *fencedScanStore) AppendPlannerEvent(ctx context.Context, params runstore.AppendPlannerEventParams) error {
	s.fenceMu.Lock()
	defer s.fenceMu.Unlock()
	if params.SchedulerClaimID != "" && params.SchedulerClaimID != s.claim {
		return runstore.ErrConflict
	}
	if err := s.memoryStore.AppendPlannerEvent(ctx, params); err != nil {
		return err
	}
	var event struct{ Kind string }
	_ = json.Unmarshal(params.Event, &event)
	if s.loseAck != "" && event.Kind == s.loseAck {
		s.loseAck = ""
		return errors.New("commit acknowledgement lost")
	}
	return nil
}

func (s *fencedScanStore) setClaim(claim string) {
	s.fenceMu.Lock()
	defer s.fenceMu.Unlock()
	s.claim = claim
}

func (s *fencedScanStore) loseAcknowledgement(kind string) {
	s.fenceMu.Lock()
	defer s.fenceMu.Unlock()
	s.loseAck = kind
}

func scanTestRef(name string) *contracts.ArtifactRef {
	revision := "rev-" + name
	return &contracts.ArtifactRef{Namespace: "scan", Name: name, Revision: &revision}
}

func scanTestPlan(ids ...string) planner.ScanState {
	state := planner.ScanState{Plan: scanTestRef("plan"), PlanDigest: "sha256:" + strings.Repeat("a", 64), Jobs: []planner.ScanJobRecord{}}
	for _, id := range ids {
		state.Jobs = append(state.Jobs, planner.ScanJobRecord{
			ID: id, Worker: "sqlmap", Status: planner.ScanJobPending,
			InputArtifacts: map[string]contracts.ArtifactRef{"request": *scanTestRef(id)},
		})
	}
	return state
}

func scanTestCompletion() planner.Completion {
	return planner.Completion{Result: &contracts.StageContentResult{
		APIVersion: contracts.APIVersion, Outcome: contracts.StageFailed, Summary: "Scan coverage is incomplete",
		Error:     &contracts.TerminationError{Code: "scan_incomplete", Message: "Scan coverage is incomplete", Retryable: false},
		Artifacts: map[string]contracts.ArtifactRef{"report": *scanTestRef("aggregate")},
	}}
}

func beginScanTest(t *testing.T, service *Service) planner.ScanSessionStart {
	t.Helper()
	started, err := service.BeginScan(t.Context(), "stage-scan", "claim-1")
	if err != nil || !started.Invoke {
		t.Fatalf("BeginScan = (%+v, %v)", started, err)
	}
	return started
}

func claimScanTest(t *testing.T, service *Service, identity planner.ScanSessionIdentity, id string) {
	t.Helper()
	if won, err := service.ClaimScanJob(t.Context(), identity, id); !won || err != nil {
		t.Fatalf("ClaimScanJob %s = (%t, %v)", id, won, err)
	}
}
