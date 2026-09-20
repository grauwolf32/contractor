package session

import (
	"context"
	"errors"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/planner"
	"github.com/grauwolf32/contractor/internal/runstore"
)

func TestPostgresScanClaimsFenceTakeoverAndConcurrentDispatch(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 45*time.Second)
	defer cancel()
	pool := isolatedPlannerPool(t, ctx)
	store := runstore.NewPostgresStore(pool)
	createPlannerStage(t, ctx, store)
	if _, err := store.ClaimRunnableRun(ctx, "scan-claim-1", time.Minute); err != nil {
		t.Fatal(err)
	}
	service, _ := New(store, Options{})
	started, err := service.BeginScan(ctx, "stage-planner", "scan-claim-1")
	if err != nil || !started.Invoke {
		t.Fatalf("BeginScan = (%+v, %v)", started, err)
	}
	plan := scanTestPlan("done", "uncertain", "pending")
	if err := service.InitializeScan(ctx, started.Identity, plan); err != nil {
		t.Fatal(err)
	}
	var winners atomic.Int32
	var group sync.WaitGroup
	for range 12 {
		group.Go(func() {
			won, err := service.ClaimScanJob(ctx, started.Identity, "done")
			if err != nil {
				t.Errorf("claim = %v", err)
			}
			if won {
				winners.Add(1)
			}
		})
	}
	group.Wait()
	if winners.Load() != 1 {
		t.Fatalf("Postgres dispatch winners = %d", winners.Load())
	}
	done := cloneScanJob(plan.Jobs[0])
	done.Status, done.Report = planner.ScanJobCompleted, scanTestRef("report")
	if err := service.FinishScanJob(ctx, started.Identity, done); err != nil {
		t.Fatal(err)
	}
	if won, err := service.ClaimScanJob(ctx, started.Identity, "uncertain"); !won || err != nil {
		t.Fatalf("uncertain job intent = (%t, %v)", won, err)
	}
	if _, err := pool.Exec(ctx, `UPDATE workflow_runs SET scheduler_claimed_at = clock_timestamp() - interval '2 seconds', scheduler_claim_expires_at = clock_timestamp() - interval '1 second' WHERE run_id = 'run-planner'`); err != nil {
		t.Fatal(err)
	}
	if won, err := service.ClaimScanJob(ctx, started.Identity, "pending"); won || !errors.Is(err, runstore.ErrConflict) {
		t.Fatalf("expired claim = (%t, %v)", won, err)
	}
	if _, err := store.ClaimRunnableRun(ctx, "scan-claim-2", time.Minute); err != nil {
		t.Fatal(err)
	}
	restarted, _ := New(store, Options{})
	recovered, err := restarted.BeginScan(ctx, "stage-planner", "scan-claim-2")
	if err != nil || !recovered.Invoke || recovered.State.Jobs[0].Status != planner.ScanJobCompleted ||
		recovered.State.Jobs[1].Status != planner.ScanJobUnknown || recovered.State.Jobs[2].Status != planner.ScanJobPending {
		t.Fatalf("Postgres recovery = (%+v, %v)", recovered, err)
	}
	if _, err := restarted.BeginScan(ctx, "stage-planner", "scan-claim-2"); !errors.Is(err, planner.ErrInvocationInProgress) {
		t.Fatalf("same-claim recovery = %v", err)
	}
	for _, id := range []string{"done", "uncertain"} {
		if won, err := restarted.ClaimScanJob(ctx, recovered.Identity, id); won || err != nil {
			t.Fatalf("terminal replay = (%t, %v)", won, err)
		}
	}
	if won, err := restarted.ClaimScanJob(ctx, recovered.Identity, "pending"); !won || err != nil {
		t.Fatalf("pending recovery = (%t, %v)", won, err)
	}
	if err := service.FinishScanJob(ctx, started.Identity, done); !errors.Is(err, runstore.ErrConflict) {
		t.Fatalf("old-owner finish = %v", err)
	}
	last := cloneScanJob(plan.Jobs[2])
	last.Status, last.Code = planner.ScanJobIncomplete, "scan_cancelled"
	if err := restarted.FinishScanJob(ctx, recovered.Identity, last); err != nil {
		t.Fatal(err)
	}
	if err := restarted.CompleteScan(ctx, recovered.Identity, scanTestCompletion()); err != nil {
		t.Fatal(err)
	}
	events, err := store.ListRunEvents(ctx, "run-planner", 0, 100)
	if err != nil || len(events) == 0 {
		t.Fatalf("read durable public scan events = (%d, %v)", len(events), err)
	}
}
