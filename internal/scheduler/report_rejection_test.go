package scheduler

import (
	"context"
	"errors"
	"testing"

	"github.com/grauwolf32/contractor/internal/runstore"
)

// TestSchedulerKeepsStageOutcomeWhenFinalReportIsRejected pins the Scheduler
// side of AWC-R115 with OBS-R37 and OBS-R38: when the Control Plane rejects a
// Runtime final report, for example one whose completion diagnostics carry an
// unknown kind or phase, the Stage and Run keep their outcome and the
// allocation is recorded with the incomplete placeholder report.
func TestSchedulerKeepsStageOutcomeWhenFinalReportIsRejected(t *testing.T) {
	harness := newSchedulerHarness(t)
	harness.workers.finalizeError = errors.New(
		"finalize allocation: Runtime Agent returned an invalid lifecycle response",
	)

	worked, err := harness.scheduler.RunOnce(context.Background())
	if err != nil || !worked {
		t.Fatalf("RunOnce = (%v, %v)", worked, err)
	}
	if harness.store.run.State != runstore.RunSucceeded || len(harness.store.stages) != 1 ||
		harness.store.stages[0].State != runstore.StageSucceeded {
		t.Fatalf("terminal state = run:%s stages:%+v", harness.store.run.State, harness.store.stages)
	}
	if harness.workers.finalizeCalls != 1 || harness.workers.releaseCalls != 1 {
		t.Fatalf("finalize/release calls = %d/%d", harness.workers.finalizeCalls, harness.workers.releaseCalls)
	}
	if len(harness.store.reports) != 1 {
		t.Fatalf("persisted reports = %+v", harness.store.reports)
	}
	worker := harness.store.reports[0].Report.Worker
	if worker.Complete || worker.Completion != nil || len(worker.Errors) != 1 ||
		worker.Errors[0].Code != "allocation_report_unavailable" {
		t.Fatalf("rejected report placeholder = %+v", worker)
	}
}
