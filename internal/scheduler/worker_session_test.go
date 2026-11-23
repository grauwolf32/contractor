package scheduler

import (
	"bytes"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/runstore"
)

func TestSchedulerRejectsMissingPersistedSessionBeforeExecution(t *testing.T) {
	for _, test := range []struct {
		name         string
		stageState   runstore.StageExecutionState
		workflowMode bool
	}{
		{name: "queued Workflow", workflowMode: true},
		{name: "recovering Workflow", stageState: runstore.StageRunning, workflowMode: true},
		{name: "preparing Stage", stageState: runstore.StagePreparing},
		{name: "recovering Stage", stageState: runstore.StageRunning},
	} {
		t.Run(test.name, func(t *testing.T) {
			h := newSchedulerHarness(t)
			stage := h.workflow.Stages[h.workflow.EntryStage]
			// An inferred shared value would otherwise match this immutable Workflow.
			stage.Session = contracts.WorkerSessionShared
			h.workflow.Stages[h.workflow.EntryStage] = stage
			installHarnessWorkflow(t, h)
			if test.stageState != "" {
				h.store.stages = []runstore.StageExecution{h.persistedExecution(t, test.stageState)}
			}

			raw := &h.store.run.WorkflowSnapshot
			reason := "unsupported_workflow_shape"
			if !test.workflowMode {
				raw = &h.store.stages[0].StageSpecSnapshot
				reason = "scheduler_state_invalid"
			}
			field := []byte(`"session":"shared",`)
			if bytes.Count(*raw, field) != 1 {
				t.Fatal("fixture must contain one explicit shared Stage session")
			}
			*raw = bytes.Replace(*raw, field, nil, 1)
			retained := append([]byte(nil), (*raw)...)
			stageCount := len(h.store.stages)

			worked, err := h.scheduler.RunOnce(t.Context())
			if err != nil || !worked || h.store.run.State != runstore.RunFailed ||
				h.store.run.StateReason.Code != reason {
				t.Fatalf("missing session: worked=%t err=%v state=%s reason=%+v",
					worked, err, h.store.run.State, h.store.run.StateReason)
			}
			if len(h.store.stages) != stageCount || h.allocator.reserveCalls != 0 ||
				h.workers.prepareCalls != 0 || h.planners.createCalls != 0 || h.planners.runCalls != 0 {
				t.Fatal("missing session authority reached execution")
			}
			if !bytes.Equal(*raw, retained) {
				t.Fatal("missing session authority was rewritten")
			}
		})
	}
}
