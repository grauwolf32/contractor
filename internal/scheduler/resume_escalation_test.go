package scheduler

import (
	"context"
	"testing"

	"github.com/grauwolf32/contractor/internal/planner"
	"github.com/grauwolf32/contractor/internal/runstore"
)

func TestSchedulerManualEscalationResumeKeepsPolicyAndExhaustedBudgetAfterRestart(t *testing.T) {
	for _, outcome := range []string{"failed", "interrupted"} {
		t.Run(outcome, func(t *testing.T) {
			h := newSchedulerHarness(t)
			strong := configureEscalationWorkflow(t, h, outcome, 1)
			if outcome == "failed" {
				h.planners.result = failedStageResult("permanent", false)
			} else {
				failure := planner.NewError("planner_rejected", "cannot continue", false, nil)
				h.planners.runErrors = []error{failure, failure, failure}
			}
			for i := 0; i < 2; i++ {
				if worked, err := h.scheduler.RunOnce(context.Background()); err != nil || !worked {
					t.Fatalf("automatic attempt: worked=%v err=%v", worked, err)
				}
			}
			if h.store.run.State != runstore.RunFailed || len(h.store.stages) != 2 {
				t.Fatalf("expected exhausted automatic history: %+v", h.store.stages)
			}
			source := h.store.stages[1]
			// ResumeFailedRun supplies a new attempt with these exact immutable
			// fields; its transactional receipt is covered by the PostgreSQL test.
			h.store.stages = append(h.store.stages, runstore.StageExecution{
				StageExecutionID: "manual-resume", RunID: source.RunID, StageName: source.StageName,
				Attempt: source.Attempt + 1, PreviousExecutionID: &source.StageExecutionID,
				ResumeSourceExecutionID: &source.StageExecutionID,
				ExecutionConfigVariant:  source.ExecutionConfigVariant, EscalationOrdinal: source.EscalationOrdinal,
				StageSpecSchemaVersion: source.StageSpecSchemaVersion, StageSpecSnapshot: source.StageSpecSnapshot,
				StageContextSchemaVersion: source.StageContextSchemaVersion, StageContext: source.StageContext,
				State: runstore.StagePreparing, CreatedAt: h.clock.now,
			})
			h.store.run.State = runstore.RunRunning
			h.store.run.FinishedAt = nil
			restarted, err := New(h.store, h.persistence, h.artifacts, h.allocator, h.workers, h.planners, h.scheduler.options)
			if err != nil {
				t.Fatal(err)
			}
			if worked, err := restarted.RunOnce(context.Background()); err != nil || !worked {
				t.Fatalf("manual attempt after restart: worked=%v err=%v", worked, err)
			}
			if h.store.run.State != runstore.RunFailed || len(h.store.stages) != 3 || len(h.persistence.decisions) != 3 {
				t.Fatalf("manual continuation reset budget: state=%s stages=%d decisions=%+v", h.store.run.State, len(h.store.stages), h.persistence.decisions)
			}
			last := h.persistence.decisions[2]
			if !last.EscalationExhausted || last.EscalationOrdinal == nil || *last.EscalationOrdinal != 1 {
				t.Fatalf("manual attempt consumed an automatic ordinal: %+v", last)
			}
			if len(h.workers.preparedSettings) != 3 || h.workers.preparedSettings[2]["builder"].ModelPolicy.Ref != strong.Ref {
				t.Fatalf("manual continuation lost its strong policy: %+v", h.workers.preparedSettings)
			}
		})
	}
}

func TestEscalationHistoryOnlyCountsValidatedAutomaticRoots(t *testing.T) {
	for _, variant := range []runstore.StageExecutionConfigVariant{
		runstore.StageExecutionConfigFailedEscalation, runstore.StageExecutionConfigInterruptedEscalation,
	} {
		t.Run(string(variant), func(t *testing.T) {
			for _, invalid := range []string{"", "duplicate", "missing source", "wrong ordinal", "wrong attempt", "nonterminal source", "gap"} {
				t.Run(invalid, func(t *testing.T) {
					h := newSchedulerHarness(t)
					one, two := 1, 2
					root := runstore.StageExecution{StageExecutionID: "auto-1", RunID: h.store.run.RunID, StageName: "copy", Attempt: 2, State: runstore.StageFailed, ExecutionConfigVariant: variant, EscalationOrdinal: &one}
					manual := root
					manual.StageExecutionID, manual.Attempt = "manual-1", 3
					manual.PreviousExecutionID, manual.ResumeSourceExecutionID = &root.StageExecutionID, &root.StageExecutionID
					automatic := root
					automatic.StageExecutionID, automatic.Attempt, automatic.EscalationOrdinal = "auto-2", 4, &two
					h.store.stages = []runstore.StageExecution{root, manual, automatic}
					switch invalid {
					case "duplicate":
						h.store.stages[1].ResumeSourceExecutionID = nil
					case "missing source":
						missing := "missing"
						h.store.stages[1].ResumeSourceExecutionID = &missing
					case "wrong ordinal":
						h.store.stages[1].EscalationOrdinal = &two
					case "wrong attempt":
						h.store.stages[1].Attempt++
					case "nonterminal source":
						h.store.stages[0].State = runstore.StageRunning
					case "gap":
						h.store.stages = []runstore.StageExecution{automatic}
					}
					used, err := h.scheduler.escalationAttempts(context.Background(), root.RunID, root.StageName, variant)
					if invalid == "" && (err != nil || used != 2) {
						t.Fatalf("valid history: used=%d err=%v", used, err)
					}
					if invalid != "" && err == nil {
						t.Fatalf("corrupt %s history accepted: used=%d", invalid, used)
					}
				})
			}
		})
	}
}
