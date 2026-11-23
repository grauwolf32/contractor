package scheduler

import (
	"context"
	"errors"
	"testing"

	"github.com/grauwolf32/contractor/internal/config/resolved"
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
					// No replay repairs a corrupt history, so the Run ends
					// instead of counting it again on every pass (SCH-R102).
					if invalid != "" && !errors.Is(err, errInvalidRunState) {
						t.Fatalf("corrupt %s history = (%d, %v), want durable state the Scheduler cannot progress", invalid, used, err)
					}
				})
			}
		})
	}
}

// Every Transition check of a progression reads only the Run's pinned
// Workflow snapshot and its Stage history, so a failed one is durable state
// the Scheduler cannot progress and ends the Run (SCH-R102). A store failure
// while reading the history is not: a later pass may read it.
func TestBuildProgressionClassifiesUnprogressableTransitions(t *testing.T) {
	escalate := &resolved.EscalateTransition{MaxAttempts: 1, Then: resolved.TransitionAction{Kind: resolved.TransitionFail}}
	for _, test := range []struct {
		name      string
		action    resolved.TransitionAction
		variant   runstore.StageExecutionConfigVariant
		attempt   int
		stageName string
		runID     string
		invalid   bool
	}{
		{name: "retry without policy", action: resolved.TransitionAction{Kind: resolved.TransitionRetry}, invalid: true},
		{
			name: "escalate without policy", action: resolved.TransitionAction{Kind: resolved.TransitionEscalate},
			variant: runstore.StageExecutionConfigFailedEscalation, invalid: true,
		},
		{
			name: "escalate without outcome identity", invalid: true,
			action: resolved.TransitionAction{Kind: resolved.TransitionEscalate, Escalate: escalate},
		},
		{
			// The exhausted retry selects its then action, which may only be
			// next or fail.
			name: "escalate after retry", attempt: 2, invalid: true,
			action: resolved.TransitionAction{Kind: resolved.TransitionRetry, Retry: &resolved.RetryTransition{
				MaxAttempts: 2, Then: resolved.TransitionAction{Kind: resolved.TransitionEscalate, Escalate: escalate},
			}},
		},
		{
			name: "next to a missing Stage", invalid: true,
			action: resolved.TransitionAction{Kind: resolved.TransitionNext, NextStage: "missing"},
		},
		{
			name: "retry of a missing Stage", stageName: "missing", invalid: true,
			action: resolved.TransitionAction{Kind: resolved.TransitionRetry, Retry: &resolved.RetryTransition{
				MaxAttempts: 2, Then: resolved.TransitionAction{Kind: resolved.TransitionFail},
			}},
		},
		{name: "unknown action", action: resolved.TransitionAction{Kind: "loop"}, invalid: true},
		{
			name: "unreadable history", runID: "run-unreadable",
			action:  resolved.TransitionAction{Kind: resolved.TransitionEscalate, Escalate: escalate},
			variant: runstore.StageExecutionConfigFailedEscalation,
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			h := newSchedulerHarness(t)
			workflow, err := executableWorkflow{workflow: h.workflow}.selectStage(h.workflow.EntryStage)
			if err != nil {
				t.Fatal(err)
			}
			run := h.store.run
			if test.runID != "" {
				run.RunID = test.runID
			}
			stageName := workflow.stageName
			if test.stageName != "" {
				stageName = test.stageName
			}
			variant := test.variant
			if variant == "" {
				variant = runstore.StageExecutionConfigBase
			}
			execution := runstore.StageExecution{
				StageExecutionID: "stage-1", RunID: run.RunID, StageName: stageName, Attempt: max(test.attempt, 1),
				ExecutionConfigVariant: runstore.StageExecutionConfigBase, State: runstore.StageAborting,
			}
			_, err = h.scheduler.buildProgression(
				context.Background(), run, workflow, execution, test.action, true, "stage_failed", variant,
			)
			if err == nil || errors.Is(err, errInvalidRunState) != test.invalid {
				t.Fatalf("progression error = %v, want durable state the Scheduler cannot progress: %v", err, test.invalid)
			}
			if !test.invalid && !errors.Is(err, runstore.ErrNotFound) {
				t.Fatalf("progression error = %v, want the store failure", err)
			}
		})
	}
}
