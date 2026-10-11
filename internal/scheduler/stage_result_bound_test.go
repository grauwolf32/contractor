package scheduler

import (
	"context"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/planner"
	"github.com/grauwolf32/contractor/internal/runstore"
)

type resolverInspector struct{ resolver ArtifactResolver }

func (i resolverInspector) Inspect(ctx context.Context, runID string, ref contracts.ArtifactRef) (planner.ArtifactMetadata, error) {
	resolved, err := i.resolver.Resolve(ctx, runID, ref)
	if err != nil {
		return planner.ArtifactMetadata{}, err
	}
	return planner.ArtifactMetadata{MediaType: resolved.MediaType}, nil
}

// A StageResult the Planner accepts must not be rejected by the Scheduler for
// its size: both apply StageContentResult.Validate, which counts compact
// UTF-8 JSON, so HTML characters cost one byte each on both paths.
func TestStageResultBoundVerdictMatchesPlanner(t *testing.T) {
	for _, test := range []struct {
		name    string
		summary string
		valid   bool
	}{
		{"HTML summary at the summary bound", strings.Repeat("<", contracts.MaxStageResultSummaryBytes), true},
		{"HTML summary over the summary bound", strings.Repeat("<", contracts.MaxStageResultSummaryBytes+1), false},
		{"required escaping over the encoded bound", "x" + strings.Repeat("\x01", 65000), false},
	} {
		t.Run(test.name, func(t *testing.T) {
			h := newSchedulerHarness(t)
			h.planners.result.Summary = test.summary
			stage := h.workflow.Stages[h.workflow.EntryStage]
			plannerErr := planner.ValidateCandidate(
				t.Context(), h.store.run.RunID, stage.Result.Artifacts, h.planners.result.Clone(),
				resolverInspector{resolver: h.artifacts},
			)
			if (plannerErr == nil) != test.valid {
				t.Fatalf("Planner verdict valid=%v: %v", plannerErr == nil, plannerErr)
			}
			worked, err := h.scheduler.RunOnce(t.Context())
			if err != nil || !worked {
				t.Fatalf("RunOnce = (%v, %v)", worked, err)
			}
			execution := h.store.stages[0]
			_, finalizing := eventIndex(h.events.values, "enter_finalizing")
			if test.valid {
				if !finalizing || execution.State != runstore.StageSucceeded || execution.Termination != nil {
					t.Fatalf("Scheduler rejected a candidate the Planner accepted: state=%s termination=%+v", execution.State, execution.Termination)
				}
				return
			}
			if finalizing || execution.State != runstore.StageInterrupted || execution.Termination == nil ||
				execution.Termination.Code != "result_contract_violation" || execution.Termination.Retryable {
				t.Fatalf("Scheduler accepted a candidate the Planner rejected: state=%s termination=%+v", execution.State, execution.Termination)
			}
		})
	}
}
