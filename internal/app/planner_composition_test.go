package app

import (
	"slices"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/planner"
	plannerscan "github.com/grauwolf32/contractor/internal/planner/scan"
	"github.com/grauwolf32/contractor/internal/planner/streamline"
	"google.golang.org/adk/v2/model"
)

// The Server composes exactly one PlannerFactory for each registered
// PlannerFactory descriptor (CAT-R79).
func TestComposedPlannerFactoriesMatchTheDescriptorRegistry(t *testing.T) {
	registry, err := composePlanners(plannerAdapters{
		sessions:      struct{ planner.SessionService }{},
		planSessions:  struct{ planner.PlanSessionService }{},
		adkSessions:   struct{ streamline.ADKSessionFactory }{},
		scanSessions:  struct{ planner.ScanSessionService }{},
		invoker:       struct{ planner.WorkerInvoker }{},
		inspector:     struct{ planner.ArtifactInspector }{},
		stateReader:   struct{ planner.WorkerStateReader }{},
		model:         func(planner.ModelAccess) (model.LLM, error) { return nil, nil },
		limits:        streamline.DefaultLimits(),
		scanArtifacts: struct{ plannerscan.ArtifactStore }{},
	})
	if err != nil {
		t.Fatal(err)
	}
	composed, described := registry.Refs(), resolved.PlannerSelectors()
	for _, ref := range described {
		if !slices.Contains(composed, ref) {
			t.Errorf("PlannerFactory descriptor %q has no composed factory", ref)
		}
	}
	for _, ref := range composed {
		if !slices.Contains(described, ref) {
			t.Errorf("composed PlannerFactory %q has no descriptor", ref)
		}
	}
}

func TestPlannerCompositionRejectsAnUnmatchedRef(t *testing.T) {
	for name, tc := range map[string]struct{ composed, described []string }{
		"descriptor without a factory": {
			composed:  []string{resolved.PassthroughPlanner},
			described: []string{resolved.PassthroughPlanner, "next@1"},
		},
		"factory without a descriptor": {
			composed:  []string{resolved.PassthroughPlanner, "next@1"},
			described: []string{resolved.PassthroughPlanner},
		},
	} {
		t.Run(name, func(t *testing.T) {
			err := matchPlannerDescriptors(tc.composed, tc.described)
			if err == nil || !strings.Contains(err.Error(), `"next@1"`) {
				t.Fatalf("error = %v, want one naming next@1", err)
			}
		})
	}
	if err := matchPlannerDescriptors(resolved.PlannerSelectors(), resolved.PlannerSelectors()); err != nil {
		t.Fatal(err)
	}
}
