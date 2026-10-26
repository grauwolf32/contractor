//go:build integration

package auditcontroller

import (
	"context"
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/findingintake"
	"github.com/grauwolf32/contractor/internal/randomid"
)

func TestPostgresTypedAdmissionContinuesBeyondOnePrimarySliceBeforeAtomicAcceptance(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 2*time.Minute)
	defer cancel()
	catalog := typedRoutingControllerConfig(t, func(files map[string]string) {
		profile := files["audit-profiles/checklist.yaml"]
		profile = strings.Replace(profile, "maxItemsPerRound: 8", "maxItemsPerRound: 160", 1)
		profile = strings.Replace(profile, "maxItemsTotal: 8", "maxItemsTotal: 160", 1)
		profile = strings.Replace(profile, "maxSubmittedRuns: 16", "maxSubmittedRuns: 320", 1)
		files["audit-profiles/checklist.yaml"] = profile
	})
	h := newPostgresControllerHarnessWithConfig(t, ctx, 0, 1, catalog)
	intake, err := findingintake.New(h.pool)
	if err != nil {
		t.Fatal(err)
	}
	c := h.controllerWithCollector(t, intake)
	c.roundBuilder, c.options.NewID = h.service, randomid.New
	for range 2 {
		preparationStep(t, ctx, c)
	}
	executions, err := h.audits.ListExecutions(ctx, h.started.Audit.AuditID)
	if err != nil || len(executions) != 1 || executions[0].RunID == nil {
		t.Fatalf("seed execution count=%d error=%v", len(executions), err)
	}
	checks := make([]auditdomain.ProposedCheck, 130)
	for ordinal := range checks {
		checks[ordinal] = auditdomain.ProposedCheck{Objective: fmt.Sprintf("Verify bounded source check %03d.", ordinal), Method: "verify"}
	}
	submitChildFindingProposal(t, ctx, intake, childFindingGrant(t, ctx, h, *executions[0].RunID), "continued-checks", nil, checks...)
	finishTypedRoutingCheck(t, ctx, h, executions[0])
	var previous auditstore.AdmissionProgress
	var accepted bool
	for step := 0; step < 400; step++ {
		preparationStep(t, ctx, c)
		progress := readTypedAdmissionProgress(t, ctx, h)
		if progress.SelectedCount-previous.SelectedCount > 1 || progress.StagedCount-previous.StagedCount > 1 {
			t.Fatalf("one controller slice changed unbounded membership: before=%d/%d after=%d/%d", previous.SelectedCount, previous.StagedCount, progress.SelectedCount, progress.StagedCount)
		}
		current, err := h.audits.Get(ctx, h.started.Audit.OwnerID, h.started.Audit.AuditID)
		if err != nil {
			t.Fatal(err)
		}
		if current.CurrentRoundID != nil && *current.CurrentRoundID != h.started.Round.RoundID {
			if progress.SelectedCount != 130 || progress.StagedCount != 130 || progress.PreparedRoot == nil {
				t.Fatalf("partial Round became current: selected=%d staged=%d", progress.SelectedCount, progress.StagedCount)
			}
			accepted = true
			break
		}
		if current.SubmittedRunCount != 1 {
			t.Fatal("hidden proposed work dispatched a child before atomic acceptance")
		}
		if progress.SelectedCount == 129 && progress.StagedCount == 0 {
			// Restart after crossing the per-slice primary ceiling while the
			// immutable containing Round still has one candidate remaining.
			c = h.controllerWithCollector(t, intake)
			c.roundBuilder, c.options.NewID = h.service, randomid.New
		}
		previous = progress
	}
	if !accepted {
		t.Fatalf("130 candidates did not complete durable continuation: selected=%d staged=%d", previous.SelectedCount, previous.StagedCount)
	}
}
