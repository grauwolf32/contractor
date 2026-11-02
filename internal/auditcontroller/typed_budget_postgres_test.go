//go:build integration

package auditcontroller

import (
	"context"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"

	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/findingintake"
	"github.com/grauwolf32/contractor/internal/helpers/randomid"
)

func TestPostgresTypedAdmissionDefersAtNonTimeRoundBudgetWithoutTargetRetry(t *testing.T) {
	for _, budget := range []string{"round", "submission"} {
		t.Run(budget, func(t *testing.T) { testTypedNonTimeBudget(t, budget) })
	}
}

func testTypedNonTimeBudget(t *testing.T, budget string) {
	t.Helper()
	code := "round_budget_exhausted"
	if budget == "submission" {
		code = "submission_budget_exhausted"
	}
	ctx, cancel := context.WithTimeout(t.Context(), time.Minute)
	defer cancel()
	catalog := typedRoutingControllerConfig(t, func(files map[string]string) {
		profile := files["audit-profiles/checklist.yaml"]
		if budget == "submission" {
			profile = strings.Replace(profile, "maxSubmittedRuns: 16", "maxSubmittedRuns: 1", 1)
			profile = strings.Replace(profile, "maxItemsPerRound: 8", "maxItemsPerRound: 2", 1)
			profile = strings.Replace(profile, "maxItemsTotal: 8", "maxItemsTotal: 2", 1)
			profile = strings.Replace(profile, "batchSize: 1", "batchSize: 2", 1)
		} else {
			profile = strings.Replace(profile, "maxRounds: 3", "maxRounds: 1", 1)
		}
		files["audit-profiles/checklist.yaml"] = profile
	})
	h := newPostgresControllerHarnessWithConfig(t, ctx, 0, 1, catalog)
	intake, err := findingintake.New(h.pool, artifactpgtest.NewStorage())
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
	receipt := submitChildFindingProposal(t, ctx, intake, childFindingGrant(t, ctx, h, *executions[0].RunID), "deferred-typed-check", nil,
		auditdomain.ProposedCheck{Objective: "A valid typed check still consumes a Round.", Method: "verify"})
	finishTypedRoutingCheck(t, ctx, h, executions[0])
	var audit auditstore.Audit
	for range 20 {
		if _, err := c.RunOnce(ctx); err != nil {
			t.Fatal(err)
		}
		audit, err = h.audits.Get(ctx, h.started.Audit.OwnerID, h.started.Audit.AuditID)
		if err != nil || audit.State == auditstore.AuditFinalizing {
			break
		}
	}
	if err != nil || audit.State != auditstore.AuditFinalizing || audit.StopReason == nil || audit.StopReason.Code != code || audit.SubmittedRunCount != 1 || audit.ReservedRunCount != 1 {
		t.Fatalf("non-time budget: state=%s reason=%+v submitted=%d reserved=%d error=%v", audit.State, audit.StopReason, audit.SubmittedRunCount, audit.ReservedRunCount, err)
	}
	var deferred int
	if err := h.pool.QueryRow(ctx, `SELECT count(*) FROM audit_artifact_links WHERE audit_id=$1 AND source_provenance->>'schema'=$2 AND source_provenance->>'receipt_id'=$3 AND source_provenance->>'disposition'='deferred'`, audit.AuditID, auditstore.AdmissionDecisionSchema, receipt.ReceiptID).Scan(&deferred); err != nil || deferred != 1 {
		t.Fatalf("bounded defer disposition count=%d error=%v", deferred, err)
	}
	progress := readTypedAdmissionProgress(t, ctx, h)
	if progress.SelectedCount != 0 || progress.StopCode != code || progress.Root != nil {
		t.Fatal("exhausted non-time budget created partial accepted work")
	}
}
