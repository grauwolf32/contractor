//go:build integration

package auditcontroller

import (
	"context"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditservice"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/findingintake"
	"github.com/grauwolf32/contractor/internal/helpers/randomid"
)

func TestPostgresTypedAdmissionWaitsForExplicitProjectionRebuildOrOriginalDeadline(t *testing.T) {
	for _, outcome := range []string{"rebuild", "deadline", "cancel"} {
		t.Run(outcome, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(t.Context(), time.Minute)
			defer cancel()
			h := newPostgresControllerHarnessWithConfig(t, ctx, 0, 1, typedRoutingControllerConfig(t))
			intake, err := findingintake.New(h.pool)
			if err != nil {
				t.Fatal(err)
			}
			now := time.Now().UTC()
			h.service, err = auditservice.New(auditservice.Options{Pool: h.pool, Profiles: h.snapshot,
				TransactionLLMCredentials: controllerTransactionCredentialLookup(), CredentialGuard: controllerCredentialGuard{},
				Now: func() time.Time { return now }})
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
			receipt := submitChildFindingProposal(t, ctx, intake, childFindingGrant(t, ctx, h, *executions[0].RunID), "pending-projection", nil,
				auditdomain.ProposedCheck{Objective: "Verify only this accepted receipt.", Method: "verify"})
			// Model the preceding schema's absent private index. The trusted
			// receipt, original bytes and hold still come from real intake.
			for _, statement := range []string{
				`ALTER TABLE finding_proposal_receipts DISABLE TRIGGER finding_proposal_receipts_protect_immutable`,
				`UPDATE finding_proposal_receipts SET checks_projection=NULL`,
				`ALTER TABLE finding_proposal_receipts ENABLE TRIGGER finding_proposal_receipts_protect_immutable`,
			} {
				if _, err := h.pool.Exec(ctx, statement); err != nil {
					t.Fatal(err)
				}
			}
			finishTypedRoutingCheck(t, ctx, h, executions[0])
			var blocked auditstore.AdmissionProgress
			for range 20 {
				preparationStep(t, ctx, c)
				blocked = readTypedAdmissionProgress(t, ctx, h)
				if blocked.BlockedReceiptID != "" {
					break
				}
			}
			if blocked.BlockedReceiptID != receipt.ReceiptID || blocked.BlockedCode != "source_projection_pending" || blocked.SelectedCount != 0 || blocked.Exhausted {
				t.Fatalf("missing index did not block exact receipt: %+v", blocked)
			}
			for range 3 {
				if worked, err := c.RunOnce(ctx); err != nil || worked {
					t.Fatalf("unchanged readiness wait: worked=%t error=%v", worked, err)
				}
			}
			if after := readTypedAdmissionProgress(t, ctx, h); after.Revision != blocked.Revision || !after.DeadlineAt.Equal(blocked.DeadlineAt) {
				t.Fatal("readiness polling changed progress or renewed the deadline")
			}
			var decisions int
			if err := h.pool.QueryRow(ctx, `SELECT count(*) FROM audit_artifact_links WHERE audit_id=$1 AND source_provenance->>'schema'=$2`, h.started.Audit.AuditID, auditstore.AdmissionDecisionSchema).Scan(&decisions); err != nil || decisions != 0 {
				t.Fatalf("unindexed receipt acquired an invented ordinal decision: %d %v", decisions, err)
			}
			if outcome == "deadline" {
				now = blocked.DeadlineAt
				preparationStep(t, ctx, c)
				audit, err := h.audits.Get(ctx, h.started.Audit.OwnerID, h.started.Audit.AuditID)
				if err != nil || audit.State != auditstore.AuditFinalizing || audit.StopReason == nil || audit.StopReason.Code != "next_round_validation_deadline" || audit.SubmittedRunCount != 1 {
					t.Fatalf("finite admission deadline: state=%s stop=%+v runs=%d error=%v", audit.State, audit.StopReason, audit.SubmittedRunCount, err)
				}
				return
			}
			if outcome == "cancel" {
				audit, err := h.audits.Get(ctx, h.started.Audit.OwnerID, h.started.Audit.AuditID)
				if err != nil {
					t.Fatal(err)
				}
				if _, err := h.service.Cancel(ctx, auditservice.MutationParams{OwnerID: audit.OwnerID, AuditID: audit.AuditID, ExpectedRevision: audit.Revision, IdempotencyKey: "cancel-pending-typed", RequestDigest: postgresDigest("cancel-pending-typed")}); err != nil {
					t.Fatal(err)
				}
				for range 12 {
					if _, err := c.RunOnce(ctx); err != nil {
						t.Fatal(err)
					}
					audit, err = h.audits.Get(ctx, audit.OwnerID, audit.AuditID)
					if err != nil || audit.State == auditstore.AuditCancelled {
						break
					}
				}
				if err != nil || audit.State != auditstore.AuditCancelled || audit.SubmittedRunCount != 1 {
					t.Fatalf("cancel readiness wait: state=%s submitted=%d error=%v", audit.State, audit.SubmittedRunCount, err)
				}
				if after := readTypedAdmissionProgress(t, ctx, h); after.Revision != blocked.Revision || after.SelectedCount != 0 {
					t.Fatal("cancellation advanced typed admission")
				}
				return
			}
			if report, err := intake.RebuildChecksProjectionBatch(ctx, 128); err != nil || report.Rebuilt != 1 {
				t.Fatalf("explicit rebuild: %+v %v", report, err)
			}
			// A fresh controller resumes the same generation only after the
			// explicit migration has made this exact receipt ready.
			c = h.controllerWithCollector(t, intake)
			c.roundBuilder, c.options.NewID = h.service, randomid.New
			preparationStep(t, ctx, c)
			after := readTypedAdmissionProgress(t, ctx, h)
			if after.Generation != blocked.Generation || !after.DeadlineAt.Equal(blocked.DeadlineAt) || after.SelectedCount != 1 || after.BlockedReceiptID != "" {
				t.Fatalf("rebuild did not resume the original intent: %+v", after)
			}
		})
	}
}
