//go:build integration

package auditcontroller

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditservice"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/findingintake"
	"github.com/grauwolf32/contractor/internal/randomid"
)

// Lose only replies after the real service has committed each durable phase.
// A fresh controller then resumes the existing checkpoint and artifact links.
type lostTypedAdmissionReplies struct {
	RoundBuilder
	t       *testing.T
	h       *postgresControllerHarness
	lost    map[string]bool
	failure error
	anchor  auditstore.AdmissionProgress
}

func (b *lostTypedAdmissionReplies) PrepareNextRound(ctx context.Context, claim auditstore.ControllerClaim, snapshot auditstore.ReconcileSnapshot) (auditstore.AcceptRoundParams, *auditstore.StopReason, error) {
	before := readTypedAdmissionProgress(b.t, ctx, b.h)
	var debitBefore int64
	if err := b.h.pool.QueryRow(ctx, `SELECT retained_evidence_bytes FROM audits WHERE audit_id=$1`, claim.AuditID).Scan(&debitBefore); err != nil {
		b.t.Fatal(err)
	}
	params, reason, err := b.RoundBuilder.PrepareNextRound(ctx, claim, snapshot)
	if err != nil && !errors.Is(err, auditservice.ErrNextRoundAdmissionPending) {
		return params, reason, err
	}
	after := readTypedAdmissionProgress(b.t, ctx, b.h)
	phase := ""
	switch {
	case after.PublishedPages > before.PublishedPages:
		phase = "page"
	case before.Root == nil && after.Root != nil:
		phase = "root"
	case after.StagedPages > before.StagedPages && after.StagedPages == after.PublishedPages:
		phase = "stage"
	}
	if phase == "" || b.lost[phase] {
		return params, reason, err
	}
	if phase == "page" {
		b.anchor = after
	} else if after.Generation != b.anchor.Generation || !after.DeadlineAt.Equal(b.anchor.DeadlineAt) || !after.SourceCutoffAt.Equal(b.anchor.SourceCutoffAt) || after.SourceUpper != b.anchor.SourceUpper {
		b.t.Fatal("lost reply restarted the source generation or validation allowance")
	}
	if after.StagedCount != after.SelectedCount && params.RoundID != "" {
		b.t.Fatal("root publication returned acceptance before private staging completed")
	}
	debitAfter := assertSingleTypedAdmissionChain(b.t, ctx, b.h, after)
	if phase == "stage" && debitAfter != debitBefore {
		b.t.Fatal("staging recharged immutable published artifacts")
	}
	if count, err := b.h.audits.CountItems(ctx, claim.AuditID); err != nil || count != 1 {
		b.t.Fatalf("private membership became visible after lost %s reply: count=%d error=%v", phase, count, err)
	}
	b.lost[phase] = true
	return auditstore.AcceptRoundParams{}, nil, b.failure
}

func assertSingleTypedAdmissionChain(t *testing.T, ctx context.Context, h *postgresControllerHarness, state auditstore.AdmissionProgress) int64 {
	t.Helper()
	var pages, roots, selections, rows int
	if err := h.pool.QueryRow(ctx, `SELECT count(*) FILTER(WHERE media_type=$2),count(*) FILTER(WHERE media_type=$3),count(*) FILTER(WHERE media_type LIKE 'application/vnd.contractor.audit.round-selection%') FROM audit_artifact_links WHERE audit_id=$1`, state.AuditID, auditdomain.PreparedRoundPageMediaType, auditdomain.PreparedRoundRootMediaType).Scan(&pages, &roots, &selections); err != nil {
		t.Fatal(err)
	}
	expectedRoots := 0
	if state.Root != nil {
		expectedRoots = 1
	}
	if pages != state.PublishedPages || roots != expectedRoots || selections != 0 {
		t.Fatalf("published chain contains duplicate artifacts: pages=%d/%d roots=%d/%d selections=%d", pages, state.PublishedPages, roots, expectedRoots, selections)
	}
	if err := h.pool.QueryRow(ctx, `SELECT count(*) FROM audit_items WHERE audit_id=$1 AND round_id=$2`, state.AuditID, state.RoundID).Scan(&rows); err != nil {
		t.Fatal(err)
	}
	if rows != state.StagedCount {
		t.Fatalf("staging duplicated membership: rows=%d staged=%d", rows, state.StagedCount)
	}
	var debit, distinctBytes int64
	if err := h.pool.QueryRow(ctx, `SELECT retained_evidence_bytes,(SELECT COALESCE(sum(size_bytes),0) FROM (SELECT DISTINCT artifact_ref,artifact_digest,size_bytes FROM audit_artifact_links WHERE audit_id=$1) AS exact) FROM audits WHERE audit_id=$1`, state.AuditID).Scan(&debit, &distinctBytes); err != nil {
		t.Fatal(err)
	}
	if debit != distinctBytes {
		t.Fatalf("retained evidence debit=%d, exact distinct linked bytes=%d", debit, distinctBytes)
	}
	return debit
}

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
	lost := &lostTypedAdmissionReplies{RoundBuilder: h.service, t: t, h: h, lost: map[string]bool{}, failure: errors.New("committed admission reply was lost")}
	c.roundBuilder, c.options.NewID = lost, randomid.New
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
		if _, err := c.RunOnce(ctx); err != nil {
			if !errors.Is(err, lost.failure) {
				t.Fatal(err)
			}
			c = h.controllerWithCollector(t, intake)
			c.roundBuilder, c.options.NewID = lost, randomid.New
		}
		progress := readTypedAdmissionProgress(t, ctx, h)
		if progress.SelectedCount-previous.SelectedCount > 1 || progress.StagedCount-previous.StagedCount > 1 {
			t.Fatalf("one controller slice changed unbounded membership: before=%d/%d after=%d/%d", previous.SelectedCount, previous.StagedCount, progress.SelectedCount, progress.StagedCount)
		}
		current, err := h.audits.Get(ctx, h.started.Audit.OwnerID, h.started.Audit.AuditID)
		if err != nil {
			t.Fatal(err)
		}
		if current.CurrentRoundID != nil && *current.CurrentRoundID != h.started.Round.RoundID {
			if progress.SelectedCount != 130 || progress.StagedCount != 130 || progress.PublishedPages != 130 || progress.StagedPages != 130 || progress.Root == nil {
				t.Fatalf("partial Round became current: selected=%d staged=%d", progress.SelectedCount, progress.StagedCount)
			}
			assertSingleTypedAdmissionChain(t, ctx, h, progress)
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
			c.roundBuilder, c.options.NewID = lost, randomid.New
		}
		previous = progress
	}
	if !accepted {
		t.Fatalf("130 candidates did not complete durable continuation: selected=%d staged=%d", previous.SelectedCount, previous.StagedCount)
	}
	if len(lost.lost) != 3 {
		t.Fatalf("did not replay all committed phase replies: %+v", lost.lost)
	}
	for range 8 {
		preparationStep(t, ctx, c)
		executions, err = h.audits.ListExecutions(ctx, h.started.Audit.AuditID)
		if err != nil {
			t.Fatal(err)
		}
		if len(executions) > 1 {
			break
		}
	}
	if len(executions) != 2 || executions[1].RunID == nil || executions[1].State != auditstore.ExecutionSubmitted {
		t.Fatalf("lost admission replies duplicated the first child: executions=%+v error=%v", executions, err)
	}
	current, err := h.audits.Get(ctx, h.started.Audit.OwnerID, h.started.Audit.AuditID)
	if err != nil || current.SubmittedRunCount != 2 || current.ReservedRunCount != 2 {
		t.Fatalf("lost admission replies duplicated submission debit: submitted=%d reserved=%d error=%v", current.SubmittedRunCount, current.ReservedRunCount, err)
	}
}
