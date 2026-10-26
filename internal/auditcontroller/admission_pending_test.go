package auditcontroller

import (
	"fmt"
	"testing"

	"github.com/grauwolf32/contractor/internal/auditservice"
	"github.com/grauwolf32/contractor/internal/auditstore"
)

func TestControllerKeepsDurableNextRoundAdmissionPending(t *testing.T) {
	h := newControllerHarness(t, 1, 1)
	h.store.round.State = auditstore.RoundClosed
	h.store.items = nil
	h.store.executions = nil
	h.store.audit.Limits.MaxRounds = 2
	h.store.audit.Limits.MaxItemsTotal = 2
	builder := &fakeRoundBuilder{err: fmt.Errorf("bounded candidate page: %w", auditservice.ErrNextRoundAdmissionPending)}
	h.controller.roundBuilder = builder
	for range 2 {
		if worked, err := h.controller.RunOnce(h.ctx); err != nil || !worked {
			t.Fatalf("admission progress = (%t, %v)", worked, err)
		}
		audit := h.store.auditSnapshot()
		if audit.State != auditstore.AuditActive || audit.Dispatch != auditstore.DispatchOpen || audit.StopReason != nil || h.store.round.Ordinal != 1 || h.store.round.State != auditstore.RoundClosed {
			t.Fatalf("pending generation closed or changed accepted membership: audit=%+v round=%+v", audit, h.store.round)
		}
		if h.creator.createdCount() != 0 {
			t.Fatal("pending admission created a child Run")
		}
	}
	if builder.calls.Load() != 2 {
		t.Fatal("pending generation did not resume on the next reconcile")
	}
}
