package auditcontroller

import (
	"context"
	"testing"

	"github.com/grauwolf32/contractor/internal/auditstore"
)

type waitingCollector struct{ err error }

func (c waitingCollector) Collect(context.Context, auditstore.ControllerClaim, auditstore.ReconcileSnapshot, auditstore.Execution) (bool, error) {
	return false, c.err
}
func (c waitingCollector) Finalize(context.Context, auditstore.ControllerClaim, auditstore.ReconcileSnapshot) (bool, error) {
	return false, nil
}

func TestCollectingNoOpStopsBeforeSettlement(t *testing.T) {
	for _, tc := range []struct {
		name string
		err  error
	}{{"waiting", nil}, {"revision conflict", auditstore.ErrPrecondition}} {
		t.Run(tc.name, func(t *testing.T) {
			h := newControllerHarness(t, 1, 1)
			h.controller.collector = waitingCollector{err: tc.err}
			h.store.audit.State = auditstore.AuditCancelling
			h.store.audit.Dispatch = auditstore.DispatchClosed
			h.store.executions = []auditstore.Execution{{ExecutionID: "collecting", State: auditstore.ExecutionCollecting}}
			revision := h.store.audit.Revision
			if changed, err := h.controller.RunOnce(h.ctx); changed || err != nil {
				t.Fatalf("reconcile = %t, %v", changed, err)
			}
			if h.store.audit.Revision != revision || len(h.store.items) != 1 || h.store.items[0].State != auditstore.ItemReady {
				t.Fatal("collection no-op allowed settlement to advance")
			}
		})
	}
}
