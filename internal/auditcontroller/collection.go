package auditcontroller

import (
	"context"
	"errors"

	"github.com/grauwolf32/contractor/internal/auditimport"
	"github.com/grauwolf32/contractor/internal/auditstore"
)

func (c *Controller) collectOne(ctx context.Context, claim auditstore.ControllerClaim, snapshot auditstore.ReconcileSnapshot) *reconcileResult {
	audit := snapshot.Audit
	if c.collector != nil {
		for _, execution := range snapshot.Executions {
			if execution.State != auditstore.ExecutionCollecting {
				continue
			}
			changed, collectErr := c.collector.Collect(ctx, claim, snapshot, execution)
			if errors.Is(collectErr, auditstore.ErrPrecondition) {
				return reconciliationDone(false, nil)
			}
			if errors.Is(collectErr, auditimport.ErrPermanent) {
				return reconciliationDone(c.failAuditImport(ctx, claim, audit, "collection-contract-invalid"))
			}
			return reconciliationDone(changed, collectErr)
		}
	}
	return nil
}
