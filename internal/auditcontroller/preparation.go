package auditcontroller

import (
	"context"
	"errors"

	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
)

// Preparation uses the ordinary role intent, Run and receipt machinery. Its
// accepted outputs persist independently of Round history and remain reusable.
func (c *Controller) progressPreparation(ctx context.Context, claim auditstore.ControllerClaim, snapshot auditstore.ReconcileSnapshot) *reconcileResult {
	audit := snapshot.Audit
	if audit.State != auditstore.AuditActive {
		return nil
	}
	if audit.Phase == auditdomain.AuditPhaseInventory {
		if c.initialRoundBuilder == nil {
			return reconciliationDone(false, nil)
		}
		params, reason, err := c.initialRoundBuilder.PrepareInitialRound(ctx, claim, snapshot)
		if err != nil {
			return reconciliationDone(false, err)
		}
		if reason != nil {
			return reconciliationDone(c.closeForRoleFailure(ctx, claim, audit, reason))
		}
		_, inserted, err := c.store.AcceptInitialRound(ctx, params)
		if errors.Is(err, auditstore.ErrPrecondition) {
			return reconciliationDone(false, nil)
		}
		return reconciliationDone(inserted, err)
	}
	if audit.Phase != auditdomain.AuditPhasePreparing {
		return nil
	}
	changed, complete, reason, err := c.reconcileRolePhase(ctx, claim, snapshot, auditstore.ExecutionPrepare)
	if changed || err != nil {
		return reconciliationDone(changed, err)
	}
	if reason != nil {
		if reason.Code != "submission_budget_exhausted" {
			reason.Code = "preparation_failed"
			for _, receipt := range snapshot.RoleReceipts {
				if receipt.ErrorCode != nil && *receipt.ErrorCode == "evidence-budget-exhausted" {
					reason.Code = "evidence_budget_exhausted"
					break
				}
			}
		}
		return reconciliationDone(c.closeForRoleFailure(ctx, claim, audit, reason))
	}
	if !complete {
		return reconciliationDone(false, nil)
	}
	_, err = c.store.CompletePreparation(ctx, claim, audit.Revision)
	if errors.Is(err, auditstore.ErrPrecondition) {
		return reconciliationDone(false, nil)
	}
	return reconciliationDone(err == nil, err)
}
