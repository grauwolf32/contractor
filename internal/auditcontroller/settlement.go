package auditcontroller

import (
	"context"
	"errors"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditimport"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/runstore"
)

func (c *Controller) settleClosedAudit(ctx context.Context, claim auditstore.ControllerClaim, snapshot auditstore.ReconcileSnapshot) (bool, error) {
	audit := snapshot.Audit
	if changed, err := c.failOneUnboundIntent(ctx, claim, snapshot); changed || err != nil {
		return changed, err
	}
	if audit.State == auditstore.AuditCancelling || audit.State == auditstore.AuditDeleting || deadlineClosure(audit) {
		if changed, err := c.cancelOneSubmittedRun(ctx, snapshot); changed || err != nil {
			return changed, err
		}
	}
	settled, err := c.store.SettleUndispatched(ctx, claim, auditstore.MaxReconcileRows)
	if err != nil {
		return false, err
	}
	if settled != 0 {
		return true, nil
	}
	if audit.Hold == auditstore.HoldHeld {
		_, changed, err := c.store.ReleaseDispatchHold(ctx, claim)
		if changed || err != nil {
			return changed, err
		}
	}
	if !auditSettlementBarrier(snapshot) {
		return false, nil
	}
	switch audit.State {
	case auditstore.AuditFinalizing:
		if snapshot.Round != nil && snapshot.Round.State != auditstore.RoundClosed {
			_, err := c.store.TransitionRound(ctx, auditstore.RoundTransitionParams{
				Claim: claim, RoundID: snapshot.Round.RoundID,
				ExpectedRevision: snapshot.Round.Revision,
				ExpectedState:    snapshot.Round.State, TargetState: auditstore.RoundClosed,
			})
			return err == nil, err
		}
		if c.collector == nil {
			return false, nil
		}
		changed, finalizeErr := c.collector.Finalize(ctx, claim, snapshot)
		if errors.Is(finalizeErr, auditimport.ErrPermanent) {
			return c.failAuditImport(ctx, claim, audit, "report-contract-invalid")
		}
		return changed, finalizeErr
	case auditstore.AuditCancelling:
		target := auditstore.AuditCancelled
		if audit.DeletionRequestedAt != nil {
			target = auditstore.AuditDeleting
		}
		_, err := c.store.TransitionClaimed(ctx, auditstore.ClaimedTransitionParams{
			Claim: claim, ExpectedRevision: audit.Revision,
			ExpectedState: auditstore.AuditCancelling, TargetState: target,
			Reason: audit.StopReason,
		})
		return err == nil, err
	case auditstore.AuditDeleting:
		runID, found, err := c.store.NextLiveRunForDeletion(ctx, claim)
		if err != nil {
			return false, err
		}
		if found {
			err = c.runs.DeleteReleasedTerminalRun(ctx, audit.OwnerID, runID)
			var blocked *runstore.RunNotDeletableError
			if errors.As(err, &blocked) || errors.Is(err, runstore.ErrNotFound) {
				return false, nil
			}
			return err == nil, err
		}
		err = c.store.PurgeClaimed(ctx, claim, auditdomain.ArtifactNamespace(audit.AuditID))
		return err == nil, err
	}
	return false, nil
}
