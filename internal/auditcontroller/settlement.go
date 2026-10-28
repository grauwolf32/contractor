package auditcontroller

import (
	"context"
	"errors"

	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditimport"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/rundeletion"
	"github.com/grauwolf32/contractor/internal/runstore"
)

func (c *Controller) settleClosedAudit(ctx context.Context, claim auditstore.ControllerClaim, snapshot auditstore.ReconcileSnapshot) (bool, error) {
	audit := snapshot.Audit
	if changed, err := c.failUnboundIntentBatch(ctx, claim, snapshot); changed || err != nil {
		return changed, err
	}
	if audit.State == auditstore.AuditCancelling || audit.State == auditstore.AuditDeleting {
		if changed, err := c.requestCancellationBatch(ctx, snapshot); changed || err != nil {
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
	if audit.PentestControlled {
		pending, err := c.store.ListPentestPendingAllocations(ctx, claim)
		if err != nil || len(pending) != 0 {
			return false, err
		}
	}
	switch audit.State {
	case auditstore.AuditFinalizing:
		if audit.CurrentRoundID == nil {
			_, err := c.store.TransitionClaimed(ctx, auditstore.ClaimedTransitionParams{
				Claim: claim, ExpectedRevision: audit.Revision,
				ExpectedState: auditstore.AuditFinalizing, TargetState: auditstore.AuditFailed,
				Reason: audit.StopReason,
			})
			return err == nil, err
		}
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
		deleted, drained, err := c.deleteReleasedRunBatch(ctx, claim, audit.OwnerID)
		if deleted || !drained || err != nil {
			return deleted, err
		}
		err = c.store.PurgeClaimed(ctx, claim, auditdomain.ArtifactNamespace(audit.AuditID))
		return err == nil, err
	}
	return false, nil
}

// deleteReleasedRunBatch deletes up to the batch bound of the deleting
// Audit's remaining child Runs, each in its own atomic deletion. Every
// candidate comes from a fresh claim-proven read, so a lost claim ends the
// batch before the next deletion. A Run that is not deletable yet, or that
// vanished, ends the batch; it is not progress. drained reports that the
// batch proved no child Run is left, so this step may purge: a step that
// deleted a Run leaves the purge, a separate barrier, to the next step.
func (c *Controller) deleteReleasedRunBatch(
	ctx context.Context, claim auditstore.ControllerClaim, ownerID string,
) (bool, bool, error) {
	batch := c.newDrainBatch()
	for batch.open(ctx) {
		runID, found, err := c.store.NextLiveRunForDeletion(ctx, claim)
		if err != nil {
			deleted, err := batch.stop(ctx, err)
			return deleted, false, err
		}
		if !found {
			return batch.committed > 0, true, nil
		}
		err = c.deleter.DeleteReleasedTerminalRun(ctx, ownerID, runID)
		var blocked *rundeletion.NotDeletableError
		if errors.As(err, &blocked) || errors.Is(err, runstore.ErrNotFound) {
			break
		}
		if err != nil {
			deleted, err := batch.stop(ctx, err)
			return deleted, false, err
		}
		batch.commit()
	}
	return batch.committed > 0, false, nil
}
