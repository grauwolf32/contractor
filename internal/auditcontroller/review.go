package auditcontroller

import (
	"context"

	"github.com/grauwolf32/contractor/internal/auditstore"
)

func (c *Controller) reconcileReviewAndDeadline(ctx context.Context, claim auditstore.ControllerClaim, snapshot auditstore.ReconcileSnapshot) *reconcileResult {
	audit := snapshot.Audit
	if audit.State == auditstore.AuditWaitingReview {
		if changed, expireErr := c.store.ExpireReportReview(ctx, claim, audit.Revision); changed || expireErr != nil {
			return reconciliationDone(changed, expireErr)
		}
		if audit.Dispatch == auditstore.DispatchOpen && audit.DeadlineAt != nil && !c.now().Before(*audit.DeadlineAt) {
			reason := auditstore.StopReason{
				Code:    "deadline_exhausted",
				Message: "The Audit time limit was reached. Extend or disable the limit to continue.",
			}
			_, err := c.store.TransitionClaimed(ctx, auditstore.ClaimedTransitionParams{
				Claim: claim, ExpectedRevision: audit.Revision,
				ExpectedState: auditstore.AuditWaitingReview,
				TargetState:   auditstore.AuditPaused, Reason: &reason,
			})
			return reconciliationDone(err == nil, err)
		}
	}

	if audit.State == auditstore.AuditActive {
		if reason := c.dispatchClosureReason(snapshot); reason != nil {
			_, err := c.store.TransitionClaimed(ctx, auditstore.ClaimedTransitionParams{
				Claim: claim, ExpectedRevision: audit.Revision,
				ExpectedState: auditstore.AuditActive, TargetState: deadlineTarget(reason),
				Reason: reason,
			})
			return reconciliationDone(err == nil, err)
		}
	}

	return nil
}
