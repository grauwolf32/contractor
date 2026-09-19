package auditcontroller

import (
	"context"
	"errors"
	"github.com/grauwolf32/contractor/internal/auditstore"
)

func (c *Controller) progressRound(ctx context.Context, claim auditstore.ControllerClaim, snapshot auditstore.ReconcileSnapshot) *reconcileResult {
	audit := snapshot.Audit
	if audit.State == auditstore.AuditActive && snapshot.Round != nil {
		switch snapshot.Round.State {
		case auditstore.RoundAccepted:
			changed, complete, reason, err := c.reconcileRolePhase(
				ctx, claim, snapshot, auditstore.ExecutionDiscovery,
			)
			if changed || err != nil {
				return reconciliationDone(changed, err)
			}
			if reason != nil {
				return reconciliationDone(c.closeForRoleFailure(ctx, claim, audit, reason))
			}
			if complete {
				_, err := c.store.TransitionRound(ctx, auditstore.RoundTransitionParams{
					Claim: claim, RoundID: snapshot.Round.RoundID,
					ExpectedRevision: snapshot.Round.Revision,
					ExpectedState:    auditstore.RoundAccepted, TargetState: auditstore.RoundExecuting,
				})
				return reconciliationDone(err == nil, err)
			}
		case auditstore.RoundExecuting:
			if audit.OutstandingRunCount == 0 && len(snapshot.Items) == 0 &&
				len(snapshot.Executions) == 0 && !snapshot.MoreItems && !snapshot.MoreExecutions {
				_, err := c.store.TransitionRound(ctx, auditstore.RoundTransitionParams{
					Claim: claim, RoundID: snapshot.Round.RoundID,
					ExpectedRevision: snapshot.Round.Revision,
					ExpectedState:    auditstore.RoundExecuting, TargetState: auditstore.RoundAssessing,
				})
				return reconciliationDone(err == nil, err)
			}
			if audit.OutstandingRunCount == 0 && len(snapshot.Executions) == 0 &&
				!snapshot.MoreItems && !snapshot.MoreExecutions &&
				onlyAwaitingReview(snapshot.Items) {
				_, err := c.store.TransitionClaimed(ctx, auditstore.ClaimedTransitionParams{
					Claim: claim, ExpectedRevision: audit.Revision,
					ExpectedState: auditstore.AuditActive,
					TargetState:   auditstore.AuditWaitingReview,
				})
				return reconciliationDone(err == nil, err)
			}
		case auditstore.RoundAssessing:
			changed, complete, reason, err := c.reconcileRolePhase(
				ctx, claim, snapshot, auditstore.ExecutionAssessment,
			)
			if changed || err != nil {
				return reconciliationDone(changed, err)
			}
			if reason != nil {
				return reconciliationDone(c.closeForRoleFailure(ctx, claim, audit, reason))
			}
			if complete {
				_, err := c.store.TransitionRound(ctx, auditstore.RoundTransitionParams{
					Claim: claim, RoundID: snapshot.Round.RoundID,
					ExpectedRevision: snapshot.Round.Revision,
					ExpectedState:    auditstore.RoundAssessing, TargetState: auditstore.RoundClosed,
				})
				return reconciliationDone(err == nil, err)
			}
		case auditstore.RoundClosed:
			if audit.OutstandingRunCount == 0 && len(snapshot.Items) == 0 && len(snapshot.Executions) == 0 {
				var reason *auditstore.StopReason
				if c.roundBuilder != nil {
					params, closureReason, buildErr := c.roundBuilder.PrepareNextRound(ctx, claim, snapshot)
					if buildErr != nil {
						return reconciliationDone(false, buildErr)
					}
					if params.RoundID != "" {
						_, _, acceptErr := c.store.AcceptNextRound(ctx, params)
						if errors.Is(acceptErr, auditstore.ErrPrecondition) {
							return reconciliationDone(false, nil)
						}
						return reconciliationDone(acceptErr == nil, acceptErr)
					}
					reason = closureReason
				}
				if reason == nil {
					reason = &auditstore.StopReason{
						Code: "round_complete", Message: "The immutable Audit round reached its settlement barrier.",
					}
				}
				_, err := c.store.TransitionClaimed(ctx, auditstore.ClaimedTransitionParams{
					Claim: claim, ExpectedRevision: audit.Revision,
					ExpectedState: auditstore.AuditActive, TargetState: auditstore.AuditFinalizing,
					Reason: reason,
				})
				return reconciliationDone(err == nil, err)
			}
		}
	}
	return nil
}
