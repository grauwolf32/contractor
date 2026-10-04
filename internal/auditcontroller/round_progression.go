package auditcontroller

import (
	"context"
	"errors"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditservice"
	"github.com/grauwolf32/contractor/internal/auditstore"
)

func nextRoundValidationError(err error) bool {
	// Failed storage I/O proves nothing about stored bytes and may succeed on
	// retry, so it is never a deterministic validation failure.
	if errors.Is(err, artifacts.ErrBlobIO) {
		return false
	}
	var validation *auditdomain.ValidationError
	return errors.As(err, &validation) || errors.Is(err, auditstore.ErrInvalid) ||
		errors.Is(err, artifacts.ErrArtifactIntegrity)
}

func (c *Controller) closeInvalidNextRound(
	ctx context.Context, claim auditstore.ControllerClaim, audit auditstore.Audit,
) *reconcileResult {
	reason := &auditstore.StopReason{
		Code:    "next_round_invalid",
		Message: "The next Audit Round failed deterministic proposal or item validation.",
	}
	return reconciliationDone(c.closeForRoleFailure(ctx, claim, audit, reason))
}

// closeInconsistentNextRound stops an Audit whose retained state cannot form
// a next Round. The stop reason names the violated invariant; the log keeps
// the full error for operators.
func (c *Controller) closeInconsistentNextRound(
	ctx context.Context, claim auditstore.ControllerClaim, audit auditstore.Audit,
	inconsistent *auditservice.RoundPreparationError,
) *reconcileResult {
	c.options.Logger.Error("Audit next Round preparation reached an inconsistent state",
		"audit_id", audit.AuditID, "error", inconsistent)
	reason := &auditstore.StopReason{
		Code:    "next_round_contract_invalid",
		Message: "The next Audit Round cannot be prepared because " + inconsistent.Diagnostic + ".",
	}
	return reconciliationDone(c.closeForRoleFailure(ctx, claim, audit, reason))
}

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
				!snapshot.MoreExecutions && snapshot.OnlyAwaitingReview {
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
						var inconsistent *auditservice.RoundPreparationError
						if errors.As(buildErr, &inconsistent) {
							return c.closeInconsistentNextRound(ctx, claim, audit, inconsistent)
						}
						if nextRoundValidationError(buildErr) {
							return c.closeInvalidNextRound(ctx, claim, audit)
						}
						return reconciliationDone(false, buildErr)
					}
					if params.RoundID != "" {
						_, _, acceptErr := c.store.AcceptNextRound(ctx, params)
						if errors.Is(acceptErr, auditstore.ErrPrecondition) {
							return reconciliationDone(false, nil)
						}
						if nextRoundValidationError(acceptErr) {
							return c.closeInvalidNextRound(ctx, claim, audit)
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
