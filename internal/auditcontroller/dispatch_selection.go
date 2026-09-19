package auditcontroller

import (
	"context"
	"errors"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/config"
)

func (c *Controller) dispatchReadyBatch(ctx context.Context, claim auditstore.ControllerClaim, snapshot auditstore.ReconcileSnapshot) (bool, error) {
	audit := snapshot.Audit
	if audit.State != auditstore.AuditActive || snapshot.Round == nil ||
		snapshot.Round.State != auditstore.RoundExecuting {
		return false, nil
	}

	selected := make([]CheckExecutionMember, 0, min(audit.Limits.BatchSize, auditstore.MaxCollectionItems))
	var binding config.ResolvedAuditWorkflowBinding
	profile, profileErr := config.DecodeResolvedAuditProfileSnapshot(audit.ProfileSnapshot)
	for _, item := range snapshot.Items {
		if item.State != auditstore.ItemReady || item.RoundID != snapshot.Round.RoundID {
			continue
		}
		if len(selected) != 0 && (!sameBatchEnvelope(selected[0].Item, item) ||
			profileErr == nil && !sameItemParameterEnvelope(binding, selected[0].Item, item)) {
			continue
		}
		if len(selected) == 0 && profileErr == nil {
			binding = profile.Workflows[item.WorkflowRole]
		}
		attempt, err := c.store.NextItemAttempt(ctx, claim, item.ItemID)
		if errors.Is(err, auditstore.ErrPrecondition) {
			return false, nil
		}
		if err != nil {
			return false, err
		}
		if attempt > audit.Limits.MaxItemRunAttempts {
			reason := auditstore.StopReason{
				Code:    "item_attempt_budget_exhausted",
				Message: "An Audit item remained ready after its configured attempt budget was exhausted.",
			}
			_, err := c.store.TransitionClaimed(ctx, auditstore.ClaimedTransitionParams{
				Claim: claim, ExpectedRevision: audit.Revision,
				ExpectedState: auditstore.AuditActive, TargetState: auditstore.AuditFinalizing,
				Reason: &reason,
			})
			return err == nil, err
		}
		selected = append(selected, CheckExecutionMember{Item: item, Attempt: attempt})
		if len(selected) == audit.Limits.BatchSize || len(selected) == auditstore.MaxCollectionItems {
			break
		}
	}
	if len(selected) != 0 {
		return c.dispatch(ctx, claim, snapshot, selected)
	}
	return false, nil
}

func sameBatchEnvelope(left, right auditstore.Item) bool {
	return left.WorkflowRole == right.WorkflowRole && left.ApprovalKind == right.ApprovalKind &&
		left.ApprovalDigest == right.ApprovalDigest
}

func sameItemParameterEnvelope(
	binding config.ResolvedAuditWorkflowBinding,
	left auditstore.Item,
	right auditstore.Item,
) bool {
	for _, mapping := range binding.Parameters {
		if mapping.Source != config.AuditParameterItemField {
			continue
		}
		switch mapping.Name {
		case "itemKey":
			if left.ItemKey != right.ItemKey {
				return false
			}
		case "subjectKey":
			if left.SubjectKey != right.SubjectKey {
				return false
			}
		case "kind":
			if left.Kind != right.Kind {
				return false
			}
		default:
			return false
		}
	}
	return true
}
