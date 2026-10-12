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
	if snapshot.MaxConcurrentRuns < 1 {
		return false, errors.New("Audit dispatch window is invalid")
	}
	if audit.OutstandingRunCount >= snapshot.MaxConcurrentRuns {
		return false, nil
	}

	candidates := batchCandidates(snapshot)
	if len(candidates) == 0 {
		return false, nil
	}
	itemIDs := make([]string, len(candidates))
	for index, item := range candidates {
		itemIDs[index] = item.ItemID
	}
	attempts, err := c.store.NextItemAttempts(ctx, claim, itemIDs)
	if err != nil {
		return false, err
	}
	selected := make([]CheckExecutionMember, 0, len(candidates))
	for _, item := range candidates {
		attempt, ready := attempts[item.ItemID]
		if !ready {
			// The item left ready after the snapshot; a later authoritative
			// snapshot decides the batch.
			return false, nil
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
	}
	return c.dispatch(ctx, claim, snapshot, selected)
}

// batchCandidates is the ordered check batch the next dispatch may reserve:
// the lowest-ordinal ready item of the current Round and the later ready items
// that share its batch and parameter envelope, up to the batch size. Their
// next attempts are read together, so selecting a batch costs one statement
// whatever its size.
func batchCandidates(snapshot auditstore.ReconcileSnapshot) []auditstore.Item {
	audit := snapshot.Audit
	limit := min(audit.Limits.BatchSize, auditstore.MaxCollectionItems)
	if limit < 1 {
		return nil
	}
	candidates := make([]auditstore.Item, 0, limit)
	var binding config.ResolvedAuditWorkflowBinding
	profile, profileErr := config.DecodeResolvedAuditProfileSnapshot(audit.ProfileSnapshot)
	for _, item := range snapshot.ReadyItems {
		if item.State != auditstore.ItemReady || item.RoundID != snapshot.Round.RoundID {
			continue
		}
		if len(candidates) != 0 && (!sameBatchEnvelope(candidates[0], item) ||
			profileErr == nil && !sameItemParameterEnvelope(binding, candidates[0], item)) {
			continue
		}
		if len(candidates) == 0 && profileErr == nil {
			binding = profile.Workflows[item.WorkflowRole]
		}
		candidates = append(candidates, item)
		if len(candidates) == limit {
			break
		}
	}
	return candidates
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
