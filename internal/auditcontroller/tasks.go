package auditcontroller

import (
	"context"
	"errors"

	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/config/auditprofile"
	"github.com/grauwolf32/contractor/internal/runservice"
)

// reconcileTasks uses the existing claim pass, RunService and importer. Work
// selection is distinct from a Run's technical state; every selected attempt
// remains an ordinary immutable execution/membership association.
func (c *Controller) reconcileTasks(ctx context.Context, claim auditstore.ControllerClaim, snapshot auditstore.ReconcileSnapshot, pinned auditprofile.PinnedProgram) (bool, error) {
	audit := snapshot.Audit
	var programErr error
	pinned.Program, programErr = auditdomain.ExecutionProgram(pinned.Program, audit.BaselineSnapshot)
	if programErr != nil {
		return false, programErr
	}
	if audit.State == auditstore.AuditActive && audit.DeadlineAt != nil && !c.now().Before(*audit.DeadlineAt) {
		_, err := c.store.TransitionClaimed(ctx, auditstore.ClaimedTransitionParams{Claim: claim, ExpectedRevision: audit.Revision, ExpectedState: auditstore.AuditActive, TargetState: auditstore.AuditPaused, Reason: &auditstore.StopReason{Code: "deadline_exhausted", Message: "The Audit time limit was reached. Existing Runs can finish; extend or disable the limit to continue."}})
		return err == nil, err
	}
	if changed, err := c.observeTerminalBatch(ctx, claim, snapshot); changed || err != nil {
		return changed, err
	}
	if result := c.collectOne(ctx, claim, snapshot); result != nil {
		return result.changed, result.err
	}
	if audit.State == auditstore.AuditPaused {
		// Unbound intents survive Pause. They already reserve an attempt and
		// resume with the original identity; no empty paused Audit completes.
		return false, nil
	}
	if audit.State == auditstore.AuditCancelling || audit.State == auditstore.AuditFinalizing || audit.State == auditstore.AuditDeleting {
		return c.settleClosedTasks(ctx, claim, snapshot)
	}
	if audit.State != auditstore.AuditActive {
		return false, auditstore.ErrPrecondition
	}
	// A reserved intent consumes the Run budget already. Reconcile it before
	// deciding whether the budget permits another reservation.
	for _, execution := range snapshot.Executions {
		if execution.State == auditstore.ExecutionIntent && execution.RunID == nil {
			return c.createTaskRun(ctx, claim, execution)
		}
	}
	if audit.InitialHypothesisOffset < len(pinned.Program.Hypotheses) || audit.InitialWorkOffset < len(pinned.Program.InitialWork) && auditstore.RemainingTaskCapacity(audit) > 0 {
		if c.options.TaskIntake == nil {
			return false, errors.New("Audit task intake is not configured")
		}
		result, err := c.options.TaskIntake.AcceptInitialTasks(ctx, claim, snapshot)
		if errors.Is(err, auditstore.ErrPrecondition) {
			return false, nil
		}
		if permanentSubmissionError(err) {
			return c.closeActiveAudit(ctx, claim, audit, &auditstore.StopReason{Code: "initial-work-invalid", Message: "The pinned initial work cannot be accepted."})
		}
		if err != nil || result.Changed {
			return result.Changed, err
		}
	}
	if len(snapshot.ReadyTasks) > 0 {
		if audit.ReservedRunCount >= audit.Limits.MaxSubmittedRuns {
			return c.closeActiveAudit(ctx, claim, audit, &auditstore.StopReason{Code: "submission_budget_exhausted", Message: "The Audit child Run submission budget was exhausted."})
		}
		selected := []string{}
		first := snapshot.ReadyTasks[0]
		batch := 1
		for _, mapping := range pinned.Program.Workflows[first.Document.Workflow].Inputs {
			if mapping.Source == "task-context" && pinned.Program.Workflows[first.Document.Workflow].Outputs["proposals"] == "" {
				batch = audit.Limits.BatchSize
				break
			}
		}
		for _, task := range snapshot.ReadyTasks {
			if task.BatchDigest == first.BatchDigest && task.Document.Workflow == first.Document.Workflow {
				selected = append(selected, task.Document.TaskID)
				if len(selected) == batch {
					break
				}
			}
		}
		execution, _, err := c.store.SelectTasks(ctx, auditstore.SelectTasksParams{Claim: claim, TaskIDs: selected})
		if errors.Is(err, auditstore.ErrPrecondition) {
			return false, nil
		}
		if err != nil {
			if c.refusedWrite(audit, "select Tasks", err) {
				return c.closeActiveAudit(ctx, claim, audit, &auditstore.StopReason{Code: "dispatch-contract-invalid", Message: "The accepted work could not satisfy the dispatch contract."})
			}
			return false, err
		}
		return c.createTaskRun(ctx, claim, execution)
	}
	if snapshot.HasUnsettledTasks || snapshot.MoreExecutions || len(snapshot.Executions) != 0 || audit.OutstandingRunCount != 0 {
		return false, nil
	}
	if audit.InitialWorkOffset < len(pinned.Program.InitialWork) {
		return c.closeActiveAudit(ctx, claim, audit, &auditstore.StopReason{Code: "task_budget_exhausted", Message: "The Audit task budget was exhausted before all initial work was accepted."})
	}
	if pinned.Program.Continuation != nil {
		if c.options.TaskIntake == nil {
			return false, errors.New("Audit continuation intake is not configured")
		}
		result, err := c.options.TaskIntake.AcceptContinuation(ctx, claim, snapshot)
		if errors.Is(err, auditstore.ErrPrecondition) {
			return false, nil
		}
		if permanentSubmissionError(err) {
			return c.closeActiveAudit(ctx, claim, audit, &auditstore.StopReason{Code: "continuation-invalid", Message: "The pinned continuation cannot be accepted."})
		}
		if err != nil || result.Changed {
			return result.Changed, err
		}
		if result.GapCode != "" {
			return c.closeActiveAudit(ctx, claim, audit, &auditstore.StopReason{Code: result.GapCode, Message: "The continuation budget was exhausted."})
		}
	}
	return c.closeActiveAudit(ctx, claim, audit, nil)
}

func (c *Controller) createTaskRun(ctx context.Context, claim auditstore.ControllerClaim, execution auditstore.Execution) (bool, error) {
	tasks, err := c.reads.ListExecutionTasks(ctx, claim.AuditID, execution.ExecutionID)
	if err != nil {
		return false, err
	}
	if len(tasks) == 0 || len(tasks) > auditstore.MaxCollectionItems {
		return c.failInvalidIntent(ctx, claim, execution.ExecutionID)
	}
	ids := make([]string, 0, len(tasks))
	for _, task := range tasks {
		ids = append(ids, task.Document.TaskID)
	}
	return c.createRun(ctx, claim, runservice.AuditCreateParams{Claim: claim, ExecutionID: execution.ExecutionID, TaskIDs: ids, RequestDigest: execution.RequestDigest})
}

func (c *Controller) settleClosedTasks(ctx context.Context, claim auditstore.ControllerClaim, snapshot auditstore.ReconcileSnapshot) (bool, error) {
	audit := snapshot.Audit
	if changed, err := c.failUnboundIntentBatch(ctx, claim, snapshot); changed || err != nil {
		return changed, err
	}
	if audit.State == auditstore.AuditCancelling || audit.State == auditstore.AuditDeleting {
		if changed, err := c.requestCancellationBatch(ctx, snapshot); changed || err != nil {
			return changed, err
		}
	}
	if count, err := c.store.SettleUndispatchedTasks(ctx, claim, auditstore.MaxReconcileRows); count != 0 || err != nil {
		return count != 0, err
	}
	if audit.Hold == auditstore.HoldHeld {
		if _, changed, err := c.store.ReleaseDispatchHold(ctx, claim); changed || err != nil {
			return changed, err
		}
	}
	if snapshot.HasUnsettledTasks || snapshot.MoreExecutions || len(snapshot.Executions) != 0 || audit.OutstandingRunCount != 0 {
		return false, nil
	}
	if audit.State == auditstore.AuditDeleting {
		deleted, drained, err := c.deleteReleasedRunBatch(ctx, claim, audit.OwnerID)
		if deleted || !drained || err != nil {
			return deleted, err
		}
		err = c.store.PurgeClaimed(ctx, claim, auditdomain.ArtifactNamespace(audit.AuditID))
		return err == nil, err
	}
	target := auditstore.AuditCompleted
	if audit.State == auditstore.AuditCancelling {
		target = auditstore.AuditCancelled
		if audit.DeletionRequestedAt != nil {
			target = auditstore.AuditDeleting
		}
	}
	_, err := c.store.TransitionClaimed(ctx, auditstore.ClaimedTransitionParams{Claim: claim, ExpectedRevision: audit.Revision, ExpectedState: audit.State, TargetState: target, Reason: audit.StopReason})
	return err == nil, err
}
