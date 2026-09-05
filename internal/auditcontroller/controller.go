package auditcontroller

import (
	"context"
	cryptorand "crypto/rand"
	"encoding/hex"
	"errors"
	"fmt"
	"sync"
	"sync/atomic"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/runservice"
	"github.com/grauwolf32/contractor/internal/runstore"
)

// Wake is an edge-triggered latency hint. Periodic PostgreSQL reconciliation
// remains the authoritative recovery mechanism.
func (c *Controller) Wake() {
	select {
	case c.wake <- struct{}{}:
	default:
	}
}

func (c *Controller) Run(ctx context.Context) error {
	if !c.beginRun() {
		return ErrAlreadyRunning
	}
	defer c.endRun()
	for {
		worked, err := c.RunOnce(ctx)
		if ctx.Err() != nil {
			return nil
		}
		if err != nil {
			c.options.Logger.Error("Audit Controller iteration failed", "error", err)
		}
		if worked && err == nil {
			continue
		}
		select {
		case <-ctx.Done():
			return nil
		case <-c.wake:
		case <-c.after(c.options.PollInterval):
		}
	}
}

// RunOnce fairly claims a bounded Audit set and reconciles each claim
// independently. A failed Audit is returned in the joined diagnostic only
// after every other claimed Audit had an opportunity to progress.
func (c *Controller) RunOnce(ctx context.Context) (bool, error) {
	claimContext, cancelClaim := context.WithTimeout(ctx, c.options.OperationTimeout)
	claims, err := c.store.Claim(claimContext, auditstore.ClaimParams{
		HolderID: c.options.HolderID, Lease: c.options.ClaimLease, Limit: c.options.ClaimBatch,
	})
	cancelClaim()
	if err != nil {
		return false, err
	}
	if len(claims) == 0 {
		return false, nil
	}
	var worked atomic.Bool
	errorsByClaim := make([]error, len(claims))
	var wait sync.WaitGroup
	for index, claim := range claims {
		wait.Add(1)
		go func(index int, claim auditstore.ControllerClaim) {
			defer wait.Done()
			operationContext, cancel := context.WithTimeout(ctx, c.options.OperationTimeout)
			changed, reconcileErr := c.reconcile(operationContext, claim)
			cancel()
			if changed {
				worked.Store(true)
			}
			releaseContext, releaseCancel := context.WithTimeout(context.Background(), c.options.OperationTimeout)
			releaseErr := c.store.ReleaseClaim(releaseContext, claim)
			releaseCancel()
			if errors.Is(releaseErr, auditstore.ErrClaimLost) {
				releaseErr = nil
			}
			errorsByClaim[index] = errors.Join(reconcileErr, releaseErr)
		}(index, claim)
	}
	wait.Wait()
	return worked.Load(), errors.Join(errorsByClaim...)
}

func (c *Controller) reconcile(
	ctx context.Context, claim auditstore.ControllerClaim,
) (bool, error) {
	snapshot, err := c.store.GetReconcileSnapshot(ctx, claim)
	if err != nil {
		return false, err
	}
	audit := snapshot.Audit

	if audit.State == auditstore.AuditActive {
		if reason := c.dispatchClosureReason(snapshot); reason != nil {
			_, err := c.store.TransitionClaimed(ctx, auditstore.ClaimedTransitionParams{
				Claim: claim, ExpectedRevision: audit.Revision,
				ExpectedState: auditstore.AuditActive, TargetState: auditstore.AuditFinalizing,
				Reason: reason,
			})
			return err == nil, err
		}
		if snapshot.Round != nil && snapshot.Round.State == auditstore.RoundAccepted {
			_, err := c.store.TransitionRound(ctx, auditstore.RoundTransitionParams{
				Claim: claim, RoundID: snapshot.Round.RoundID,
				ExpectedRevision: snapshot.Round.Revision,
				ExpectedState:    auditstore.RoundAccepted, TargetState: auditstore.RoundExecuting,
			})
			return err == nil, err
		}
	}

	if changed, err := c.observeOneTerminal(ctx, claim, snapshot); changed || err != nil {
		return changed, err
	}

	closed := audit.Dispatch == auditstore.DispatchClosed ||
		audit.State == auditstore.AuditCancelling || audit.State == auditstore.AuditFinalizing
	if closed {
		if changed, err := c.failOneUnboundIntent(ctx, claim, snapshot); changed || err != nil {
			return changed, err
		}
		if audit.State == auditstore.AuditCancelling || deadlineClosure(audit) {
			return c.cancelOneSubmittedRun(ctx, snapshot)
		}
		return false, nil
	}
	if audit.State != auditstore.AuditActive || snapshot.Round == nil ||
		snapshot.Round.State != auditstore.RoundExecuting {
		return false, nil
	}

	if changed, err := c.resumeOneIntent(ctx, claim, snapshot); changed || err != nil {
		return changed, err
	}
	for _, item := range snapshot.Items {
		if item.State != auditstore.ItemReady || item.RoundID != snapshot.Round.RoundID {
			continue
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
		return c.dispatch(ctx, claim, snapshot, item, attempt)
	}
	return false, nil
}

func (c *Controller) dispatchClosureReason(
	snapshot auditstore.ReconcileSnapshot,
) *auditstore.StopReason {
	audit := snapshot.Audit
	if audit.DeadlineAt != nil && !c.now().Before(*audit.DeadlineAt) {
		return &auditstore.StopReason{
			Code: "deadline_exhausted", Message: "The Audit wall-time deadline was reached; no new Runs may be submitted.",
		}
	}
	if audit.Limits.BatchSize != 1 || snapshot.Round == nil || audit.CurrentRoundID == nil ||
		snapshot.Round.RoundID != *audit.CurrentRoundID {
		return &auditstore.StopReason{
			Code: "controller_contract_invalid", Message: "The active Audit is outside the one-round Controller contract.",
		}
	}
	if audit.ReservedRunCount >= audit.Limits.MaxSubmittedRuns {
		for _, item := range snapshot.Items {
			if item.State == auditstore.ItemReady {
				return &auditstore.StopReason{
					Code: "submission_budget_exhausted", Message: "The Audit child Run submission budget was exhausted.",
				}
			}
		}
	}
	return nil
}

func (c *Controller) observeOneTerminal(
	ctx context.Context,
	claim auditstore.ControllerClaim,
	snapshot auditstore.ReconcileSnapshot,
) (bool, error) {
	for _, execution := range snapshot.Executions {
		if execution.State != auditstore.ExecutionSubmitted || execution.RunID == nil {
			continue
		}
		run, err := c.runs.GetRun(ctx, *execution.RunID)
		if err != nil {
			return false, fmt.Errorf("read Audit child Run %q: %w", *execution.RunID, err)
		}
		if !terminalRun(run.State) {
			continue
		}
		cursor, err := c.runs.GetRunEventCursor(ctx, run.RunID)
		if err != nil {
			return false, fmt.Errorf("read terminal Audit child Run cursor %q: %w", run.RunID, err)
		}
		if cursor.Sequence < 1 {
			return false, fmt.Errorf("terminal Audit child Run %q has no durable lifecycle event", run.RunID)
		}
		_, err = c.store.ObserveTerminal(ctx, auditstore.ObserveTerminalParams{
			Claim: claim, ExecutionID: execution.ExecutionID, RunID: run.RunID,
			Generation: cursor.Generation, Sequence: uint64(cursor.Sequence),
		})
		return err == nil, err
	}
	return false, nil
}

func (c *Controller) failOneUnboundIntent(
	ctx context.Context,
	claim auditstore.ControllerClaim,
	snapshot auditstore.ReconcileSnapshot,
) (bool, error) {
	for _, execution := range snapshot.Executions {
		if execution.State == auditstore.ExecutionIntent && execution.RunID == nil {
			_, err := c.store.ObserveSubmissionFailure(ctx, auditstore.ObserveSubmissionFailureParams{
				Claim: claim, ExecutionID: execution.ExecutionID,
			})
			return err == nil, err
		}
	}
	return false, nil
}

func (c *Controller) resumeOneIntent(
	ctx context.Context,
	claim auditstore.ControllerClaim,
	snapshot auditstore.ReconcileSnapshot,
) (bool, error) {
	for _, execution := range snapshot.Executions {
		if execution.State != auditstore.ExecutionIntent || execution.RunID != nil {
			continue
		}
		members, err := c.store.ListExecutionItems(ctx, execution.ExecutionID)
		if err != nil {
			return false, err
		}
		if len(members) != 1 {
			return c.failInvalidIntent(ctx, claim, execution.ExecutionID)
		}
		item, exists := snapshotItem(snapshot.Items, members[0].ItemID)
		if !exists {
			return c.failInvalidIntent(ctx, claim, execution.ExecutionID)
		}
		prepared, err := c.builder.Prepare(ctx, snapshot, item, members[0].ItemAttempt)
		if err != nil {
			if errors.Is(err, ErrInvalidSubmission) {
				return c.failInvalidIntent(ctx, claim, execution.ExecutionID)
			}
			return false, err
		}
		if !matchingIntent(execution, members[0], prepared.Intent) {
			return c.failInvalidIntent(ctx, claim, execution.ExecutionID)
		}
		return c.createRun(ctx, claim, prepared)
	}
	return false, nil
}

func (c *Controller) dispatch(
	ctx context.Context,
	claim auditstore.ControllerClaim,
	snapshot auditstore.ReconcileSnapshot,
	item auditstore.Item,
	attempt int,
) (bool, error) {
	prepared, err := c.builder.Prepare(ctx, snapshot, item, attempt)
	if err != nil {
		if errors.Is(err, ErrInvalidSubmission) {
			reason := auditstore.StopReason{
				Code: "dispatch_contract_invalid", Message: "Pinned Audit dispatch data failed deterministic validation.",
			}
			_, transitionErr := c.store.TransitionClaimed(ctx, auditstore.ClaimedTransitionParams{
				Claim: claim, ExpectedRevision: snapshot.Audit.Revision,
				ExpectedState: auditstore.AuditActive, TargetState: auditstore.AuditFinalizing,
				Reason: &reason,
			})
			return transitionErr == nil, transitionErr
		}
		return false, err
	}
	prepared.Intent.Claim = claim
	execution, _, err := c.store.CreateExecutionIntent(ctx, prepared.Intent)
	if errors.Is(err, auditstore.ErrPrecondition) {
		// The derived dispatch window, deadline, budget, item state, or a
		// concurrent fence won. A later authoritative snapshot decides which.
		return false, nil
	}
	if err != nil {
		return false, err
	}
	if execution.State != auditstore.ExecutionIntent {
		return false, fmt.Errorf("new Audit execution %q has unexpected state %q", execution.ExecutionID, execution.State)
	}
	return c.createRun(ctx, claim, prepared)
}

func (c *Controller) createRun(
	ctx context.Context,
	claim auditstore.ControllerClaim,
	prepared PreparedSubmission,
) (bool, error) {
	prepared.Run.Claim = claim
	prepared.Run.NewRunID = func() (string, error) { return c.newID("run_") }
	result, err := c.creator.CreateAudit(ctx, prepared.Run)
	if err == nil {
		c.notifier.Wake()
		return result.Created || result.Replayed, nil
	}
	if permanentSubmissionError(err) {
		return c.failInvalidIntent(ctx, claim, prepared.Intent.ExecutionID)
	}
	return false, err
}

func (c *Controller) failInvalidIntent(
	ctx context.Context, claim auditstore.ControllerClaim, executionID string,
) (bool, error) {
	_, err := c.store.ObserveSubmissionFailure(ctx, auditstore.ObserveSubmissionFailureParams{
		Claim: claim, ExecutionID: executionID,
	})
	return err == nil, err
}

func (c *Controller) cancelOneSubmittedRun(
	ctx context.Context, snapshot auditstore.ReconcileSnapshot,
) (bool, error) {
	for _, execution := range snapshot.Executions {
		if execution.State != auditstore.ExecutionSubmitted || execution.RunID == nil {
			continue
		}
		run, err := c.runs.GetRun(ctx, *execution.RunID)
		if err != nil {
			return false, err
		}
		if run.State != runstore.RunInitializing && run.State != runstore.RunRunning {
			continue
		}
		reason := "Parent Audit dispatch closed before the child Run completed."
		requestedBy := snapshot.Audit.OwnerID
		cancelled, err := c.runs.RequestRunCancellation(ctx, run.RunID, runstore.WorkflowRunCancellation{
			Code: runstore.CancellationUserRequested, RequestedAt: c.now().UTC().Round(0),
			RequestedBy: &requestedBy, Reason: &reason,
		})
		if err != nil {
			return false, err
		}
		if cancelled.State == runstore.RunCancelling {
			c.notifier.Cancel(run.RunID)
		}
		return true, nil
	}
	return false, nil
}

func matchingIntent(
	execution auditstore.Execution,
	member auditstore.ExecutionItem,
	prepared auditstore.CreateExecutionIntentParams,
) bool {
	return execution.ExecutionID == prepared.ExecutionID && execution.Role == prepared.Role &&
		execution.RoundID != nil && prepared.RoundID != nil && *execution.RoundID == *prepared.RoundID &&
		execution.SubmissionKey == prepared.SubmissionKey && execution.RequestDigest == prepared.RequestDigest &&
		execution.Manifest.Digest == prepared.Manifest.Digest && sameExactRef(execution.Manifest.Ref, prepared.Manifest.Ref) &&
		len(prepared.Members) == 1 && member.ExecutionItemID == prepared.Members[0].ExecutionItemID &&
		member.ItemID == prepared.Members[0].ItemID && member.BatchOrdinal == 0 &&
		member.ItemAttempt == prepared.Members[0].ItemAttempt && member.Task.Digest == prepared.Members[0].Task.Digest &&
		sameExactRef(member.Task.Ref, prepared.Members[0].Task.Ref)
}

func snapshotItem(items []auditstore.Item, itemID string) (auditstore.Item, bool) {
	for _, item := range items {
		if item.ItemID == itemID {
			return item, true
		}
	}
	return auditstore.Item{}, false
}

func deadlineClosure(audit auditstore.Audit) bool {
	return audit.State == auditstore.AuditFinalizing && audit.StopReason != nil &&
		audit.StopReason.Code == "deadline_exhausted"
}

func terminalRun(state runstore.WorkflowRunState) bool {
	return state == runstore.RunSucceeded || state == runstore.RunFailed || state == runstore.RunCancelled
}

func permanentSubmissionError(err error) bool {
	return errors.Is(err, runservice.ErrInvalid) || errors.Is(err, auditstore.ErrConflict) ||
		errors.Is(err, artifacts.ErrArtifactNotFound) || errors.Is(err, artifacts.ErrArtifactIntegrity)
}

func (c *Controller) beginRun() bool {
	c.runMu.Lock()
	defer c.runMu.Unlock()
	if c.running {
		return false
	}
	c.running = true
	return true
}

func (c *Controller) endRun() {
	c.runMu.Lock()
	c.running = false
	c.runMu.Unlock()
}

func (c *Controller) now() time.Time {
	c.clockMu.Lock()
	defer c.clockMu.Unlock()
	return c.options.Clock.Now()
}

func (c *Controller) after(duration time.Duration) <-chan time.Time {
	c.clockMu.Lock()
	defer c.clockMu.Unlock()
	return c.options.Clock.After(duration)
}

func (c *Controller) newID(prefix string) (string, error) {
	c.idMu.Lock()
	defer c.idMu.Unlock()
	return c.options.NewID(prefix)
}

func randomID(prefix string) (string, error) {
	var raw [16]byte
	if _, err := cryptorand.Read(raw[:]); err != nil {
		return "", err
	}
	return prefix + hex.EncodeToString(raw[:]), nil
}
