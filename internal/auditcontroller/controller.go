package auditcontroller

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/grauwolf32/contractor/internal/agentskills"
	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/artifacts/blob"
	artifactpostgres "github.com/grauwolf32/contractor/internal/artifacts/postgres"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/claimpass"
	"github.com/grauwolf32/contractor/internal/config/auditprofile"
	"github.com/grauwolf32/contractor/internal/contracts"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/runservice"
	"github.com/grauwolf32/contractor/internal/runstore"
)

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
		case <-c.after(c.options.PollInterval):
		}
	}
}

// RunOnce fairly claims a bounded Audit set and reconciles each claim
// independently through the shared claim lifecycle (PER-R26). A failed Audit
// is returned in the joined diagnostic only after every other claimed Audit
// had an opportunity to progress. Each release has the ordinary operation
// budget.
func (c *Controller) RunOnce(ctx context.Context) (bool, error) {
	claimContext, cancelClaim := context.WithTimeout(ctx, c.options.OperationTimeout)
	claims, err := c.store.Claim(claimContext, auditstore.ClaimParams{
		HolderID: c.options.HolderID, Lease: c.options.ClaimLease, Limit: c.options.ClaimBatch,
	})
	cancelClaim()
	if err != nil {
		return false, err
	}
	return claimpass.Run(ctx, claims, claimpass.Lifecycle[auditstore.ControllerClaim]{
		Handle: c.reconcile, Release: c.store.ReleaseClaim,
		ReleaseBudget: c.options.OperationTimeout, Lost: auditstore.ErrClaimLost,
	})
}

func (c *Controller) reconcile(
	ctx context.Context, claim auditstore.ControllerClaim,
) (bool, error) {
	readContext, cancelRead := context.WithTimeout(ctx, c.options.OperationTimeout)
	snapshot, err := c.store.GetReconcileSnapshot(readContext, claim)
	if err != nil {
		cancelRead()
		return false, err
	}
	var cancel context.CancelFunc
	if snapshot.Audit.State == auditstore.AuditDeleting {
		cancelRead()
		renewContext, cancelRenew := context.WithTimeout(ctx, c.options.OperationTimeout)
		err := c.store.RenewClaim(renewContext, claim, deletionClaimLease)
		cancelRenew()
		if err != nil {
			return false, err
		}
		ctx, cancel = persistencepostgres.WithCleanupBudget(ctx)
		defer c.reportLongDeletionStep(claim, c.now())
	} else {
		// Ordinary work keeps sharing one budget with its snapshot read.
		ctx, cancel = readContext, cancelRead
	}
	defer cancel()
	pinned, err := auditprofile.DecodePinnedProgram(snapshot.Audit.ProfileSnapshot)
	if err != nil {
		return false, fmt.Errorf("%w: pinned program is invalid", ErrInvalidSubmission)
	}
	return c.reconcileTasks(ctx, claim, snapshot, pinned)
}

// reportLongDeletionStep logs a deleting Audit's step that outlasted the
// ordinary operation budget. Such a step deletes up to the drain bound of
// child Runs under the cleanup budget, and the Audits claimed with it wait
// for the pass to end; the log is how that accepted risk is detected
// (AUX-R97).
func (c *Controller) reportLongDeletionStep(claim auditstore.ControllerClaim, started time.Time) {
	if elapsed := c.now().Sub(started); elapsed > c.options.OperationTimeout {
		c.options.Logger.Warn("Audit deletion step outlasted the ordinary operation budget",
			"audit_id", claim.AuditID, "duration", elapsed, "operation_timeout", c.options.OperationTimeout)
	}
}

// reconcileResult distinguishes continuing a phase from stopping without a change.
type reconcileResult struct {
	changed bool
	err     error
}

func reconciliationDone(changed bool, err error) *reconcileResult {
	return &reconcileResult{changed: changed, err: err}
}

func (c *Controller) closeActiveAudit(
	ctx context.Context,
	claim auditstore.ControllerClaim,
	audit auditstore.Audit,
	reason *auditstore.StopReason,
) (bool, error) {
	if reason == nil {
		_, err := c.store.TransitionClaimed(ctx, auditstore.ClaimedTransitionParams{Claim: claim, ExpectedRevision: audit.Revision, ExpectedState: auditstore.AuditActive, TargetState: auditstore.AuditFinalizing})
		return err == nil, err
	}
	bounded := *reason
	// PostgreSQL text holds no NUL, so it is replaced like an invalid byte.
	bounded.Message = strings.ReplaceAll(strings.ToValidUTF8(bounded.Message, "\uFFFD"), "\x00", "\uFFFD")
	if len(bounded.Message) > auditstore.MaxStopReasonMessageBytes {
		// Cutting inside a multi-byte character leaves a partial one, which
		// ToValidUTF8 drops.
		bounded.Message = strings.ToValidUTF8(bounded.Message[:auditstore.MaxStopReasonMessageBytes], "")
	}
	_, err := c.store.TransitionClaimed(ctx, auditstore.ClaimedTransitionParams{
		Claim: claim, ExpectedRevision: audit.Revision,
		ExpectedState: auditstore.AuditActive, TargetState: auditstore.AuditFinalizing,
		Reason: &bounded,
	})
	return err == nil, err
}

// refusedWrite reports whether the Audit store refused a Controller write as
// invalid input (auditstore.ErrInvalid): its validation or a named CHECK
// constraint rejected values the write derives from durable Audit state, so
// the same write is refused on every pass and the caller closes the Audit
// instead of retrying it (AUX-R99). A transient failure, such as a
// serialization failure or a lost connection, is never ErrInvalid and stays
// retryable. The stop reason does not carry the store's cause, so the
// refusal is logged with it.
func (c *Controller) refusedWrite(audit auditstore.Audit, operation string, err error) bool {
	if !errors.Is(err, auditstore.ErrInvalid) {
		return false
	}
	c.options.Logger.Error("Audit store refused a Controller write as invalid",
		"audit_id", audit.AuditID, "operation", operation, "error", err)
	return true
}

// roleDispatchInvalid is the stop reason of a role execution whose pinned
// dispatch data failed deterministic validation, in the submission builder
// or in the store that refused its intent.
func (c *Controller) failAuditImport(
	ctx context.Context,
	claim auditstore.ControllerClaim,
	audit auditstore.Audit,
	code string,
) (bool, error) {
	reason := auditstore.StopReason{
		Code: code, Message: "Trusted Audit result processing could not satisfy its durable contract.",
	}
	target := auditstore.AuditFailed
	if audit.State == auditstore.AuditActive {
		target = auditstore.AuditFinalizing
	} else if audit.State != auditstore.AuditFinalizing {
		return false, fmt.Errorf("Audit import failed while Audit %q was %q", audit.AuditID, audit.State)
	}
	_, err := c.store.TransitionClaimed(ctx, auditstore.ClaimedTransitionParams{
		Claim: claim, ExpectedRevision: audit.Revision,
		ExpectedState: audit.State, TargetState: target,
		Reason: &reason,
	})
	return err == nil, err
}

func (c *Controller) newDrainBatch() drainBatch {
	return drainBatch{limit: c.drainLimit}
}

// open reports whether the phase may start another mutation: fewer than the
// bound are committed and, after the first, the step's operation budget has
// not ended. The first mutation runs under that budget like any other step
// mutation and reports its end itself.
func (b *drainBatch) open(ctx context.Context) bool {
	return b.committed < b.limit && (b.committed == 0 || ctx.Err() == nil)
}

func (b *drainBatch) commit() { b.committed++ }

// stop ends the phase with err; the step reports progress when any mutation
// of the batch committed. When the step's budget ended under a later
// mutation, that mutation committed or rolled back whole; each drain mutation
// is idempotent and the next step reads its outcome from a fresh snapshot,
// so the end of the budget stops the batch rather than failing the step.
func (b *drainBatch) stop(ctx context.Context, err error) (bool, error) {
	if err != nil && b.committed > 0 && ctx.Err() != nil {
		return true, nil
	}
	return b.committed > 0, err
}

// observeTerminalBatch records the terminal observation of up to the batch
// bound of submitted executions whose child Run the snapshot shows terminal,
// in snapshot order. A terminal Run state is final, so a candidate cannot
// become ineligible except by its own observation, which the store makes
// idempotent; collection stays a separate step for each execution.
func (c *Controller) observeTerminalBatch(
	ctx context.Context,
	claim auditstore.ControllerClaim,
	snapshot auditstore.ReconcileSnapshot,
) (bool, error) {
	batch := c.newDrainBatch()
	for _, execution := range snapshot.Executions {
		if !batch.open(ctx) {
			break
		}
		if execution.State != auditstore.ExecutionSubmitted || execution.RunID == nil {
			continue
		}
		runID := *execution.RunID
		state, err := childRunState(snapshot, runID)
		if err != nil {
			return batch.stop(ctx, err)
		}
		if !terminalRun(state) {
			continue
		}
		cursor, err := c.runs.GetRunEventCursor(ctx, runID)
		if err != nil {
			return batch.stop(ctx, fmt.Errorf("read terminal Audit child Run cursor %q: %w", runID, err))
		}
		if cursor.Sequence < 1 {
			return batch.stop(ctx, fmt.Errorf("terminal Audit child Run %q has no durable lifecycle event", runID))
		}
		if _, err := c.store.ObserveTerminal(ctx, auditstore.ObserveTerminalParams{
			Claim: claim, ExecutionID: execution.ExecutionID, RunID: runID,
			Generation: cursor.Generation, Sequence: uint64(cursor.Sequence),
		}); err != nil {
			return batch.stop(ctx, err)
		}
		batch.commit()
	}
	return batch.stop(ctx, nil)
}

// childRunState returns the snapshot's lifecycle state of a child Run bound
// to a window execution. The snapshot reads every such state with the
// executions, so terminal observation and cancellation read no Run per
// execution.
func childRunState(snapshot auditstore.ReconcileSnapshot, runID string) (runstore.WorkflowRunState, error) {
	state, found := snapshot.RunStates[runID]
	if !found {
		return "", fmt.Errorf("read Audit child Run %q: %w", runID, runstore.ErrNotFound)
	}
	return state, nil
}

// failUnboundIntentBatch closes up to the batch bound of intents that never
// got a child Run in a closed Audit. Dispatch never reopens once closed, so a
// concurrent owner mutation cannot make a later candidate eligible for
// submission again; a Run bound concurrently makes the store report a stale
// precondition, which ends the batch.
func (c *Controller) failUnboundIntentBatch(
	ctx context.Context,
	claim auditstore.ControllerClaim,
	snapshot auditstore.ReconcileSnapshot,
) (bool, error) {
	batch := c.newDrainBatch()
	for _, execution := range snapshot.Executions {
		if !batch.open(ctx) {
			break
		}
		if execution.State != auditstore.ExecutionIntent || execution.RunID != nil {
			continue
		}
		if _, err := c.store.ObserveSubmissionFailure(ctx, auditstore.ObserveSubmissionFailureParams{
			Claim: claim, ExecutionID: execution.ExecutionID,
		}); err != nil {
			return batch.stop(ctx, err)
		}
		batch.commit()
	}
	return batch.stop(ctx, nil)
}

func (c *Controller) createRun(
	ctx context.Context,
	claim auditstore.ControllerClaim,
	prepared runservice.AuditCreateParams,
) (bool, error) {
	prepared.Claim = claim
	prepared.NewRunID = func() (string, error) { return c.newID("run_") }
	result, err := c.creator.CreateAudit(ctx, prepared)
	if err == nil {
		c.notifier.Wake()
		return result.Created || result.Replayed, nil
	}
	if permanentSubmissionError(err) {
		return c.failInvalidIntent(ctx, claim, prepared.ExecutionID)
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

// requestCancellationBatch requests cancellation of up to the batch bound of
// child Runs that the snapshot shows still running, for a cancelling or
// deleting Audit. A request is idempotent and never stands for the Run's
// terminal transition: each child keeps the settlement barrier closed until
// it is terminal, observed and collected. The Audit cannot leave cancelling
// or deleting for a state that needs its children, and the step's budget
// ends before its claim lease, so no request outlives the claim.
func (c *Controller) requestCancellationBatch(
	ctx context.Context, snapshot auditstore.ReconcileSnapshot,
) (bool, error) {
	batch := c.newDrainBatch()
	for _, execution := range snapshot.Executions {
		if !batch.open(ctx) {
			break
		}
		if execution.State != auditstore.ExecutionSubmitted || execution.RunID == nil {
			continue
		}
		runID := *execution.RunID
		state, err := childRunState(snapshot, runID)
		if err != nil {
			return batch.stop(ctx, err)
		}
		if state != runstore.RunInitializing && state != runstore.RunPending && state != runstore.RunRunning && state != runstore.RunWaiting {
			continue
		}
		reason := "Parent Audit dispatch closed before the child Run completed."
		requestedBy := snapshot.Audit.OwnerID
		cancelled, err := c.runs.RequestRunCancellation(ctx, runID, runstore.WorkflowRunCancellation{
			Code: runstore.CancellationUserRequested, RequestedAt: c.now().UTC().Round(0),
			RequestedBy: &requestedBy, Reason: &reason,
		})
		if err != nil {
			return batch.stop(ctx, err)
		}
		if cancelled.State == runstore.RunCancelling {
			c.notifier.Cancel(runID)
		}
		batch.commit()
	}
	return batch.stop(ctx, nil)
}

func terminalRun(state runstore.WorkflowRunState) bool {
	return runstore.RunLifecycleTerminal.Includes(state)
}

// permanentSubmissionError reports a definite rejection of the pinned
// submission, which no retry of the same intent can change: the Run Service,
// the Run store or the Audit store refused it as invalid, its replay
// conflicts, an exact input is missing or corrupt, or a non-retryable Run
// Skill failure such as a start-pinned Skill source that no longer exists.
// The intent then fails instead of being resumed (AUX-R99, FAIL-AUX-1). Any
// other error leaves the intent to retry.
func permanentSubmissionError(err error) bool {
	var skill *agentskills.RunSkillError
	var document *auditdomain.ValidationError
	return errors.Is(err, runservice.ErrInvalid) || errors.Is(err, runstore.ErrInvalid) ||
		errors.Is(err, auditstore.ErrInvalid) || errors.Is(err, auditstore.ErrConflict) ||
		artifacts.IsExactReferenceRejection(err) || artifactpostgres.IsExactRevisionMissing(err) ||
		errors.Is(err, artifacts.ErrHistoryProtected) || errors.Is(err, artifacts.ErrArtifactConflict) ||
		errors.Is(err, artifacts.ErrRunRevisionLimit) || errors.Is(err, artifacts.ErrArtifactFrozen) ||
		errors.Is(err, artifacts.ErrReservedNamespace) || errors.Is(err, artifacts.ErrInvalidMediaType) ||
		errors.Is(err, blob.ErrTooLarge) || errors.Is(err, artifacts.ErrVersionedWriteTarget) ||
		errors.Is(err, contracts.ErrExactArtifactMismatch) || errors.As(err, &document) ||
		errors.As(err, &skill) && !skill.Retryable
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

type drainBatch struct {
	limit     int
	committed int
}
