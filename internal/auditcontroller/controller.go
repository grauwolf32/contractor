package auditcontroller

import (
	"context"
	"errors"
	"fmt"
	"sort"
	"strings"
	"time"

	"github.com/grauwolf32/contractor/internal/agentskills"
	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/claimpass"
	"github.com/grauwolf32/contractor/internal/config/auditprofile"
	"github.com/grauwolf32/contractor/internal/config/resolved"
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
	audit := snapshot.Audit
	// A non-nil phase result stops this pass, including retryable no-op outcomes.
	if result := c.reconcileReviewAndDeadline(ctx, claim, snapshot); result != nil {
		return result.changed, result.err
	}
	// A batched drain phase ends the step when it changed anything: later
	// phases need a fresh snapshot of the revision it advanced (AUX-R97).
	if changed, err := c.observeTerminalBatch(ctx, claim, snapshot); changed || err != nil {
		return changed, err
	}
	if result := c.collectOne(ctx, claim, snapshot); result != nil {
		return result.changed, result.err
	}
	if audit.State == auditstore.AuditActive {
		if changed, err := c.resumeOneIntent(ctx, claim, snapshot); changed || err != nil {
			return changed, err
		}
	}

	if result := c.progressRound(ctx, claim, snapshot); result != nil {
		return result.changed, result.err
	}
	closed := audit.Dispatch == auditstore.DispatchClosed ||
		audit.State == auditstore.AuditCancelling || audit.State == auditstore.AuditFinalizing ||
		audit.State == auditstore.AuditDeleting
	if closed {
		return c.settleClosedAudit(ctx, claim, snapshot)
	}
	if result := c.progressPreparation(ctx, claim, snapshot); result != nil {
		return result.changed, result.err
	}
	return c.dispatchReadyBatch(ctx, claim, snapshot)
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

func (c *Controller) reconcileRolePhase(
	ctx context.Context,
	claim auditstore.ControllerClaim,
	snapshot auditstore.ReconcileSnapshot,
	kind auditstore.ExecutionRole,
) (changed bool, complete bool, reason *auditstore.StopReason, err error) {
	profile, decodeErr := auditprofile.DecodeResolvedAuditProfileSnapshot(snapshot.Audit.ProfileSnapshot)
	if decodeErr != nil || (kind != auditstore.ExecutionPrepare && snapshot.Round == nil) {
		return false, false, &auditstore.StopReason{
			Code: "role_contract_invalid", Message: "The pinned Audit role configuration is invalid.",
		}, nil
	}
	roles := roleNames(profile, kind)
	if len(roles) == 0 {
		return false, true, nil, nil
	}
	pendingDependency := false
	for _, workflowRole := range roles {
		latest, found := latestRoleExecution(snapshot.RoleExecutions, kind, workflowRole)
		if found {
			if latest.State != auditstore.ExecutionCollected {
				if changed, handled, resumeErr := c.resumeRoundResultsInput(ctx, claim, snapshot, profile.Workflows[workflowRole], latest); handled {
					return changed, false, nil, resumeErr
				}
				return false, false, nil, nil
			}
			disposition, receiptFound := roleExecutionDisposition(snapshot, latest.ExecutionID)
			if !receiptFound {
				return false, false, nil, nil
			}
			if disposition == auditstore.CollectionAccepted {
				continue
			}
			if !roleDispositionRetryable(disposition, receiptErrorCode(snapshot, latest.ExecutionID)) {
				return false, false, &auditstore.StopReason{
					Code: "role_execution_not_retryable",
					Message: fmt.Sprintf(
						"Audit Workflow role %q ended with a non-retryable disposition.", workflowRole,
					),
				}, nil
			}
			maxAttempts := snapshot.Audit.Limits.MaxItemRunAttempts
			if kind == auditstore.ExecutionPrepare {
				maxAttempts = profile.Workflows[workflowRole].MaxRunAttempts
			}
			if latest.RoleAttempt == nil || *latest.RoleAttempt >= maxAttempts {
				return false, false, &auditstore.StopReason{
					Code: "role_attempt_budget_exhausted",
					Message: fmt.Sprintf(
						"Audit Workflow role %q exhausted its bounded execution attempts.", workflowRole,
					),
				}, nil
			}
		}
		if !roleDependenciesSatisfied(profile, snapshot, workflowRole) {
			pendingDependency = true
			continue
		}
		attempt := 1
		if found && latest.RoleAttempt != nil {
			attempt = *latest.RoleAttempt + 1
		}
		if snapshot.Audit.ReservedRunCount >= snapshot.Audit.Limits.MaxSubmittedRuns {
			return false, false, &auditstore.StopReason{
				Code:    "submission_budget_exhausted",
				Message: "The Audit child Run submission budget was exhausted before a required role completed.",
			}, nil
		}
		if !snapshot.DispatchWindowOpen(resolved.WorkflowLaneClass(profile.Workflows[workflowRole].Workflow)) {
			return false, false, nil, nil
		}
		return c.dispatchRole(ctx, claim, snapshot, workflowRole, attempt)
	}
	if pendingDependency {
		return false, false, &auditstore.StopReason{
			Code:    "role_dependency_unresolved",
			Message: "A pinned Audit Workflow role dependency could not be satisfied.",
		}, nil
	}
	return false, true, nil, nil
}

// closeActiveAudit closes dispatch of the active Audit and moves it to
// finalizing with reason. The closing transition cannot itself be closed, so
// the message is bounded to valid UTF-8 within the store's stop reason bound:
// a builder may compose it from a diagnostic of any length, and a refused
// closure would be retried on every pass (AUX-R100).
func (c *Controller) closeActiveAudit(
	ctx context.Context,
	claim auditstore.ControllerClaim,
	audit auditstore.Audit,
	reason *auditstore.StopReason,
) (bool, error) {
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
func roleDispatchInvalid() *auditstore.StopReason {
	return &auditstore.StopReason{
		Code:    "role_dispatch_contract_invalid",
		Message: "Pinned Audit role dispatch data failed deterministic validation.",
	}
}

// dispatchInvalid is the stop reason of a check execution whose pinned
// dispatch data failed deterministic validation, in the submission builder
// or in the store that refused its intent.
func dispatchInvalid() *auditstore.StopReason {
	return &auditstore.StopReason{
		Code: "dispatch_contract_invalid", Message: "Pinned Audit dispatch data failed deterministic validation.",
	}
}

func (c *Controller) dispatchRole(
	ctx context.Context,
	claim auditstore.ControllerClaim,
	snapshot auditstore.ReconcileSnapshot,
	workflowRole string,
	attempt int,
) (bool, bool, *auditstore.StopReason, error) {
	prepared, err := c.builder.WithClaim(claim).PrepareRole(ctx, snapshot, workflowRole, attempt)
	if err != nil {
		if errors.Is(err, errRoundResultsPending) {
			return true, false, nil, nil
		}
		if errors.Is(err, ErrInvalidSubmission) {
			return false, false, roleDispatchInvalid(), nil
		}
		return false, false, nil, err
	}
	prepared.Intent.Claim = claim
	execution, _, err := c.store.CreateExecutionIntent(ctx, prepared.Intent)
	if errors.Is(err, auditstore.ErrPrecondition) {
		return false, false, nil, nil
	}
	if c.refusedWrite(snapshot.Audit, "create role execution intent", err) {
		return false, false, roleDispatchInvalid(), nil
	}
	if err != nil {
		return false, false, nil, err
	}
	if execution.State != auditstore.ExecutionIntent {
		return false, false, nil, fmt.Errorf(
			"new Audit role execution %q has unexpected state %q", execution.ExecutionID, execution.State,
		)
	}
	changed, err := c.createRun(ctx, claim, prepared)
	return changed, false, nil, err
}

func roleNames(profile auditprofile.ResolvedAuditProfile, kind auditstore.ExecutionRole) []string {
	result := make([]string, 0)
	for name, binding := range profile.Workflows {
		if auditstore.ExecutionRole(binding.Kind) == kind {
			result = append(result, name)
		}
	}
	sort.Strings(result)
	return result
}

func latestRoleExecution(
	executions []auditstore.Execution, kind auditstore.ExecutionRole, workflowRole string,
) (auditstore.Execution, bool) {
	var result auditstore.Execution
	found := false
	for _, execution := range executions {
		if execution.Role != kind || execution.WorkflowRole != workflowRole || execution.RoleAttempt == nil {
			continue
		}
		if !found || result.RoleAttempt == nil || *execution.RoleAttempt > *result.RoleAttempt {
			result, found = execution, true
		}
	}
	return result, found
}

func executionDisposition(
	receipts []auditstore.CollectionReceiptSummary, executionID string,
) (auditstore.CollectionDisposition, bool) {
	for _, receipt := range receipts {
		if receipt.ExecutionID == executionID {
			return receipt.Disposition, true
		}
	}
	return "", false
}

func roleExecutionDisposition(
	snapshot auditstore.ReconcileSnapshot, executionID string,
) (auditstore.CollectionDisposition, bool) {
	if disposition, found := executionDisposition(snapshot.RoleReceipts, executionID); found {
		return disposition, true
	}
	return executionDisposition(snapshot.Receipts, executionID)
}

func receiptErrorCode(snapshot auditstore.ReconcileSnapshot, executionID string) *string {
	for _, receipts := range [][]auditstore.CollectionReceiptSummary{snapshot.RoleReceipts, snapshot.Receipts} {
		for _, receipt := range receipts {
			if receipt.ExecutionID == executionID {
				return receipt.ErrorCode
			}
		}
	}
	return nil
}

func roleDispositionRetryable(disposition auditstore.CollectionDisposition, errorCode *string) bool {
	if disposition == auditstore.CollectionExecutionFailed || disposition == auditstore.CollectionMissingOutput {
		return true
	}
	return disposition == auditstore.CollectionInvalidResult &&
		(errorCode == nil || *errorCode != "evidence-budget-exhausted")
}

func roleDependenciesSatisfied(
	profile auditprofile.ResolvedAuditProfile,
	snapshot auditstore.ReconcileSnapshot,
	workflowRole string,
) bool {
	binding := profile.Workflows[workflowRole]
	for _, mapping := range binding.Inputs {
		if mapping.Source != auditprofile.AuditInputFromRetainedOutput && mapping.Source != auditprofile.AuditInputFromPreparation {
			continue
		}
		source, exists := profile.Workflows[mapping.Role]
		if !exists {
			return false
		}
		execution, found := latestRoleExecution(
			snapshot.RoleExecutions, auditstore.ExecutionRole(source.Kind), mapping.Role,
		)
		if !found || execution.State != auditstore.ExecutionCollected {
			return false
		}
		disposition, found := roleExecutionDisposition(snapshot, execution.ExecutionID)
		if !found || disposition != auditstore.CollectionAccepted {
			return false
		}
	}
	return true
}

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

func (c *Controller) dispatchClosureReason(
	snapshot auditstore.ReconcileSnapshot,
) *auditstore.StopReason {
	audit := snapshot.Audit
	if audit.DeadlineAt != nil && !c.now().Before(*audit.DeadlineAt) {
		return &auditstore.StopReason{
			Code: "deadline_exhausted", Message: "The Audit time limit was reached. Existing Runs can finish; extend or disable the limit to continue.",
		}
	}
	if (audit.Phase == auditdomain.AuditPhasePreparing || audit.Phase == auditdomain.AuditPhaseInventory) &&
		snapshot.Round == nil && audit.CurrentRoundID == nil {
		return nil
	}
	if audit.Limits.BatchSize < 1 || audit.Limits.BatchSize > auditstore.MaxCollectionItems ||
		snapshot.Round == nil || audit.CurrentRoundID == nil ||
		snapshot.Round.RoundID != *audit.CurrentRoundID {
		return &auditstore.StopReason{
			Code: "controller_contract_invalid", Message: "The active Audit is outside the one-round Controller contract.",
		}
	}
	if audit.ReservedRunCount >= audit.Limits.MaxSubmittedRuns && len(snapshot.ReadyItems) != 0 {
		return &auditstore.StopReason{
			Code: "submission_budget_exhausted", Message: "The Audit child Run submission budget was exhausted.",
		}
	}
	return nil
}

// drainBatch counts the mutations of one batched drain phase of a reconcile
// step against the fixed per-step bound (AUX-R97, LIM-AUX-14). Snapshot rows
// are only candidates: every mutation keeps its own store-side claim, state
// and Run checks, and the first error, including a lost claim or a stale
// precondition, ends the batch with the mutations already committed.
type drainBatch struct {
	limit     int
	committed int
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

func (c *Controller) resumeOneIntent(
	ctx context.Context,
	claim auditstore.ControllerClaim,
	snapshot auditstore.ReconcileSnapshot,
) (bool, error) {
	for _, execution := range snapshot.Executions {
		if execution.State != auditstore.ExecutionIntent || execution.RunID != nil {
			continue
		}
		members, err := c.reads.ListExecutionItems(ctx, execution.ExecutionID)
		if err != nil {
			return false, err
		}
		var prepared PreparedSubmission
		if execution.Role == auditstore.ExecutionCheck {
			if len(members) == 0 || len(members) > snapshot.Audit.Limits.BatchSize {
				return c.failInvalidIntent(ctx, claim, execution.ExecutionID)
			}
			itemIDs := make([]string, len(members))
			for index, member := range members {
				itemIDs[index] = member.ItemID
			}
			items, readErr := c.reads.ListItemsByIDs(ctx, snapshot.Audit.AuditID, itemIDs)
			if readErr != nil {
				return false, readErr
			}
			itemsByID := make(map[string]auditstore.Item, len(items))
			for _, item := range items {
				itemsByID[item.ItemID] = item
			}
			selected := make([]CheckExecutionMember, len(members))
			for index, member := range members {
				item, exists := itemsByID[member.ItemID]
				if !exists {
					return c.failInvalidIntent(ctx, claim, execution.ExecutionID)
				}
				selected[index] = CheckExecutionMember{Item: item, Attempt: member.ItemAttempt}
			}
			prepared, err = c.builder.WithClaim(claim).PrepareBatch(ctx, snapshot, selected)
			if err == nil && !matchingIntent(execution, members, prepared.Intent) {
				err = ErrInvalidSubmission
			}
		} else {
			if len(members) != 0 || execution.RoleAttempt == nil {
				return c.failInvalidIntent(ctx, claim, execution.ExecutionID)
			}
			prepared, err = c.builder.WithClaim(claim).PrepareRole(
				ctx, snapshot, execution.WorkflowRole, *execution.RoleAttempt,
			)
			if err == nil && !matchingRoleIntent(execution, prepared.Intent) {
				err = ErrInvalidSubmission
			}
		}
		if err != nil {
			if errors.Is(err, ErrInvalidSubmission) {
				return c.failInvalidIntent(ctx, claim, execution.ExecutionID)
			}
			return false, err
		}
		return c.createRun(ctx, claim, prepared)
	}
	return false, nil
}

func matchingRoleIntent(
	execution auditstore.Execution, prepared auditstore.CreateExecutionIntentParams,
) bool {
	roundMatches := execution.RoundID == nil && prepared.RoundID == nil ||
		execution.RoundID != nil && prepared.RoundID != nil && *execution.RoundID == *prepared.RoundID
	preparationMatches := execution.Preparation == nil && prepared.Preparation == nil ||
		execution.Preparation != nil && prepared.Preparation != nil &&
			sameExactInputMap(execution.Preparation.Inputs, prepared.Preparation.Inputs) &&
			sameParameters(execution.Preparation.Parameters, prepared.Preparation.Parameters)
	return execution.ExecutionID == prepared.ExecutionID && execution.Role == prepared.Role &&
		execution.WorkflowRole == prepared.WorkflowRole && roundMatches && preparationMatches &&
		execution.RoleAttempt != nil && prepared.RoleAttempt != nil &&
		*execution.RoleAttempt == *prepared.RoleAttempt && execution.SubmissionKey == prepared.SubmissionKey &&
		execution.RequestDigest == prepared.RequestDigest && execution.Manifest.Digest == prepared.Manifest.Digest &&
		execution.Manifest.Ref.SameExact(prepared.Manifest.Ref) && len(prepared.Members) == 0
}

func (c *Controller) dispatch(
	ctx context.Context,
	claim auditstore.ControllerClaim,
	snapshot auditstore.ReconcileSnapshot,
	selected []CheckExecutionMember,
) (bool, error) {
	prepared, err := c.builder.WithClaim(claim).PrepareBatch(ctx, snapshot, selected)
	if err != nil {
		if errors.Is(err, ErrInvalidSubmission) {
			return c.closeActiveAudit(ctx, claim, snapshot.Audit, dispatchInvalid())
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
	if c.refusedWrite(snapshot.Audit, "create check execution intent", err) {
		return c.closeActiveAudit(ctx, claim, snapshot.Audit, dispatchInvalid())
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

func matchingIntent(
	execution auditstore.Execution,
	members []auditstore.ExecutionItem,
	prepared auditstore.CreateExecutionIntentParams,
) bool {
	if execution.ExecutionID != prepared.ExecutionID || execution.Role != prepared.Role ||
		execution.WorkflowRole != prepared.WorkflowRole ||
		execution.RoundID == nil || prepared.RoundID == nil || *execution.RoundID != *prepared.RoundID ||
		execution.SubmissionKey != prepared.SubmissionKey || execution.RequestDigest != prepared.RequestDigest ||
		execution.Manifest.Digest != prepared.Manifest.Digest || !execution.Manifest.Ref.SameExact(prepared.Manifest.Ref) ||
		len(members) != len(prepared.Members) {
		return false
	}
	for index, member := range members {
		candidate := prepared.Members[index]
		if member.ExecutionItemID != candidate.ExecutionItemID || member.ItemID != candidate.ItemID ||
			member.BatchOrdinal != candidate.BatchOrdinal || member.BatchOrdinal != index ||
			member.ItemAttempt != candidate.ItemAttempt || member.Task.Digest != candidate.Task.Digest ||
			!member.Task.Ref.SameExact(candidate.Task.Ref) || !sameExactArtifacts(member.Inputs, candidate.Inputs) {
			return false
		}
	}
	return true
}

func sameExactArtifacts(left, right []auditstore.ExactArtifact) bool {
	if len(left) != len(right) {
		return false
	}
	for index := range left {
		if left[index].Digest != right[index].Digest || left[index].MediaType != right[index].MediaType ||
			left[index].SizeBytes != right[index].SizeBytes || !left[index].Ref.SameExact(right[index].Ref) {
			return false
		}
	}
	return true
}

func auditSettlementBarrier(snapshot auditstore.ReconcileSnapshot) bool {
	return snapshot.Audit.OutstandingRunCount == 0 && len(snapshot.Items) == 0 &&
		len(snapshot.Executions) == 0 && !snapshot.MoreItems && !snapshot.MoreExecutions &&
		snapshot.Audit.Hold != auditstore.HoldHeld
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
		artifacts.IsExactReferenceRejection(err) || artifacts.IsExactRevisionMissing(err) ||
		errors.Is(err, artifacts.ErrHistoryProtected) || errors.Is(err, artifacts.ErrArtifactConflict) ||
		errors.Is(err, artifacts.ErrRunRevisionLimit) || errors.Is(err, artifacts.ErrArtifactFrozen) ||
		errors.Is(err, artifacts.ErrReservedNamespace) || errors.Is(err, artifacts.ErrInvalidMediaType) ||
		errors.Is(err, artifacts.ErrPayloadTooLarge) || errors.Is(err, artifacts.ErrVersionedWriteTarget) ||
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

func deadlineTarget(reason *auditstore.StopReason) auditstore.AuditState {
	if reason != nil && reason.Code == "deadline_exhausted" {
		return auditstore.AuditPaused
	}
	return auditstore.AuditFinalizing
}
