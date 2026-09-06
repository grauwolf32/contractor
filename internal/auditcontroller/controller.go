package auditcontroller

import (
	"context"
	cryptorand "crypto/rand"
	"encoding/hex"
	"errors"
	"fmt"
	"sort"
	"sync"
	"sync/atomic"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditimport"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/config"
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

	if audit.State == auditstore.AuditWaitingReview {
		if changed, expireErr := c.store.ExpireReportReview(ctx, claim, audit.Revision); changed || expireErr != nil {
			return changed, expireErr
		}
		if audit.DeadlineAt != nil && !c.now().Before(*audit.DeadlineAt) {
			reason := auditstore.StopReason{
				Code:    "deadline_exhausted",
				Message: "The Audit wall-time deadline elapsed while exact human approval was pending.",
			}
			_, err := c.store.TransitionClaimed(ctx, auditstore.ClaimedTransitionParams{
				Claim: claim, ExpectedRevision: audit.Revision,
				ExpectedState: auditstore.AuditWaitingReview,
				TargetState:   auditstore.AuditFinalizing, Reason: &reason,
			})
			return err == nil, err
		}
	}

	if audit.State == auditstore.AuditActive {
		if reason := c.dispatchClosureReason(snapshot); reason != nil {
			_, err := c.store.TransitionClaimed(ctx, auditstore.ClaimedTransitionParams{
				Claim: claim, ExpectedRevision: audit.Revision,
				ExpectedState: auditstore.AuditActive, TargetState: auditstore.AuditFinalizing,
				Reason: reason,
			})
			return err == nil, err
		}
	}

	if changed, err := c.observeOneTerminal(ctx, claim, snapshot); changed || err != nil {
		return changed, err
	}
	if c.collector != nil {
		for _, execution := range snapshot.Executions {
			if execution.State != auditstore.ExecutionCollecting {
				continue
			}
			changed, collectErr := c.collector.Collect(ctx, claim, snapshot, execution)
			if errors.Is(collectErr, auditstore.ErrPrecondition) {
				return false, nil
			}
			if errors.Is(collectErr, auditimport.ErrPermanent) {
				return c.failAuditImport(ctx, claim, audit, "collection-contract-invalid")
			}
			return changed, collectErr
		}
	}
	if audit.State == auditstore.AuditActive {
		if changed, err := c.resumeOneIntent(ctx, claim, snapshot); changed || err != nil {
			return changed, err
		}
	}

	if audit.State == auditstore.AuditActive && snapshot.Round != nil {
		switch snapshot.Round.State {
		case auditstore.RoundAccepted:
			changed, complete, reason, err := c.reconcileRolePhase(
				ctx, claim, snapshot, auditstore.ExecutionDiscovery,
			)
			if changed || err != nil {
				return changed, err
			}
			if reason != nil {
				return c.closeForRoleFailure(ctx, claim, audit, reason)
			}
			if complete {
				_, err := c.store.TransitionRound(ctx, auditstore.RoundTransitionParams{
					Claim: claim, RoundID: snapshot.Round.RoundID,
					ExpectedRevision: snapshot.Round.Revision,
					ExpectedState:    auditstore.RoundAccepted, TargetState: auditstore.RoundExecuting,
				})
				return err == nil, err
			}
		case auditstore.RoundExecuting:
			if audit.OutstandingRunCount == 0 && len(snapshot.Items) == 0 &&
				len(snapshot.Executions) == 0 && !snapshot.MoreItems && !snapshot.MoreExecutions {
				_, err := c.store.TransitionRound(ctx, auditstore.RoundTransitionParams{
					Claim: claim, RoundID: snapshot.Round.RoundID,
					ExpectedRevision: snapshot.Round.Revision,
					ExpectedState:    auditstore.RoundExecuting, TargetState: auditstore.RoundAssessing,
				})
				return err == nil, err
			}
			if audit.OutstandingRunCount == 0 && len(snapshot.Executions) == 0 &&
				!snapshot.MoreItems && !snapshot.MoreExecutions &&
				onlyAwaitingReview(snapshot.Items) {
				_, err := c.store.TransitionClaimed(ctx, auditstore.ClaimedTransitionParams{
					Claim: claim, ExpectedRevision: audit.Revision,
					ExpectedState: auditstore.AuditActive,
					TargetState:   auditstore.AuditWaitingReview,
				})
				return err == nil, err
			}
		case auditstore.RoundAssessing:
			changed, complete, reason, err := c.reconcileRolePhase(
				ctx, claim, snapshot, auditstore.ExecutionAssessment,
			)
			if changed || err != nil {
				return changed, err
			}
			if reason != nil {
				return c.closeForRoleFailure(ctx, claim, audit, reason)
			}
			if complete {
				_, err := c.store.TransitionRound(ctx, auditstore.RoundTransitionParams{
					Claim: claim, RoundID: snapshot.Round.RoundID,
					ExpectedRevision: snapshot.Round.Revision,
					ExpectedState:    auditstore.RoundAssessing, TargetState: auditstore.RoundClosed,
				})
				return err == nil, err
			}
		case auditstore.RoundClosed:
			if audit.OutstandingRunCount == 0 && len(snapshot.Items) == 0 && len(snapshot.Executions) == 0 {
				var reason *auditstore.StopReason
				if c.roundBuilder != nil {
					params, closureReason, buildErr := c.roundBuilder.PrepareNextRound(ctx, claim, snapshot)
					if buildErr != nil {
						return false, buildErr
					}
					if params.RoundID != "" {
						_, _, acceptErr := c.store.AcceptNextRound(ctx, params)
						if errors.Is(acceptErr, auditstore.ErrPrecondition) {
							return false, nil
						}
						return acceptErr == nil, acceptErr
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
				return err == nil, err
			}
		}
	}
	closed := audit.Dispatch == auditstore.DispatchClosed ||
		audit.State == auditstore.AuditCancelling || audit.State == auditstore.AuditFinalizing ||
		audit.State == auditstore.AuditDeleting
	if closed {
		if changed, err := c.failOneUnboundIntent(ctx, claim, snapshot); changed || err != nil {
			return changed, err
		}
		if audit.State == auditstore.AuditCancelling || audit.State == auditstore.AuditDeleting || deadlineClosure(audit) {
			if changed, err := c.cancelOneSubmittedRun(ctx, snapshot); changed || err != nil {
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
		switch audit.State {
		case auditstore.AuditFinalizing:
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
			runID, found, err := c.store.NextLiveRunForDeletion(ctx, claim)
			if err != nil {
				return false, err
			}
			if found {
				err = c.runs.DeleteReleasedTerminalRun(ctx, audit.OwnerID, runID)
				var blocked *runstore.RunNotDeletableError
				if errors.As(err, &blocked) || errors.Is(err, runstore.ErrNotFound) {
					return false, nil
				}
				return err == nil, err
			}
			err = c.store.PurgeClaimed(ctx, claim, auditdomain.ArtifactNamespace(audit.AuditID))
			return err == nil, err
		}
		return false, nil
	}
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

func (c *Controller) reconcileRolePhase(
	ctx context.Context,
	claim auditstore.ControllerClaim,
	snapshot auditstore.ReconcileSnapshot,
	kind auditstore.ExecutionRole,
) (changed bool, complete bool, reason *auditstore.StopReason, err error) {
	profile, decodeErr := config.DecodeResolvedAuditProfileSnapshot(snapshot.Audit.ProfileSnapshot)
	if decodeErr != nil || snapshot.Round == nil {
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
			if latest.RoleAttempt == nil || *latest.RoleAttempt >= snapshot.Audit.Limits.MaxItemRunAttempts {
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

func (c *Controller) closeForRoleFailure(
	ctx context.Context,
	claim auditstore.ControllerClaim,
	audit auditstore.Audit,
	reason *auditstore.StopReason,
) (bool, error) {
	_, err := c.store.TransitionClaimed(ctx, auditstore.ClaimedTransitionParams{
		Claim: claim, ExpectedRevision: audit.Revision,
		ExpectedState: auditstore.AuditActive, TargetState: auditstore.AuditFinalizing,
		Reason: reason,
	})
	return err == nil, err
}

func (c *Controller) dispatchRole(
	ctx context.Context,
	claim auditstore.ControllerClaim,
	snapshot auditstore.ReconcileSnapshot,
	workflowRole string,
	attempt int,
) (bool, bool, *auditstore.StopReason, error) {
	prepared, err := c.builder.PrepareRole(ctx, snapshot, workflowRole, attempt)
	if err != nil {
		if errors.Is(err, ErrInvalidSubmission) {
			return false, false, &auditstore.StopReason{
				Code:    "role_dispatch_contract_invalid",
				Message: "Pinned Audit role dispatch data failed deterministic validation.",
			}, nil
		}
		return false, false, nil, err
	}
	prepared.Intent.Claim = claim
	execution, _, err := c.store.CreateExecutionIntent(ctx, prepared.Intent)
	if errors.Is(err, auditstore.ErrPrecondition) {
		return false, false, nil, nil
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

func roleNames(profile config.ResolvedAuditProfile, kind auditstore.ExecutionRole) []string {
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
	profile config.ResolvedAuditProfile,
	snapshot auditstore.ReconcileSnapshot,
	workflowRole string,
) bool {
	binding := profile.Workflows[workflowRole]
	for _, mapping := range binding.Inputs {
		if mapping.Source != config.AuditInputFromRetainedOutput {
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
			Code: "deadline_exhausted", Message: "The Audit wall-time deadline was reached; no new Runs may be submitted.",
		}
	}
	if audit.Limits.BatchSize < 1 || audit.Limits.BatchSize > auditstore.MaxCollectionItems ||
		snapshot.Round == nil || audit.CurrentRoundID == nil ||
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
		var prepared PreparedSubmission
		if execution.Role == auditstore.ExecutionCheck {
			if len(members) == 0 || len(members) > snapshot.Audit.Limits.BatchSize {
				return c.failInvalidIntent(ctx, claim, execution.ExecutionID)
			}
			selected := make([]CheckExecutionMember, len(members))
			for index, member := range members {
				item, exists := snapshotItem(snapshot.Items, member.ItemID)
				if !exists {
					return c.failInvalidIntent(ctx, claim, execution.ExecutionID)
				}
				selected[index] = CheckExecutionMember{Item: item, Attempt: member.ItemAttempt}
			}
			prepared, err = c.builder.PrepareBatch(ctx, snapshot, selected)
			if err == nil && !matchingIntent(execution, members, prepared.Intent) {
				err = ErrInvalidSubmission
			}
		} else {
			if len(members) != 0 || execution.RoleAttempt == nil {
				return c.failInvalidIntent(ctx, claim, execution.ExecutionID)
			}
			prepared, err = c.builder.PrepareRole(
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
	return execution.ExecutionID == prepared.ExecutionID && execution.Role == prepared.Role &&
		execution.WorkflowRole == prepared.WorkflowRole && execution.RoundID != nil && prepared.RoundID != nil &&
		*execution.RoundID == *prepared.RoundID && execution.RoleAttempt != nil && prepared.RoleAttempt != nil &&
		*execution.RoleAttempt == *prepared.RoleAttempt && execution.SubmissionKey == prepared.SubmissionKey &&
		execution.RequestDigest == prepared.RequestDigest && execution.Manifest.Digest == prepared.Manifest.Digest &&
		sameExactRef(execution.Manifest.Ref, prepared.Manifest.Ref) && len(prepared.Members) == 0
}

func (c *Controller) dispatch(
	ctx context.Context,
	claim auditstore.ControllerClaim,
	snapshot auditstore.ReconcileSnapshot,
	selected []CheckExecutionMember,
) (bool, error) {
	prepared, err := c.builder.PrepareBatch(ctx, snapshot, selected)
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
	members []auditstore.ExecutionItem,
	prepared auditstore.CreateExecutionIntentParams,
) bool {
	if execution.ExecutionID != prepared.ExecutionID || execution.Role != prepared.Role ||
		execution.WorkflowRole != prepared.WorkflowRole ||
		execution.RoundID == nil || prepared.RoundID == nil || *execution.RoundID != *prepared.RoundID ||
		execution.SubmissionKey != prepared.SubmissionKey || execution.RequestDigest != prepared.RequestDigest ||
		execution.Manifest.Digest != prepared.Manifest.Digest || !sameExactRef(execution.Manifest.Ref, prepared.Manifest.Ref) ||
		len(members) != len(prepared.Members) {
		return false
	}
	for index, member := range members {
		candidate := prepared.Members[index]
		if member.ExecutionItemID != candidate.ExecutionItemID || member.ItemID != candidate.ItemID ||
			member.BatchOrdinal != candidate.BatchOrdinal || member.BatchOrdinal != index ||
			member.ItemAttempt != candidate.ItemAttempt || member.Task.Digest != candidate.Task.Digest ||
			!sameExactRef(member.Task.Ref, candidate.Task.Ref) || !sameExactArtifacts(member.Inputs, candidate.Inputs) {
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
			left[index].SizeBytes != right[index].SizeBytes || !sameExactRef(left[index].Ref, right[index].Ref) {
			return false
		}
	}
	return true
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

func auditSettlementBarrier(snapshot auditstore.ReconcileSnapshot) bool {
	return snapshot.Audit.OutstandingRunCount == 0 && len(snapshot.Items) == 0 &&
		len(snapshot.Executions) == 0 && !snapshot.MoreItems && !snapshot.MoreExecutions &&
		snapshot.Audit.Hold != auditstore.HoldHeld
}

func onlyAwaitingReview(items []auditstore.Item) bool {
	if len(items) == 0 {
		return false
	}
	for _, item := range items {
		if item.State != auditstore.ItemAwaitingReview {
			return false
		}
	}
	return true
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
