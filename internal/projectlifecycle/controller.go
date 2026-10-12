// Package projectlifecycle owns restart-safe cleanup of Projects that have
// durably entered the deleting lifecycle.
package projectlifecycle

import (
	"context"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"math"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditstore/hookstore"
	"github.com/grauwolf32/contractor/internal/evalstore"
	"github.com/grauwolf32/contractor/internal/findingintake"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/grauwolf32/contractor/internal/randomid"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

const (
	defaultPollInterval     = time.Second
	defaultClaimDuration    = time.Minute
	defaultOperationTimeout = 15 * time.Second
	// defaultPurgeTimeout matches the Artifact purger's own cleanup budget.
	defaultPurgeTimeout = 2 * time.Minute
)

var errNoDeletion = errors.New("no Project deletion is claimable")

type RunStore interface {
	RequestRunCancellation(context.Context, string, runstore.WorkflowRunCancellation) (runstore.WorkflowRun, error)
}

type RunDeleter interface {
	DeleteReleasedTerminalRun(context.Context, string, string) error
}

type RunCancellationNotifier interface {
	Cancel(string)
}

type Clock interface {
	Now() time.Time
	After(time.Duration) <-chan time.Time
}

type Options struct {
	PollInterval  time.Duration
	ClaimDuration time.Duration
	// OperationTimeout bounds claim bookkeeping and the cancelling/draining
	// phases; PurgeTimeout bounds one Run or final Project purge.
	OperationTimeout time.Duration
	PurgeTimeout     time.Duration
	Clock            Clock
	NewID            func(string) (string, error)
	Logger           *slog.Logger
}

type Controller struct {
	pool     *pgxpool.Pool
	runs     RunStore
	deleter  RunDeleter
	notifier RunCancellationNotifier
	options  Options
	wake     chan struct{}
}

func New(
	pool *pgxpool.Pool,
	runs RunStore,
	deleter RunDeleter,
	notifier RunCancellationNotifier,
	options Options,
) (*Controller, error) {
	if pool == nil || runs == nil || deleter == nil || notifier == nil {
		return nil, errors.New("Project deletion controller dependencies are incomplete")
	}
	if options.PollInterval == 0 {
		options.PollInterval = defaultPollInterval
	}
	if options.ClaimDuration == 0 {
		options.ClaimDuration = defaultClaimDuration
	}
	if options.OperationTimeout == 0 {
		options.OperationTimeout = defaultOperationTimeout
	}
	if options.PurgeTimeout == 0 {
		options.PurgeTimeout = defaultPurgeTimeout
	}
	if options.Clock == nil {
		options.Clock = realClock{}
	}
	if options.NewID == nil {
		options.NewID = randomid.New
	}
	if options.Logger == nil {
		options.Logger = slog.New(slog.NewTextHandler(io.Discard, nil))
	}
	if options.PollInterval <= 0 || options.ClaimDuration.Microseconds() <= 0 ||
		options.OperationTimeout <= 0 || options.PurgeTimeout <= 0 {
		return nil, errors.New("Project deletion controller durations must be positive")
	}
	return &Controller{
		pool: pool, runs: runs, deleter: deleter, notifier: notifier, options: options,
		wake: make(chan struct{}, 1),
	}, nil
}

// Wake requests prompt reconciliation after DELETE commits. It is
// edge-triggered, non-blocking, and durability never depends on delivery.
func (c *Controller) Wake() {
	select {
	case c.wake <- struct{}{}:
	default:
	}
}

func (c *Controller) Run(ctx context.Context) error {
	for {
		worked, err := c.RunOnce(ctx)
		if ctx.Err() != nil {
			return nil
		}
		if err != nil {
			c.options.Logger.Error("Project deletion iteration failed", "error", err)
		}
		if worked && err == nil {
			continue
		}
		select {
		case <-ctx.Done():
			return nil
		case <-c.wake:
		case <-c.options.Clock.After(c.options.PollInterval):
		}
	}
}

// RunOnce claims one deleting Project and advances at most one bounded cleanup
// action. A false result means the controller is waiting for ordinary Run
// cancellation or allocation release.
func (c *Controller) RunOnce(ctx context.Context) (bool, error) {
	claimID, err := c.options.NewID("project_deletion_claim_")
	if err != nil {
		return false, fmt.Errorf("generate Project deletion claim ID: %w", err)
	}
	claimContext, cancelClaim := context.WithTimeout(ctx, c.options.OperationTimeout)
	claim, err := c.claim(claimContext, claimID)
	cancelClaim()
	if errors.Is(err, errNoDeletion) {
		return false, nil
	}
	if err != nil {
		return false, err
	}

	// Each phase has its own budget: a large purge must not consume the
	// ordinary operation budget and then fail on every attempt.
	advanceContext, cancelAdvance := context.WithTimeout(ctx, c.phaseTimeout(claim.Phase))
	worked, completed, err := c.advance(advanceContext, claim)
	cancelAdvance()
	if completed {
		return worked, err
	}
	// Claim bookkeeping runs on a fresh bounded context, never on the phase
	// context the failed attempt may already have exhausted.
	bookkeepingContext, cancelBookkeeping := context.WithTimeout(
		context.WithoutCancel(ctx), c.options.OperationTimeout,
	)
	defer cancelBookkeeping()
	if err == nil && worked {
		if releaseErr := c.releaseClaim(bookkeepingContext, claim); releaseErr != nil {
			return false, releaseErr
		}
		return true, nil
	}
	// Keep this Project briefly ineligible when it cannot progress. That lets
	// another deleting Project run without sacrificing restart recovery.
	if deferErr := c.deferClaim(bookkeepingContext, claim); deferErr != nil && err == nil {
		return false, deferErr
	}
	return false, err
}

func isPurgePhase(phase projectstore.DeletionPhase) bool {
	return phase == projectstore.DeletionPurgingRuns || phase == projectstore.DeletionPurgingArtifacts
}

func (c *Controller) phaseTimeout(phase projectstore.DeletionPhase) time.Duration {
	if isPurgePhase(phase) {
		return c.options.PurgeTimeout
	}
	return c.options.OperationTimeout
}

// purgeClaimDuration keeps the ordinary claim margin over the longer purge
// budget, so a purge cannot outlive its claim and race a second controller.
func (c *Controller) purgeClaimDuration() time.Duration {
	extra := max(0, c.options.PurgeTimeout-c.options.OperationTimeout)
	if extra > math.MaxInt64-c.options.ClaimDuration {
		return math.MaxInt64
	}
	return c.options.ClaimDuration + extra
}

func (c *Controller) claim(ctx context.Context, claimID string) (projectstore.DeletionClaim, error) {
	claim, found, err := projectstore.NewPostgresStore(c.pool).ClaimDeletion(ctx, projectstore.ClaimDeletionParams{
		ClaimID: claimID, Duration: c.options.ClaimDuration, PurgeDuration: c.purgeClaimDuration(),
	})
	if err != nil {
		return projectstore.DeletionClaim{}, err
	}
	if !found {
		return projectstore.DeletionClaim{}, errNoDeletion
	}
	return claim, nil
}

func (c *Controller) advance(
	ctx context.Context,
	claim projectstore.DeletionClaim,
) (worked bool, completed bool, err error) {
	if claim.Phase == projectstore.DeletionCancelling || claim.Phase == projectstore.DeletionDraining {
		// Recheck while draining so deletions already stranded by a skipped
		// Audit lock can recover after a controller restart or upgrade.
		if changed, err := c.requestOneAuditDeletion(ctx, claim); changed || err != nil {
			return changed, false, err
		}
	}
	switch claim.Phase {
	case projectstore.DeletionCancelling:
		return c.cancelOneRun(ctx, claim)
	case projectstore.DeletionDraining:
		return c.waitForDrain(ctx, claim)
	case projectstore.DeletionPurgingRuns:
		return c.purgeOneRun(ctx, claim)
	case projectstore.DeletionPurgingArtifacts:
		deleted, err := c.purgeProject(ctx, claim)
		if err != nil {
			return false, false, err
		}
		return true, deleted, nil
	default:
		return false, false, errors.New("Project deletion phase is invalid")
	}
}

// requestOneAuditDeletion commits the Project fence into each child Audit
// before generic Run cancellation advances. The Audit controller remains the
// sole owner of collection, hold release, child-Run deletion and Audit purge.
func (c *Controller) requestOneAuditDeletion(
	ctx context.Context, claim projectstore.DeletionClaim,
) (bool, error) {
	changed, err := hookstore.NewProjectDeletion(c.pool).RequestProjectOwnedDeletion(ctx, claim)
	if err != nil {
		return false, fmt.Errorf("request Project Audit deletion: %w", err)
	}
	return changed, nil
}

func (c *Controller) cancelOneRun(
	ctx context.Context,
	claim projectstore.DeletionClaim,
) (bool, bool, error) {
	var runID string
	err := c.pool.QueryRow(ctx, `
SELECT run_id
FROM workflow_runs
WHERE project_id = $1 AND owner_id = $2
  AND state IN ('initializing', 'pending', 'running', 'waiting')
ORDER BY created_at, run_id
LIMIT 1`, claim.ProjectID, claim.OwnerID).Scan(&runID)
	if errors.Is(err, pgx.ErrNoRows) {
		// SKIP LOCKED in requestOneAuditDeletion may have skipped the last
		// Audit. Do not leave cancelling until every Audit has received the
		// fence, even when ordinary Run cancellation is already complete.
		var auditsPending bool
		if err := c.pool.QueryRow(ctx, `
SELECT EXISTS (
    SELECT 1 FROM audits
    WHERE project_id = $1 AND deletion_requested_at IS NULL
)`, claim.ProjectID).Scan(&auditsPending); err != nil {
			return false, false, fmt.Errorf("inspect Project Audit deletion requests: %w", err)
		}
		if auditsPending {
			return false, false, nil
		}
		return c.advancePhase(ctx, claim, projectstore.DeletionDraining)
	}
	if err != nil {
		return false, false, fmt.Errorf("list Project Runs to cancel: %w", err)
	}
	reason := "Project deletion"
	requestedBy := claim.OwnerID
	run, err := c.runs.RequestRunCancellation(ctx, runID, runstore.WorkflowRunCancellation{
		Code: runstore.CancellationUserRequested, RequestedAt: c.options.Clock.Now().UTC().Round(0),
		RequestedBy: &requestedBy, Reason: &reason,
	})
	if err != nil {
		return false, false, fmt.Errorf("cancel Project Run %q: %w", runID, err)
	}
	if run.State == runstore.RunCancelling {
		c.notifier.Cancel(runID)
	}
	return true, false, nil
}

func (c *Controller) waitForDrain(
	ctx context.Context,
	claim projectstore.DeletionClaim,
) (bool, bool, error) {
	if blocked, err := evalstore.NewPostgresStore(c.pool).ProjectBlocked(ctx, claim.OwnerID, claim.ProjectID); err != nil || blocked {
		return false, false, err
	}
	var blocked bool
	err := c.pool.QueryRow(ctx, waitForDrainSQL, claim.ProjectID).Scan(&blocked)
	if err != nil {
		return false, false, fmt.Errorf("inspect Project Run drain: %w", err)
	}
	if blocked {
		return false, false, nil
	}
	return c.advancePhase(ctx, claim, projectstore.DeletionPurgingRuns)
}

func (c *Controller) purgeOneRun(
	ctx context.Context,
	claim projectstore.DeletionClaim,
) (bool, bool, error) {
	var runID string
	err := c.pool.QueryRow(ctx, `
SELECT run_id
FROM workflow_runs
WHERE project_id = $1 AND owner_id = $2
ORDER BY created_at, run_id
LIMIT 1`, claim.ProjectID, claim.OwnerID).Scan(&runID)
	if errors.Is(err, pgx.ErrNoRows) {
		return c.advancePhase(ctx, claim, projectstore.DeletionPurgingArtifacts)
	}
	if err != nil {
		return false, false, fmt.Errorf("list Project Runs to purge: %w", err)
	}
	if err := c.deleter.DeleteReleasedTerminalRun(ctx, claim.OwnerID, runID); err != nil {
		return false, false, fmt.Errorf("purge Project Run %q: %w", runID, err)
	}
	return true, false, nil
}

func (c *Controller) advancePhase(
	ctx context.Context,
	claim projectstore.DeletionClaim,
	next projectstore.DeletionPhase,
) (bool, bool, error) {
	if err := projectstore.NewPostgresStore(c.pool).AdvanceDeletion(ctx, claim, next); err != nil {
		return false, false, err
	}
	return true, false, nil
}

func (c *Controller) releaseClaim(ctx context.Context, claim projectstore.DeletionClaim) error {
	return projectstore.NewPostgresStore(c.pool).ReleaseDeletionClaim(ctx, claim)
}

func (c *Controller) deferClaim(ctx context.Context, claim projectstore.DeletionClaim) error {
	return projectstore.NewPostgresStore(c.pool).DeferDeletionClaim(ctx, claim, c.options.PollInterval)
}

// purgeProject advances the Project purge by one bounded transaction
// (LIM-DEL-3, LIM-DEL-2). Every transaction first purges the Project's private
// eval storage, a no-op once done, so no Artifact is removed while eval work
// could still need it. It then deletes one batch of the Project's finding
// proposal receipts (DEL-R61) and, once none is left, removes one ProjectScope
// purge batch, or, when the scope is already empty, the scope and the Project
// row. It reports whether the Project is gone.
func (c *Controller) purgeProject(ctx context.Context, claim projectstore.DeletionClaim) (bool, error) {
	var deleted bool
	err := persistencepostgres.InTxWithRetry(ctx, c.pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		deleted = false
		projects := projectstore.NewPostgresTxStore(tx)
		if err := projects.LockDeletionClaim(ctx, claim); err != nil {
			return fmt.Errorf("lock Project for Artifact purge: %w", err)
		}
		var runsRemain, auditsRemain bool
		if err := tx.QueryRow(ctx, `
SELECT EXISTS (SELECT 1 FROM workflow_runs WHERE project_id = $1),
       EXISTS (SELECT 1 FROM audits WHERE project_id = $1)`,
			claim.ProjectID,
		).Scan(&runsRemain, &auditsRemain); err != nil {
			return fmt.Errorf("verify Project Run purge: %w", err)
		}
		if runsRemain {
			return errors.New("Project still has WorkflowRuns during Artifact purge")
		}
		if auditsRemain {
			return errors.New("Project still has Audits during Artifact purge")
		}
		if err := evalstore.NewTxStore(tx).PurgeProject(ctx, evalstore.Scope{OwnerID: claim.OwnerID, ProjectID: claim.ProjectID}); err != nil {
			return err
		}
		receiptsGone, err := findingintake.PurgeProjectReceipts(ctx, tx, claim.ProjectID)
		if err != nil {
			return err
		}
		if !receiptsGone {
			return nil
		}
		purger, err := artifacts.NewPostgresPurger(tx)
		if err != nil {
			return err
		}
		scopeGone, err := purger.PurgeProject(ctx, claim.ProjectID)
		if err != nil {
			return fmt.Errorf("purge Project Artifacts: %w", err)
		}
		if !scopeGone {
			return nil
		}
		if err := projects.DeleteClaimedProject(ctx, claim); err != nil {
			return err
		}
		deleted = true
		return nil
	})
	return deleted, err
}

type realClock struct{}

func (realClock) Now() time.Time                                { return time.Now() }
func (realClock) After(duration time.Duration) <-chan time.Time { return time.After(duration) }
