// Package projectlifecycle owns restart-safe cleanup of Projects that have
// durably entered the deleting lifecycle.
package projectlifecycle

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

const (
	defaultPollInterval     = time.Second
	defaultClaimDuration    = time.Minute
	defaultOperationTimeout = 15 * time.Second
)

var errNoDeletion = errors.New("no Project deletion is claimable")

type RunStore interface {
	RequestRunCancellation(context.Context, string, runstore.WorkflowRunCancellation) (runstore.WorkflowRun, error)
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
	PollInterval     time.Duration
	ClaimDuration    time.Duration
	OperationTimeout time.Duration
	Clock            Clock
	NewID            func(string) (string, error)
	Logger           *slog.Logger
}

type Controller struct {
	pool     *pgxpool.Pool
	runs     RunStore
	notifier RunCancellationNotifier
	options  Options
	wake     chan struct{}
}

func New(
	pool *pgxpool.Pool,
	runs RunStore,
	notifier RunCancellationNotifier,
	options Options,
) (*Controller, error) {
	if pool == nil || runs == nil || notifier == nil {
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
	if options.Clock == nil {
		options.Clock = realClock{}
	}
	if options.NewID == nil {
		options.NewID = randomID
	}
	if options.Logger == nil {
		options.Logger = slog.New(slog.NewTextHandler(io.Discard, nil))
	}
	if options.PollInterval <= 0 || options.ClaimDuration.Microseconds() <= 0 ||
		options.OperationTimeout <= 0 {
		return nil, errors.New("Project deletion controller durations must be positive")
	}
	return &Controller{
		pool: pool, runs: runs, notifier: notifier, options: options,
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
	operationContext, cancel := context.WithTimeout(ctx, c.options.OperationTimeout)
	defer cancel()
	claim, err := c.claim(operationContext, claimID)
	if errors.Is(err, errNoDeletion) {
		return false, nil
	}
	if err != nil {
		return false, err
	}

	worked, completed, err := c.advance(operationContext, claim)
	if completed {
		return worked, err
	}
	if err == nil && worked {
		if releaseErr := c.releaseClaim(operationContext, claim); releaseErr != nil {
			return false, releaseErr
		}
		return true, nil
	}
	// Keep this Project briefly ineligible when it cannot progress. That lets
	// another deleting Project run without sacrificing restart recovery.
	if deferErr := c.deferClaim(operationContext, claim); deferErr != nil && err == nil {
		return false, deferErr
	}
	return false, err
}

type deletionClaim struct {
	ProjectID string
	OwnerID   string
	ClaimID   string
	Phase     projectstore.DeletionPhase
}

func (c *Controller) claim(ctx context.Context, claimID string) (deletionClaim, error) {
	var claim deletionClaim
	claim.ClaimID = claimID
	err := c.pool.QueryRow(ctx, `
WITH candidate AS (
    SELECT project_id
    FROM projects
    WHERE lifecycle_state = 'deleting'
      AND (deletion_claim_id IS NULL OR deletion_claim_expires_at <= clock_timestamp())
    ORDER BY deletion_requested_at, project_id
    FOR UPDATE SKIP LOCKED
    LIMIT 1
)
UPDATE projects AS project
SET deletion_claim_id = $1,
    deletion_claimed_at = clock_timestamp(),
    deletion_claim_expires_at = clock_timestamp() + ($2::bigint * interval '1 microsecond')
FROM candidate
WHERE project.project_id = candidate.project_id
RETURNING project.project_id, project.owner_id, project.deletion_phase`,
		claimID, c.options.ClaimDuration.Microseconds(),
	).Scan(&claim.ProjectID, &claim.OwnerID, &claim.Phase)
	if errors.Is(err, pgx.ErrNoRows) {
		return deletionClaim{}, errNoDeletion
	}
	if err != nil {
		return deletionClaim{}, fmt.Errorf("claim Project deletion: %w", err)
	}
	if !claim.Phase.Valid() {
		return deletionClaim{}, errors.New("claimed Project deletion phase is invalid")
	}
	return claim, nil
}

func (c *Controller) advance(
	ctx context.Context,
	claim deletionClaim,
) (worked bool, completed bool, err error) {
	switch claim.Phase {
	case projectstore.DeletionCancelling:
		return c.cancelOneRun(ctx, claim)
	case projectstore.DeletionDraining:
		return c.waitForDrain(ctx, claim)
	case projectstore.DeletionPurgingRuns:
		return c.purgeOneRun(ctx, claim)
	case projectstore.DeletionPurgingArtifacts:
		if err := c.purgeProject(ctx, claim); err != nil {
			return false, false, err
		}
		return true, true, nil
	default:
		return false, false, errors.New("Project deletion phase is invalid")
	}
}

func (c *Controller) cancelOneRun(
	ctx context.Context,
	claim deletionClaim,
) (bool, bool, error) {
	var runID string
	err := c.pool.QueryRow(ctx, `
SELECT run_id
FROM workflow_runs
WHERE project_id = $1 AND owner_id = $2
  AND state IN ('initializing', 'running')
ORDER BY created_at, run_id
LIMIT 1`, claim.ProjectID, claim.OwnerID).Scan(&runID)
	if errors.Is(err, pgx.ErrNoRows) {
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
	claim deletionClaim,
) (bool, bool, error) {
	var blocked bool
	err := c.pool.QueryRow(ctx, `
SELECT EXISTS (
    SELECT 1 FROM workflow_runs
    WHERE project_id = $1
      AND state IN ('initializing', 'running', 'cancelling')
) OR EXISTS (
    SELECT 1
    FROM workflow_runs AS run
    JOIN stage_executions AS execution ON execution.run_id = run.run_id
    JOIN stage_allocations AS allocation
      ON allocation.stage_execution_id = execution.stage_execution_id
    WHERE run.project_id = $1
      AND allocation.release_completed_at IS NULL
)`, claim.ProjectID).Scan(&blocked)
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
	claim deletionClaim,
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
	if err := c.runs.DeleteReleasedTerminalRun(ctx, claim.OwnerID, runID); err != nil {
		return false, false, fmt.Errorf("purge Project Run %q: %w", runID, err)
	}
	return true, false, nil
}

func (c *Controller) advancePhase(
	ctx context.Context,
	claim deletionClaim,
	next projectstore.DeletionPhase,
) (bool, bool, error) {
	tag, err := c.pool.Exec(ctx, `
UPDATE projects
SET deletion_phase = $4,
    revision = revision + 1,
    updated_at = GREATEST(clock_timestamp(), updated_at + interval '1 microsecond')
WHERE project_id = $1 AND owner_id = $2
  AND lifecycle_state = 'deleting' AND deletion_phase = $3
  AND deletion_claim_id = $5`,
		claim.ProjectID, claim.OwnerID, claim.Phase, next, claim.ClaimID,
	)
	if err != nil {
		return false, false, fmt.Errorf("advance Project deletion phase: %w", err)
	}
	if tag.RowsAffected() != 1 {
		return false, false, errors.New("Project deletion claim was lost")
	}
	return true, false, nil
}

func (c *Controller) releaseClaim(ctx context.Context, claim deletionClaim) error {
	tag, err := c.pool.Exec(ctx, `
UPDATE projects
SET deletion_claim_id = NULL,
    deletion_claimed_at = NULL,
    deletion_claim_expires_at = NULL
WHERE project_id = $1 AND owner_id = $2 AND deletion_claim_id = $3`,
		claim.ProjectID, claim.OwnerID, claim.ClaimID,
	)
	if err != nil {
		return fmt.Errorf("release Project deletion claim: %w", err)
	}
	if tag.RowsAffected() != 1 {
		return errors.New("Project deletion claim was lost")
	}
	return nil
}

func (c *Controller) deferClaim(ctx context.Context, claim deletionClaim) error {
	tag, err := c.pool.Exec(ctx, `
UPDATE projects
SET deletion_claim_expires_at = GREATEST(
        clock_timestamp() + ($4::bigint * interval '1 microsecond'),
        deletion_claimed_at + interval '1 microsecond'
    )
WHERE project_id = $1 AND owner_id = $2 AND deletion_claim_id = $3`,
		claim.ProjectID, claim.OwnerID, claim.ClaimID, c.options.PollInterval.Microseconds(),
	)
	if err != nil {
		return fmt.Errorf("defer Project deletion claim: %w", err)
	}
	if tag.RowsAffected() != 1 {
		return errors.New("Project deletion claim was lost")
	}
	return nil
}

func (c *Controller) purgeProject(ctx context.Context, claim deletionClaim) error {
	return persistencepostgres.InTx(ctx, c.pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		var locked bool
		err := tx.QueryRow(ctx, `
SELECT true
FROM projects
WHERE project_id = $1 AND owner_id = $2
  AND lifecycle_state = 'deleting' AND deletion_phase = 'purging_artifacts'
  AND deletion_claim_id = $3
FOR UPDATE`, claim.ProjectID, claim.OwnerID, claim.ClaimID).Scan(&locked)
		if errors.Is(err, pgx.ErrNoRows) {
			return errors.New("Project deletion claim was lost")
		}
		if err != nil {
			return fmt.Errorf("lock Project for final purge: %w", err)
		}
		var runsRemain bool
		if err := tx.QueryRow(ctx, `
SELECT EXISTS (SELECT 1 FROM workflow_runs WHERE project_id = $1)`,
			claim.ProjectID,
		).Scan(&runsRemain); err != nil {
			return fmt.Errorf("verify Project Run purge: %w", err)
		}
		if runsRemain {
			return errors.New("Project still has WorkflowRuns during final purge")
		}
		purger, err := artifacts.NewPostgresPurger(tx)
		if err != nil {
			return err
		}
		if err := purger.PurgeProject(ctx, claim.ProjectID); err != nil {
			return fmt.Errorf("purge Project Artifacts: %w", err)
		}
		tag, err := tx.Exec(ctx, `DELETE FROM projects WHERE project_id = $1 AND owner_id = $2`,
			claim.ProjectID, claim.OwnerID,
		)
		if err != nil {
			return fmt.Errorf("delete Project: %w", err)
		}
		if tag.RowsAffected() != 1 {
			return errors.New("Project deletion claim was lost")
		}
		return nil
	})
}

type realClock struct{}

func (realClock) Now() time.Time                                { return time.Now() }
func (realClock) After(duration time.Duration) <-chan time.Time { return time.After(duration) }

func randomID(prefix string) (string, error) {
	buffer := make([]byte, 16)
	if _, err := rand.Read(buffer); err != nil {
		return "", err
	}
	return prefix + hex.EncodeToString(buffer), nil
}
