// Package scancontroller reconciles bounded sweep members into ordinary Runs.
package scancontroller

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"sync"
	"time"

	"github.com/grauwolf32/contractor/internal/apierror"
	"github.com/grauwolf32/contractor/internal/claimpass"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/scanstore"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

// claimReleaseBudget bounds the release of a tick's claim.
const claimReleaseBudget = 5 * time.Second

type Options struct {
	Pool      *pgxpool.Pool
	HolderID  string
	Admit     func(context.Context, persistencepostgres.ControllerClaim, scanstore.Snapshot, scanstore.Member) error
	CancelRun func(context.Context, string, string) (runstore.WorkflowRun, error)
	Now       func() time.Time
	Logger    *slog.Logger
}
type active struct {
	cancel context.CancelFunc
	done   chan struct{}
}
type Controller struct {
	options Options
	mu      sync.Mutex
	blocked map[string]bool
	active  map[string]active
}

func New(o Options) (*Controller, error) {
	if o.Pool == nil || o.HolderID == "" || o.Admit == nil || o.CancelRun == nil || o.Now == nil || o.Logger == nil {
		return nil, scanstore.ErrInvalid
	}
	return &Controller{options: o, blocked: map[string]bool{}, active: map[string]active{}}, nil
}
func (c *Controller) Run(ctx context.Context) error {
	timer := time.NewTicker(time.Second)
	defer timer.Stop()
	for {
		if err := ctx.Err(); err != nil {
			return nil
		}
		if err := c.RunOnce(ctx); err != nil && ctx.Err() == nil {
			c.options.Logger.Warn("scan sweep tick failed", "error", err)
		}
		select {
		case <-ctx.Done():
			return nil
		case <-timer.C:
		}
	}
}

// Drain blocks this parent's new work, cancels and joins its current tick,
// then executes the terminal purge under the caller's context. SQL parent
// locks and claim epochs fence other controller instances as well.
func (c *Controller) Drain(ctx context.Context, id string, fn func() error) error {
	c.mu.Lock()
	if c.blocked[id] {
		c.mu.Unlock()
		return scanstore.ErrConflict
	}
	c.blocked[id] = true
	a, found := c.active[id]
	if found {
		a.cancel()
	}
	c.mu.Unlock()
	defer func() { c.mu.Lock(); delete(c.blocked, id); c.mu.Unlock() }()
	if found {
		select {
		case <-a.done:
		case <-ctx.Done():
			return ctx.Err()
		}
	}
	return fn()
}

// RunOnce leases one sweep and ticks it. The claim is given back after the
// tick under its own budget, before Drain sees the tick end; a definite
// release failure joins the tick's error, and a claim another holder took
// over is not a failure (PER-R26, FAIL-SCN-15).
func (c *Controller) RunOnce(parent context.Context) (err error) {
	claim, found, err := scanstore.New(c.options.Pool).Claim(parent, c.options.HolderID, time.Minute)
	if err != nil || !found {
		return err
	}
	released := false
	release := func() {
		if released {
			return
		}
		released = true
		err = errors.Join(err, c.release(parent, claim))
	}
	defer release()

	ctx, cancel := context.WithTimeout(parent, 30*time.Second)
	defer cancel()
	a := active{cancel: cancel, done: make(chan struct{})}
	c.mu.Lock()
	if c.blocked[claim.SubjectID] {
		c.mu.Unlock()
		return nil
	}
	c.active[claim.SubjectID] = a
	c.mu.Unlock()
	defer func() { release(); c.mu.Lock(); delete(c.active, claim.SubjectID); close(a.done); c.mu.Unlock() }()
	snapshot, err := scanstore.New(c.options.Pool).Snapshot(ctx, claim.SubjectID)
	snapshotValid := err == nil
	if err != nil && !errors.Is(err, scanstore.ErrInvalid) {
		return err
	}
	var work []scanstore.Member
	err = persistencepostgres.InTx(ctx, c.options.Pool, pgx.TxOptions{}, func(tx pgx.Tx) error { var err error; work, err = scanstore.New(tx).Work(ctx, claim); return err })
	if err != nil {
		return err
	}
	bytes := 0
	for _, m := range work {
		bytes += len(m.Child)
		if bytes > scanstore.TickBytes {
			break
		}
		if err := ctx.Err(); err != nil {
			return err
		}
		sweep, err := scanstore.New(c.options.Pool).Get(ctx, snapshot.OwnerID, claim.SubjectID)
		if err != nil {
			return err
		}
		if m.State == "admitted" {
			settled := m.RunState == nil || runstore.RunLifecycleTerminal.Includes(*m.RunState)
			reason := ""
			if m.RunState == nil {
				reason = "run_removed"
			}
			if sweep.State == "cancelling" && !settled {
				run, err := c.options.CancelRun(ctx, *m.RunID, snapshot.OwnerID)
				if err != nil {
					return err
				}
				settled = runstore.RunLifecycleTerminal.Includes(run.State)
			}
			if err := c.record(ctx, claim, m, "admitted", reason, m.RunID, settled, 2); err != nil {
				return err
			}
			continue
		}
		if sweep.State == "cancelling" {
			if err := c.record(ctx, claim, m, "cancelled", "sweep_cancelled", nil, true, 0); err != nil {
				return err
			}
			continue
		}
		if !snapshotValid {
			if err := c.record(ctx, claim, m, "failed", "snapshot_invalid", nil, true, 0); err != nil {
				return err
			}
			continue
		}
		if !c.options.Now().Before(sweep.CreatedAt.Add(scanstore.AdmissionDeadline)) || m.Attempts >= 100 {
			if err := c.record(ctx, claim, m, "failed", "admission_exhausted", nil, true, 0); err != nil {
				return err
			}
			continue
		}
		operation, cancel := context.WithTimeout(ctx, 10*time.Second)
		err = c.options.Admit(operation, claim, snapshot, m)
		cancel()
		if err == nil {
			continue
		}
		if ctx.Err() != nil {
			return ctx.Err()
		}
		// The child UOW records the Run and member together. Record checks the
		// authoritative current member before changing a pending outcome, so a
		// lost committed acknowledgement cannot erase the admitted fact.
		classifiedError, classified := apierror.Of(err)
		var code apierror.Kind
		if classified {
			code = classifiedError.ErrorKind()
		}
		permanent := classified && !code.Retryable()
		if permanent {
			if err := c.record(ctx, claim, m, "failed", code.Code(), nil, true, 0); err != nil {
				return err
			}
		} else {
			backoff := min(300, 1<<min(m.Attempts, 8))
			if err := c.record(ctx, claim, m, "pending", "admission_retry", nil, false, backoff); err != nil {
				return err
			}
		}
	}
	return nil
}

// release gives the claim of one tick back. Its error names the sweep and
// carries only the store's error, which holds no request content.
func (c *Controller) release(parent context.Context, claim persistencepostgres.ControllerClaim) error {
	err := claimpass.Release(parent, claimReleaseBudget, persistencepostgres.ErrControllerClaimLost, func(ctx context.Context) error {
		return scanstore.New(c.options.Pool).Release(ctx, claim)
	})
	if err != nil {
		return fmt.Errorf("release the claim of scan sweep %s: %w", claim.SubjectID, err)
	}
	return nil
}

func (c *Controller) record(ctx context.Context, claim persistencepostgres.ControllerClaim, m scanstore.Member, state, reason string, runID *string, settled bool, delay int) error {
	return persistencepostgres.InTxWithRetry(ctx, c.options.Pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		store := scanstore.New(tx)
		if err := store.Lock(ctx, claim); err != nil {
			return err
		}
		if m.State == "pending" {
			var cancelling bool
			if err := store.Cancelling(ctx, claim.SubjectID, &cancelling); err != nil {
				return err
			}
			if cancelling {
				state, reason, settled, delay = "cancelled", "sweep_cancelled", true, 0
			}
		}
		err := store.Record(ctx, claim, m.Ordinal, state, reason, runID, settled, delay)
		if errors.Is(err, persistencepostgres.ErrControllerClaimLost) {
			return err
		}
		return err
	})
}
