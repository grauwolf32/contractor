// Package evalcoordinator reconciles durable experiments. The existing Scheduler
// retains all Run execution and placement authority.
package evalcoordinator

import (
	"context"
	"errors"
	"log/slog"
	"sync/atomic"
	"time"

	"github.com/grauwolf32/contractor/internal/claimpass"
	"github.com/grauwolf32/contractor/internal/evalstore"
	"github.com/grauwolf32/contractor/internal/helpers/randomid"
)

const (
	defaultPollInterval     = time.Second
	defaultClaimLease       = time.Minute
	defaultOperationTimeout = 20 * time.Second
	defaultBatchSize        = 4
	maxBatchSize            = 32
	minPollInterval         = 10 * time.Millisecond
	claimReleaseTimeout     = 5 * time.Second
)

type Store interface {
	Claim(context.Context, string, time.Duration, int) ([]evalstore.Claim, error)
	ReleaseClaim(context.Context, evalstore.Claim) error
}

type Service interface {
	Tick(context.Context, evalstore.Claim) (bool, error)
}

// Clock schedules the wait between passes. Tests inject one to drive the
// poll cadence without sleeping; the default is the system clock.
type Clock interface {
	After(time.Duration) <-chan time.Time
}

type Options struct {
	HolderID                              string
	PollInterval, Lease, OperationTimeout time.Duration
	Batch                                 int
	Clock                                 Clock
	Logger                                *slog.Logger
}

type Coordinator struct {
	store   Store
	service Service
	options Options
	wake    chan struct{}
	running atomic.Bool
}

func New(store Store, service Service, o Options) (*Coordinator, error) {
	if store == nil || service == nil {
		return nil, errors.New("eval coordinator dependencies are incomplete")
	}
	if o.HolderID == "" {
		holderID, err := randomid.New("eval-")
		if err != nil {
			return nil, err
		}
		o.HolderID = holderID
	}
	if o.PollInterval == 0 {
		o.PollInterval = defaultPollInterval
	}
	if o.Lease == 0 {
		o.Lease = defaultClaimLease
	}
	if o.OperationTimeout == 0 {
		o.OperationTimeout = defaultOperationTimeout
	}
	if o.Batch == 0 {
		o.Batch = defaultBatchSize
	}
	if o.Clock == nil {
		o.Clock = systemClock{}
	}
	if o.Logger == nil {
		o.Logger = slog.Default()
	}
	if o.PollInterval < minPollInterval ||
		o.Lease < time.Second ||
		o.Lease > 5*time.Minute ||
		o.OperationTimeout <= 0 ||
		o.OperationTimeout >= o.Lease ||
		o.Batch < 1 ||
		o.Batch > maxBatchSize {
		return nil, errors.New("eval coordinator bounds are invalid")
	}
	return &Coordinator{store: store, service: service, options: o, wake: make(chan struct{}, 1)}, nil
}

func (c *Coordinator) Wake() {
	select {
	case c.wake <- struct{}{}:
	default:
	}
}

func (c *Coordinator) Run(ctx context.Context) error {
	if !c.running.CompareAndSwap(false, true) {
		return errors.New("eval coordinator is already running")
	}
	defer c.running.Store(false)
	for {
		_, err := c.RunOnce(ctx)
		if ctx.Err() != nil {
			return nil
		}
		if err != nil {
			c.options.Logger.Error("Eval reconciliation deferred", "error", err)
		}
		// Observing an active execution is work but must not turn into a hot
		// poll: every pass waits for the poll interval or a wake hint.
		select {
		case <-ctx.Done():
			return nil
		case <-c.wake:
		case <-c.options.Clock.After(c.options.PollInterval):
		}
	}
}

// RunOnce leases a bounded batch of experiments and ticks each one through
// the shared claim lifecycle (PER-R26): every tick has its own operation
// budget and every release the fixed release budget.
func (c *Coordinator) RunOnce(ctx context.Context) (bool, error) {
	claimCtx, cancel := context.WithTimeout(ctx, c.options.OperationTimeout)
	claims, err := c.store.Claim(claimCtx, c.options.HolderID, c.options.Lease, c.options.Batch)
	cancel()
	if err != nil {
		return false, err
	}
	return claimpass.Run(ctx, claims, claimpass.Lifecycle[evalstore.Claim]{
		Handle: c.tick, Release: c.store.ReleaseClaim,
		ReleaseBudget: claimReleaseTimeout, Lost: evalstore.ErrClaimLost,
	})
}

func (c *Coordinator) tick(ctx context.Context, claim evalstore.Claim) (bool, error) {
	tickCtx, cancel := context.WithTimeout(ctx, c.options.OperationTimeout)
	defer cancel()
	return c.service.Tick(tickCtx, claim)
}

type systemClock struct{}

func (systemClock) After(duration time.Duration) <-chan time.Time { return time.After(duration) }
