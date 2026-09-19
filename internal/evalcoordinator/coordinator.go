// Package evalcoordinator reconciles durable experiments. The existing Scheduler
// retains all Run execution and placement authority.
package evalcoordinator

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"errors"
	"log/slog"
	"sync"
	"sync/atomic"
	"time"

	"github.com/grauwolf32/contractor/internal/evalstore"
)

type Store interface {
	Claim(context.Context, string, time.Duration, int) ([]evalstore.Claim, error)
	ReleaseClaim(context.Context, evalstore.Claim) error
}
type Service interface {
	Tick(context.Context, evalstore.Claim) (bool, error)
}
type Options struct {
	HolderID                              string
	PollInterval, Lease, OperationTimeout time.Duration
	Batch                                 int
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
		var b [16]byte
		if _, err := rand.Read(b[:]); err != nil {
			return nil, err
		}
		o.HolderID = "eval-" + hex.EncodeToString(b[:])
	}
	if o.PollInterval == 0 {
		o.PollInterval = time.Second
	}
	if o.Lease == 0 {
		o.Lease = time.Minute
	}
	if o.OperationTimeout == 0 {
		o.OperationTimeout = 20 * time.Second
	}
	if o.Batch == 0 {
		o.Batch = 4
	}
	if o.Logger == nil {
		o.Logger = slog.Default()
	}
	if o.PollInterval < 10*time.Millisecond || o.Lease < time.Second || o.Lease > 5*time.Minute || o.OperationTimeout <= 0 || o.OperationTimeout >= o.Lease || o.Batch < 1 || o.Batch > 32 {
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
		// Observing an active execution is work but must not turn into a hot poll.
		timer := time.NewTimer(c.options.PollInterval)
		select {
		case <-ctx.Done():
			timer.Stop()
			return nil
		case <-c.wake:
			timer.Stop()
		case <-timer.C:
		}
	}
}
func (c *Coordinator) RunOnce(ctx context.Context) (bool, error) {
	claimCtx, cancel := context.WithTimeout(ctx, c.options.OperationTimeout)
	claims, err := c.store.Claim(claimCtx, c.options.HolderID, c.options.Lease, c.options.Batch)
	cancel()
	if err != nil {
		return false, err
	}
	var changed atomic.Bool
	errs := make([]error, len(claims))
	var wg sync.WaitGroup
	for i, claim := range claims {
		wg.Add(1)
		go func(i int, claim evalstore.Claim) {
			defer wg.Done()
			tickCtx, cancel := context.WithTimeout(ctx, c.options.OperationTimeout)
			worked, err := c.service.Tick(tickCtx, claim)
			cancel()
			if worked {
				changed.Store(true)
			}
			releaseCtx, releaseCancel := context.WithTimeout(context.WithoutCancel(ctx), 5*time.Second)
			releaseErr := c.store.ReleaseClaim(releaseCtx, claim)
			releaseCancel()
			if errors.Is(releaseErr, evalstore.ErrClaimLost) {
				releaseErr = nil
			}
			errs[i] = errors.Join(err, releaseErr)
		}(i, claim)
	}
	wg.Wait()
	return changed.Load(), errors.Join(errs...)
}
