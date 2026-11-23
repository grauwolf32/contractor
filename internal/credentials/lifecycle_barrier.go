package credentials

import (
	"context"
	"errors"
	"sync"
)

// LifecycleBarrier serializes credential creation/deletion against commits
// that introduce durable references. The first deployment has one active
// Control Plane, so a process-local barrier is the authoritative mutation
// gate; PostgreSQL constraints remain the durable integrity boundary.
type LifecycleBarrier struct {
	mu sync.RWMutex
}

func NewLifecycleBarrier() *LifecycleBarrier { return &LifecycleBarrier{} }

func (b *LifecycleBarrier) WithCredentialReferences(ctx context.Context, fn func() error) error {
	if b == nil || fn == nil {
		return errors.New("credential lifecycle barrier is not configured")
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	b.mu.RLock()
	defer b.mu.RUnlock()
	if err := ctx.Err(); err != nil {
		return err
	}
	return fn()
}

func (b *LifecycleBarrier) WithCredentialMutation(ctx context.Context, fn func() error) error {
	if b == nil || fn == nil {
		return errors.New("credential lifecycle barrier is not configured")
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	if err := ctx.Err(); err != nil {
		return err
	}
	return fn()
}

func (b *LifecycleBarrier) lockMutation(ctx context.Context) error {
	if b == nil {
		return errors.New("credential lifecycle barrier is not configured")
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	b.mu.Lock()
	if err := ctx.Err(); err != nil {
		b.mu.Unlock()
		return err
	}
	return nil
}

func (b *LifecycleBarrier) unlockMutation() { b.mu.Unlock() }
