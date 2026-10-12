package telemetry

import (
	"context"
	"errors"
	"log/slog"
	"time"
)

// Durable telemetry retention (OBS-R43, OBS-R44): one cleanup pass per
// retentionInterval deletes at most retentionBatch expired rows from each
// telemetry table. When a pass deletes a full batch, expired rows may remain,
// so the loop drains them after retentionDrainPause instead of waiting for the
// next interval.
const (
	retentionInterval   = 24 * time.Hour
	retentionBatch      = 500
	retentionDrainPause = time.Second
)

type expiredTelemetryCleaner interface {
	CleanupExpired(context.Context, time.Time, int) (int64, error)
}

// Retention removes expired durable telemetry in a maintenance loop of its
// own. It is not Workflow execution and holds no Scheduler lane, so a lower
// Run limit or a paused owner cannot disable it (SCH-R159).
type Retention struct {
	cleaner     expiredTelemetryCleaner
	passTimeout time.Duration
	logger      *slog.Logger
	now         func() time.Time
	after       func(time.Duration) <-chan time.Time
}

// NewRetention cleans the tables of repository. passTimeout bounds one
// cleanup pass across all telemetry tables.
func NewRetention(repository *Repository, passTimeout time.Duration, logger *slog.Logger) (*Retention, error) {
	if repository == nil || logger == nil {
		return nil, errors.New("telemetry retention dependencies are incomplete")
	}
	if passTimeout <= 0 {
		return nil, errors.New("telemetry retention pass timeout must be positive")
	}
	return &Retention{
		cleaner: repository, passTimeout: passTimeout, logger: logger,
		now: time.Now, after: time.After,
	}, nil
}

// Run cleans until ctx is done and then returns nil. A failed pass is logged
// and retried at the next interval; it never stops the loop.
func (r *Retention) Run(ctx context.Context) error {
	for {
		if ctx.Err() != nil {
			return nil
		}
		passContext, cancel := context.WithTimeout(ctx, r.passTimeout)
		deleted, err := r.cleaner.CleanupExpired(passContext, r.now(), retentionBatch)
		cancel()
		if err != nil && ctx.Err() == nil {
			r.logger.Warn("telemetry retention cleanup failed", "error", err)
		} else if deleted > 0 {
			r.logger.Info("expired telemetry removed", "rows", deleted)
		}
		wait := retentionInterval
		if err == nil && deleted >= retentionBatch {
			wait = retentionDrainPause
		}
		select {
		case <-ctx.Done():
			return nil
		case <-r.after(wait):
		}
	}
}
