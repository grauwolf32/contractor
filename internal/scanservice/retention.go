package scanservice

import (
	"context"
	"time"

	"github.com/grauwolf32/contractor/internal/scanstore"
)

// Run reclaims expired sweeps at most 25 members per maintenance tick. Expiry
// hides the resource before physical deletion; its create receipt is retained.
func (s *Service) Run(ctx context.Context) error {
	ticker := time.NewTicker(time.Minute)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return nil
		case <-ticker.C:
			operation, cancel := context.WithTimeout(ctx, 10*time.Second)
			err := scanstore.New(s.options.Pool).PurgeExpired(operation, time.Now())
			cancel()
			if err != nil {
				if ctx.Err() != nil {
					return nil
				}
				s.options.Logger.Warn("scan sweep retention tick failed")
			}
		}
	}
}
