package gitimport

import (
	"fmt"
	"time"

	"github.com/grauwolf32/contractor/internal/sourcezip"
)

// Limits are restart-only process budgets. Fixed protocol parser bounds and the
// generic compressed Artifact ceiling are independent of these operator values.
type Limits struct {
	Timeout          time.Duration
	MaxReceivedBytes int64
	MaxDecodedBytes  int64
	Source           sourcezip.Limits
}

func DefaultLimits() Limits {
	return Limits{Timeout: Deadline, MaxReceivedBytes: MaxReceivedBytes, MaxDecodedBytes: MaxDecodedBytes, Source: sourcezip.DefaultLimits()}
}

func (l Limits) Validate() error {
	if l.Timeout <= 0 || l.Timeout > 30*time.Minute ||
		l.MaxReceivedBytes <= 0 || l.MaxReceivedBytes > 1<<30 ||
		l.MaxDecodedBytes <= 0 || l.MaxDecodedBytes > 2<<30 ||
		l.Source.MaxExpandedBytes > l.MaxDecodedBytes {
		return fmt.Errorf("Git import limits require timeout in (0,30m], received bytes in 1..1 GiB, and decoded bytes in 1..2 GiB and not below expanded source bytes")
	}
	return l.Source.Validate()
}

func (c Config) EffectiveLimits() Limits {
	if c.Limits != nil {
		return *c.Limits
	}
	return DefaultLimits()
}
