// Package streamline implements the bounded streamline@1 PlannerFactory and
// the Server-owned execution host shared with router@1. The model loop runs
// in Runtime; the Scheduler uses the framework-neutral planner.Factory boundary.
package streamline

import (
	"fmt"
	"time"

	"github.com/grauwolf32/contractor/internal/contracts"
)

const (
	defaultMaxModelCalls  = 32
	defaultMaxTokens      = 200_000
	defaultMaxWorkerCalls = 64
	defaultMaxWallTime    = 30 * time.Minute
)

type Limits struct {
	MaxModelCalls  int
	MaxTokens      int64
	MaxWorkerCalls int
	MaxWallTime    time.Duration
}

func DefaultLimits() Limits {
	return Limits{
		MaxModelCalls: defaultMaxModelCalls, MaxTokens: defaultMaxTokens,
		MaxWorkerCalls: defaultMaxWorkerCalls, MaxWallTime: defaultMaxWallTime,
	}
}

func normalizeLimits(value Limits) (Limits, error) {
	defaults := DefaultLimits()
	if value.MaxModelCalls == 0 {
		value.MaxModelCalls = defaults.MaxModelCalls
	}
	if value.MaxTokens == 0 {
		value.MaxTokens = defaults.MaxTokens
	}
	if value.MaxWorkerCalls == 0 {
		value.MaxWorkerCalls = defaults.MaxWorkerCalls
	}
	if value.MaxWallTime == 0 {
		value.MaxWallTime = defaults.MaxWallTime
	}
	if value.MaxModelCalls <= 0 || value.MaxModelCalls > 1_000 ||
		value.MaxTokens <= 0 || value.MaxTokens > contracts.MaxWorkerTotalTokens ||
		value.MaxWorkerCalls <= 0 || value.MaxWorkerCalls > contracts.MaxPlannerWorkerCalls ||
		value.MaxWallTime <= 0 || value.MaxWallTime > 24*time.Hour {
		return Limits{}, fmt.Errorf("model-backed Planner limits are outside bounded ranges")
	}
	return value, nil
}
