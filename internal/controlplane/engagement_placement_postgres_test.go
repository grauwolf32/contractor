package controlplane

import (
	"context"
	"errors"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts/runtimesettings"
	"github.com/grauwolf32/contractor/internal/runstore"
)

func TestEngagementPlacementRequiresSelectedCapabilityWithoutChangingConfig(t *testing.T) {
	ctx := t.Context()
	pool := isolatedPlacementPool(t, ctx)
	fixture := newPlacementFixture(t, ctx, pool, nil)
	without := fixture.registerCandidate(t, ctx, "metadata-only", "1", nil)
	with := fixture.registerCandidate(t, ctx, "controlled", "2", []runtimesettings.RuntimeAdapterRef{runtimesettings.RuntimeAdapterHTTPTarget})
	calls := 0
	fixture.allocator.auditAdapterRequirements = func(_ context.Context, runID, stageID string) ([]runtimesettings.RuntimeAdapterRef, error) {
		calls++
		if runID != fixture.request().RunID || stageID != fixture.stageExecutionID {
			t.Fatal("placement used foreign binding")
		}
		return []runtimesettings.RuntimeAdapterRef{runtimesettings.RuntimeAdapterHTTPTarget}, nil
	}
	reservations, err := fixture.allocator.ReserveAllContext(ctx, fixture.request())
	if err != nil || len(reservations) != 1 || reservations[0].Grant.RuntimeAgentID != with || reservations[0].Grant.RuntimeAgentID == without || calls != 1 {
		t.Fatalf("reservations=%+v calls=%d err=%v", reservations, calls, err)
	}
	if len(reservations[0].ResolvedRuntimeConfig.RequiredRuntimeAdapters) != 0 {
		t.Fatal("Audit requirement leaked into RuntimeConfig provenance")
	}
}

func TestEngagementPlacementFailsClosedOnRequirementRead(t *testing.T) {
	for _, unavailable := range []bool{false, true} {
		t.Run(map[bool]string{false: "missing-adapter", true: "unavailable-baseline"}[unavailable], func(t *testing.T) {
			ctx := t.Context()
			pool := isolatedPlacementPool(t, ctx)
			fixture := newPlacementFixture(t, ctx, pool, nil)
			fixture.registerCandidate(t, ctx, "metadata-only", "1", nil)
			fixture.allocator.auditAdapterRequirements = func(context.Context, string, string) ([]runtimesettings.RuntimeAdapterRef, error) {
				if unavailable {
					return nil, context.DeadlineExceeded
				}
				return []runtimesettings.RuntimeAdapterRef{runtimesettings.RuntimeAdapterHTTPTarget}, nil
			}
			_, err := fixture.allocator.ReserveAllContext(ctx, fixture.request())
			if unavailable && !errors.Is(err, context.DeadlineExceeded) || !unavailable && !errors.Is(err, ErrInsufficientCapacity) {
				t.Fatal(err)
			}
			allocations, err := runstore.NewPostgresStore(pool).ListStageAllocations(ctx, fixture.stageExecutionID)
			if err != nil || len(allocations) != 0 {
				t.Fatal("ineligible allocation became durable", allocations, err)
			}
		})
	}
}

func TestEngagementPlacementCombinesConfiguredAndAuditAdapters(t *testing.T) {
	for _, test := range []struct {
		name      string
		adapters  []runtimesettings.RuntimeAdapterRef
		placeable bool
	}{
		{"both", []runtimesettings.RuntimeAdapterRef{runtimesettings.RuntimeAdapterHTTPTarget, runtimesettings.RuntimeAdapterOTLPHTTP}, true},
		{"missing-telemetry", []runtimesettings.RuntimeAdapterRef{runtimesettings.RuntimeAdapterHTTPTarget}, false},
		{"missing-target", []runtimesettings.RuntimeAdapterRef{runtimesettings.RuntimeAdapterOTLPHTTP}, false},
	} {
		t.Run(test.name, func(t *testing.T) {
			ctx := t.Context()
			pool := isolatedPlacementPool(t, ctx)
			fixture := newPlacementFixture(t, ctx, pool, []string{"debug"})
			fixture.registerCandidate(t, ctx, "controlled-telemetry", "1", test.adapters)
			fixture.allocator.auditAdapterRequirements = func(context.Context, string, string) ([]runtimesettings.RuntimeAdapterRef, error) {
				return []runtimesettings.RuntimeAdapterRef{runtimesettings.RuntimeAdapterHTTPTarget}, nil
			}
			reservations, err := fixture.allocator.ReserveAllContext(ctx, fixture.request())
			if !test.placeable {
				if !errors.Is(err, ErrInsufficientCapacity) {
					t.Fatalf("placement error = %v", err)
				}
				allocations, readErr := runstore.NewPostgresStore(pool).ListStageAllocations(ctx, fixture.stageExecutionID)
				if readErr != nil || len(allocations) != 0 {
					t.Fatalf("partial allocations = %v, err = %v", allocations, readErr)
				}
				return
			}
			if err != nil || len(reservations) != 1 {
				t.Fatalf("compatible placement: count=%d err=%v", len(reservations), err)
			}
			adapters := reservations[0].ResolvedRuntimeConfig.RequiredRuntimeAdapters
			if len(adapters) != 1 || adapters[0] != runtimesettings.RuntimeAdapterOTLPHTTP {
				t.Fatalf("Audit requirement changed RuntimeConfig provenance: %v", adapters)
			}
		})
	}
}
