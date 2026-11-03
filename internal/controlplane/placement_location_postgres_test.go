package controlplane

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"

	"github.com/grauwolf32/contractor/internal/contracts/control"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/jackc/pgx/v5/pgxpool"
)

func registerPlacementLocation(t *testing.T, f *placementFixture, instance, digit string, labels []string) {
	t.Helper()
	registration := testRegistration(instance)
	registration.PlacementLabels = labels
	registration.SupportedRuntimes = []string{f.template.Runtime.RuntimeID + "@" + f.template.Runtime.Version}
	registration.SupportedSandboxProfiles = []string{f.template.SandboxProfile.SandboxProfileID + "@" + f.template.SandboxProfile.Version}
	registration.SupportedToolsets = nil
	for _, selected := range f.template.Toolsets {
		registration.SupportedToolsets = append(registration.SupportedToolsets, control.ToolsetCapability{Ref: selected.Ref.ToolsetID + "@" + selected.Ref.Version, Tools: append([]string{}, selected.Tools...)})
	}
	f.registerProcess(t, f.ctx, instance, digit, nil, registration)
}

func TestPlacementWrongLocationFleetDoesNotReadCandidateConfiguration(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 45*time.Second)
	defer cancel()
	pool := isolatedPlacementPool(t, ctx)
	f := newPlacementFixture(t, ctx, pool, nil, []string{"cloud"})
	for index := 1; index <= 15; index++ {
		registerPlacementLocation(t, f, fmt.Sprintf("local-%d", index), fmt.Sprintf("%x", index), []string{"local/lab"})
	}
	trace := &placementQueryCounter{}
	configuration := pool.Config().Copy()
	configuration.ConnConfig.Tracer = trace
	countedPool, err := pgxpool.NewWithConfig(ctx, configuration)
	if err != nil {
		t.Fatal(err)
	}
	defer countedPool.Close()
	f.allocator.pool = countedPool
	if _, err := f.allocator.ReserveAllContext(ctx, f.request()); !errors.Is(err, ErrInsufficientCapacity) {
		t.Fatalf("wrong-location fleet placement = %v", err)
	}
	if queries := trace.queries.Load(); queries != 0 {
		t.Fatalf("wrong-location candidates caused %d configuration queries", queries)
	}
}

func TestPlacementLocationWaitsForMatchingFleetAndRetainsProvenance(t *testing.T) {
	for _, requirement := range []string{"cloud", "cloud/aws"} {
		t.Run(requirement, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
			defer cancel()
			pool := isolatedPlacementPool(t, ctx)
			f := newPlacementFixture(t, ctx, pool, nil, []string{requirement})
			registerPlacementLocation(t, f, "local", "1", []string{"local/lab"})
			registerPlacementLocation(t, f, "undeclared", "2", nil)
			if requirement == "cloud/aws" {
				registerPlacementLocation(t, f, "other-provider", "3", []string{"cloud/hetzner"})
			}
			request := f.request()
			// Reserve two Workers atomically; a single matching slot is insufficient.
			second := request.Bindings[0]
			second.LogicalAgentName, second.Namespace = "second", "second"
			request.Bindings = append(request.Bindings, second)
			registerPlacementLocation(t, f, "matching-first", "4", []string{"cloud/aws"})
			if _, err := f.allocator.ReserveAllContext(ctx, request); !errors.Is(err, ErrInsufficientCapacity) {
				t.Fatalf("incomplete matching error = %v", err)
			}
			if len(f.registry.SnapshotOperations().Allocations) != 0 {
				t.Fatal("partial placement consumed a slot")
			}
			store := runstore.NewPostgresStore(pool, artifactpgtest.NewStorage())
			stage, err := store.GetStageExecution(ctx, f.stageExecutionID)
			if err != nil || stage.AdmittedAt != nil {
				t.Fatalf("capacity waiting was admitted: %+v, %v", stage, err)
			}
			registerPlacementLocation(t, f, "matching-later", "5", []string{"cloud/aws/eu"})
			placed, err := f.allocator.ReserveAllContext(ctx, request)
			if err != nil || len(placed) != 2 {
				t.Fatalf("late fleet placement = %+v, %v", placed, err)
			}
			for _, allocation := range placed {
				if allocation.Grant.RuntimeInstanceID != "matching-first" && allocation.Grant.RuntimeInstanceID != "matching-later" {
					t.Fatalf("wrong location: %+v", allocation)
				}
			}
			if replayed, err := f.allocator.ReserveAllContext(ctx, request); err != nil || replayed[0].Grant.AllocationID != placed[0].Grant.AllocationID {
				t.Fatalf("placement receipt = %+v, %v", replayed, err)
			}
			changed := request
			changed.RequiredPlacementLabels = []string{"local"}
			if _, err := f.allocator.ReserveAllContext(ctx, changed); !errors.Is(err, ErrReservationConflict) {
				t.Fatalf("changed receipt = %v", err)
			}
			retained, err := store.ListStageAllocations(ctx, f.stageExecutionID)
			if err != nil || len(retained) != 2 {
				t.Fatalf("retained allocations = %+v, %v", retained, err)
			}
			for index, allocation := range retained {
				var original Reservation
				for _, candidate := range placed {
					if candidate.Grant.AllocationID == allocation.AllocationID {
						original = candidate
					}
				}
				if len(allocation.RuntimePlacementLabels) == 0 || !slices.Equal(allocation.RuntimePlacementLabels, original.PlacementLabels) {
					t.Fatalf("actual labels lost: %+v", allocation)
				}
				if _, err := pool.Exec(ctx, `UPDATE stage_allocations SET runtime_placement_labels=ARRAY['local/lab'] WHERE allocation_id=$1`, allocation.AllocationID); err == nil {
					t.Fatal("allocation location was mutable")
				}
				retained[index].RuntimePlacementLabels[0] = "local/lab"
			}
			// A new Control Plane has no fleet registrations; history still reads
			// the exact selected declarations from PostgreSQL.
			fresh := newTestRegistry(t, newTestClock())
			if len(fresh.SnapshotOperations().RuntimeAgents) != 0 {
				t.Fatal("new Registry unexpectedly has live fleet")
			}
			retained, err = runstore.NewPostgresStore(pool, artifactpgtest.NewStorage()).ListStageAllocations(ctx, f.stageExecutionID)
			if err != nil || len(retained) != 2 || retained[0].RuntimePlacementLabels[0] == "local/lab" {
				t.Fatalf("history depends on live fleet or aliases caller memory: %+v, %v", retained, err)
			}
			if _, err := pool.Exec(ctx, `ALTER TABLE stage_allocations DROP CONSTRAINT stage_allocations_placement_labels_valid`); err != nil {
				t.Fatal(err)
			}
			if _, err := pool.Exec(ctx, `ALTER TABLE stage_allocations DISABLE TRIGGER stage_allocations_protect_placement_labels`); err != nil {
				t.Fatal(err)
			}
			if _, err := pool.Exec(ctx, `UPDATE stage_allocations SET runtime_placement_labels=ARRAY['cloud/*'] WHERE allocation_id=$1`, retained[0].AllocationID); err != nil {
				t.Fatal(err)
			}
			if _, err := store.ListStageAllocations(ctx, f.stageExecutionID); err == nil {
				t.Fatal("history reader accepted corrupted actual labels")
			}
		})
	}
}
