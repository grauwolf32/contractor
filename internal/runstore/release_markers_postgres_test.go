package runstore

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

// markerStatementTrace counts release-marker statements and the existence
// checks after a marker that left a named allocation unmarked.
type markerStatementTrace struct {
	mu                     sync.Mutex
	markers, verifications int
}

func (trace *markerStatementTrace) TraceQueryStart(ctx context.Context, _ *pgx.Conn, data pgx.TraceQueryStartData) context.Context {
	statement := strings.TrimSpace(data.SQL)
	trace.mu.Lock()
	defer trace.mu.Unlock()
	switch {
	case strings.HasPrefix(statement, "UPDATE stage_allocations"):
		trace.markers++
	case strings.Contains(statement, "FROM stage_allocations WHERE allocation_id"):
		trace.verifications++
	}
	return ctx
}

func (*markerStatementTrace) TraceQueryEnd(context.Context, *pgx.Conn, pgx.TraceQueryEndData) {}

func (trace *markerStatementTrace) take() (int, int) {
	trace.mu.Lock()
	defer trace.mu.Unlock()
	markers, verifications := trace.markers, trace.verifications
	trace.markers, trace.verifications = 0, 0
	return markers, verifications
}

// The release markers name at most stageAllocationMarkBatch allocations per
// statement (LIM-SCH-22) and keep each allocation's outcome: a duplicate is
// marked once, a completed release is left as it is, and only a missing
// allocation is ErrNotFound, for that allocation, while the others are still
// marked. An invalid ID refuses the call before any statement.
func TestPostgresReleaseMarkersBatchAndKeepPerAllocationOutcomes(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 45*time.Second)
	defer cancel()
	pool := isolatedRunStorePool(t, ctx)
	store := NewPostgresStore(pool, artifactpgtest.NewStorage())
	createTestRun(t, ctx, store, "run-markers")
	if _, err := store.CreateStageExecution(ctx, testActiveStageParams("run-markers", "stage-markers", "copy")); err != nil {
		t.Fatal(err)
	}
	const count = stageAllocationMarkBatch + 2
	ids := make([]string, count)
	for index := range ids {
		ids[index] = fmt.Sprintf("allocation-%03d", index)
		allocation := testStageAllocation("stage-markers", ids[index])
		allocation.LogicalAgentName = fmt.Sprintf("agent-%03d", index)
		allocation.Namespace = allocation.LogicalAgentName
		if err := store.RecordStageAllocation(ctx, allocation); err != nil {
			t.Fatal(err)
		}
	}
	trace := &markerStatementTrace{}
	configuration := pool.Config().Copy()
	configuration.ConnConfig.Tracer = trace
	counted, err := pgxpool.NewWithConfig(ctx, configuration)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(counted.Close)
	marked := NewPostgresStore(counted, artifactpgtest.NewStorage())
	rows := func() map[string]StageAllocation {
		t.Helper()
		allocations, err := store.ListStageAllocations(ctx, "stage-markers")
		if err != nil {
			t.Fatal(err)
		}
		result := make(map[string]StageAllocation, len(allocations))
		for _, allocation := range allocations {
			result[allocation.AllocationID] = allocation
		}
		return result
	}

	if err := marked.MarkStageAllocationReleaseAttempt(ctx, append(ids, ids[0])...); err != nil {
		t.Fatal(err)
	}
	if markers, verifications := trace.take(); markers != 2 || verifications != 0 {
		t.Fatalf("attempts of %d allocations = %d markers and %d checks, want 2 and 0", count, markers, verifications)
	}
	for _, allocation := range rows() {
		if allocation.ReleaseAttemptedAt == nil || allocation.ReleaseCompletedAt != nil {
			t.Fatalf("allocation %s after the attempt marker = %+v", allocation.AllocationID, allocation)
		}
	}

	err = marked.MarkStageAllocationReleased(ctx, ids[0], "allocation-missing", ids[1], ids[2])
	if !errors.Is(err, ErrNotFound) || !strings.Contains(err.Error(), `"allocation-missing"`) ||
		strings.Contains(err.Error(), ids[1]) {
		t.Fatalf("completion with a missing allocation = %v, want ErrNotFound for it alone", err)
	}
	if markers, verifications := trace.take(); markers != 1 || verifications != 1 {
		t.Fatalf("completion with a missing allocation = %d markers and %d checks, want 1 and 1", markers, verifications)
	}
	completed := rows()
	for index, id := range ids {
		allocation := completed[id]
		if (allocation.ReleaseCompletedAt != nil) != (index < 3) {
			t.Fatalf("allocation %s completed = %v", id, allocation.ReleaseCompletedAt)
		}
		if allocation.ReleaseCompletedAt != nil && allocation.ReleaseCompletedAt.Before(*allocation.ReleaseAttemptedAt) {
			t.Fatalf("allocation %s completed before its attempt: %+v", id, allocation)
		}
	}

	// A completed release is not an error and keeps both of its timestamps.
	if err := marked.MarkStageAllocationReleaseAttempt(ctx, ids[0], ids[3]); err != nil {
		t.Fatal(err)
	}
	if err := marked.MarkStageAllocationReleased(ctx, ids[0]); err != nil {
		t.Fatal(err)
	}
	if markers, verifications := trace.take(); markers != 2 || verifications != 2 {
		t.Fatalf("markers naming a completed release = %d markers and %d checks, want 2 and 2", markers, verifications)
	}
	again := rows()
	if before, after := completed[ids[0]], again[ids[0]]; !after.ReleaseAttemptedAt.Equal(*before.ReleaseAttemptedAt) ||
		!after.ReleaseCompletedAt.Equal(*before.ReleaseCompletedAt) {
		t.Fatalf("markers changed a completed release: before %+v after %+v", before, after)
	}
	if !again[ids[3]].ReleaseAttemptedAt.After(*completed[ids[3]].ReleaseAttemptedAt) {
		t.Fatal("a repeated attempt marker did not record the new attempt")
	}

	for name, call := range map[string]func() error{
		"attempt with an empty ID":    func() error { return marked.MarkStageAllocationReleaseAttempt(ctx, ids[4], "") },
		"completion with an empty ID": func() error { return marked.MarkStageAllocationReleased(ctx, " ", ids[4]) },
	} {
		if err := call(); !errors.Is(err, ErrInvalid) {
			t.Fatalf("%s = %v, want ErrInvalid", name, err)
		}
	}
	if err := marked.MarkStageAllocationReleased(ctx); err != nil {
		t.Fatalf("marker without allocations = %v", err)
	}
	if markers, verifications := trace.take(); markers != 0 || verifications != 0 {
		t.Fatalf("refused and empty markers sent %d markers and %d checks, want none", markers, verifications)
	}
	if rows()[ids[4]].ReleaseCompletedAt != nil {
		t.Fatal("a refused marker completed an allocation")
	}
}
