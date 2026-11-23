package scheduler

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"

	"github.com/grauwolf32/contractor/internal/controlplane"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

// releaseMarkerTrace counts the release-marker statements and the existence
// checks that follow a marker which left a named allocation unmarked.
type releaseMarkerTrace struct {
	mu                     sync.Mutex
	markers, verifications int
}

func (trace *releaseMarkerTrace) TraceQueryStart(ctx context.Context, _ *pgx.Conn, data pgx.TraceQueryStartData) context.Context {
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

func (*releaseMarkerTrace) TraceQueryEnd(context.Context, *pgx.Conn, pgx.TraceQueryEndData) {}

func (trace *releaseMarkerTrace) take() (int, int) {
	trace.mu.Lock()
	defer trace.mu.Unlock()
	markers, verifications := trace.markers, trace.verifications
	trace.markers, trace.verifications = 0, 0
	return markers, verifications
}

// multiAllocationStage is a terminal attempt of run-1 on PostgreSQL whose
// Stage binds the placed Agent and extra more, each with its own durable row
// and live grant, as a multi-Agent Stage pins them. Its store counts the
// release markers.
type multiAllocationStage struct {
	deferredPlacementHarness
	trace        *releaseMarkerTrace
	execution    runstore.StageExecution
	reservations []controlplane.Reservation
	// earlier is the allocation of the attempt before it, also terminal with
	// an incomplete release.
	earlier string
}

func newMultiAllocationStage(t *testing.T, ctx context.Context, extra int) multiAllocationStage {
	t.Helper()
	h := newDeferredPlacementHarness(t, ctx, nil)
	var placed []controlplane.Reservation
	allocations := createTerminalAttempts(t, ctx, h, 2, func(attempt int, _ string, reservations []controlplane.Reservation) {
		if attempt == 2 {
			placed = reservations
		}
	})
	allocator := h.workers.allocator
	reservations := append([]controlplane.Reservation(nil), placed...)
	for index := 1; index <= extra; index++ {
		reservation := placed[0]
		reservation.Grant.AllocationID = fmt.Sprintf("%s-agent-%d", placed[0].Grant.AllocationID, index)
		reservation.Grant.LogicalAgentName = fmt.Sprintf("agent-%d", index)
		reservation.Grant.Namespace = reservation.Grant.LogicalAgentName
		reservation.Grant.RuntimeInstanceID = fmt.Sprintf("runtime-%d", index+1)
		if err := h.store.RecordStageAllocation(ctx, stageAllocationFromReservation(reservation)); err != nil {
			t.Fatal(err)
		}
		allocator.grants[reservation.Grant.AllocationID] = reservation.Grant
		reservations = append(reservations, reservation)
	}
	allocator.cached = append([]controlplane.Reservation(nil), reservations...)
	execution, err := h.store.GetStageExecution(ctx, historicalAttemptID(2))
	if err != nil {
		t.Fatal(err)
	}
	trace := &releaseMarkerTrace{}
	configuration := h.pool.Config().Copy()
	configuration.ConnConfig.Tracer = trace
	counted, err := pgxpool.NewWithConfig(ctx, configuration)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(counted.Close)
	h.scheduler.store = runstore.NewPostgresStore(counted, artifactpgtest.NewStorage())
	return multiAllocationStage{
		deferredPlacementHarness: h, trace: trace, execution: execution,
		reservations: reservations, earlier: allocations[historicalAttemptID(1)],
	}
}

// rows returns the durable allocation rows of both attempts by ID.
func (f multiAllocationStage) rows(t *testing.T, ctx context.Context) map[string]runstore.StageAllocation {
	t.Helper()
	batch, err := f.store.ListStageAllocationsBatch(ctx, []string{historicalAttemptID(1), historicalAttemptID(2)})
	if err != nil {
		t.Fatal(err)
	}
	result := make(map[string]runstore.StageAllocation)
	for _, rows := range batch {
		for _, row := range rows {
			result[row.AllocationID] = row
		}
	}
	return result
}

// A lane releasing a six-Agent Stage records the six release attempts with
// one marker and, after its release batch, marks released with one more
// marker exactly the allocations whose grants are gone; the one the Runtime
// refused keeps its grant, its fence and its incomplete row. The marker per
// allocation took 6 attempt and 5 completion statements.
func TestPostgresTerminalReleaseMarksEachPhaseOnce(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 45*time.Second)
	defer cancel()
	f := newMultiAllocationStage(t, ctx, 5)
	refused := f.reservations[2].Grant.AllocationID
	f.workers.releaseErrors = map[string]error{refused: errors.New("injected Runtime release failure")}
	if err := f.workers.allocator.SetWriteFence(refused); err != nil {
		t.Fatal(err)
	}
	f.trace.take()

	err := f.scheduler.releaseTerminal(f.execution.StageExecutionID, f.reservations)
	if err == nil || !strings.Contains(err.Error(), "injected Runtime release failure") {
		t.Fatalf("partial terminal release = %v, want the Runtime failure", err)
	}
	if markers, verifications := f.trace.take(); markers != 2 || verifications != 0 {
		t.Fatalf("release of 6 allocations = %d markers and %d existence checks, want 2 and 0", markers, verifications)
	}
	rows := f.rows(t, ctx)
	for _, reservation := range f.reservations {
		row := rows[reservation.Grant.AllocationID]
		if row.ReleaseAttemptedAt == nil {
			t.Fatalf("allocation %s has no release attempt", row.AllocationID)
		}
		if completed := row.ReleaseCompletedAt != nil; completed == (row.AllocationID == refused) {
			t.Fatalf("allocation %s completed = %t", row.AllocationID, completed)
		}
	}
	if _, err := f.workers.allocator.GetGrant(refused); err != nil || !f.workers.allocator.fenced[refused] {
		t.Fatalf("refused grant = %v, fenced %t; want it live and fenced", err, f.workers.allocator.fenced[refused])
	}
	if earlier := rows[f.earlier]; earlier.ReleaseAttemptedAt != nil || earlier.ReleaseCompletedAt != nil {
		t.Fatalf("lane release marked the earlier attempt's allocation: %+v", earlier)
	}
}

// Recovery of one terminal Stage whose allocations ended differently records
// all release attempts with one marker, marks the already gone grant
// released with one, releases the live and the diverged grants in one batch
// and marks those it released with one more. The refused grant stays
// incomplete, the completed row is not touched, and the earlier terminal
// attempt, which a full batch could have taken, is left to its own operation.
// The marker per allocation took 13 statements for these rows.
func TestPostgresTerminalReleaseRecoveryMarksEachPhaseOnce(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 45*time.Second)
	defer cancel()
	f := newMultiAllocationStage(t, ctx, 5)
	allocator := f.workers.allocator
	id := func(index int) string { return f.reservations[index].Grant.AllocationID }
	refused, gone, diverged, completed := id(1), id(2), id(3), id(4)
	f.workers.releaseErrors = map[string]error{refused: errors.New("injected Runtime release failure")}
	delete(allocator.grants, gone)
	for index := range allocator.cached {
		if allocator.cached[index].Grant.AllocationID == diverged {
			allocator.cached[index].Grant.StageExecutionID = "stage-unknown"
			allocator.grants[diverged] = allocator.cached[index].Grant
		}
	}
	if err := f.store.MarkStageAllocationReleased(ctx, completed); err != nil {
		t.Fatal(err)
	}
	delete(allocator.grants, completed)
	before := f.rows(t, ctx)[completed]
	if !f.scheduler.beginTerminalRelease(f.execution.StageExecutionID) {
		t.Fatal("Stage release gate is taken")
	}
	f.trace.take()

	worked, err := f.scheduler.recoverTerminalStage(ctx, f.execution)
	if !worked || err == nil || !strings.Contains(err.Error(), "injected Runtime release failure") {
		t.Fatalf("recovery = (%t, %v), want work and the Runtime failure", worked, err)
	}
	if markers, verifications := f.trace.take(); markers != 3 || verifications != 0 {
		t.Fatalf("recovery of 5 incomplete allocations = %d markers and %d existence checks, want 3 and 0",
			markers, verifications)
	}
	rows := f.rows(t, ctx)
	for _, reservation := range f.reservations {
		row := rows[reservation.Grant.AllocationID]
		if row.ReleaseAttemptedAt == nil || (row.ReleaseCompletedAt != nil) == (row.AllocationID == refused) {
			t.Fatalf("allocation %s after recovery: attempted %v completed %v", row.AllocationID,
				row.ReleaseAttemptedAt, row.ReleaseCompletedAt)
		}
	}
	if after := rows[completed]; !after.ReleaseAttemptedAt.Equal(*before.ReleaseAttemptedAt) ||
		!after.ReleaseCompletedAt.Equal(*before.ReleaseCompletedAt) {
		t.Fatalf("recovery changed the completed release: before %+v after %+v", before, after)
	}
	if !allocator.fenced[diverged] {
		t.Fatal("diverged grant was released without its write fence")
	}
	if _, err := allocator.GetGrant(refused); err != nil {
		t.Fatalf("refused grant after recovery = %v, want it live", err)
	}
	if earlier := rows[f.earlier]; earlier.ReleaseAttemptedAt != nil || earlier.ReleaseCompletedAt != nil {
		t.Fatalf("recovery of one Stage marked the earlier attempt's allocation: %+v", earlier)
	}
	if f.scheduler.beginTerminalRelease(f.execution.StageExecutionID) {
		f.scheduler.endTerminalRelease(f.execution.StageExecutionID)
	} else {
		t.Fatal("recovery kept the Stage release gate")
	}

	// The next operation retries only the refused grant: one attempt and,
	// once the Runtime answers, one completion marker.
	f.workers.releaseErrors = nil
	if !f.scheduler.beginTerminalRelease(f.execution.StageExecutionID) {
		t.Fatal("Stage release gate is taken")
	}
	if worked, err := f.scheduler.recoverTerminalStage(ctx, f.execution); !worked || err != nil {
		t.Fatalf("retry = (%t, %v)", worked, err)
	}
	if markers, verifications := f.trace.take(); markers != 2 || verifications != 0 {
		t.Fatalf("retry of 1 allocation = %d markers and %d existence checks, want 2 and 0", markers, verifications)
	}
	if row := f.rows(t, ctx)[refused]; row.ReleaseCompletedAt == nil {
		t.Fatalf("refused allocation after the retry = %+v", row)
	}
}
