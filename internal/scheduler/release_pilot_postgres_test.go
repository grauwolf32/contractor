package scheduler

import (
	"context"
	"errors"
	"reflect"
	"testing"
	"time"
)

// A completion marker can commit even when its response is lost. Recovery
// must inspect durable per-allocation outcomes, retain a failed peer, and
// leave already completed markers and the terminal Stage fact unchanged.
func TestPostgresReleasePilotLostMarkerReply(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 45*time.Second)
	defer cancel()
	f := newMultiAllocationStage(t, ctx, 1)
	failed, released := f.reservations[0].Grant.AllocationID, f.reservations[1].Grant.AllocationID
	before, err := f.store.GetStageExecution(ctx, f.execution.StageExecutionID)
	if err != nil {
		t.Fatal(err)
	}
	lost := &releasePilotLostMarkerReply{Store: f.scheduler.store}
	f.scheduler.store = lost
	f.workers.releaseErrors = map[string]error{failed: errors.New("cleanup unavailable")}
	if err := f.scheduler.releaseTerminal(f.execution.StageExecutionID, f.reservations); !errors.Is(err, errReleasePilotMarkerReply) {
		t.Fatalf("lost committed marker reply = %v", err)
	}
	rows := f.rows(t, ctx)
	completed := rows[released]
	if completed.ReleaseCompletedAt == nil || completed.ReleaseAttemptedAt == nil || rows[failed].ReleaseCompletedAt != nil {
		t.Fatalf("partial release markers: %+v", rows)
	}
	if _, err := f.workers.allocator.GetGrant(failed); err != nil {
		t.Fatalf("failed grant disappeared: %v", err)
	}
	f.workers.releaseErrors = nil
	if !f.scheduler.beginTerminalRelease(f.execution.StageExecutionID) {
		t.Fatal("release gate busy")
	}
	if worked, err := f.scheduler.recoverTerminalStage(ctx, f.execution); !worked || err != nil {
		t.Fatalf("recovery = %v %v", worked, err)
	}
	rows = f.rows(t, ctx)
	if rows[failed].ReleaseCompletedAt == nil || !reflect.DeepEqual(completed, rows[released]) {
		t.Fatalf("recovery changed completed peer: %+v", rows)
	}
	if !f.scheduler.beginTerminalRelease(f.execution.StageExecutionID) {
		t.Fatal("release gate busy after recovery")
	}
	if worked, err := f.scheduler.recoverTerminalStage(ctx, f.execution); worked || err != nil {
		t.Fatalf("settled recovery repeated work: %v %v", worked, err)
	}
	if after := f.rows(t, ctx); !reflect.DeepEqual(rows, after) {
		t.Fatal("settled replay changed durable allocation facts")
	}
	after, err := f.store.GetStageExecution(ctx, f.execution.StageExecutionID)
	if err != nil || !reflect.DeepEqual(before, after) {
		t.Fatalf("cleanup rewrote terminal Stage: %v", err)
	}
}

var errReleasePilotMarkerReply = errors.New("injected lost PostgreSQL release-marker reply")

type releasePilotLostMarkerReply struct {
	Store
	lost bool
}

func (s *releasePilotLostMarkerReply) MarkStageAllocationReleased(ctx context.Context, ids ...string) error {
	if err := s.Store.MarkStageAllocationReleased(ctx, ids...); err != nil {
		return err
	}
	if !s.lost {
		s.lost = true
		return errReleasePilotMarkerReply
	}
	return nil
}
