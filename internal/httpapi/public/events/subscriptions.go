package events

import (
	"context"
	"errors"
	"math"
	"time"

	"github.com/grauwolf32/contractor/internal/apierror"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/controlplane"
	"github.com/grauwolf32/contractor/internal/runstore"
)

func (s *socket) pumpSubscription(ctx context.Context, current *subscription, after *Cursor) {
	switch current.stream.Kind {
	case StreamRun:
		s.pumpRun(ctx, current, after)
	case StreamOperations:
		s.pumpOperations(ctx, current, after)
	}
}

// pumpRun authorizes a Run subscription and validates its cursor, then
// hands it to the Run's poller, which replays and follows its events until
// the subscription or the poller ends it.
func (s *socket) pumpRun(ctx context.Context, current *subscription, after *Cursor) {
	run, err := s.hub.runs.GetRun(ctx, current.stream.ID)
	if err != nil {
		if ctx.Err() == nil {
			if errors.Is(err, runstore.ErrNotFound) {
				s.sendError(current.id, apierror.NotFound, "Run stream was not found")
			} else {
				s.sendError(current.id, apierror.Overloaded, "Run stream is temporarily unavailable")
			}
		}
		return
	}
	if run.OwnerID != s.session.Principal.UserID {
		s.sendError(current.id, apierror.NotFound, "Run stream was not found")
		return
	}
	latest, err := s.hub.runs.GetRunEventCursor(ctx, current.stream.ID)
	if err != nil || latest.Sequence < 0 || !contracts.ValidResourceID(latest.Generation) {
		if ctx.Err() == nil {
			s.sendError(current.id, apierror.Overloaded, "Run stream is temporarily unavailable")
		}
		return
	}
	start := latest.Sequence
	startCursor := runCursor(latest)
	if after != nil {
		sequence, parseErr := parseCursor(*after)
		if parseErr != nil || sequence > math.MaxInt64 {
			s.sendResync(current.id, current.stream, "cursor_unavailable")
			return
		}
		if after.Generation != latest.Generation {
			s.sendResync(current.id, current.stream, "generation_changed")
			return
		}
		if int64(sequence) > latest.Sequence {
			s.sendResync(current.id, current.stream, "cursor_unavailable")
			return
		}
		start = int64(sequence)
		startCursor = *after
	}
	if !s.sendFrame(subscribedFrame{
		Version: ProtocolVersion, Type: "subscribed", SubscriptionID: current.id,
		Stream: current.stream, Cursor: startCursor,
	}) {
		return
	}
	subscriber := &runSubscriber{
		socket: s, subscription: current, ctx: ctx,
		generation: latest.Generation, last: start, ended: make(chan struct{}),
	}
	poller, attached := s.hub.attachRun(current.stream.ID, subscriber)
	if !attached {
		return
	}
	select {
	case <-ctx.Done():
	case <-subscriber.ended:
	}
	s.hub.detachRun(poller, subscriber)
	// After the detach the poller can no longer end the subscription, so its
	// outcome is final even when it ended while being cancelled.
	select {
	case <-subscriber.ended:
		if subscriber.closeCode != 0 {
			s.stop(subscriber.closeCode, subscriber.closeReason)
		}
	default:
	}
}

func (s *socket) pumpOperations(ctx context.Context, current *subscription, after *Cursor) {
	updates, cancelUpdates := s.hub.operations.SubscribeOperations()
	defer cancelUpdates()
	snapshot := s.hub.operations.SnapshotOperations()
	if !contracts.ValidResourceID(snapshot.Cursor.Generation) {
		s.sendError(current.id, apierror.Overloaded, "Operations stream is temporarily unavailable")
		return
	}
	start := snapshot.Cursor
	startCursor := operationsCursor(start)
	var replay []controlplane.OperationsChange
	if after != nil {
		sequence, err := parseCursor(*after)
		if err != nil {
			s.sendResync(current.id, current.stream, "cursor_unavailable")
			return
		}
		start = controlplane.OperationsCursor{Generation: after.Generation, Revision: sequence}
		replay, _, err = s.hub.operations.ReplayOperations(start)
		if err != nil {
			s.sendOperationsResync(current, err)
			return
		}
		startCursor = *after
	}
	if !s.sendFrame(subscribedFrame{
		Version: ProtocolVersion, Type: "subscribed", SubscriptionID: current.id,
		Stream: current.stream, Cursor: startCursor,
	}) {
		return
	}
	last := start.Revision
	if !s.sendOperationsChanges(current, &last, replay) {
		return
	}
	if !s.catchUpOperations(current, start.Generation, &last) {
		return
	}
	ticker := time.NewTicker(s.hub.catchUp)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case _, ok := <-updates:
			if !ok {
				return
			}
		case <-ticker.C:
		}
		if !s.catchUpOperations(current, start.Generation, &last) {
			return
		}
	}
}

func (s *socket) catchUpOperations(current *subscription, generation string, last *uint64) bool {
	changes, cursor, err := s.hub.operations.ReplayOperations(controlplane.OperationsCursor{
		Generation: generation, Revision: *last,
	})
	if err != nil {
		s.sendOperationsResync(current, err)
		return false
	}
	if !s.sendOperationsChanges(current, last, changes) {
		return false
	}
	if *last != cursor.Revision {
		s.sendResync(current.id, current.stream, "sequence_gap")
		return false
	}
	return true
}

func (s *socket) sendOperationsChanges(
	current *subscription,
	last *uint64,
	changes []controlplane.OperationsChange,
) bool {
	for _, change := range changes {
		if change.Cursor.Revision != *last+1 {
			s.sendResync(current.id, current.stream, "sequence_gap")
			return false
		}
		encoded, err := operationsEventServerFrame(current.id, current.stream, change)
		if err != nil {
			s.sendResync(current.id, current.stream, "cursor_unavailable")
			return false
		}
		if !s.sendEncoded(encoded) {
			return false
		}
		*last = change.Cursor.Revision
	}
	return true
}

func (s *socket) sendOperationsResync(current *subscription, err error) {
	reason := "cursor_unavailable"
	switch {
	case errors.Is(err, controlplane.ErrOperationsGeneration):
		reason = "generation_changed"
	case errors.Is(err, controlplane.ErrOperationsGap):
		reason = "sequence_gap"
	}
	s.sendResync(current.id, current.stream, reason)
}
