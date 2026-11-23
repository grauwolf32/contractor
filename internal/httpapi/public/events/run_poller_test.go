package events

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strconv"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/gorilla/websocket"
	"github.com/grauwolf32/contractor/internal/runstore"
)

// API-R72: the idle subscriptions of one Run, from every socket a session may
// open, share one poller. It reads the Run once per catch-up interval whatever
// the number of subscriptions, that interval is the safety interval while the
// notification listener is established, and one notification costs one read
// that reaches every subscription.
func TestIdleRunSubscriptionsShareOneCatchUpRead(t *testing.T) {
	clock := newFakeClock()
	listener := newFakeRunListener()
	runs := &countingRunRepository{fakeRunRepository: newFakeRunRepository()}
	harness := newEventHarness(t, func(options *Options) {
		options.Runs, options.RunNotifications, options.clock = runs, listener, clock
	})
	const runID = "run-idle"
	runs.addRun(runID, "user-1", "run-generation-idle")
	listen := listener.next(t)
	listen.ready()

	connections := make([]*websocket.Conn, maximumSocketsPerSession)
	for index := range connections {
		connections[index] = harness.dial(t)
	}
	subscriptionID := func(socket, slot int) string { return fmt.Sprintf("idle-%d-%02d", socket, slot) }
	var waiting *fakeTimer
	subscriptions := int64(0)
	for index, connection := range connections {
		for slot := range maximumSubscriptions {
			subscribeRunStream(t, connection, subscriptionID(index, slot), runID, nil)
			if waiting == nil {
				// The first subscription starts the poller, which arms its
				// first timer before it serves that subscription.
				clock.next(t, runSafetyCatchUpInterval)
			}
			// Every subscription wakes the poller for one catch-up from its
			// own cursor; the poller then waits for the next interval.
			waiting = clock.next(t, runSafetyCatchUpInterval)
			subscriptions++
			if reads := runs.reads.Load(); reads != subscriptions {
				t.Fatalf("%d subscriptions cost %d Run event reads", subscriptions, reads)
			}
		}
	}
	if pollers, refs, served := runPollerState(harness.hub, runID); pollers != 1 ||
		refs != int(subscriptions) || served != int(subscriptions) {
		t.Fatalf("Run pollers=%d refs=%d served=%d for %d subscriptions", pollers, refs, served, subscriptions)
	}

	for interval := 1; interval <= 3; interval++ {
		waiting = catchUpOnce(t, runs, clock, runSafetyCatchUpInterval, func() { waiting.fire <- time.Time{} })
	}
	runs.append(testLifecycleEvent(runID, 1))
	waiting = catchUpOnce(t, runs, clock, runSafetyCatchUpInterval, func() { listen.consume(runID) })
	for index, connection := range connections {
		received := make(map[string]bool, maximumSubscriptions)
		for range maximumSubscriptions {
			frame := readServerJSON(t, connection)
			id, _ := frame["subscriptionId"].(string)
			if frame["type"] != "event" || nestedString(t, frame, "cursor", "sequence") != "1" || received[id] {
				t.Fatalf("socket %d notified frame = %#v", index, frame)
			}
			received[id] = true
		}
		for slot := range maximumSubscriptions {
			if !received[subscriptionID(index, slot)] {
				t.Fatalf("subscription %s missed the notified event", subscriptionID(index, slot))
			}
		}
	}
	catchUpOnce(t, runs, clock, runSafetyCatchUpInterval, func() { waiting.fire <- time.Time{} })

	for index, connection := range connections {
		for slot := range maximumSubscriptions {
			writeClientJSON(t, connection, map[string]any{
				"version": ProtocolVersion, "type": "unsubscribe", "subscriptionId": subscriptionID(index, slot),
			})
			if frame := readServerJSON(t, connection); frame["type"] != "unsubscribed" ||
				frame["subscriptionId"] != subscriptionID(index, slot) {
				t.Fatalf("unsubscribe ack = %#v", frame)
			}
		}
	}
	if pollers, _, _ := runPollerState(harness.hub, runID); pollers != 0 {
		t.Fatalf("the last unsubscribe left %d Run pollers", pollers)
	}
}

// API-R72: without an established listener a Run poller catches up every
// catch-up interval. Once the listener is established it catches up at once,
// which delivers a commit that was never notified, and then waits the safety
// interval; when the listener fails it catches up at once and returns to the
// catch-up interval until the listener is established again.
func TestRunCatchUpFollowsTheNotificationListener(t *testing.T) {
	const catchUp = 250 * time.Millisecond
	clock := newFakeClock()
	listener := newFakeRunListener()
	runs := &countingRunRepository{fakeRunRepository: newFakeRunRepository()}
	harness := newEventHarness(t, func(options *Options) {
		options.Runs, options.RunNotifications, options.clock = runs, listener, clock
		options.CatchUpInterval = catchUp
	})
	const runID = "run-listener"
	runs.addRun(runID, "user-1", "run-generation-listener")
	first := listener.next(t)
	connection := harness.dial(t)
	subscribeRunStream(t, connection, "listener", runID, nil)
	clock.next(t, catchUp)
	waiting := clock.next(t, catchUp)
	waiting = catchUpOnce(t, runs, clock, catchUp, func() { waiting.fire <- time.Time{} })

	runs.append(testLifecycleEvent(runID, 1))
	waiting = catchUpOnce(t, runs, clock, runSafetyCatchUpInterval, first.ready)
	expectRunEvents(t, connection, "listener", 1, 1)
	waiting = catchUpOnce(t, runs, clock, runSafetyCatchUpInterval, func() { waiting.fire <- time.Time{} })

	waiting = catchUpOnce(t, runs, clock, catchUp, func() { first.fail <- errors.New("listener connection lost") })
	retry := clock.next(t, runListenerRetryDelay)
	runs.append(testLifecycleEvent(runID, 2))
	catchUpOnce(t, runs, clock, catchUp, func() { waiting.fire <- time.Time{} })
	expectRunEvents(t, connection, "listener", 2, 2)

	retry.fire <- time.Time{}
	second := listener.next(t)
	catchUpOnce(t, runs, clock, runSafetyCatchUpInterval, second.ready)
	runs.append(testLifecycleEvent(runID, 3))
	catchUpOnce(t, runs, clock, runSafetyCatchUpInterval, func() { second.consume(runID) })
	expectRunEvents(t, connection, "listener", 3, 3)
}

// API-R49, API-R72: subscriptions to one Run from different cursors share its
// poller, and each still receives exactly the events after its own cursor, in
// order, across replay pages and live delivery.
func TestRunSubscriptionsFromDifferentCursorsShareOnePoller(t *testing.T) {
	harness := newEventHarness(t, func(options *Options) { options.clock = newFakeClock() })
	const (
		runID      = "run-cursors"
		generation = "run-generation-cursors"
		committed  = runReplayPageSize + 72
	)
	harness.runs.addRun(runID, "user-1", generation)
	for sequence := int64(1); sequence <= committed; sequence++ {
		harness.runs.append(testLifecycleEvent(runID, sequence))
	}
	early, late := harness.dial(t), harness.dial(t)
	subscribeRunStream(t, early, "early", runID, &Cursor{Generation: generation, Sequence: "0"})
	subscribeRunStream(t, late, "late", runID, &Cursor{Generation: generation, Sequence: "150"})
	expectRunEvents(t, early, "early", 1, committed)
	expectRunEvents(t, late, "late", 151, committed)
	if pollers, refs, served := runPollerState(harness.hub, runID); pollers != 1 || refs != 2 || served != 2 {
		t.Fatalf("Run pollers=%d refs=%d served=%d for two subscriptions", pollers, refs, served)
	}
	harness.runs.append(testLifecycleEvent(runID, committed+1))
	harness.hub.notifyRun(runID)
	expectRunEvents(t, early, "early", committed+1, committed+1)
	expectRunEvents(t, late, "late", committed+1, committed+1)
}

// A failed shared read ends every subscription of the Run with the retryable
// error that each subscription received from its own read before.
func TestFailedRunReadEndsEverySubscriptionWithRetryableError(t *testing.T) {
	runs := &failingRunRepository{fakeRunRepository: newFakeRunRepository()}
	harness := newEventHarness(t, func(options *Options) {
		options.Runs, options.clock = runs, newFakeClock()
	})
	const runID = "run-failing"
	runs.addRun(runID, "user-1", "run-generation-failing")
	connection := harness.dial(t)
	subscribeRunStream(t, connection, "first", runID, nil)
	subscribeRunStream(t, connection, "second", runID, nil)
	runs.failing.Store(true)
	harness.hub.notifyRun(runID)
	failed := make(map[string]bool, 2)
	for range 2 {
		frame := readServerJSON(t, connection)
		id, _ := frame["subscriptionId"].(string)
		if frame["type"] != "error" || frame["code"] != "overloaded" || frame["retryable"] != true || failed[id] {
			t.Fatalf("failed Run read frame = %#v", frame)
		}
		failed[id] = true
	}
	if !failed["first"] || !failed["second"] {
		t.Fatalf("subscriptions ended by the failed read = %v", failed)
	}
}

// API-R64: a poller that cannot enqueue a frame for one subscription ends
// only that subscription and leaves closing its socket with 1013 to the
// subscription, so it never blocks on a slow socket and keeps serving the
// other subscriptions of the Run.
func TestRunPollerEndsOnlyTheSlowSubscription(t *testing.T) {
	harness := newEventHarness(t, func(options *Options) { options.clock = newFakeClock() })
	const runID = "run-slow"
	harness.runs.addRun(runID, "user-1", "run-generation-slow")
	harness.runs.append(testLifecycleEvent(runID, 1))
	slow, slowSubscriber := newBareRunSubscriber(t, harness.hub, "slow", runID)
	fast, fastSubscriber := newBareRunSubscriber(t, harness.hub, "fast", runID)
	for range maximumQueuedFrames {
		if queued, full := slow.offer([]byte(`{}`)); !queued || full {
			t.Fatal("bounded queue rejected an in-limit frame")
		}
	}
	slowPoller, attached := harness.hub.attachRun(runID, slowSubscriber)
	if !attached {
		t.Fatal("open Hub refused a Run subscription")
	}
	defer harness.hub.detachRun(slowPoller, slowSubscriber)
	fastPoller, attached := harness.hub.attachRun(runID, fastSubscriber)
	if !attached {
		t.Fatal("open Hub refused a Run subscription")
	}
	defer harness.hub.detachRun(fastPoller, fastSubscriber)

	select {
	case <-slowSubscriber.ended:
	case <-time.After(5 * time.Second):
		t.Fatal("the poller did not end the slow subscription")
	}
	if slowSubscriber.closeCode != websocket.CloseTryAgainLater || slow.closeCode.Load() != 0 || slow.ctx.Err() != nil {
		t.Fatalf("slow subscription close=%d socket close=%d socket err=%v",
			slowSubscriber.closeCode, slow.closeCode.Load(), slow.ctx.Err())
	}
	select {
	case encoded := <-fast.queue.channel():
		var frame map[string]any
		if err := json.Unmarshal(encoded, &frame); err != nil || frame["subscriptionId"] != "fast" ||
			nestedString(t, frame, "cursor", "sequence") != "1" {
			t.Fatalf("fast subscription frame = %s, %v", encoded, err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("the slow subscription stopped delivery to the fast one")
	}
	select {
	case <-fastSubscriber.ended:
		t.Fatal("the slow subscription ended the fast one")
	default:
	}
}

// API-R72, API-R110: a page that ends on its data budget while committed
// events follow does not end the catch-up. The poller reads on at once and
// delivers every event in order in the same catch-up, without another
// wake-up or timer, and the page that reports no further committed event
// ends it, also when that page fills its budget exactly.
func TestRunCatchUpContinuesAcrossByteShortPages(t *testing.T) {
	clock := newFakeClock()
	runs := &countingRunRepository{fakeRunRepository: newFakeRunRepository()}
	harness := newEventHarness(t, func(options *Options) { options.Runs, options.clock = runs, clock })
	const runID, generation = "run-byte-short", "run-generation-byte-short"
	runs.addRun(runID, "user-1", generation)
	// Run and Stage lifecycle events alternate and a page holds exactly one
	// of each, far below its event count: ten events take five pages.
	mixed := func(sequence int64) runstore.WorkflowRunEvent {
		if sequence%2 == 1 {
			return testLifecycleEvent(runID, sequence)
		}
		return testStageLifecycleEvent(runID, sequence)
	}
	runs.setPageDataBytes(len(mixed(1).Data) + len(mixed(2).Data))
	for sequence := int64(1); sequence <= 10; sequence++ {
		runs.append(mixed(sequence))
	}
	connection := harness.dial(t)
	subscribeRunStream(t, connection, "byte-short", runID, &Cursor{Generation: generation, Sequence: "0"})
	// The subscription starts the poller, which arms its first timer before
	// the subscription wakes it; it arms the next one when the catch-up ends.
	clock.next(t, harness.hub.catchUp)
	clock.next(t, harness.hub.catchUp)
	if reads := runs.reads.Load(); reads != 5 {
		t.Fatalf("one catch-up across byte-short pages cost %d reads, want 5", reads)
	}
	expectRunEvents(t, connection, "byte-short", 1, 10)

	runs.append(mixed(11))
	runs.append(mixed(12))
	catchUpOnce(t, runs, clock, harness.hub.catchUp, func() { harness.hub.notifyRun(runID) })
	expectRunEvents(t, connection, "byte-short", 11, 12)
}

// API-R110: a catch-up ends at the page that reports no further committed
// event, even when that page holds its full event count.
func TestRunCatchUpEndsAtTheLastCommittedEvent(t *testing.T) {
	clock := newFakeClock()
	runs := &countingRunRepository{fakeRunRepository: newFakeRunRepository()}
	harness := newEventHarness(t, func(options *Options) { options.Runs, options.clock = runs, clock })
	const runID, generation = "run-full-page", "run-generation-full-page"
	runs.addRun(runID, "user-1", generation)
	for sequence := int64(1); sequence <= runReplayPageSize; sequence++ {
		runs.append(testLifecycleEvent(runID, sequence))
	}
	connection := harness.dial(t)
	subscribeRunStream(t, connection, "full-page", runID, &Cursor{Generation: generation, Sequence: "0"})
	clock.next(t, harness.hub.catchUp)
	clock.next(t, harness.hub.catchUp)
	if reads := runs.reads.Load(); reads != 1 {
		t.Fatalf("a catch-up of exactly one full page cost %d reads, want 1", reads)
	}
	expectRunEvents(t, connection, "full-page", 1, runReplayPageSize)
}

// API-R49, API-R72, API-R110: subscriptions from different cursors each
// receive exactly the events after their own cursor across byte-short pages.
func TestRunSubscriptionsFromDifferentCursorsReplayAcrossByteShortPages(t *testing.T) {
	runs := newFakeRunRepository()
	harness := newEventHarness(t, func(options *Options) { options.Runs, options.clock = runs, newFakeClock() })
	const runID, generation = "run-byte-short-cursors", "run-generation-byte-short-cursors"
	runs.addRun(runID, "user-1", generation)
	runs.setPageDataBytes(3 * len(testLifecycleEvent(runID, 1).Data))
	for sequence := int64(1); sequence <= 20; sequence++ {
		runs.append(testLifecycleEvent(runID, sequence))
	}
	early, late := harness.dial(t), harness.dial(t)
	subscribeRunStream(t, early, "early", runID, &Cursor{Generation: generation, Sequence: "0"})
	subscribeRunStream(t, late, "late", runID, &Cursor{Generation: generation, Sequence: "11"})
	expectRunEvents(t, early, "early", 1, 20)
	expectRunEvents(t, late, "late", 12, 20)
}

// API-R60, API-R110: a gap or an undeliverable event behind a byte-short page
// ends the subscription with resync_required in the same catch-up, as within
// one page, and a failed read behind it ends the subscription with the
// retryable error; no event is skipped and no timer is needed.
func TestRunCatchUpChecksEventsBehindByteShortPages(t *testing.T) {
	for _, test := range []struct {
		name   string
		fourth func(runID string) runstore.WorkflowRunEvent
		failed bool
		final  func(frame map[string]any) bool
	}{
		{
			name:   "sequence gap",
			fourth: func(runID string) runstore.WorkflowRunEvent { return testLifecycleEvent(runID, 5) },
			final: func(frame map[string]any) bool {
				return frame["type"] == "resync_required" && frame["reason"] == "sequence_gap"
			},
		},
		{
			name: "undeliverable event",
			fourth: func(runID string) runstore.WorkflowRunEvent {
				event := testLifecycleEvent(runID, 4)
				event.Data = json.RawMessage(`{"runId":"run-other","resource":"run","state":"running"}`)
				return event
			},
			final: func(frame map[string]any) bool {
				return frame["type"] == "resync_required" && frame["reason"] == "cursor_unavailable"
			},
		},
		{
			name:   "failed read",
			fourth: func(runID string) runstore.WorkflowRunEvent { return testLifecycleEvent(runID, 4) },
			failed: true,
			final: func(frame map[string]any) bool {
				return frame["type"] == "error" && frame["code"] == "overloaded" && frame["retryable"] == true
			},
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			runs := &countingRunRepository{fakeRunRepository: newFakeRunRepository()}
			if test.failed {
				runs.during = func(_ context.Context, read int64) error {
					if read > 1 {
						return errors.New("database unavailable")
					}
					return nil
				}
			}
			harness := newEventHarness(t, func(options *Options) { options.Runs, options.clock = runs, newFakeClock() })
			runID := "run-behind-" + strings.ReplaceAll(test.name, " ", "-")
			generation := "generation-" + runID
			runs.addRun(runID, "user-1", generation)
			runs.setPageDataBytes(3 * len(testLifecycleEvent(runID, 1).Data))
			for sequence := int64(1); sequence <= 3; sequence++ {
				runs.append(testLifecycleEvent(runID, sequence))
			}
			runs.append(test.fourth(runID))
			connection := harness.dial(t)
			subscribeRunStream(t, connection, "behind", runID, &Cursor{Generation: generation, Sequence: "0"})
			expectRunEvents(t, connection, "behind", 1, 3)
			if frame := readServerJSON(t, connection); frame["subscriptionId"] != "behind" || !test.final(frame) {
				t.Fatalf("frame behind a byte-short page = %#v", frame)
			}
		})
	}
}

// API-R64, API-R110: reading behind a byte-short page stops with the
// subscriptions that need it. A subscription the poller ends as a slow
// consumer, a socket that closes during a read and a Hub that shuts down
// during a read each end the catch-up at once, although the last page
// reported further committed events.
func TestRunCatchUpBehindByteShortPagesStopsWithItsSubscriptions(t *testing.T) {
	for _, test := range []struct {
		name string
		// prefill frames are queued before the subscription attaches.
		prefill int
		// during acts in the given read.
		during    func(hub *Hub, current *socket)
		duringRun int64
		reads     int64
		events    int
		ended     bool
	}{
		{name: "slow consumer", prefill: maximumQueuedFrames - 2, reads: 1, events: 2, ended: true},
		{
			name: "socket closed during a read", during: func(_ *Hub, current *socket) { current.cancel() },
			duringRun: 2, reads: 2, events: 3,
		},
		{
			name: "Hub shut down during a read", during: func(hub *Hub, _ *socket) { hub.cancel() },
			duringRun: 2, reads: 2, events: 3,
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			clock := newFakeClock()
			runs := &countingRunRepository{fakeRunRepository: newFakeRunRepository()}
			harness := newEventHarness(t, func(options *Options) { options.Runs, options.clock = runs, clock })
			runID := "run-stop-" + strings.ReplaceAll(test.name, " ", "-")
			runs.addRun(runID, "user-1", "generation-"+runID)
			runs.setPageDataBytes(3 * len(testLifecycleEvent(runID, 1).Data))
			for sequence := int64(1); sequence <= 10; sequence++ {
				runs.append(testLifecycleEvent(runID, sequence))
			}
			current, subscriber := newBareRunSubscriber(t, harness.hub, "stop", runID)
			for range test.prefill {
				if queued, full := current.offer([]byte(`{}`)); !queued || full {
					t.Fatal("bounded queue rejected an in-limit frame")
				}
			}
			if test.during != nil {
				runs.during = func(_ context.Context, read int64) error {
					if read == test.duringRun {
						test.during(harness.hub, current)
					}
					return nil
				}
			}
			poller, attached := harness.hub.attachRun(runID, subscriber)
			if !attached {
				t.Fatal("open Hub refused a Run subscription")
			}
			defer harness.hub.detachRun(poller, subscriber)
			clock.next(t, harness.hub.catchUp)
			clock.next(t, harness.hub.catchUp)
			if reads := runs.reads.Load(); reads != test.reads {
				t.Fatalf("catch-up read %d pages, want %d", reads, test.reads)
			}
			for range test.prefill {
				<-current.queue.channel()
			}
			for sequence := int64(1); sequence <= int64(test.events); sequence++ {
				select {
				case encoded := <-current.queue.channel():
					var frame map[string]any
					if err := json.Unmarshal(encoded, &frame); err != nil || frame["type"] != "event" ||
						nestedString(t, frame, "cursor", "sequence") != strconv.FormatInt(sequence, 10) {
						t.Fatalf("frame %d = %s, %v", sequence, encoded, err)
					}
				default:
					t.Fatalf("the subscription received %d events, want %d", sequence-1, test.events)
				}
			}
			select {
			case encoded := <-current.queue.channel():
				t.Fatalf("unexpected frame after %d events: %s", test.events, encoded)
			default:
			}
			select {
			case <-subscriber.ended:
				if !test.ended || subscriber.closeCode != websocket.CloseTryAgainLater {
					t.Fatalf("subscription ended with close %d", subscriber.closeCode)
				}
			default:
				if test.ended {
					t.Fatal("the slow subscription was not ended")
				}
			}
		})
	}
}

// catchUpOnce triggers one wake-up of an idle Run poller and requires that it
// read the Run exactly once before it waits again for want.
func catchUpOnce(
	t *testing.T,
	runs *countingRunRepository,
	clock *fakeClock,
	want time.Duration,
	trigger func(),
) *fakeTimer {
	t.Helper()
	before := runs.reads.Load()
	trigger()
	waiting := clock.next(t, want)
	if reads := runs.reads.Load() - before; reads != 1 {
		t.Fatalf("one catch-up cost %d Run event reads", reads)
	}
	return waiting
}

func subscribeRunStream(t *testing.T, connection *websocket.Conn, id, runID string, after *Cursor) {
	t.Helper()
	frame := map[string]any{
		"version": ProtocolVersion, "type": "subscribe", "subscriptionId": id,
		"stream": map[string]any{"kind": "run", "id": runID},
	}
	if after != nil {
		frame["after"] = map[string]any{"generation": after.Generation, "sequence": after.Sequence}
	}
	writeClientJSON(t, connection, frame)
	if ack := readServerJSON(t, connection); ack["type"] != "subscribed" || ack["subscriptionId"] != id {
		t.Fatalf("Run subscription ack = %#v", ack)
	}
}

func expectRunEvents(t *testing.T, connection *websocket.Conn, id string, first, last int64) {
	t.Helper()
	for sequence := first; sequence <= last; sequence++ {
		frame := readServerJSON(t, connection)
		if frame["type"] != "event" || frame["subscriptionId"] != id ||
			nestedString(t, frame, "cursor", "sequence") != strconv.FormatInt(sequence, 10) {
			t.Fatalf("Run event %d for %s = %#v", sequence, id, frame)
		}
	}
}

// runPollerState reports how many Run pollers the Hub runs and, for runID,
// how many subscriptions are attached to its poller and how many it serves.
func runPollerState(hub *Hub, runID string) (pollers, refs, served int) {
	hub.mu.Lock()
	defer hub.mu.Unlock()
	poller := hub.runPollers[runID]
	if poller == nil {
		return len(hub.runPollers), 0, 0
	}
	poller.mu.Lock()
	defer poller.mu.Unlock()
	return len(hub.runPollers), poller.refs, len(poller.subscribers)
}

// newBareRunSubscriber builds a Run subscription on a socket without a
// connection or writer, so its queue keeps every frame the poller enqueues.
func newBareRunSubscriber(t *testing.T, hub *Hub, id, runID string) (*socket, *runSubscriber) {
	t.Helper()
	ctx, cancel := context.WithCancel(hub.ctx)
	t.Cleanup(cancel)
	current := &socket{hub: hub, ctx: ctx, cancel: cancel, queue: newOutboundQueue(), subs: map[string]*subscription{}}
	return current, &runSubscriber{
		socket: current, subscription: &subscription{id: id, stream: Stream{Kind: StreamRun, ID: runID}},
		ctx: ctx, generation: "run-generation-" + id, ended: make(chan struct{}),
	}
}

// countingRunRepository counts Run event reads, the cost that must not grow
// with the number of subscriptions. A test that sets during acts in each read
// before it is served; an error fails that read.
type countingRunRepository struct {
	*fakeRunRepository
	reads  atomic.Int64
	during func(ctx context.Context, read int64) error
}

func (r *countingRunRepository) ListRunEvents(
	ctx context.Context,
	runID string,
	after int64,
	limit int,
) (runstore.WorkflowRunEventPage, error) {
	read := r.reads.Add(1)
	if r.during != nil {
		if err := r.during(ctx, read); err != nil {
			return runstore.WorkflowRunEventPage{}, err
		}
	}
	return r.fakeRunRepository.ListRunEvents(ctx, runID, after, limit)
}

// failingRunRepository fails every Run event read once failing is set.
type failingRunRepository struct {
	*fakeRunRepository
	failing atomic.Bool
}

func (r *failingRunRepository) ListRunEvents(
	ctx context.Context,
	runID string,
	after int64,
	limit int,
) (runstore.WorkflowRunEventPage, error) {
	if r.failing.Load() {
		return runstore.WorkflowRunEventPage{}, errors.New("database unavailable")
	}
	return r.fakeRunRepository.ListRunEvents(ctx, runID, after, limit)
}

// fakeTimer is one timer requested from a fakeClock; the test fires it.
type fakeTimer struct {
	duration time.Duration
	fire     chan time.Time
}

// fakeClock hands every requested timer to the test instead of waiting for
// real time, so the test decides when each interval has passed.
type fakeClock struct {
	requests chan *fakeTimer
	// stash keeps the requests of other durations that next has seen; only
	// the test goroutine uses it.
	stash []*fakeTimer
}

func newFakeClock() *fakeClock {
	return &fakeClock{requests: make(chan *fakeTimer, 4096)}
}

func (c *fakeClock) After(duration time.Duration) <-chan time.Time {
	timer := &fakeTimer{duration: duration, fire: make(chan time.Time, 1)}
	c.requests <- timer
	return timer.fire
}

// next returns the oldest timer of the given duration requested so far,
// waiting for one when there is none.
func (c *fakeClock) next(t *testing.T, duration time.Duration) *fakeTimer {
	t.Helper()
	for index, timer := range c.stash {
		if timer.duration == duration {
			c.stash = append(c.stash[:index], c.stash[index+1:]...)
			return timer
		}
	}
	deadline := time.NewTimer(5 * time.Second)
	defer deadline.Stop()
	for {
		select {
		case timer := <-c.requests:
			if timer.duration == duration {
				return timer
			}
			c.stash = append(c.stash, timer)
		case <-deadline.C:
			t.Fatalf("no %s timer was requested; other timers: %d", duration, len(c.stash))
			return nil
		}
	}
}

// fakeRunListener hands every Listen call to the test, which decides when the
// listener is established, what it notifies and when it fails.
type fakeRunListener struct {
	calls chan *fakeListen
}

type fakeListen struct {
	ready   func()
	consume func(string)
	fail    chan error
}

func newFakeRunListener() *fakeRunListener {
	return &fakeRunListener{calls: make(chan *fakeListen, 4)}
}

func (l *fakeRunListener) Listen(ctx context.Context, ready func(), consume func(string)) error {
	call := &fakeListen{ready: ready, consume: consume, fail: make(chan error, 1)}
	select {
	case l.calls <- call:
	case <-ctx.Done():
		return ctx.Err()
	}
	select {
	case err := <-call.fail:
		return err
	case <-ctx.Done():
		return ctx.Err()
	}
}

func (l *fakeRunListener) next(t *testing.T) *fakeListen {
	t.Helper()
	select {
	case call := <-l.calls:
		return call
	case <-time.After(5 * time.Second):
		t.Fatal("the Hub did not start the Run notification listener")
		return nil
	}
}
