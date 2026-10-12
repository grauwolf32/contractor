package events

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strconv"
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
// with the number of subscriptions.
type countingRunRepository struct {
	*fakeRunRepository
	reads atomic.Int64
}

func (r *countingRunRepository) ListRunEvents(
	ctx context.Context,
	runID string,
	after int64,
	limit int,
) ([]runstore.WorkflowRunEvent, error) {
	r.reads.Add(1)
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
) ([]runstore.WorkflowRunEvent, error) {
	if r.failing.Load() {
		return nil, errors.New("database unavailable")
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
