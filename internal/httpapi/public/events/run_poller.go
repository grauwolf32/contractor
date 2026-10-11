package events

import (
	"context"
	"sync"
	"time"

	"github.com/gorilla/websocket"
	"github.com/grauwolf32/contractor/internal/runstore"
)

const (
	runReplayPageSize = 128
	// runSafetyCatchUpInterval replaces the catch-up interval of every Run
	// poller while the Run notification listener is established
	// (LIM-API-22). Notifications then wake the pollers for every commit; the
	// periodic read only bounds how long a lost notification delays delivery.
	runSafetyCatchUpInterval = 15 * time.Second
	// runListenerRetryDelay separates attempts to establish the Run
	// notification listener.
	runListenerRetryDelay = time.Second
)

// clock schedules the Run catch-up and listener retry timers; tests replace
// it to count reads per interval without waiting for real time.
type clock interface {
	After(time.Duration) <-chan time.Time
}

type realClock struct{}

func (realClock) After(duration time.Duration) <-chan time.Time { return time.After(duration) }

// runPoller reads the events of one Run for every subscription of that Run
// (API-R72), so the cost of a catch-up does not grow with the number of
// subscriptions. A committed-event notification, a new subscription and a
// change of the listener state wake it; without a wake-up it catches up once
// per catch-up interval. It only enqueues frames and never blocks on a socket:
// a subscription it ends closes its own socket when that is needed.
type runPoller struct {
	hub    *Hub
	runID  string
	ctx    context.Context
	cancel context.CancelFunc
	// wake holds at most one pending wake-up, so wake-ups that arrive during
	// a round coalesce into one more round.
	wake chan struct{}
	// refs counts attached subscriptions and is guarded by hub.mu; the last
	// detach removes the poller from the Hub and stops it.
	refs int

	// mu serializes delivery with attaching and detaching subscriptions.
	mu          sync.Mutex
	subscribers map[*runSubscriber]struct{}
}

// runSubscriber is one Run subscription that a runPoller serves after the
// subscription has sent its subscribed frame.
type runSubscriber struct {
	socket       *socket
	subscription *subscription
	ctx          context.Context
	generation   string
	// last is the sequence of the last event delivered to the subscription;
	// the poller's mu guards it.
	last int64
	// ended is closed when the poller ends the subscription. A non-zero
	// closeCode then asks the subscription to close its socket with it.
	ended       chan struct{}
	closeCode   int
	closeReason string
}

func newRunPoller(hub *Hub, runID string) *runPoller {
	ctx, cancel := context.WithCancel(hub.ctx)
	return &runPoller{
		hub: hub, runID: runID, ctx: ctx, cancel: cancel,
		wake: make(chan struct{}, 1), subscribers: make(map[*runSubscriber]struct{}),
	}
}

// run catches up after every wake-up and after every catch-up interval that
// passes without one. The interval is chosen anew before each wait; a change
// of the listener state also wakes the poller, so the new interval applies
// at once.
func (p *runPoller) run() {
	defer p.hub.pollers.Done()
	for {
		expired := p.hub.clock.After(p.hub.runCatchUpInterval())
		select {
		case <-p.ctx.Done():
			return
		case <-p.wake:
		case <-expired:
		}
		p.catchUp()
	}
}

// signal wakes the poller without blocking.
func (p *runPoller) signal() {
	select {
	case p.wake <- struct{}{}:
	default:
	}
}

// add starts serving a subscription and then wakes the poller, so a round
// that did not see the subscription is always followed by one that does.
func (p *runPoller) add(subscriber *runSubscriber) {
	p.mu.Lock()
	p.subscribers[subscriber] = struct{}{}
	p.mu.Unlock()
	p.signal()
}

// remove stops serving a subscription. It returns only after any delivery
// that could still enqueue a frame for the subscription, so no frame can
// follow its unsubscribed frame (API-R46).
func (p *runPoller) remove(subscriber *runSubscriber) {
	p.mu.Lock()
	delete(p.subscribers, subscriber)
	p.mu.Unlock()
}

// catchUp delivers every committed event after the last delivered sequence of
// each subscription. It reads one page at a time from the lowest of those
// sequences, so idle subscriptions cost one read per round however many
// there are.
func (p *runPoller) catchUp() {
	for p.ctx.Err() == nil {
		from, ok := p.lowestDelivered()
		if !ok {
			return
		}
		events, err := p.hub.runs.ListRunEvents(p.ctx, p.runID, from, runReplayPageSize)
		if err != nil {
			if p.ctx.Err() == nil {
				p.endAll("overloaded", "Run stream is temporarily unavailable")
			}
			return
		}
		p.deliver(from, events)
		if len(events) < runReplayPageSize {
			return
		}
	}
}

func (p *runPoller) lowestDelivered() (int64, bool) {
	p.mu.Lock()
	defer p.mu.Unlock()
	lowest, found := int64(0), false
	for subscriber := range p.subscribers {
		if subscriber.ctx.Err() == nil && (!found || subscriber.last < lowest) {
			lowest, found = subscriber.last, true
		}
	}
	return lowest, found
}

// deliver hands one page read after sequence from to every subscription that
// has received everything up to from. A subscription that attached during the
// read may need earlier events; its attach woke the poller, so the next round
// reads from its sequence.
func (p *runPoller) deliver(from int64, events []runstore.WorkflowRunEvent) {
	p.mu.Lock()
	defer p.mu.Unlock()
	for subscriber := range p.subscribers {
		if subscriber.last >= from && subscriber.ctx.Err() == nil {
			p.deliverLocked(subscriber, events)
		}
	}
}

// deliverLocked enqueues the events after the subscription's last delivered
// sequence in order. Any other sequence than the next one is a gap that
// requires a resync, as does an event that cannot be published.
func (p *runPoller) deliverLocked(subscriber *runSubscriber, events []runstore.WorkflowRunEvent) {
	current := subscriber.subscription
	for _, event := range events {
		if event.RunID != p.runID {
			p.endLocked(subscriber, p.resync(subscriber, "sequence_gap"))
			return
		}
		if event.SequenceNumber <= subscriber.last {
			continue
		}
		if event.SequenceNumber != subscriber.last+1 {
			p.endLocked(subscriber, p.resync(subscriber, "sequence_gap"))
			return
		}
		encoded, err := runEventServerFrame(current.id, current.stream, subscriber.generation, event)
		if err != nil {
			p.endLocked(subscriber, p.resync(subscriber, "cursor_unavailable"))
			return
		}
		if !p.sendLocked(subscriber, encoded) {
			p.endLocked(subscriber, nil)
			return
		}
		subscriber.last = event.SequenceNumber
	}
}

func (p *runPoller) resync(subscriber *runSubscriber, reason string) resyncRequiredFrame {
	return resyncRequiredFrame{
		Version: ProtocolVersion, Type: "resync_required",
		SubscriptionID: subscriber.subscription.id, Stream: subscriber.subscription.stream, Reason: reason,
	}
}

// endAll ends every subscription after a failed read with a retryable error,
// and a subscription that is already being cancelled without one.
func (p *runPoller) endAll(code, message string) {
	p.mu.Lock()
	defer p.mu.Unlock()
	for subscriber := range p.subscribers {
		if subscriber.ctx.Err() != nil {
			p.endLocked(subscriber, nil)
			continue
		}
		p.endLocked(subscriber, errorFrame{
			Version: ProtocolVersion, Type: "error", SubscriptionID: subscriber.subscription.id,
			Code: code, Message: message, Retryable: true,
		})
	}
}

// sendLocked offers one frame to the subscription's socket and reports whether
// it was queued. A full queue of an open socket makes it a slow consumer: the
// subscription then closes the socket with status 1013 (API-R64) itself, so a
// blocking close never holds the poller.
func (p *runPoller) sendLocked(subscriber *runSubscriber, encoded []byte) bool {
	queued, full := subscriber.socket.offer(encoded)
	if full {
		subscriber.closeCode, subscriber.closeReason = websocket.CloseTryAgainLater, "slow consumer"
	}
	return queued
}

// endLocked ends a subscription after offering its final frame, if any.
func (p *runPoller) endLocked(subscriber *runSubscriber, final any) {
	if final != nil {
		encoded, err := marshalServerFrame(final)
		if err != nil {
			subscriber.closeCode, subscriber.closeReason = websocket.CloseInternalServerErr, "invalid Server frame"
		} else {
			p.sendLocked(subscriber, encoded)
		}
	}
	delete(p.subscribers, subscriber)
	close(subscriber.ended)
}
