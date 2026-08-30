package events

import (
	"context"
	"sync"
	"sync/atomic"
	"time"

	"github.com/gorilla/websocket"
	"github.com/grauwolf32/contractor/internal/auth"
)

type socket struct {
	hub     *Hub
	conn    *websocket.Conn
	session auth.Session
	ctx     context.Context
	cancel  context.CancelFunc
	queue   *outboundQueue

	closeOnce   sync.Once
	closeCode   atomic.Int32
	lastOutput  atomic.Int64
	pendingPing atomic.Int64

	subMu sync.Mutex
	subs  map[string]*subscription
	subWG sync.WaitGroup
	ioWG  sync.WaitGroup
}

type subscription struct {
	id     string
	stream Stream
	cancel context.CancelFunc
	done   chan struct{}
}

func newSocket(hub *Hub, connection *websocket.Conn, session auth.Session) *socket {
	ctx, cancel := context.WithCancel(hub.ctx)
	return &socket{
		hub: hub, conn: connection, session: session,
		ctx: ctx, cancel: cancel, queue: newOutboundQueue(),
		subs: make(map[string]*subscription),
	}
}

func (s *socket) run() {
	now := time.Now().UnixNano()
	s.lastOutput.Store(now)
	s.conn.SetReadLimit(maximumClientFrameBytes)
	s.conn.SetPongHandler(func(string) error {
		s.pendingPing.Store(0)
		return nil
	})
	s.ioWG.Add(2)
	go s.writeLoop()
	go s.sessionLoop()
	s.readLoop()
	s.cancel()
	s.cancelSubscriptions()
	s.stop(websocket.CloseNormalClosure, "connection closed")
	s.ioWG.Wait()
}

func (s *socket) readLoop() {
	for s.ctx.Err() == nil {
		messageType, payload, err := s.conn.ReadMessage()
		if err != nil {
			return
		}
		if messageType != websocket.TextMessage {
			s.stop(websocket.CloseUnsupportedData, "text frames are required")
			return
		}
		frame, err := decodeClientFrame(payload)
		if err != nil {
			s.sendError("", "invalid_frame", "frame does not satisfy the protocol", false)
			continue
		}
		switch frame.typ {
		case "subscribe":
			s.startSubscription(frame)
		case "unsubscribe":
			s.unsubscribe(frame.subscriptionID)
		}
	}
}

func (s *socket) writeLoop() {
	defer s.ioWG.Done()
	tickEvery := s.hub.pingInterval / 4
	if tickEvery > time.Second {
		tickEvery = time.Second
	}
	if tickEvery <= 0 {
		tickEvery = time.Millisecond
	}
	ticker := time.NewTicker(tickEvery)
	defer ticker.Stop()
	for {
		select {
		case <-s.ctx.Done():
			return
		case frame := <-s.queue.channel():
			s.queue.consumed(frame)
			if err := s.conn.SetWriteDeadline(time.Now().Add(s.hub.writeTimeout)); err != nil {
				s.stop(websocket.CloseInternalServerErr, "write deadline failed")
				return
			}
			if err := s.conn.WriteMessage(websocket.TextMessage, frame); err != nil {
				s.stop(websocket.CloseGoingAway, "write failed")
				return
			}
			s.lastOutput.Store(time.Now().UnixNano())
		case <-ticker.C:
			now := time.Now()
			sendPing, timedOut := s.heartbeatState(now)
			if timedOut {
				s.stop(websocket.CloseGoingAway, "pong timeout")
				return
			}
			if !sendPing {
				continue
			}
			s.pendingPing.Store(now.UnixNano())
			if err := s.conn.WriteControl(
				websocket.PingMessage, nil, now.Add(s.hub.writeTimeout),
			); err != nil {
				s.stop(websocket.CloseGoingAway, "ping failed")
				return
			}
			s.lastOutput.Store(now.UnixNano())
		}
	}
}

func (s *socket) heartbeatState(now time.Time) (sendPing bool, timedOut bool) {
	if sentAt := s.pendingPing.Load(); sentAt != 0 {
		return false, now.Sub(time.Unix(0, sentAt)) >= s.hub.pongTimeout
	}
	return now.Sub(time.Unix(0, s.lastOutput.Load())) >= s.hub.pingInterval, false
}

func (s *socket) sessionLoop() {
	defer s.ioWG.Done()
	ticker := time.NewTicker(s.hub.sessionCheck)
	defer ticker.Stop()
	for {
		select {
		case <-s.ctx.Done():
			return
		case <-ticker.C:
			if _, err := s.hub.authentication.Check(s.session.Handle); err != nil {
				s.stop(websocket.ClosePolicyViolation, "session expired")
				return
			}
		}
	}
}

func (s *socket) startSubscription(frame clientFrame) {
	s.subMu.Lock()
	if _, duplicate := s.subs[frame.subscriptionID]; duplicate {
		s.subMu.Unlock()
		s.sendError(frame.subscriptionID, "invalid_frame", "subscriptionId is already active", false)
		return
	}
	if len(s.subs) >= maximumSubscriptions {
		s.subMu.Unlock()
		s.sendError(frame.subscriptionID, "subscription_limit", "subscription limit reached", false)
		return
	}
	ctx, cancel := context.WithCancel(s.ctx)
	current := &subscription{
		id: frame.subscriptionID, stream: frame.stream,
		cancel: cancel, done: make(chan struct{}),
	}
	s.subs[current.id] = current
	s.subWG.Add(1)
	s.subMu.Unlock()
	go func() {
		defer s.subWG.Done()
		defer close(current.done)
		defer func() {
			s.subMu.Lock()
			if s.subs[current.id] == current {
				delete(s.subs, current.id)
			}
			s.subMu.Unlock()
		}()
		s.pumpSubscription(ctx, current, frame.after)
	}()
}

func (s *socket) unsubscribe(id string) {
	s.subMu.Lock()
	current := s.subs[id]
	s.subMu.Unlock()
	if current == nil {
		s.sendError(id, "not_found", "subscription was not found", false)
		return
	}
	current.cancel()
	select {
	case <-current.done:
	case <-s.ctx.Done():
		return
	}
	s.sendFrame(unsubscribedFrame{
		Version: ProtocolVersion, Type: "unsubscribed", SubscriptionID: id,
	})
}

func (s *socket) cancelSubscriptions() {
	s.subMu.Lock()
	current := make([]*subscription, 0, len(s.subs))
	for _, value := range s.subs {
		current = append(current, value)
	}
	s.subMu.Unlock()
	for _, value := range current {
		value.cancel()
	}
	s.subWG.Wait()
}

func (s *socket) sendFrame(frame any) bool {
	encoded, err := marshalServerFrame(frame)
	if err != nil {
		s.stop(websocket.CloseInternalServerErr, "invalid Server frame")
		return false
	}
	return s.sendEncoded(encoded)
}

func (s *socket) sendEncoded(encoded []byte) bool {
	if s.ctx.Err() != nil {
		return false
	}
	if !s.queue.enqueue(encoded) {
		s.stop(websocket.CloseTryAgainLater, "slow consumer")
		return false
	}
	return true
}

func (s *socket) sendError(subscriptionID, code, message string, retryable bool) bool {
	return s.sendFrame(errorFrame{
		Version: ProtocolVersion, Type: "error", SubscriptionID: subscriptionID,
		Code: code, Message: message, Retryable: retryable,
	})
}

func (s *socket) sendResync(subscriptionID string, stream Stream, reason string) bool {
	return s.sendFrame(resyncRequiredFrame{
		Version: ProtocolVersion, Type: "resync_required",
		SubscriptionID: subscriptionID, Stream: stream, Reason: reason,
	})
}

func (s *socket) stop(code int, reason string) {
	s.closeOnce.Do(func() {
		s.closeCode.Store(int32(code))
		s.queue.close()
		s.cancel()
		if s.conn == nil {
			return
		}
		_ = s.conn.WriteControl(
			websocket.CloseMessage,
			websocket.FormatCloseMessage(code, reason),
			time.Now().Add(s.hub.writeTimeout),
		)
		_ = s.conn.Close()
	})
}
