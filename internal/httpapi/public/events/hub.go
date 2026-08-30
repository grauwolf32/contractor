package events

import (
	"context"
	"errors"
	"log/slog"
	"net/http"
	"sync"
	"time"

	"github.com/gorilla/websocket"
	"github.com/grauwolf32/contractor/internal/auth"
	"github.com/grauwolf32/contractor/internal/controlplane"
	"github.com/grauwolf32/contractor/internal/runstore"
)

var (
	ErrInvalidHandshake = errors.New("invalid WebSocket event handshake")
	ErrInvalidOrigin    = errors.New("invalid WebSocket event origin")
	ErrSession          = errors.New("browser session is unavailable")
	ErrSocketLimit      = errors.New("browser session WebSocket limit reached")
)

const maximumSocketsPerSession = 8

type RunRepository interface {
	GetRun(context.Context, string) (runstore.WorkflowRun, error)
	GetRunEventCursor(context.Context, string) (runstore.WorkflowRunEventCursor, error)
	ListRunEvents(context.Context, string, int64, int) ([]runstore.WorkflowRunEvent, error)
}

type OperationsSource interface {
	SnapshotOperations() controlplane.OperationsSnapshot
	ReplayOperations(controlplane.OperationsCursor) ([]controlplane.OperationsChange, controlplane.OperationsCursor, error)
	SubscribeOperations() (<-chan struct{}, func())
}

type RunNotificationSource interface {
	Listen(context.Context, func(string)) error
}

type Options struct {
	Context              context.Context
	Authentication       *auth.Service
	Origins              auth.OriginPolicy
	Runs                 RunRepository
	Operations           OperationsSource
	RunNotifications     RunNotificationSource
	Logger               *slog.Logger
	CatchUpInterval      time.Duration
	SessionCheckInterval time.Duration
	PingInterval         time.Duration
	PongTimeout          time.Duration
	WriteTimeout         time.Duration
}

type Hub struct {
	ctx            context.Context
	cancel         context.CancelFunc
	authentication *auth.Service
	origins        auth.OriginPolicy
	runs           RunRepository
	operations     OperationsSource
	notifications  RunNotificationSource
	logger         *slog.Logger
	catchUp        time.Duration
	sessionCheck   time.Duration
	pingInterval   time.Duration
	pongTimeout    time.Duration
	writeTimeout   time.Duration

	mu             sync.Mutex
	closed         bool
	sessionSockets map[auth.SessionHandle]map[*socketSlot]struct{}
	runWatchers    map[string]map[uint64]chan struct{}
	nextRunWatcher uint64
	background     sync.WaitGroup
	active         sync.WaitGroup
}

type socketSlot struct {
	handle   auth.SessionHandle
	socket   *socket
	revoked  bool
	released bool
}

func NewHub(options Options) (*Hub, error) {
	if options.Authentication == nil || options.Runs == nil || options.Operations == nil ||
		len(options.Origins.Values()) == 0 {
		return nil, errors.New("WebSocket event dependencies are incomplete")
	}
	if options.Context == nil {
		options.Context = context.Background()
	}
	if options.Logger == nil {
		options.Logger = slog.New(slog.NewTextHandler(discardWriter{}, nil))
	}
	if options.CatchUpInterval == 0 {
		options.CatchUpInterval = time.Second
	}
	if options.SessionCheckInterval == 0 {
		options.SessionCheckInterval = time.Second
	}
	if options.PingInterval == 0 {
		options.PingInterval = 20 * time.Second
	}
	if options.PongTimeout == 0 {
		options.PongTimeout = 60 * time.Second
	}
	if options.WriteTimeout == 0 {
		options.WriteTimeout = 10 * time.Second
	}
	if options.CatchUpInterval <= 0 || options.SessionCheckInterval <= 0 ||
		options.PingInterval <= 0 || options.PongTimeout <= options.PingInterval ||
		options.WriteTimeout <= 0 {
		return nil, errors.New("WebSocket event durations are invalid")
	}
	ctx, cancel := context.WithCancel(options.Context)
	hub := &Hub{
		ctx: ctx, cancel: cancel, authentication: options.Authentication,
		origins: options.Origins, runs: options.Runs, operations: options.Operations,
		notifications: options.RunNotifications, logger: options.Logger,
		catchUp: options.CatchUpInterval, sessionCheck: options.SessionCheckInterval,
		pingInterval: options.PingInterval, pongTimeout: options.PongTimeout,
		writeTimeout:   options.WriteTimeout,
		sessionSockets: make(map[auth.SessionHandle]map[*socketSlot]struct{}),
		runWatchers:    make(map[string]map[uint64]chan struct{}),
	}
	hub.startBackground()
	return hub, nil
}

func (h *Hub) Close() {
	h.cancel()
	h.closeAll(websocket.CloseGoingAway, "Server shutting down")
	h.background.Wait()
	h.active.Wait()
}

func (h *Hub) Serve(w http.ResponseWriter, r *http.Request, session auth.Session) error {
	if r.Method != http.MethodGet || r.URL.RawQuery != "" || !websocket.IsWebSocketUpgrade(r) {
		return ErrInvalidHandshake
	}
	origins := r.Header.Values("Origin")
	if len(origins) != 1 || !h.origins.Allows(origins[0]) {
		return ErrInvalidOrigin
	}
	protocols := r.Header.Values("Sec-WebSocket-Protocol")
	if len(protocols) != 1 || protocols[0] != ProtocolVersion {
		return ErrInvalidHandshake
	}
	if _, err := h.authentication.Check(session.Handle); err != nil {
		return ErrSession
	}
	slot, err := h.reserveSocket(session.Handle)
	if err != nil {
		return err
	}
	defer h.releaseSocket(slot)
	if _, err := h.authentication.Check(session.Handle); err != nil {
		return ErrSession
	}
	upgrader := websocket.Upgrader{
		ReadBufferSize: 4096, WriteBufferSize: 4096,
		Subprotocols: []string{ProtocolVersion},
		CheckOrigin: func(request *http.Request) bool {
			values := request.Header.Values("Origin")
			return len(values) == 1 && h.origins.Allows(values[0])
		},
	}
	connection, err := upgrader.Upgrade(w, r, nil)
	if err != nil {
		// gorilla/websocket has already written the bounded HTTP handshake
		// failure; writing a second public error response would corrupt it.
		return nil
	}
	current := newSocket(h, connection, session)
	if !h.attachSocket(slot, current) {
		current.stop(websocket.ClosePolicyViolation, "session revoked")
		return nil
	}
	current.run()
	return nil
}

func (h *Hub) reserveSocket(handle auth.SessionHandle) (*socketSlot, error) {
	h.mu.Lock()
	defer h.mu.Unlock()
	if h.closed || h.ctx.Err() != nil {
		return nil, ErrSession
	}
	sockets := h.sessionSockets[handle]
	if len(sockets) >= maximumSocketsPerSession {
		return nil, ErrSocketLimit
	}
	if sockets == nil {
		sockets = make(map[*socketSlot]struct{})
		h.sessionSockets[handle] = sockets
	}
	slot := &socketSlot{handle: handle}
	sockets[slot] = struct{}{}
	h.active.Add(1)
	return slot, nil
}

func (h *Hub) attachSocket(slot *socketSlot, current *socket) bool {
	h.mu.Lock()
	defer h.mu.Unlock()
	if slot.released || slot.revoked || h.closed || h.ctx.Err() != nil {
		return false
	}
	slot.socket = current
	return true
}

func (h *Hub) releaseSocket(slot *socketSlot) {
	h.mu.Lock()
	if slot.released {
		h.mu.Unlock()
		return
	}
	slot.released = true
	if sockets := h.sessionSockets[slot.handle]; sockets != nil {
		delete(sockets, slot)
		if len(sockets) == 0 {
			delete(h.sessionSockets, slot.handle)
		}
	}
	h.mu.Unlock()
	h.active.Done()
}

func (h *Hub) startBackground() {
	revocations, cancelRevocations := h.authentication.SubscribeRevocations()
	h.background.Add(1)
	go func() {
		defer h.background.Done()
		defer cancelRevocations()
		for {
			select {
			case <-h.ctx.Done():
				return
			case handle, ok := <-revocations:
				if !ok {
					return
				}
				h.revokeSession(handle)
			}
		}
	}()
	if h.notifications != nil {
		h.background.Add(1)
		go h.listenRunNotifications()
	}
	h.background.Add(1)
	go func() {
		defer h.background.Done()
		<-h.ctx.Done()
		h.closeAll(websocket.CloseGoingAway, "Server shutting down")
	}()
}

func (h *Hub) listenRunNotifications() {
	defer h.background.Done()
	for h.ctx.Err() == nil {
		err := h.notifications.Listen(h.ctx, h.notifyRun)
		if h.ctx.Err() != nil {
			return
		}
		if err != nil {
			h.logger.Warn("WorkflowRun event notification listener will retry")
		}
		timer := time.NewTimer(time.Second)
		select {
		case <-h.ctx.Done():
			timer.Stop()
			return
		case <-timer.C:
		}
	}
}

func (h *Hub) revokeSession(handle auth.SessionHandle) {
	h.mu.Lock()
	var sockets []*socket
	for slot := range h.sessionSockets[handle] {
		slot.revoked = true
		if slot.socket != nil {
			sockets = append(sockets, slot.socket)
		}
	}
	h.mu.Unlock()
	for _, current := range sockets {
		current.stop(websocket.ClosePolicyViolation, "session revoked")
	}
}

func (h *Hub) closeAll(code int, reason string) {
	h.mu.Lock()
	if !h.closed {
		h.closed = true
	}
	var sockets []*socket
	for _, slots := range h.sessionSockets {
		for slot := range slots {
			slot.revoked = true
			if slot.socket != nil {
				sockets = append(sockets, slot.socket)
			}
		}
	}
	for runID, watchers := range h.runWatchers {
		for id, watcher := range watchers {
			delete(watchers, id)
			close(watcher)
		}
		delete(h.runWatchers, runID)
	}
	h.mu.Unlock()
	for _, current := range sockets {
		current.stop(code, reason)
	}
}

func (h *Hub) subscribeRun(runID string) (<-chan struct{}, func()) {
	h.mu.Lock()
	if h.closed {
		closed := make(chan struct{})
		close(closed)
		h.mu.Unlock()
		return closed, func() {}
	}
	h.nextRunWatcher++
	id := h.nextRunWatcher
	updates := make(chan struct{}, 1)
	watchers := h.runWatchers[runID]
	if watchers == nil {
		watchers = make(map[uint64]chan struct{})
		h.runWatchers[runID] = watchers
	}
	watchers[id] = updates
	h.mu.Unlock()
	var once sync.Once
	cancel := func() {
		once.Do(func() {
			h.mu.Lock()
			if watchers := h.runWatchers[runID]; watchers != nil {
				if watcher, ok := watchers[id]; ok {
					delete(watchers, id)
					close(watcher)
				}
				if len(watchers) == 0 {
					delete(h.runWatchers, runID)
				}
			}
			h.mu.Unlock()
		})
	}
	return updates, cancel
}

func (h *Hub) notifyRun(runID string) {
	h.mu.Lock()
	defer h.mu.Unlock()
	for _, watcher := range h.runWatchers[runID] {
		select {
		case watcher <- struct{}{}:
		default:
		}
	}
}

type discardWriter struct{}

func (discardWriter) Write(value []byte) (int, error) { return len(value), nil }
