package events

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"sort"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/gorilla/websocket"
	"github.com/grauwolf32/contractor/internal/auth"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/controlplane"
	"github.com/grauwolf32/contractor/internal/runstore"
)

const eventTestOrigin = "https://ui.events.test"

type fakeRunRepository struct {
	mu         sync.Mutex
	runs       map[string]runstore.WorkflowRun
	generation map[string]string
	events     map[string][]runstore.WorkflowRunEvent
}

func newFakeRunRepository() *fakeRunRepository {
	return &fakeRunRepository{
		runs: make(map[string]runstore.WorkflowRun), generation: make(map[string]string),
		events: make(map[string][]runstore.WorkflowRunEvent),
	}
}

func (f *fakeRunRepository) GetRun(_ context.Context, runID string) (runstore.WorkflowRun, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	run, ok := f.runs[runID]
	if !ok {
		return runstore.WorkflowRun{}, runstore.ErrNotFound
	}
	return run, nil
}

func (f *fakeRunRepository) GetRunEventCursor(
	_ context.Context,
	runID string,
) (runstore.WorkflowRunEventCursor, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	if _, ok := f.runs[runID]; !ok {
		return runstore.WorkflowRunEventCursor{}, runstore.ErrNotFound
	}
	sequence := int64(0)
	if events := f.events[runID]; len(events) != 0 {
		sequence = events[len(events)-1].SequenceNumber
	}
	return runstore.WorkflowRunEventCursor{Generation: f.generation[runID], Sequence: sequence}, nil
}

func (f *fakeRunRepository) ListRunEvents(
	ctx context.Context,
	runID string,
	after int64,
	limit int,
) ([]runstore.WorkflowRunEvent, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	f.mu.Lock()
	defer f.mu.Unlock()
	result := make([]runstore.WorkflowRunEvent, 0, limit)
	for _, event := range f.events[runID] {
		if event.SequenceNumber <= after {
			continue
		}
		result = append(result, event)
		if len(result) == limit {
			break
		}
	}
	return result, nil
}

func (f *fakeRunRepository) addRun(runID, owner, generation string) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.runs[runID] = runstore.WorkflowRun{RunID: runID, OwnerID: owner}
	f.generation[runID] = generation
}

func (f *fakeRunRepository) append(event runstore.WorkflowRunEvent) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.events[event.RunID] = append(f.events[event.RunID], event)
	sort.Slice(f.events[event.RunID], func(i, j int) bool {
		return f.events[event.RunID][i].SequenceNumber < f.events[event.RunID][j].SequenceNumber
	})
}

type eventHarness struct {
	hub        *Hub
	auth       *auth.Service
	login      auth.Login
	runs       *fakeRunRepository
	operations *controlplane.InMemoryRegistry
	server     *httptest.Server
}

func newEventHarness(t *testing.T) *eventHarness {
	t.Helper()
	password := []byte("correct horse battery staple")
	hash, err := auth.HashPassword(password)
	if err != nil {
		t.Fatal(err)
	}
	bootstrap, err := auth.NewBootstrap("user-1", "admin", hash)
	if err != nil {
		t.Fatal(err)
	}
	authentication, err := auth.NewService(bootstrap, auth.Options{})
	if err != nil {
		t.Fatal(err)
	}
	login, err := authentication.Login("admin", password, "192.0.2.1")
	if err != nil {
		t.Fatal(err)
	}
	origins, err := auth.NewOriginPolicy([]string{eventTestOrigin}, false)
	if err != nil {
		t.Fatal(err)
	}
	operations, err := controlplane.NewRegistry(controlplane.RegistryOptions{})
	if err != nil {
		t.Fatal(err)
	}
	runs := newFakeRunRepository()
	hub, err := NewHub(Options{
		Context: t.Context(), Authentication: authentication, Origins: origins,
		Runs: runs, Operations: operations,
		CatchUpInterval: 10 * time.Millisecond, SessionCheckInterval: 10 * time.Millisecond,
	})
	if err != nil {
		t.Fatal(err)
	}
	harness := &eventHarness{hub: hub, auth: authentication, login: login, runs: runs, operations: operations}
	harness.server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		err := hub.Serve(w, r, login.Session)
		switch {
		case err == nil:
		case errors.Is(err, ErrInvalidOrigin):
			http.Error(w, "forbidden", http.StatusForbidden)
		case errors.Is(err, ErrSession):
			http.Error(w, "unauthorized", http.StatusUnauthorized)
		case errors.Is(err, ErrSocketLimit):
			http.Error(w, "limited", http.StatusTooManyRequests)
		default:
			http.Error(w, "invalid", http.StatusBadRequest)
		}
	}))
	t.Cleanup(func() {
		hub.Close()
		harness.server.Close()
	})
	return harness
}

func (h *eventHarness) dial(t *testing.T) *websocket.Conn {
	t.Helper()
	return h.dialWith(t, eventTestOrigin, []string{ProtocolVersion})
}

func (h *eventHarness) dialWith(t *testing.T, origin string, protocols []string) *websocket.Conn {
	t.Helper()
	dialer := websocket.Dialer{Subprotocols: protocols, HandshakeTimeout: 2 * time.Second}
	header := http.Header{}
	header.Set("Origin", origin)
	url := "ws" + strings.TrimPrefix(h.server.URL, "http")
	connection, response, err := dialer.Dial(url, header)
	if err != nil {
		status := 0
		if response != nil {
			status = response.StatusCode
			_ = response.Body.Close()
		}
		t.Fatalf("dial event WebSocket: status=%d error=%v", status, err)
	}
	if connection.Subprotocol() != ProtocolVersion {
		connection.Close()
		t.Fatalf("negotiated subprotocol = %q", connection.Subprotocol())
	}
	t.Cleanup(func() { _ = connection.Close() })
	return connection
}

func TestRunSubscriptionReplaysThenFollowsAndUnsubscribesInOrder(t *testing.T) {
	harness := newEventHarness(t)
	harness.runs.addRun("run-1", "user-1", "run-generation-1")
	for sequence := int64(42); sequence <= 45; sequence++ {
		harness.runs.append(testLifecycleEvent("run-1", sequence))
	}
	connection := harness.dial(t)
	writeClientJSON(t, connection, map[string]any{
		"version": ProtocolVersion, "type": "subscribe", "subscriptionId": "run-detail",
		"stream": map[string]any{"kind": "run", "id": "run-1"},
		"after":  map[string]any{"generation": "run-generation-1", "sequence": "41"},
	})
	ack := readServerJSON(t, connection)
	if ack["type"] != "subscribed" || nestedString(t, ack, "cursor", "sequence") != "41" {
		t.Fatalf("subscription ack = %#v", ack)
	}
	for sequence := int64(42); sequence <= 45; sequence++ {
		frame := readServerJSON(t, connection)
		if frame["type"] != "event" || frame["kind"] != "lifecycle.changed" ||
			nestedString(t, frame, "cursor", "sequence") != fmt.Sprint(sequence) {
			t.Fatalf("replayed event %d = %#v", sequence, frame)
		}
	}
	harness.runs.append(testLifecycleEvent("run-1", 46))
	// Notifications are only wake-up hints and may be duplicated. Replay from
	// the last delivered sequence must still emit the durable event once.
	harness.hub.notifyRun("run-1")
	harness.hub.notifyRun("run-1")
	live := readServerJSON(t, connection)
	if nestedString(t, live, "cursor", "sequence") != "46" {
		t.Fatalf("live event = %#v", live)
	}
	writeClientJSON(t, connection, map[string]any{
		"version": ProtocolVersion, "type": "unsubscribe", "subscriptionId": "run-detail",
	})
	unsubscribed := readServerJSON(t, connection)
	if unsubscribed["type"] != "unsubscribed" || unsubscribed["subscriptionId"] != "run-detail" {
		t.Fatalf("unsubscribe ack = %#v", unsubscribed)
	}
}

func TestRunSubscriptionGapRequiresAuthoritativeResync(t *testing.T) {
	harness := newEventHarness(t)
	harness.runs.addRun("run-gap", "user-1", "run-generation-gap")
	harness.runs.append(testLifecycleEvent("run-gap", 2))
	connection := harness.dial(t)
	writeClientJSON(t, connection, map[string]any{
		"version": ProtocolVersion, "type": "subscribe", "subscriptionId": "run-gap",
		"stream": map[string]any{"kind": "run", "id": "run-gap"},
		"after":  map[string]any{"generation": "run-generation-gap", "sequence": "0"},
	})
	if frame := readServerJSON(t, connection); frame["type"] != "subscribed" {
		t.Fatalf("gap subscription ack = %#v", frame)
	}
	if frame := readServerJSON(t, connection); frame["type"] != "resync_required" || frame["reason"] != "sequence_gap" {
		t.Fatalf("Run sequence gap = %#v", frame)
	}
}

func TestHandshakeAuthorizationUnknownFramesAndSubscriptionLimitFailClosed(t *testing.T) {
	harness := newEventHarness(t)
	harness.runs.addRun("run-other", "another-user", "run-generation-other")
	for _, test := range []struct {
		origin    string
		protocols []string
		status    int
	}{
		{origin: "https://attacker.test", protocols: []string{ProtocolVersion}, status: http.StatusForbidden},
		{origin: eventTestOrigin, protocols: []string{ProtocolVersion, "another"}, status: http.StatusBadRequest},
	} {
		dialer := websocket.Dialer{Subprotocols: test.protocols}
		header := http.Header{"Origin": []string{test.origin}}
		connection, response, err := dialer.Dial("ws"+strings.TrimPrefix(harness.server.URL, "http"), header)
		if connection != nil {
			connection.Close()
		}
		if err == nil || response == nil || response.StatusCode != test.status {
			t.Fatalf("invalid handshake = connection=%v response=%v error=%v", connection, response, err)
		}
		_ = response.Body.Close()
	}

	connection := harness.dial(t)
	writeClientJSON(t, connection, map[string]any{
		"version": ProtocolVersion, "type": "subscribe", "subscriptionId": "other",
		"stream": map[string]any{"kind": "run", "id": "run-other"},
	})
	if frame := readServerJSON(t, connection); frame["type"] != "error" || frame["code"] != "not_found" {
		t.Fatalf("unauthorized Run subscription = %#v", frame)
	}
	writeClientJSON(t, connection, map[string]any{
		"version": ProtocolVersion, "type": "cancel_run", "runId": "run-other",
	})
	if frame := readServerJSON(t, connection); frame["type"] != "error" || frame["code"] != "invalid_frame" {
		t.Fatalf("unknown mutation frame = %#v", frame)
	}
	for index := 0; index <= maximumSubscriptions; index++ {
		writeClientJSON(t, connection, map[string]any{
			"version": ProtocolVersion, "type": "subscribe",
			"subscriptionId": fmt.Sprintf("operations-%02d", index),
			"stream":         map[string]any{"kind": "operations"},
		})
	}
	acknowledged, limited := 0, 0
	for range maximumSubscriptions + 1 {
		frame := readServerJSON(t, connection)
		switch {
		case frame["type"] == "subscribed":
			acknowledged++
		case frame["type"] == "error" && frame["code"] == "subscription_limit":
			limited++
		default:
			t.Fatalf("unexpected subscription-limit frame = %#v", frame)
		}
	}
	if acknowledged != maximumSubscriptions || limited != 1 {
		t.Fatalf("subscription results acknowledged=%d limited=%d", acknowledged, limited)
	}
}

func TestOperationsResyncAndLiveInvalidation(t *testing.T) {
	harness := newEventHarness(t)
	initial := harness.operations.SnapshotOperations().Cursor
	for range 256 {
		if err := harness.operations.InvalidateOperations(controlplane.OperationsCredential, ""); err != nil {
			t.Fatal(err)
		}
	}
	connection := harness.dial(t)
	writeClientJSON(t, connection, map[string]any{
		"version": ProtocolVersion, "type": "subscribe", "subscriptionId": "expired",
		"stream": map[string]any{"kind": "operations"},
		"after":  map[string]any{"generation": initial.Generation, "sequence": "0"},
	})
	if frame := readServerJSON(t, connection); frame["type"] != "resync_required" || frame["reason"] != "cursor_unavailable" {
		t.Fatalf("expired Operations cursor = %#v", frame)
	}
	writeClientJSON(t, connection, map[string]any{
		"version": ProtocolVersion, "type": "subscribe", "subscriptionId": "restart",
		"stream": map[string]any{"kind": "operations"},
		"after":  map[string]any{"generation": "operations-old-process", "sequence": "0"},
	})
	if frame := readServerJSON(t, connection); frame["type"] != "resync_required" || frame["reason"] != "generation_changed" {
		t.Fatalf("restarted Operations cursor = %#v", frame)
	}
	writeClientJSON(t, connection, map[string]any{
		"version": ProtocolVersion, "type": "subscribe", "subscriptionId": "live",
		"stream": map[string]any{"kind": "operations"},
	})
	ack := readServerJSON(t, connection)
	if ack["type"] != "subscribed" || nestedString(t, ack, "cursor", "sequence") != "256" {
		t.Fatalf("Operations ack = %#v", ack)
	}
	if err := harness.operations.InvalidateOperations(controlplane.OperationsConfiguration, "config-v1"); err != nil {
		t.Fatal(err)
	}
	event := readServerJSON(t, connection)
	if event["kind"] != "operations.changed" || nestedString(t, event, "cursor", "sequence") != "257" ||
		nestedString(t, event, "data", "resource") != "configuration" {
		t.Fatalf("Operations event = %#v", event)
	}
	if err := harness.operations.InvalidateOperations(controlplane.OperationsSchedulerSettings, ""); err != nil {
		t.Fatal(err)
	}
	settingsEvent := readServerJSON(t, connection)
	settingsData, ok := settingsEvent["data"].(map[string]any)
	if !ok || settingsEvent["kind"] != "operations.changed" ||
		nestedString(t, settingsEvent, "cursor", "sequence") != "258" ||
		nestedString(t, settingsEvent, "data", "resource") != "schedulerSettings" ||
		settingsData["resourceId"] != nil {
		t.Fatalf("Scheduler settings Operations event = %#v", settingsEvent)
	}
}

func TestSessionRevocationClosesOnlyItsSockets(t *testing.T) {
	harness := newEventHarness(t)
	connection := harness.dial(t)
	harness.auth.Destroy(harness.login.Session.Handle)
	_ = connection.SetReadDeadline(time.Now().Add(2 * time.Second))
	_, _, err := connection.ReadMessage()
	if err == nil || !websocket.IsCloseError(err, websocket.ClosePolicyViolation) {
		t.Fatalf("revoked session close error = %v", err)
	}
}

func TestSocketLimitAndCapacityRelease(t *testing.T) {
	harness := newEventHarness(t)
	slots := make([]*socketSlot, 0, maximumSocketsPerSession)
	for range maximumSocketsPerSession {
		slot, err := harness.hub.reserveSocket(harness.login.Session.Handle)
		if err != nil {
			t.Fatalf("reserve in-limit socket: %v", err)
		}
		slots = append(slots, slot)
	}
	if _, err := harness.hub.reserveSocket(harness.login.Session.Handle); !errors.Is(err, ErrSocketLimit) {
		t.Fatalf("ninth session socket error = %v", err)
	}
	harness.hub.releaseSocket(slots[0])
	slots = slots[1:]
	replacement, err := harness.hub.reserveSocket(harness.login.Session.Handle)
	if err != nil {
		t.Fatalf("reserve socket after release: %v", err)
	}
	slots = append(slots, replacement)
	for _, slot := range slots {
		harness.hub.releaseSocket(slot)
	}
}

func TestProtocolAndQueueBoundsRejectUnknownOrUnsafeData(t *testing.T) {
	for _, raw := range []string{
		`{"version":"contractor.events.v1","type":"subscribe","subscriptionId":"x","stream":{"kind":"operations","id":"forbidden"}}`,
		`{"version":"contractor.events.v1","type":"subscribe","subscriptionId":"x","stream":{"kind":"operations","id":null}}`,
		`{"version":"contractor.events.v1","type":"unsubscribe","subscriptionId":"x","stream":{"kind":"operations"}}`,
		`{"version":"contractor.events.v2","type":"unsubscribe","subscriptionId":"x"}`,
		`{"version":"contractor.events.v1","type":"subscribe","subscriptionId":"x","stream":{"kind":"run","id":"run-1"},"after":{"generation":"g","sequence":"01"}}`,
		`{"version":"contractor.events.v1","type":"subscribe","subscriptionId":"x","stream":{"kind":"run","id":"run-1"},"after":null}`,
	} {
		if _, err := decodeClientFrame([]byte(raw)); err == nil {
			t.Fatalf("invalid frame was accepted: %s", raw)
		}
	}
	activity := runstore.WorkflowRunEvent{
		RunID: "run-1", SequenceNumber: 1, EventID: "event-1",
		EventSchemaVersion: contracts.APIVersion, Kind: runstore.RunEventPlannerActivity,
		Data:       json.RawMessage(`{"stageExecutionId":"stage-1","sessionId":"session-1","invocationId":"invocation-1","activity":{"kind":"adk_event","author":"router_planner","functionCalls":null,"functionResults":null}}`),
		OccurredAt: time.Now().UTC(),
	}
	encoded, err := runEventServerFrame("sub", Stream{Kind: StreamRun, ID: "run-1"}, "generation-1", activity)
	if err != nil || strings.Contains(string(encoded), `"functionCalls":null`) ||
		!strings.Contains(string(encoded), `"eventKind":"planner.activity"`) {
		t.Fatalf("typed Planner event frame = %s, %v", encoded, err)
	}
	unsafe := activity
	unsafe.EventID = "event-unsafe"
	unsafe.Data = json.RawMessage(`{"stageExecutionId":"stage-1","sessionId":"session-1","invocationId":"invocation-1","rawToolPayload":{"token":"secret"}}`)
	if _, err := runEventServerFrame("sub", Stream{Kind: StreamRun, ID: "run-1"}, "generation-1", unsafe); err == nil {
		t.Fatal("unsafe persisted Planner payload entered a public frame")
	}

	firstContext, firstCancel := context.WithCancel(context.Background())
	defer firstCancel()
	first := &socket{hub: &Hub{writeTimeout: time.Second}, ctx: firstContext, cancel: firstCancel, queue: newOutboundQueue()}
	secondContext, secondCancel := context.WithCancel(context.Background())
	defer secondCancel()
	second := &socket{hub: &Hub{writeTimeout: time.Second}, ctx: secondContext, cancel: secondCancel, queue: newOutboundQueue()}
	for range maximumQueuedFrames {
		if !first.sendEncoded([]byte(`{}`)) {
			t.Fatal("bounded queue rejected an in-limit frame")
		}
	}
	if first.sendEncoded([]byte(`{}`)) || first.closeCode.Load() != websocket.CloseTryAgainLater || firstContext.Err() == nil {
		t.Fatalf("slow socket was not closed with 1013: code=%d err=%v", first.closeCode.Load(), firstContext.Err())
	}
	if secondContext.Err() != nil || !second.sendEncoded([]byte(`{}`)) {
		t.Fatal("overflowing one queue affected another socket")
	}
}

func TestHeartbeatTimesOutOnlyAnUnansweredPing(t *testing.T) {
	base := time.Date(2026, 8, 31, 12, 0, 0, 0, time.UTC)
	current := &socket{hub: &Hub{pingInterval: 20 * time.Second, pongTimeout: 60 * time.Second}}
	current.lastOutput.Store(base.UnixNano())

	// Fresh output suppresses ping and cannot time out merely because the
	// connection itself is older than pongTimeout.
	current.lastOutput.Store(base.Add(59 * time.Second).UnixNano())
	if send, timeout := current.heartbeatState(base.Add(61 * time.Second)); send || timeout {
		t.Fatalf("active output heartbeat state = send %t timeout %t", send, timeout)
	}

	current.lastOutput.Store(base.UnixNano())
	if send, timeout := current.heartbeatState(base.Add(20 * time.Second)); !send || timeout {
		t.Fatalf("idle heartbeat state = send %t timeout %t", send, timeout)
	}
	current.pendingPing.Store(base.Add(20 * time.Second).UnixNano())
	if send, timeout := current.heartbeatState(base.Add(79 * time.Second)); send || timeout {
		t.Fatalf("pending pong before deadline = send %t timeout %t", send, timeout)
	}
	if send, timeout := current.heartbeatState(base.Add(80 * time.Second)); send || !timeout {
		t.Fatalf("unanswered pong at deadline = send %t timeout %t", send, timeout)
	}
}

func testLifecycleEvent(runID string, sequence int64) runstore.WorkflowRunEvent {
	return runstore.WorkflowRunEvent{
		RunID: runID, SequenceNumber: sequence, EventID: fmt.Sprintf("event-%d", sequence),
		EventSchemaVersion: contracts.APIVersion, Kind: runstore.RunEventLifecycleChanged,
		Data:       json.RawMessage(fmt.Sprintf(`{"runId":%q,"resource":"run","state":"running"}`, runID)),
		OccurredAt: time.Date(2026, 8, 31, 12, 0, int(sequence%60), 0, time.UTC),
	}
}

func writeClientJSON(t *testing.T, connection *websocket.Conn, value any) {
	t.Helper()
	_ = connection.SetWriteDeadline(time.Now().Add(2 * time.Second))
	if err := connection.WriteJSON(value); err != nil {
		t.Fatal(err)
	}
}

func readServerJSON(t *testing.T, connection *websocket.Conn) map[string]any {
	t.Helper()
	_ = connection.SetReadDeadline(time.Now().Add(3 * time.Second))
	var result map[string]any
	if err := connection.ReadJSON(&result); err != nil {
		t.Fatal(err)
	}
	return result
}

func nestedString(t *testing.T, source map[string]any, object, field string) string {
	t.Helper()
	nested, ok := source[object].(map[string]any)
	if !ok {
		t.Fatalf("%s is not an object in %#v", object, source)
	}
	value, ok := nested[field].(string)
	if !ok {
		t.Fatalf("%s.%s is not a string in %#v", object, field, source)
	}
	return value
}
