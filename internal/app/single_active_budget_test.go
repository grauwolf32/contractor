package app

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"log/slog"
	"net"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgconn"
	"github.com/jackc/pgx/v5/pgproto3"
	"github.com/jackc/pgx/v5/pgtype"
	"github.com/jackc/pgx/v5/pgxpool"

	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
)

// startupBudgetRecorder wraps the dialer and the query tracer of a working
// pool configuration, so a test reads the deadline that each connection
// attempt and each statement of a session copied from it ran under. It
// delegates to the wrapped tracer, so a recorded statement deadline is the
// one the configured budget tracer produced.
type startupBudgetRecorder struct {
	dialer  pgconn.DialFunc
	tracer  pgx.QueryTracer
	mu      sync.Mutex
	dials   []time.Time
	queries []time.Time
}

func recordStartupBudgets(t *testing.T, config *pgxpool.Config) *startupBudgetRecorder {
	t.Helper()
	tracer, ok := config.ConnConfig.Tracer.(pgx.QueryTracer)
	if !ok || config.ConnConfig.DialFunc == nil {
		t.Fatal("the working pool configuration has no query budget tracer or dialer")
	}
	recorder := &startupBudgetRecorder{dialer: config.ConnConfig.DialFunc, tracer: tracer}
	config.ConnConfig.DialFunc = recorder.dial
	config.ConnConfig.Tracer = recorder
	return recorder
}

func (r *startupBudgetRecorder) dial(ctx context.Context, network, address string) (net.Conn, error) {
	deadline, _ := ctx.Deadline()
	r.mu.Lock()
	r.dials = append(r.dials, deadline)
	r.mu.Unlock()
	return r.dialer(ctx, network, address)
}

func (r *startupBudgetRecorder) TraceQueryStart(ctx context.Context, conn *pgx.Conn, data pgx.TraceQueryStartData) context.Context {
	ctx = r.tracer.TraceQueryStart(ctx, conn, data)
	deadline, _ := ctx.Deadline()
	r.mu.Lock()
	r.queries = append(r.queries, deadline)
	r.mu.Unlock()
	return ctx
}

func (r *startupBudgetRecorder) TraceQueryEnd(ctx context.Context, conn *pgx.Conn, data pgx.TraceQueryEndData) {
	r.tracer.TraceQueryEnd(ctx, conn, data)
}

// sessionDial returns the deadline of the lease session's first connection
// attempt; later dials carry the session's cancel requests or a standby
// reconnect.
func (r *startupBudgetRecorder) sessionDial(t *testing.T) []time.Time {
	t.Helper()
	dials := r.dialsSince(0)
	if len(dials) == 0 {
		t.Fatal("the lease session never dialed")
	}
	return dials[:1]
}

// dialsSince and queriesSince return the deadlines recorded after the first
// skip entries.
func (r *startupBudgetRecorder) dialsSince(skip int) []time.Time {
	r.mu.Lock()
	defer r.mu.Unlock()
	return append([]time.Time(nil), r.dials[min(skip, len(r.dials)):]...)
}

func (r *startupBudgetRecorder) queriesSince(skip int) []time.Time {
	r.mu.Lock()
	defer r.mu.Unlock()
	return append([]time.Time(nil), r.queries[min(skip, len(r.queries)):]...)
}

// assertBudgetDeadlines checks that every deadline is the configured budget
// started during the observed call: the start lies between two clock
// readings taken around the call, so the check needs no assumption about how
// long the call took.
func assertBudgetDeadlines(t *testing.T, step string, deadlines []time.Time, budget time.Duration, before, after time.Time) {
	t.Helper()
	if len(deadlines) == 0 {
		t.Fatalf("%s recorded no deadline", step)
	}
	for _, deadline := range deadlines {
		if deadline.IsZero() {
			t.Fatalf("%s ran without a deadline", step)
		}
		if deadline.Before(before.Add(budget)) || deadline.After(after.Add(budget)) {
			t.Fatalf("%s deadline is not the configured %s budget started during the call", step, budget)
		}
	}
}

// assertCallerDeadlines checks that an earlier caller deadline was kept
// exactly: neither replaced nor extended by a configured budget.
func assertCallerDeadlines(t *testing.T, step string, deadlines []time.Time, caller context.Context) {
	t.Helper()
	want, ok := caller.Deadline()
	if !ok {
		t.Fatal("caller context has no deadline")
	}
	if len(deadlines) == 0 {
		t.Fatalf("%s recorded no deadline", step)
	}
	for _, deadline := range deadlines {
		if !deadline.Equal(want) {
			t.Fatalf("%s deadline %s, want the earlier caller deadline %s", step, deadline.Format(time.RFC3339Nano), want.Format(time.RFC3339Nano))
		}
	}
}

const (
	stallingPeerProcessID uint32 = 4242
	// peerObservationGuard only keeps a client that never acts from hanging
	// the test binary; no assertion depends on how long an observation takes.
	peerObservationGuard = time.Minute
)

var stallingPeerSecretKey = []byte{0x0b, 0xad, 0xc0, 0xde}

// stallingPeer is a controlled PostgreSQL protocol peer for one lease
// session. It answers the startup handshake and the first answer statements,
// then stalls at the start of the next statement without replying or
// closing; with a negative answer it stalls right after the startup message.
// A simple query is answered with an empty result, and an extended-protocol
// statement with the one boolean false that a refused acquisition returns.
// It reports each session it accepts, the message it stalled at, each cancel
// request that arrives on a separate connection, and the end of the stalled
// session together with whether the client sent Terminate before it.
type stallingPeer struct {
	listener net.Listener
	answer   int
	sessions chan struct{}
	stalled  chan string
	cancels  chan pgproto3.CancelRequest
	ended    chan bool
	wg       sync.WaitGroup
	mu       sync.Mutex
	closed   bool
	conns    []net.Conn
}

func newStallingPeer(t *testing.T, answer int) *stallingPeer {
	t.Helper()
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	peer := &stallingPeer{
		listener: listener, answer: answer,
		sessions: make(chan struct{}, 8), stalled: make(chan string, 1), cancels: make(chan pgproto3.CancelRequest, 4), ended: make(chan bool, 1),
	}
	peer.wg.Add(1)
	go peer.accept()
	t.Cleanup(peer.close)
	return peer
}

func (p *stallingPeer) accept() {
	defer p.wg.Done()
	for {
		conn, err := p.listener.Accept()
		if err != nil {
			return
		}
		p.mu.Lock()
		if p.closed {
			p.mu.Unlock()
			_ = conn.Close()
			return
		}
		p.conns = append(p.conns, conn)
		p.wg.Add(1)
		p.mu.Unlock()
		go p.serve(conn)
	}
}

func (p *stallingPeer) serve(conn net.Conn) {
	defer p.wg.Done()
	defer conn.Close()
	backend := pgproto3.NewBackend(conn, conn)
	first, err := backend.ReceiveStartupMessage()
	if err != nil {
		return
	}
	switch message := first.(type) {
	case *pgproto3.CancelRequest:
		request := pgproto3.CancelRequest{ProcessID: message.ProcessID, SecretKey: bytes.Clone(message.SecretKey)}
		select {
		case p.cancels <- request:
		default:
		}
		// Closing the connection acknowledges the cancel request.
		return
	case *pgproto3.StartupMessage:
		select {
		case p.sessions <- struct{}{}:
		default:
		}
	default:
		return
	}
	if p.answer < 0 {
		p.stall(backend, "startup message")
		return
	}
	backend.Send(&pgproto3.AuthenticationOk{})
	backend.Send(&pgproto3.BackendKeyData{ProcessID: stallingPeerProcessID, SecretKey: stallingPeerSecretKey})
	backend.Send(&pgproto3.ReadyForQuery{TxStatus: 'I'})
	if backend.Flush() != nil {
		return
	}
	resultFormat := int16(pgtype.TextFormatCode)
	for answered := 0; ; {
		message, err := backend.Receive()
		if err != nil {
			return
		}
		switch message.(type) {
		case *pgproto3.Query, *pgproto3.Parse:
			if answered == p.answer {
				p.stall(backend, fmt.Sprintf("%T", message))
				return
			}
			answered++
		}
		switch message := message.(type) {
		case *pgproto3.Query:
			backend.Send(&pgproto3.CommandComplete{CommandTag: []byte("SELECT 1")})
			backend.Send(&pgproto3.ReadyForQuery{TxStatus: 'I'})
		case *pgproto3.Parse:
			backend.Send(&pgproto3.ParseComplete{})
		case *pgproto3.Describe:
			if message.ObjectType == 'S' {
				backend.Send(&pgproto3.ParameterDescription{ParameterOIDs: []uint32{pgtype.Int8OID}})
				backend.Send(refusedAcquisitionDescription(pgtype.TextFormatCode))
			} else {
				backend.Send(refusedAcquisitionDescription(resultFormat))
			}
		case *pgproto3.Bind:
			resultFormat = pgtype.TextFormatCode
			if len(message.ResultFormatCodes) > 0 {
				resultFormat = message.ResultFormatCodes[0]
			}
			backend.Send(&pgproto3.BindComplete{})
		case *pgproto3.Execute:
			refused := []byte("f")
			if resultFormat == pgtype.BinaryFormatCode {
				refused = []byte{0}
			}
			backend.Send(&pgproto3.DataRow{Values: [][]byte{refused}})
			backend.Send(&pgproto3.CommandComplete{CommandTag: []byte("SELECT 1")})
		case *pgproto3.Sync:
			backend.Send(&pgproto3.ReadyForQuery{TxStatus: 'I'})
		default:
			p.stall(backend, fmt.Sprintf("unanswerable %T", message))
			return
		}
		if backend.Flush() != nil {
			return
		}
	}
}

func refusedAcquisitionDescription(format int16) *pgproto3.RowDescription {
	return &pgproto3.RowDescription{Fields: []pgproto3.FieldDescription{{
		Name: []byte("pg_try_advisory_lock"), DataTypeOID: pgtype.BoolOID, DataTypeSize: 1, TypeModifier: -1, Format: format,
	}}}
}

// stall reports the message the session stalled at and then only reads,
// never replies, until the client ends the connection.
func (p *stallingPeer) stall(backend *pgproto3.Backend, at string) {
	select {
	case p.stalled <- at:
	default:
	}
	terminated := false
	for {
		message, err := backend.Receive()
		if err != nil {
			select {
			case p.ended <- terminated:
			default:
			}
			return
		}
		if _, ok := message.(*pgproto3.Terminate); ok {
			terminated = true
		}
	}
}

func (p *stallingPeer) close() {
	_ = p.listener.Close()
	p.mu.Lock()
	p.closed = true
	for _, conn := range p.conns {
		_ = conn.Close()
	}
	p.mu.Unlock()
	p.wg.Wait()
}

func awaitObservation[T any](t *testing.T, observations <-chan T, what string) T {
	t.Helper()
	guard := time.NewTimer(peerObservationGuard)
	defer guard.Stop()
	select {
	case observation := <-observations:
		return observation
	case <-guard.C:
	}
	t.Fatalf("%s was never observed", what)
	var none T
	return none
}

// assertStalledSessionCleanup waits for the client to end the stalled
// session. A session that completed its handshake also has a backend key, so
// the client must first cancel the stalled statement with that key and then
// send Terminate.
func assertStalledSessionCleanup(t *testing.T, peer *stallingPeer, handshakeCompleted bool) {
	t.Helper()
	terminated := awaitObservation(t, peer.ended, "the end of the stalled lease session")
	if !handshakeCompleted {
		return
	}
	request := awaitObservation(t, peer.cancels, "a cancel request for the stalled statement")
	if request.ProcessID != stallingPeerProcessID || !bytes.Equal(request.SecretKey, stallingPeerSecretKey) {
		t.Fatal("the cancel request does not carry the stalled session's backend key")
	}
	if !terminated {
		t.Fatal("the client ended the stalled lease session without Terminate")
	}
}

// stallingPeerPool configures the working pool exactly as Server startup
// does, from the database settings, but connects to the stalling peer.
// PoolConfig applies OpenPool's budgets without connecting, so nothing
// reaches the peer until the lease session dials.
func stallingPeerPool(t *testing.T, peer *stallingPeer, settings DatabaseSettings) (*pgxpool.Pool, *startupBudgetRecorder) {
	t.Helper()
	config, err := persistencepostgres.PoolConfig("postgres://lease@"+peer.listener.Addr().String()+"/contractor?sslmode=disable", settings.poolOptions(nil))
	if err != nil {
		t.Fatal(err)
	}
	recorder := recordStartupBudgets(t, config)
	pool, err := pgxpool.NewWithConfig(context.Background(), config)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(pool.Close)
	return pool, recorder
}

// startLeaseStartup runs the Server's startup lease steps in the background,
// so a step that never ends fails the test at the observation guard instead
// of hanging it.
func startLeaseStartup(ctx context.Context, pool *pgxpool.Pool, logger *slog.Logger) <-chan error {
	result := make(chan error, 1)
	go func() {
		lease, err := awaitControlPlaneLease(ctx, pool, "127.0.0.1:0", time.Second, logger)
		if lease != nil {
			lease.Close()
		}
		result <- err
	}()
	return result
}

// standbyHolders records the holder each standing-by log line names, so a
// test observes that startup went on to stand by.
type standbyHolders chan int64

func (standbyHolders) Enabled(context.Context, slog.Level) bool { return true }

func (h standbyHolders) Handle(_ context.Context, record slog.Record) error {
	if record.Message != "Control Plane lease is held by another Server; standing by" {
		return nil
	}
	record.Attrs(func(attr slog.Attr) bool {
		if attr.Key != "holder_pid" {
			return true
		}
		select {
		case h <- attr.Value.Int64():
		default:
		}
		return false
	})
	return nil
}

func (h standbyHolders) WithAttrs([]slog.Attr) slog.Handler { return h }
func (h standbyHolders) WithGroup(string) slog.Handler      { return h }

// The startup lease steps run under the process context, which has no
// deadline. A stall in any of them must still end: at the configured
// database budget that the lease session inherits from the working pool, or
// earlier when the caller's own deadline or cancellation comes first. Either
// way the client cancels a stalled statement and closes the session. A
// stalled step stops startup, except the holder lookup, after which the
// Server stands by with an unknown holder.
func TestControlPlaneLeaseStartupStallEndsAtItsBudgetOwner(t *testing.T) {
	for _, stage := range []struct {
		name      string
		answer    int
		stalledAt string
		// failure names the step whose error stops startup; empty when the
		// Server stands by instead.
		failure string
	}{
		{name: "handshake", answer: -1, stalledAt: "startup message", failure: "connect Control Plane lease session"},
		{name: "keepalive configuration", answer: 0, stalledAt: "*pgproto3.Query", failure: "configure Control Plane lease session keepalives"},
		{name: "first acquisition", answer: 1, stalledAt: "*pgproto3.Parse", failure: "acquire Control Plane lease"},
		{name: "holder lookup", answer: 2, stalledAt: "*pgproto3.Parse"},
	} {
		handshakeCompleted := stage.answer >= 0
		// assertStartupEnd checks how startup ended after the step's owner,
		// the configured budget or the caller, ended the stalled step with
		// cause. A Server that stood by instead returns no lease once it is
		// shut down and reports that shutdown as its context's cancellation,
		// so the process exits without an error (FAIL-PER-2).
		assertStartupEnd := func(t *testing.T, err error, cause error) {
			t.Helper()
			if stage.failure == "" {
				if !errors.Is(err, context.Canceled) || strings.Contains(err.Error(), "readiness listener") {
					t.Fatalf("standing-by Server shutdown = %v, want its context's cancellation", err)
				}
				return
			}
			if !errors.Is(err, cause) || !strings.Contains(err.Error(), stage.failure) {
				t.Fatalf("startup error = %v, want the %q step ended by %v", err, stage.failure, cause)
			}
		}
		assertStoodByWithUnknownHolder := func(t *testing.T, holders standbyHolders) {
			t.Helper()
			if holder := awaitObservation(t, holders, "standing by"); holder != 0 {
				t.Fatalf("standing by logged holder %d, want an unknown holder", holder)
			}
		}
		assertStatements := func(t *testing.T, recorder *startupBudgetRecorder) []time.Time {
			t.Helper()
			// A standing-by Server may already poll again; only the startup
			// statements are checked.
			queries := recorder.queriesSince(0)
			if len(queries) < stage.answer+1 || stage.failure != "" && len(queries) != stage.answer+1 {
				t.Fatalf("recorded %d lease session statements, want %d", len(queries), stage.answer+1)
			}
			return queries[:stage.answer+1]
		}
		t.Run(stage.name+"/configured budget", func(t *testing.T) {
			settings := DatabaseSettings{
				ConnectTimeout: 2 * time.Hour,
				QueryTimeout:   500 * time.Millisecond, StatementTimeout: 400 * time.Millisecond, LockTimeout: 300 * time.Millisecond,
			}
			if !handshakeCompleted {
				settings.ConnectTimeout = 250 * time.Millisecond
			}
			peer := newStallingPeer(t, stage.answer)
			pool, recorder := stallingPeerPool(t, peer, settings)
			ctx, cancel := context.WithCancel(t.Context())
			defer cancel()
			if _, ok := ctx.Deadline(); ok {
				t.Fatal("the startup context must have no deadline of its own")
			}
			holders := make(standbyHolders, 1)
			before := time.Now()
			result := startLeaseStartup(ctx, pool, slog.New(holders))
			if stage.failure == "" {
				// The stalled lookup ends at its budget and the Server stands
				// by. Its next acquisition probe finds the session the lookup
				// left unusable and reconnects; the test then shuts it down.
				assertStoodByWithUnknownHolder(t, holders)
				awaitObservation(t, peer.sessions, "the startup lease session")
				awaitObservation(t, peer.sessions, "the standby's reconnected lease session")
				cancel()
			}
			err := awaitObservation(t, result, "the end of the stalled startup")
			after := time.Now()
			assertStartupEnd(t, err, context.DeadlineExceeded)
			if at := awaitObservation(t, peer.stalled, "the stalled step"); at != stage.stalledAt {
				t.Fatalf("the lease session stalled at %s, want %s", at, stage.stalledAt)
			}
			assertStalledSessionCleanup(t, peer, handshakeCompleted)
			assertBudgetDeadlines(t, "lease session connect", recorder.sessionDial(t), settings.ConnectTimeout, before, after)
			if queries := assertStatements(t, recorder); handshakeCompleted {
				assertBudgetDeadlines(t, "lease session statement", queries, settings.QueryTimeout, before, after)
			}
		})
		t.Run(stage.name+"/earlier caller deadline", func(t *testing.T) {
			settings := DatabaseSettings{
				ConnectTimeout: 2 * time.Hour,
				QueryTimeout:   3 * time.Hour, StatementTimeout: 2 * time.Hour, LockTimeout: time.Hour,
			}
			peer := newStallingPeer(t, stage.answer)
			pool, recorder := stallingPeerPool(t, peer, settings)
			ctx, cancel := context.WithDeadline(t.Context(), time.Now().Add(time.Hour))
			defer cancel()
			holders := make(standbyHolders, 1)
			result := startLeaseStartup(ctx, pool, slog.New(holders))
			if at := awaitObservation(t, peer.stalled, "the stalled step"); at != stage.stalledAt {
				t.Fatalf("the lease session stalled at %s, want %s", at, stage.stalledAt)
			}
			// The caller, Server shutdown here, cancels while the step stalls.
			cancel()
			assertStartupEnd(t, awaitObservation(t, result, "the end of the cancelled startup"), context.Canceled)
			if stage.failure == "" {
				assertStoodByWithUnknownHolder(t, holders)
			}
			assertStalledSessionCleanup(t, peer, handshakeCompleted)
			assertCallerDeadlines(t, "lease session connect", recorder.sessionDial(t), ctx)
			if queries := assertStatements(t, recorder); handshakeCompleted {
				assertCallerDeadlines(t, "lease session statement", queries, ctx)
			}
		})
	}
}
