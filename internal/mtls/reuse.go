package mtls

import (
	"container/list"
	"context"
	"crypto/tls"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"sync"
	"time"

	"github.com/grauwolf32/contractor/internal/contracts"
)

// Bounds of reusable private connections (CTL-R165). The Runtime holds the
// same values in contractor_runtime._https and api/testdata/v1alpha1/
// private-connection-reuse.json pins both sides.
const (
	// PrivateListenerIdleTimeout is how long a private listener keeps an idle
	// keep-alive connection open (LIM-CTL-13). It exceeds the client idle
	// reuse bound, so a client normally never writes a request to a
	// connection the listener has already closed.
	PrivateListenerIdleTimeout = time.Minute
	// privateConnectionIdleReuse bounds how long a client keeps an idle
	// connection for reuse (LIM-CTL-14).
	privateConnectionIdleReuse = 30 * time.Second
	// privateConnectionReuseLifetime bounds the age of a connection at the
	// start of a request (LIM-CTL-15), so peer certificate rotation reaches
	// every reused connection within this time.
	privateConnectionReuseLifetime = 5 * time.Minute
	// privateConnectionsPerPeer and privateIdleConnectionsPerPeer bound the
	// open and idle connections one owner keeps per peer endpoint
	// (LIM-CTL-16). The bound holds per generation: while a retired
	// generation finishes its requests, the endpoint may carry its
	// connections in use as well, up to twice the bound, and none of them
	// takes a new request.
	privateConnectionsPerPeer     = 8
	privateIdleConnectionsPerPeer = 2
	// runtimeAgentTransportCapacity bounds the principals one owner keeps
	// connections for (LIM-CTL-17).
	runtimeAgentTransportCapacity = 256
)

// ErrRuntimeAgentTransportsClosed reports a request started after the owner
// closed its connections.
var ErrRuntimeAgentTransportsClosed = errors.New("Runtime Agent connections are closed")

// RuntimeAgentTransports owns reusable private connections from the Server to
// Runtime Agent endpoints (CTL-R163, CTL-R164). Every connection is verified
// for one expected Runtime Agent principal before net/http writes a request
// byte (BindRuntimeAgentPrincipal), and a connection verified for one
// principal is never offered to a request for another: each principal has its
// own http.Transport, which keys its connections by endpoint. The owner bounds
// the principals it keeps, the connections per endpoint, idle reuse and the
// age at which a connection stops taking requests, and Close ends every
// connection. Request deadlines stay with each request's context and client.
type RuntimeAgentTransports struct {
	base    *tls.Config
	timeout time.Duration
	// now drives idle eviction and the reuse lifetime; dialContext opens TCP
	// connections (nil dials directly). Tests replace both to run on a
	// virtual clock and to count connections.
	now         func() time.Time
	dialContext func(context.Context, string, string) (net.Conn, error)

	mu      sync.Mutex
	closed  bool
	entries map[string]*principalTransports
	// order lists entries by last activity, most recent first.
	order list.List
}

// Format redacts the owner as a whole for every verb (SEC-R88): through the
// unexported base field fmt would print the Control Plane's private key.
func (t *RuntimeAgentTransports) Format(state fmt.State, _ rune) {
	_, _ = io.WriteString(state, "mtls.RuntimeAgentTransports([REDACTED])")
}

// principalTransports holds the connections of one expected principal. A
// generation is one http.Transport; a new generation replaces the current one
// when the reuse lifetime ends, and a retired generation closes its
// connections as soon as its last request completes.
type principalTransports struct {
	runtimeAgentID string
	tlsConfig      *tls.Config
	element        *list.Element
	current        *transportGeneration
	inflight       int
	lastActive     time.Time
}

// Format redacts the connections as a whole for every verb (SEC-R88): through
// the unexported tlsConfig field fmt would print the Control Plane's private
// key.
func (principalTransports) Format(state fmt.State, _ rune) {
	_, _ = io.WriteString(state, "mtls.principalTransports([REDACTED])")
}

type transportGeneration struct {
	owner     *principalTransports
	transport *http.Transport
	createdAt time.Time
	inflight  int
	retired   bool
}

// NewRuntimeAgentTransports returns an owner of reusable connections that
// authenticate Runtime Agent endpoints with base. timeout bounds each TLS
// handshake and the wait for response headers, as before reuse.
func NewRuntimeAgentTransports(base *tls.Config, timeout time.Duration) (*RuntimeAgentTransports, error) {
	if base == nil {
		return nil, errors.New("Runtime Agent TLS client configuration is required")
	}
	if timeout <= 0 {
		return nil, errors.New("Runtime Agent request timeout must be positive")
	}
	verified := base.Clone()
	// Every new connection performs full certificate verification.
	verified.ClientSessionCache = nil
	return &RuntimeAgentTransports{
		base: verified, timeout: timeout, now: time.Now,
		entries: make(map[string]*principalTransports),
	}, nil
}

// RoundTripper returns the transport of requests that must reach the expected
// Runtime Agent principal.
func (o *RuntimeAgentTransports) RoundTripper(runtimeAgentID string) (http.RoundTripper, error) {
	if !contracts.ValidRuntimeAgentID(runtimeAgentID) {
		return nil, fmt.Errorf("%w: expected principal is invalid", ErrRuntimeAgentIdentity)
	}
	return principalRoundTripper{owner: o, runtimeAgentID: runtimeAgentID}, nil
}

// Evict closes the connections of one principal: idle ones at once, those in
// use when their response completes. A later request opens and verifies a
// new connection.
func (o *RuntimeAgentTransports) Evict(runtimeAgentID string) {
	o.mu.Lock()
	var closing []*http.Transport
	if entry := o.entries[runtimeAgentID]; entry != nil {
		closing = o.removeLocked(entry, closing)
	}
	o.mu.Unlock()
	closeIdle(closing)
}

// Close ends every connection and refuses later requests. Idle connections
// close at once; a connection in use closes when its response completes.
func (o *RuntimeAgentTransports) Close() {
	o.mu.Lock()
	var closing []*http.Transport
	if !o.closed {
		o.closed = true
		for _, entry := range o.entries {
			closing = o.removeLocked(entry, closing)
		}
	}
	o.mu.Unlock()
	closeIdle(closing)
}

func (o *RuntimeAgentTransports) begin(runtimeAgentID string) (*transportGeneration, error) {
	o.mu.Lock()
	if o.closed {
		o.mu.Unlock()
		return nil, ErrRuntimeAgentTransportsClosed
	}
	now := o.now()
	closing := o.sweepLocked(now, nil)
	entry := o.entries[runtimeAgentID]
	if entry == nil {
		bound, err := BindRuntimeAgentPrincipal(o.base, runtimeAgentID)
		if err != nil {
			o.mu.Unlock()
			closeIdle(closing)
			return nil, err
		}
		entry = &principalTransports{runtimeAgentID: runtimeAgentID, tlsConfig: bound}
		entry.element = o.order.PushFront(entry)
		o.entries[runtimeAgentID] = entry
		for len(o.entries) > runtimeAgentTransportCapacity {
			closing = o.removeLocked(o.order.Back().Value.(*principalTransports), closing)
		}
	}
	if entry.current == nil || !now.Before(entry.current.createdAt.Add(privateConnectionReuseLifetime)) {
		if entry.current != nil {
			closing = retireLocked(entry.current, closing)
		}
		entry.current = o.newGeneration(entry, now)
	}
	generation := entry.current
	generation.inflight++
	entry.inflight++
	entry.lastActive = now
	o.order.MoveToFront(entry.element)
	o.mu.Unlock()
	closeIdle(closing)
	return generation, nil
}

func (o *RuntimeAgentTransports) finish(generation *transportGeneration) {
	o.mu.Lock()
	var closing []*http.Transport
	generation.inflight--
	entry := generation.owner
	entry.inflight--
	if generation.retired && generation.inflight == 0 {
		closing = append(closing, generation.transport)
	}
	if entry.element != nil {
		entry.lastActive = o.now()
		o.order.MoveToFront(entry.element)
	}
	o.mu.Unlock()
	closeIdle(closing)
}

// sweepLocked removes principals idle for the idle reuse bound, starting with
// the least recently active one. A principal with a request in flight is
// never idle; the walk stops at the first idle principal still within the
// bound, so each call does work proportional to what it removes plus the
// requests in flight.
func (o *RuntimeAgentTransports) sweepLocked(now time.Time, closing []*http.Transport) []*http.Transport {
	for element := o.order.Back(); element != nil; {
		previous := element.Prev()
		entry := element.Value.(*principalTransports)
		if entry.inflight == 0 {
			if now.Sub(entry.lastActive) < privateConnectionIdleReuse {
				break
			}
			closing = o.removeLocked(entry, closing)
		}
		element = previous
	}
	return closing
}

func (o *RuntimeAgentTransports) removeLocked(
	entry *principalTransports, closing []*http.Transport,
) []*http.Transport {
	delete(o.entries, entry.runtimeAgentID)
	if entry.element != nil {
		o.order.Remove(entry.element)
		entry.element = nil
	}
	if entry.current != nil {
		closing = retireLocked(entry.current, closing)
		entry.current = nil
	}
	return closing
}

// retireLocked stops new requests on a generation and closes its idle
// connections at once. net/http then closes each connection still in use as
// it goes idle. A request that began before the retirement (or a net/http
// retry of one) asks the transport for a connection and so undoes that, so
// finish closes the idle connections once more after the last request.
func retireLocked(generation *transportGeneration, closing []*http.Transport) []*http.Transport {
	generation.retired = true
	return append(closing, generation.transport)
}

func (o *RuntimeAgentTransports) newGeneration(entry *principalTransports, now time.Time) *transportGeneration {
	return &transportGeneration{
		owner: entry, createdAt: now,
		transport: &http.Transport{
			DialContext: o.dialContext, TLSClientConfig: entry.tlsConfig, ForceAttemptHTTP2: false,
			TLSHandshakeTimeout: o.timeout, ResponseHeaderTimeout: o.timeout,
			MaxConnsPerHost: privateConnectionsPerPeer, MaxIdleConns: privateIdleConnectionsPerPeer,
			MaxIdleConnsPerHost: privateIdleConnectionsPerPeer, IdleConnTimeout: privateConnectionIdleReuse,
		},
	}
}

// closeIdle runs outside the owner's lock: closing a TLS connection writes
// its close alert.
func closeIdle(transports []*http.Transport) {
	for _, transport := range transports {
		transport.CloseIdleConnections()
	}
}

type principalRoundTripper struct {
	owner          *RuntimeAgentTransports
	runtimeAgentID string
}

// errPrivateRequestHeader reports a request that would take over connection
// management or let net/http repeat a write (CTL-R166).
var errPrivateRequestHeader = errors.New("private request carries a connection or idempotency-key header")

func (r principalRoundTripper) RoundTrip(request *http.Request) (*http.Response, error) {
	var generation *transportGeneration
	err := errPrivateRequestHeader
	// net/http repeats a request that carries an idempotency key after a
	// reused connection fails, and Connection belongs to the owner.
	if request.Header.Values("Connection") == nil && request.Header.Values("Idempotency-Key") == nil &&
		request.Header.Values("X-Idempotency-Key") == nil && !request.Close {
		generation, err = r.owner.begin(r.runtimeAgentID)
	}
	if err != nil {
		if request.Body != nil {
			_ = request.Body.Close()
		}
		return nil, err
	}
	response, err := generation.transport.RoundTrip(request)
	if err != nil {
		r.owner.finish(generation)
		return nil, err
	}
	response.Body = &generationBody{
		ReadCloser: response.Body, finish: func() { r.owner.finish(generation) },
	}
	return response, nil
}

// generationBody keeps its generation's request in flight until the caller
// closes the body. net/http returns a connection whose body was read to the
// end to its idle pool before the read reports the end, and closes a
// connection whose body was closed early, so a retired generation has no
// connection left in use once its last body closes.
type generationBody struct {
	io.ReadCloser
	once   sync.Once
	finish func()
}

func (b *generationBody) Close() error {
	err := b.ReadCloser.Close()
	b.once.Do(b.finish)
	return err
}
