package mtls

import (
	"context"
	"crypto/tls"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
	"github.com/grauwolf32/contractor/internal/localpki"
	"github.com/grauwolf32/contractor/internal/mtlstest"
)

// reuseFixture is one Runtime Agent endpoint whose certificate the test can
// rotate, with server-side counters of verified handshakes and requests and
// client-side counters of dialed and closed connections.
type reuseFixture struct {
	root         string
	ca           string
	controlPlane Files
	server       *httptest.Server
	certificate  atomic.Pointer[tls.Certificate]
	principal    string
	handshakes   atomic.Int64
	requests     atomic.Int64
	clock        *virtualClock
	dials        *dialCounter
}

func newReuseFixture(t *testing.T, handler http.Handler) *reuseFixture {
	t.Helper()
	root := t.TempDir()
	generator := localpki.Generator{}
	ca, err := generator.InitCA(root, false)
	if err != nil {
		t.Fatal(err)
	}
	leaf := localpki.LeafOptions{IPAddresses: []net.IP{net.ParseIP("127.0.0.1")}}
	controlPlane, err := generator.IssueControlPlane(root, localpki.ControlPlaneOptions{LeafOptions: leaf})
	if err != nil {
		t.Fatal(err)
	}
	agent, err := generator.IssueAgent(root, "agent", leaf)
	if err != nil {
		t.Fatal(err)
	}
	f := &reuseFixture{
		root: root, ca: ca.Certificate,
		controlPlane: Files{Certificate: controlPlane.Certificate, PrivateKey: controlPlane.PrivateKey, CA: ca.Certificate},
		clock:        &virtualClock{now: time.Date(2026, 10, 23, 12, 0, 0, 0, time.UTC)},
		dials:        &dialCounter{},
	}
	f.principal = f.present(t, agent)
	serverConfig, err := mtlstest.AgentServerConfig(mtlstest.Files{
		Certificate: agent.Certificate, PrivateKey: agent.PrivateKey, CA: ca.Certificate,
	})
	if err != nil {
		t.Fatal(err)
	}
	verifyClient := serverConfig.VerifyConnection
	serverConfig.VerifyConnection = func(state tls.ConnectionState) error {
		if err := verifyClient(state); err != nil {
			return err
		}
		f.handshakes.Add(1)
		return nil
	}
	// Each handshake presents the current certificate; httptest's own
	// certificate on the outer configuration is never used.
	handshakeConfig := serverConfig
	serverConfig = &tls.Config{MinVersion: tls.VersionTLS13}
	serverConfig.GetConfigForClient = func(*tls.ClientHelloInfo) (*tls.Config, error) {
		current := handshakeConfig.Clone()
		current.Certificates = []tls.Certificate{*f.certificate.Load()}
		return current, nil
	}
	f.server = httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		f.requests.Add(1)
		handler.ServeHTTP(w, r)
	}))
	f.server.TLS = serverConfig
	f.server.EnableHTTP2 = false
	f.server.Config.ErrorLog = log.New(io.Discard, "", 0)
	f.server.StartTLS()
	t.Cleanup(f.server.Close)
	return f
}

// present makes the endpoint serve the certificate at paths from its next
// handshake on and returns that certificate's principal.
func (f *reuseFixture) present(t *testing.T, paths localpki.Paths) string {
	t.Helper()
	certificate, err := tls.LoadX509KeyPair(paths.Certificate, paths.PrivateKey)
	if err != nil {
		t.Fatal(err)
	}
	principal, err := RuntimeAgentID(certificate.Leaf)
	if err != nil {
		t.Fatal(err)
	}
	f.certificate.Store(&certificate)
	return principal
}

func (f *reuseFixture) owner(t *testing.T) *RuntimeAgentTransports {
	t.Helper()
	base, err := ControlPlaneEndpointClientConfig(f.controlPlane)
	if err != nil {
		t.Fatal(err)
	}
	owner, err := NewRuntimeAgentTransports(base, 2*time.Second)
	if err != nil {
		t.Fatal(err)
	}
	owner.now = f.clock.Now
	owner.dialContext = f.dials.dial
	t.Cleanup(owner.Close)
	return owner
}

func (f *reuseFixture) client(t *testing.T, owner *RuntimeAgentTransports, principal string) *http.Client {
	t.Helper()
	transport, err := owner.RoundTripper(principal)
	if err != nil {
		t.Fatal(err)
	}
	return &http.Client{Transport: transport, Timeout: 2 * time.Second}
}

// call sends one request and reads the whole response body.
func call(ctx context.Context, client *http.Client, method, url, probe string) (string, error) {
	request, err := http.NewRequestWithContext(ctx, method, url, strings.NewReader(probe))
	if err != nil {
		return "", err
	}
	request.Header.Set("X-Probe", probe)
	response, err := client.Do(request)
	if err != nil {
		return "", err
	}
	defer response.Body.Close()
	body, err := io.ReadAll(response.Body)
	if err != nil {
		return "", err
	}
	if response.StatusCode != http.StatusOK {
		return "", fmt.Errorf("status %d", response.StatusCode)
	}
	return string(body), nil
}

func echoProbe(w http.ResponseWriter, r *http.Request) {
	_, _ = io.Copy(io.Discard, r.Body)
	_, _ = io.WriteString(w, r.Header.Get("X-Probe"))
}

type virtualClock struct {
	mu  sync.Mutex
	now time.Time
}

func (c *virtualClock) Now() time.Time {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.now
}

func (c *virtualClock) Advance(d time.Duration) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.now = c.now.Add(d)
}

type dialCounter struct {
	dialed, closed atomic.Int64
	fail           atomic.Bool
}

func (d *dialCounter) dial(ctx context.Context, network, address string) (net.Conn, error) {
	if d.fail.Load() {
		return nil, errors.New("dial refused by test")
	}
	conn, err := (&net.Dialer{}).DialContext(ctx, network, address)
	if err != nil {
		return nil, err
	}
	d.dialed.Add(1)
	return &countedConn{Conn: conn, closed: &d.closed}, nil
}

func (d *dialCounter) open() int64 { return d.dialed.Load() - d.closed.Load() }

type countedConn struct {
	net.Conn
	once   sync.Once
	closed *atomic.Int64
}

func (c *countedConn) Close() error {
	c.once.Do(func() { c.closed.Add(1) })
	return c.Conn.Close()
}

func TestRuntimeAgentTransportsReuseOneVerifiedConnection(t *testing.T) {
	f := newReuseFixture(t, http.HandlerFunc(echoProbe))
	owner := f.owner(t)
	client := f.client(t, owner, f.principal)
	for index := range 10 {
		probe := fmt.Sprintf("call-%d", index)
		method := http.MethodPost
		if index%2 == 1 {
			method = http.MethodGet
		}
		if body, err := call(context.Background(), client, method, f.server.URL, probe); err != nil || body != probe {
			t.Fatalf("call %d = (%q, %v)", index, body, err)
		}
	}
	if f.handshakes.Load() != 1 || f.dials.dialed.Load() != 1 || f.requests.Load() != 10 {
		t.Fatalf("handshakes=%d dials=%d requests=%d, want one verified connection for ten requests",
			f.handshakes.Load(), f.dials.dialed.Load(), f.requests.Load())
	}

	// The fresh-connection client this replaced verified one connection per call.
	fresh := newReuseFixture(t, http.HandlerFunc(echoProbe))
	base, err := ControlPlaneEndpointClientConfig(fresh.controlPlane)
	if err != nil {
		t.Fatal(err)
	}
	bound, err := BindRuntimeAgentPrincipal(base, fresh.principal)
	if err != nil {
		t.Fatal(err)
	}
	old := &http.Client{Transport: &http.Transport{TLSClientConfig: bound, DisableKeepAlives: true}}
	for index := range 10 {
		if _, err := call(context.Background(), old, http.MethodPost, fresh.server.URL, "old"); err != nil {
			t.Fatalf("fresh-connection call %d: %v", index, err)
		}
	}
	if fresh.handshakes.Load() != 10 {
		t.Fatalf("fresh-connection handshakes=%d, want 10", fresh.handshakes.Load())
	}

	owner.Close()
	if f.dials.open() != 0 {
		t.Fatalf("Close left %d connections open", f.dials.open())
	}
	if _, err := call(context.Background(), client, http.MethodGet, f.server.URL, "late"); !errors.Is(err, ErrRuntimeAgentTransportsClosed) {
		t.Fatalf("call after Close = %v, want ErrRuntimeAgentTransportsClosed", err)
	}
	if f.dials.dialed.Load() != 1 {
		t.Fatalf("a call after Close dialed: dials=%d", f.dials.dialed.Load())
	}
}

func TestRuntimeAgentTransportsNeverOfferAConnectionToAnotherPrincipal(t *testing.T) {
	f := newReuseFixture(t, http.HandlerFunc(echoProbe))
	owner := f.owner(t)
	if _, err := call(context.Background(), f.client(t, owner, f.principal), http.MethodPost, f.server.URL, "a"); err != nil {
		t.Fatal(err)
	}
	other := strings.Repeat("0", 64)
	_, err := call(context.Background(), f.client(t, owner, other), http.MethodPost, f.server.URL, "b")
	if !errors.Is(err, ErrRuntimeAgentIdentity) {
		t.Fatalf("request for another principal = %v, want ErrRuntimeAgentIdentity", err)
	}
	if f.requests.Load() != 1 || f.dials.dialed.Load() != 2 {
		t.Fatalf("requests=%d dials=%d: the other principal reused a connection or reached HTTP",
			f.requests.Load(), f.dials.dialed.Load())
	}
	if _, err := owner.RoundTripper("not-a-principal"); !errors.Is(err, ErrRuntimeAgentIdentity) {
		t.Fatalf("invalid principal = %v", err)
	}
}

func TestRuntimeAgentTransportsReverifyRotatedCertificatesAfterLifetime(t *testing.T) {
	for _, test := range []struct {
		name   string
		rotate func(*testing.T, *reuseFixture)
		accept bool
	}{
		{
			name: "renewed certificate for the same key", accept: true,
			rotate: func(t *testing.T, f *reuseFixture) {
				renewed, err := localpki.Generator{}.RenewAgent(f.root, "agent")
				if err != nil {
					t.Fatal(err)
				}
				if f.present(t, renewed) != f.principal {
					t.Fatal("renewal changed the principal")
				}
			},
		},
		{
			name: "new key",
			rotate: func(t *testing.T, f *reuseFixture) {
				replaced, err := localpki.Generator{}.IssueAgent(
					f.root, "replacement", localpki.LeafOptions{IPAddresses: []net.IP{net.ParseIP("127.0.0.1")}},
				)
				if err != nil {
					t.Fatal(err)
				}
				if f.present(t, replaced) == f.principal {
					t.Fatal("a new key kept the principal")
				}
			},
		},
		{
			name: "untrusted CA",
			rotate: func(t *testing.T, f *reuseFixture) {
				root := t.TempDir()
				generator := localpki.Generator{}
				if _, err := generator.InitCA(root, false); err != nil {
					t.Fatal(err)
				}
				foreign, err := generator.IssueAgent(
					root, "agent", localpki.LeafOptions{IPAddresses: []net.IP{net.ParseIP("127.0.0.1")}},
				)
				if err != nil {
					t.Fatal(err)
				}
				f.present(t, foreign)
			},
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			f := newReuseFixture(t, http.HandlerFunc(echoProbe))
			owner := f.owner(t)
			client := f.client(t, owner, f.principal)
			if _, err := call(context.Background(), client, http.MethodPost, f.server.URL, "before"); err != nil {
				t.Fatal(err)
			}
			test.rotate(t, f)
			// Within the lifetime the verified connection keeps serving; a
			// call before each idle bound keeps the principal active.
			advanced, step := time.Duration(0), privateConnectionIdleReuse-time.Second
			for advanced+step < privateConnectionReuseLifetime {
				f.clock.Advance(step)
				advanced += step
				if _, err := call(context.Background(), client, http.MethodPost, f.server.URL, "within"); err != nil {
					t.Fatal(err)
				}
			}
			f.clock.Advance(privateConnectionReuseLifetime - advanced - time.Second)
			if _, err := call(context.Background(), client, http.MethodPost, f.server.URL, "last"); err != nil {
				t.Fatal(err)
			}
			if f.handshakes.Load() != 1 {
				t.Fatalf("handshakes=%d before the lifetime ended", f.handshakes.Load())
			}
			served := f.requests.Load()
			f.clock.Advance(time.Second)
			_, err := call(context.Background(), client, http.MethodPost, f.server.URL, "after")
			if test.accept {
				if err != nil || f.handshakes.Load() != 2 || f.requests.Load() != served+1 {
					t.Fatalf("renewed certificate: err=%v handshakes=%d requests=%d", err, f.handshakes.Load(), f.requests.Load())
				}
			} else if err == nil || f.requests.Load() != served {
				t.Fatalf("rotated peer: err=%v requests=%d, want rejection before request bytes", err, f.requests.Load())
			}
			if f.dials.dialed.Load() != 2 {
				t.Fatalf("dials=%d, want one new connection after the lifetime", f.dials.dialed.Load())
			}
			// The connection of the retired generation closed once idle.
			if f.dials.open() > 1 {
				t.Fatalf("open connections=%d after retirement", f.dials.open())
			}
		})
	}
}

func TestRuntimeAgentTransportsEvictIdlePrincipalsAndBoundCapacity(t *testing.T) {
	f := newReuseFixture(t, http.HandlerFunc(echoProbe))
	owner := f.owner(t)
	client := f.client(t, owner, f.principal)
	if _, err := call(context.Background(), client, http.MethodPost, f.server.URL, "first"); err != nil {
		t.Fatal(err)
	}
	f.clock.Advance(privateConnectionIdleReuse - time.Second)
	if _, err := call(context.Background(), client, http.MethodPost, f.server.URL, "within"); err != nil {
		t.Fatal(err)
	}
	f.clock.Advance(privateConnectionIdleReuse)
	if _, err := call(context.Background(), client, http.MethodPost, f.server.URL, "idle"); err != nil {
		t.Fatal(err)
	}
	if f.handshakes.Load() != 2 || f.dials.open() != 1 {
		t.Fatalf("handshakes=%d open=%d: an idle principal was not evicted and closed",
			f.handshakes.Load(), f.dials.open())
	}

	// More principals than the capacity: the least recently active ones go.
	f.dials.fail.Store(true)
	const principals = runtimeAgentTransportCapacity + 44
	for index := range principals {
		id := fmt.Sprintf("%064x", index+1)
		if _, err := call(context.Background(), f.client(t, owner, id), http.MethodGet, f.server.URL, "x"); err == nil {
			t.Fatal("refused dial succeeded")
		}
	}
	owner.mu.Lock()
	entries, ordered := len(owner.entries), owner.order.Len()
	_, kept := owner.entries[fmt.Sprintf("%064x", principals)]
	_, original := owner.entries[f.principal]
	owner.mu.Unlock()
	if entries != runtimeAgentTransportCapacity || ordered != entries || !kept || original {
		t.Fatalf("entries=%d ordered=%d newest kept=%t evicted original kept=%t",
			entries, ordered, kept, original)
	}
	if f.dials.open() != 0 {
		t.Fatalf("capacity eviction left %d connections open", f.dials.open())
	}
}

func TestRuntimeAgentTransportsBoundConcurrentConnectionsWithoutCrossTalk(t *testing.T) {
	arrived := make(chan struct{}, 64)
	release := make(chan struct{})
	f := newReuseFixture(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		arrived <- struct{}{}
		<-release
		echoProbe(w, r)
	}))
	owner := f.owner(t)
	client := f.client(t, owner, f.principal)
	const calls = privateConnectionsPerPeer + 4
	results := make(chan error, calls)
	for index := range calls {
		go func() {
			probe := fmt.Sprintf("concurrent-%d", index)
			body, err := call(context.Background(), client, http.MethodPost, f.server.URL, probe)
			if err == nil && body != probe {
				err = fmt.Errorf("response %q answered request %q", body, probe)
			}
			results <- err
		}()
	}
	for range privateConnectionsPerPeer {
		<-arrived
	}
	close(release)
	for range calls {
		if err := <-results; err != nil {
			t.Fatal(err)
		}
	}
	if f.dials.dialed.Load() != privateConnectionsPerPeer || f.requests.Load() != calls {
		t.Fatalf("dials=%d requests=%d, want at most %d connections", f.dials.dialed.Load(), f.requests.Load(), privateConnectionsPerPeer)
	}
	owner.mu.Lock()
	inflight := owner.entries[f.principal].inflight
	owner.mu.Unlock()
	if inflight != 0 {
		t.Fatalf("inflight=%d after every body closed", inflight)
	}
	// net/http closes a connection beyond the idle bound just after the
	// response body reports its end.
	awaitCondition(t, func() bool { return f.dials.open() <= privateIdleConnectionsPerPeer },
		"idle connections stay above the idle bound")
}

// awaitCondition waits for an asynchronous effect of the other side or of the
// net/http connection loops. The deadline only bounds a failing test; no
// assertion depends on elapsed time.
func awaitCondition(t *testing.T, condition func() bool, failure string) {
	t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	for !condition() {
		if time.Now().After(deadline) {
			t.Fatal(failure)
		}
		time.Sleep(time.Millisecond)
	}
}

func TestRuntimeAgentTransportsCloseCancelledConnectionsAndFinishInFlightOnes(t *testing.T) {
	arrived := make(chan string, 4)
	release := make(chan struct{})
	f := newReuseFixture(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		probe := r.Header.Get("X-Probe")
		// Reading the body lets the server notice a closed connection.
		_, _ = io.Copy(io.Discard, r.Body)
		arrived <- probe
		if probe == "cancelled" {
			<-r.Context().Done()
			return
		}
		if probe == "held" {
			<-release
		}
		echoProbe(w, r)
	}))
	owner := f.owner(t)
	client := f.client(t, owner, f.principal)
	if _, err := call(context.Background(), client, http.MethodPost, f.server.URL, "warm"); err != nil {
		t.Fatal(err)
	}
	<-arrived
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() {
		_, err := call(ctx, client, http.MethodPost, f.server.URL, "cancelled")
		done <- err
	}()
	<-arrived
	cancel()
	if err := <-done; !errors.Is(err, context.Canceled) {
		t.Fatalf("cancelled call = %v", err)
	}
	if body, err := call(context.Background(), client, http.MethodPost, f.server.URL, "next"); err != nil || body != "next" {
		t.Fatalf("call after cancellation = (%q, %v)", body, err)
	}
	<-arrived
	if f.dials.dialed.Load() != 2 || f.dials.open() != 1 {
		t.Fatalf("dials=%d open=%d: the cancelled connection was reused or left open", f.dials.dialed.Load(), f.dials.open())
	}

	// Close during a request: the request completes and its connection closes
	// when its body closes.
	held := make(chan error, 1)
	go func() {
		body, err := call(context.Background(), client, http.MethodPost, f.server.URL, "held")
		if err == nil && body != "held" {
			err = fmt.Errorf("held response %q", body)
		}
		held <- err
	}()
	<-arrived
	owner.Close()
	close(release)
	if err := <-held; err != nil {
		t.Fatal(err)
	}
	awaitCondition(t, func() bool { return f.dials.open() == 0 },
		"Close left a connection open after the in-flight response")
}

// TestRuntimeAgentTransportsCloseIdleConnectionsAtOnceWhileOthersAreInFlight
// holds one request of a principal while its other connection is idle: Evict
// and Close end the idle connection at once and the held one when its
// response completes (CTL-R164).
func TestRuntimeAgentTransportsCloseIdleConnectionsAtOnceWhileOthersAreInFlight(t *testing.T) {
	for _, closer := range []string{"Evict", "Close"} {
		t.Run(closer, func(t *testing.T) {
			arrived := make(chan struct{}, 2)
			release := make(chan struct{})
			var releaseOnce sync.Once
			releaseHeld := func() { releaseOnce.Do(func() { close(release) }) }
			f := newReuseFixture(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.Header.Get("X-Probe") == "held" {
					arrived <- struct{}{}
					<-release
				}
				echoProbe(w, r)
			}))
			// Registered after the fixture, so a failing test releases the
			// held handler before the server closes.
			t.Cleanup(releaseHeld)
			owner := f.owner(t)
			client := f.client(t, owner, f.principal)
			held := make(chan error, 1)
			go func() {
				_, err := call(context.Background(), client, http.MethodPost, f.server.URL, "held")
				held <- err
			}()
			<-arrived
			if _, err := call(context.Background(), client, http.MethodPost, f.server.URL, "idle"); err != nil {
				t.Fatal(err)
			}
			if f.dials.dialed.Load() != 2 || f.dials.open() != 2 {
				t.Fatalf("dials=%d open=%d, want one held and one idle connection", f.dials.dialed.Load(), f.dials.open())
			}
			if closer == "Evict" {
				owner.Evict(f.principal)
			} else {
				owner.Close()
			}
			if f.dials.open() != 1 {
				t.Fatalf("%s left the idle connection open: open=%d", closer, f.dials.open())
			}
			releaseHeld()
			if err := <-held; err != nil {
				t.Fatal(err)
			}
			awaitCondition(t, func() bool { return f.dials.open() == 0 },
				closer+" left the held connection open after its response")
		})
	}
}

func TestRuntimeAgentTransportsEvictClosesOnePrincipal(t *testing.T) {
	f := newReuseFixture(t, http.HandlerFunc(echoProbe))
	owner := f.owner(t)
	client := f.client(t, owner, f.principal)
	if _, err := call(context.Background(), client, http.MethodPost, f.server.URL, "a"); err != nil {
		t.Fatal(err)
	}
	owner.Evict(f.principal)
	if f.dials.open() != 0 {
		t.Fatalf("Evict left %d connections open", f.dials.open())
	}
	if _, err := call(context.Background(), client, http.MethodPost, f.server.URL, "b"); err != nil {
		t.Fatal(err)
	}
	if f.handshakes.Load() != 2 {
		t.Fatalf("handshakes=%d, want a new verified connection after Evict", f.handshakes.Load())
	}
}

func TestRuntimeAgentTransportsNeverReplayAnAmbiguousWrite(t *testing.T) {
	var receipts atomic.Int64
	f := newReuseFixture(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("X-Probe") != "write" {
			echoProbe(w, r)
			return
		}
		_, _ = io.Copy(io.Discard, r.Body)
		receipts.Add(1)
		// The request reached the peer, which drops the connection before
		// answering: the outcome is unknown to the client.
		connection, _, err := w.(http.Hijacker).Hijack()
		if err == nil {
			_ = connection.Close()
		}
	}))
	owner := f.owner(t)
	client := f.client(t, owner, f.principal)
	if _, err := call(context.Background(), client, http.MethodPost, f.server.URL, "warm"); err != nil {
		t.Fatal(err)
	}
	if _, err := call(context.Background(), client, http.MethodPost, f.server.URL, "write"); err == nil {
		t.Fatal("an ambiguous write reported success")
	}
	if receipts.Load() != 1 {
		t.Fatalf("receipts=%d, want the ambiguous write delivered exactly once", receipts.Load())
	}
	// A header that would let net/http repeat a write, or take connection
	// management from the owner, is refused before any byte is sent.
	for _, header := range []string{"Idempotency-Key", "X-Idempotency-Key", "Connection"} {
		request, err := http.NewRequestWithContext(
			context.Background(), http.MethodPost, f.server.URL, strings.NewReader("write"),
		)
		if err != nil {
			t.Fatal(err)
		}
		request.Header.Set("X-Probe", "write")
		request.Header.Set(header, "close")
		if _, err := client.Do(request); !errors.Is(err, errPrivateRequestHeader) {
			t.Fatalf("%s request = %v, want refusal", header, err)
		}
	}
	if receipts.Load() != 1 {
		t.Fatalf("receipts=%d after refused requests", receipts.Load())
	}
}

func TestSharedPrivateConnectionReuseLimits(t *testing.T) {
	var limits struct {
		ListenerIdleTimeoutSeconds int `json:"listenerIdleTimeoutSeconds"`
		ClientIdleReuseSeconds     int `json:"clientIdleReuseSeconds"`
		ReuseLifetimeSeconds       int `json:"reuseLifetimeSeconds"`
		ConnectionsPerPeer         int `json:"connectionsPerPeer"`
		IdleConnectionsPerPeer     int `json:"idleConnectionsPerPeer"`
	}
	if err := json.Unmarshal(contractstest.ReadFile(t, "api", "testdata", "v1alpha1", "private-connection-reuse.json"), &limits); err != nil {
		t.Fatal(err)
	}
	if time.Duration(limits.ListenerIdleTimeoutSeconds)*time.Second != PrivateListenerIdleTimeout ||
		time.Duration(limits.ClientIdleReuseSeconds)*time.Second != privateConnectionIdleReuse ||
		time.Duration(limits.ReuseLifetimeSeconds)*time.Second != privateConnectionReuseLifetime ||
		limits.ConnectionsPerPeer != privateConnectionsPerPeer ||
		limits.IdleConnectionsPerPeer != privateIdleConnectionsPerPeer {
		t.Fatalf("Go reuse limits differ from the shared fixture: %+v", limits)
	}
	if privateConnectionIdleReuse >= PrivateListenerIdleTimeout {
		t.Fatal("client idle reuse must end before the peer listener closes an idle connection")
	}
}
