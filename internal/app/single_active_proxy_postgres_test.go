package app

import (
	"context"
	"io"
	"log/slog"
	"net"
	"os"
	"strconv"
	"sync"
	"testing"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"
)

// pauseProxy forwards a single TCP stream between a client and an upstream
// PostgreSQL server and can hold the upstream-to-client direction to simulate
// a transient or sustained server-to-client stall without tearing the
// connection down. The upstream socket stays open while paused, so PostgreSQL
// still sees the backend as live and keeps any session advisory lock.
type pauseProxy struct {
	listener net.Listener
	upstream string
	gateMu   sync.Mutex
	connsMu  sync.Mutex
	conns    []net.Conn
	closed   chan struct{}
}

func newPauseProxy(t *testing.T, upstream string) *pauseProxy {
	t.Helper()
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	proxy := &pauseProxy{listener: listener, upstream: upstream, closed: make(chan struct{})}
	go proxy.accept()
	t.Cleanup(proxy.close)
	return proxy
}

func (p *pauseProxy) address() string { return p.listener.Addr().String() }

func (p *pauseProxy) pause()  { p.gateMu.Lock() }
func (p *pauseProxy) resume() { p.gateMu.Unlock() }

func (p *pauseProxy) track(conn net.Conn) {
	p.connsMu.Lock()
	p.conns = append(p.conns, conn)
	p.connsMu.Unlock()
}

func (p *pauseProxy) accept() {
	for {
		client, err := p.listener.Accept()
		if err != nil {
			return
		}
		upstream, err := net.Dial("tcp", p.upstream)
		if err != nil {
			_ = client.Close()
			continue
		}
		p.track(client)
		p.track(upstream)
		// Client-to-upstream flows freely; upstream-to-client is gated so a
		// pause holds the server's bytes mid-stream.
		go p.forward(upstream, client, false)
		go p.forward(client, upstream, true)
	}
}

func (p *pauseProxy) forward(dst, src net.Conn, gated bool) {
	buffer := make([]byte, 32*1024)
	for {
		read, err := src.Read(buffer)
		if read > 0 {
			if gated {
				p.gateMu.Lock()
				p.gateMu.Unlock() //nolint:staticcheck // gate: block while paused
			}
			if _, writeErr := dst.Write(buffer[:read]); writeErr != nil {
				return
			}
		}
		if err != nil {
			return
		}
	}
}

func (p *pauseProxy) close() {
	select {
	case <-p.closed:
		return
	default:
		close(p.closed)
	}
	_ = p.listener.Close()
	p.connsMu.Lock()
	for _, conn := range p.conns {
		_ = conn.Close()
	}
	p.connsMu.Unlock()
}

// proxiedLeasePool returns a pool whose lease session connects through a
// controllable TCP proxy, plus a direct pool to the same database for
// out-of-band assertions about lock ownership.
func proxiedLeasePool(t *testing.T, ctx context.Context) (*pgxpool.Pool, *pgxpool.Pool, *pauseProxy) {
	t.Helper()
	databaseURL := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if databaseURL == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	directConfig, err := pgxpool.ParseConfig(databaseURL)
	if err != nil {
		t.Fatal(err)
	}
	upstream := net.JoinHostPort(directConfig.ConnConfig.Host, strconv.Itoa(int(directConfig.ConnConfig.Port)))
	proxy := newPauseProxy(t, upstream)

	proxiedConfig, err := pgxpool.ParseConfig(databaseURL)
	if err != nil {
		t.Fatal(err)
	}
	host, port, err := net.SplitHostPort(proxy.address())
	if err != nil {
		t.Fatal(err)
	}
	parsedPort, err := strconv.ParseUint(port, 10, 16)
	if err != nil {
		t.Fatal(err)
	}
	proxiedConfig.ConnConfig.Host = host
	proxiedConfig.ConnConfig.Port = uint16(parsedPort)
	proxiedConfig.ConnConfig.Fallbacks = nil
	proxied, err := pgxpool.NewWithConfig(ctx, proxiedConfig)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(proxied.Close)

	direct, err := pgxpool.NewWithConfig(ctx, directConfig)
	if err != nil {
		t.Fatal(err)
	}
	if err := direct.Ping(ctx); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(direct.Close)
	return proxied, direct, proxy
}

func shortLease(t *testing.T, ctx context.Context, pool *pgxpool.Pool) *controlPlaneLease {
	t.Helper()
	lease, err := openControlPlaneLease(ctx, pool)
	if err != nil {
		t.Fatal(err)
	}
	lease.poll = 40 * time.Millisecond
	lease.tolerance = 600 * time.Millisecond
	t.Cleanup(lease.Close)
	return lease
}

func TestPostgresControlPlaneLeaseToleratesTransientStall(t *testing.T) {
	ctx, stop := context.WithTimeout(t.Context(), 20*time.Second)
	defer stop()
	proxied, _, proxy := proxiedLeasePool(t, ctx)
	lease := shortLease(t, ctx, proxied)
	if acquired, err := lease.tryAcquire(ctx); err != nil || !acquired {
		t.Fatalf("lease acquire = (%v, %v)", acquired, err)
	}
	watchCtx, cancel := context.WithCancelCause(ctx)
	done := lease.watch(watchCtx, cancel)
	defer func() {
		cancel(nil)
		<-done
	}()

	// A stall shorter than the tolerance window must not stop the active Server.
	proxy.pause()
	time.Sleep(200 * time.Millisecond)
	proxy.resume()

	select {
	case <-watchCtx.Done():
		t.Fatalf("transient stall stopped the Server: %v", context.Cause(watchCtx))
	case <-time.After(900 * time.Millisecond):
	}
	if lease.lost {
		t.Fatal("lease marked lost after a transient stall")
	}
}

func TestPostgresControlPlaneLeaseStopsOnSustainedOutageBeforeTakeover(t *testing.T) {
	ctx, stop := context.WithTimeout(t.Context(), 20*time.Second)
	defer stop()
	proxied, direct, proxy := proxiedLeasePool(t, ctx)
	lease := shortLease(t, ctx, proxied)
	if acquired, err := lease.tryAcquire(ctx); err != nil || !acquired {
		t.Fatalf("lease acquire = (%v, %v)", acquired, err)
	}
	watchCtx, cancel := context.WithCancelCause(ctx)
	done := lease.watch(watchCtx, cancel)
	defer func() {
		cancel(nil)
		<-done
	}()

	// Sustained outage: hold the server-to-client direction indefinitely.
	proxy.pause()
	defer proxy.resume()

	select {
	case <-watchCtx.Done():
		cause := context.Cause(watchCtx)
		if cause == nil || cause == context.Canceled {
			t.Fatalf("active Server stopped without a lease-loss cause: %v", cause)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("active Server did not stop within the tolerance bound")
	}

	// The lock-holding session's upstream socket is still open through the
	// proxy, so PostgreSQL has not released the lock: a competing Server cannot
	// acquire it at the moment this Server stops.
	contender, err := openControlPlaneLease(ctx, direct)
	if err != nil {
		t.Fatal(err)
	}
	defer contender.Close()
	acquireCtx, cancelAcquire := context.WithTimeout(ctx, 2*time.Second)
	defer cancelAcquire()
	if acquired, err := contender.tryAcquire(acquireCtx); err != nil || acquired {
		t.Fatalf("competing Server acquired the lease before the stalled holder released it = (%v, %v)", acquired, err)
	}
}

func TestPostgresControlPlaneLeaseStandbySurvivesAcquisitionStall(t *testing.T) {
	ctx, stop := context.WithTimeout(t.Context(), 25*time.Second)
	defer stop()
	proxied, direct, proxy := proxiedLeasePool(t, ctx)
	logger := slog.New(slog.NewTextHandler(io.Discard, nil))

	holder, err := openControlPlaneLease(ctx, direct)
	if err != nil {
		t.Fatal(err)
	}
	defer holder.Close()
	if acquired, err := holder.tryAcquire(ctx); err != nil || !acquired {
		t.Fatalf("holder acquire = (%v, %v)", acquired, err)
	}

	standby := shortLease(t, ctx, proxied)
	// Lock is held elsewhere, so acquisition is refused without error.
	if acquired, err := standby.acquireOrReconnect(ctx, logger); err != nil || acquired {
		t.Fatalf("standby first probe = (%v, %v)", acquired, err)
	}

	// A sustained stall cancels the probe and drops the standby's not-yet-
	// locking connection; acquireOrReconnect must recover rather than fail.
	proxy.pause()
	stalled, err := standby.acquireOrReconnect(ctx, logger)
	if err != nil || stalled {
		t.Fatalf("standby stalled probe = (%v, %v), want (false, nil)", stalled, err)
	}
	proxy.resume()

	// After reconnect the standby can still observe the lock is held.
	if acquired, err := standby.acquireOrReconnect(ctx, logger); err != nil || acquired {
		t.Fatalf("standby probe after reconnect = (%v, %v)", acquired, err)
	}

	// Once the holder releases, the reconnected standby acquires the lease.
	holder.Close()
	acquired := false
	for deadline := time.Now().Add(5 * time.Second); time.Now().Before(deadline); {
		acquired, err = standby.acquireOrReconnect(ctx, logger)
		if err != nil {
			t.Fatalf("standby takeover probe failed: %v", err)
		}
		if acquired {
			break
		}
		time.Sleep(40 * time.Millisecond)
	}
	if !acquired {
		t.Fatal("standby did not acquire the lease after the holder released it")
	}
}
