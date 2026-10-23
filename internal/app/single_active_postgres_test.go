package app

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net"
	"net/http"
	"strings"
	"testing"
	"time"

	publicapi "github.com/grauwolf32/contractor/internal/httpapi/public"
	"github.com/grauwolf32/contractor/internal/requestid"
)

func TestPostgresControlPlaneLeaseReleasesOnNormalExit(t *testing.T) {
	pool := isolatedAppPool(t, t.Context())
	first, err := openControlPlaneLease(t.Context(), pool)
	if err != nil {
		t.Fatal(err)
	}
	defer first.Close()
	second, err := openControlPlaneLease(t.Context(), pool)
	if err != nil {
		t.Fatal(err)
	}
	defer second.Close()
	if acquired, err := first.tryAcquire(t.Context()); err != nil || !acquired {
		t.Fatalf("first lease = (%v, %v)", acquired, err)
	}
	if acquired, err := second.tryAcquire(t.Context()); err != nil || acquired {
		t.Fatalf("second Server acquired held lease = (%v, %v)", acquired, err)
	}
	first.Close()
	if acquired, err := second.tryAcquire(t.Context()); err != nil || !acquired {
		t.Fatalf("second Server did not acquire released lease = (%v, %v)", acquired, err)
	}
}

func TestPostgresControlPlaneLeaseSessionDetectsVanishedHolder(t *testing.T) {
	pool := isolatedAppPool(t, t.Context())
	lease, err := openControlPlaneLease(t.Context(), pool)
	if err != nil {
		t.Fatal(err)
	}
	defer lease.Close()
	var version string
	if err := lease.conn.QueryRow(t.Context(), "SELECT version()").Scan(&version); err != nil {
		t.Fatal(err)
	}
	userTimeout, err := leaseUserTimeoutOn(version)
	if err != nil {
		t.Fatal(err)
	}
	for name, want := range map[string]string{
		"tcp_keepalives_idle": "10", "tcp_keepalives_interval": "5",
		"tcp_keepalives_count": "3", "tcp_user_timeout": userTimeout,
	} {
		var got string
		if err := lease.conn.QueryRow(t.Context(), "SELECT current_setting($1)", name).Scan(&got); err != nil || got != want {
			t.Errorf("%s = (%q, %v), want %q on %q", name, got, err, want, version)
		}
	}
}

// leaseUserTimeoutOn returns the tcp_user_timeout the lease session must read
// back from a server whose version() is version (PER-R6). PostgreSQL applies
// the setting only where the operating system has TCP_USER_TIMEOUT, which is
// Linux, and reads it as 0 elsewhere. Deciding from the build platform rather
// than from a probe keeps the assertion able to fail on Linux: a lease session
// that stopped requesting the timeout reads the kernel default 0 there.
func leaseUserTimeoutOn(version string) (string, error) {
	build, _, compiled := strings.Cut(version, ", compiled by ")
	at := strings.LastIndex(build, " on ")
	if !compiled || at < 0 || !strings.HasPrefix(build, "PostgreSQL ") {
		return "", fmt.Errorf("unrecognised PostgreSQL version() %q", version)
	}
	if strings.Contains(build[at+len(" on "):], "-linux") {
		return "30000", nil
	}
	return "0", nil
}

func TestControlPlaneLeaseUserTimeoutFollowsServerPlatform(t *testing.T) {
	for version, want := range map[string]string{
		"PostgreSQL 17.6 (Debian 17.6-1.pgdg120+1) on x86_64-pc-linux-gnu, compiled by gcc (Debian 12.2.0-14) 12.2.0, 64-bit": "30000",
		"PostgreSQL 17.6 on aarch64-unknown-linux-musl, compiled by gcc (Alpine 14.2.0) 14.2.0, 64-bit":                       "30000",
		"PostgreSQL 17.6 on x86_64-linux, compiled by gcc-14.2.0, 64-bit":                                                     "30000",
		"PostgreSQL 17.11 (Homebrew) on aarch64-apple-darwin27.0.0, compiled by Apple clang version 21.0.0, 64-bit":           "0",
		"PostgreSQL 17.2 on amd64-portbld-freebsd14.1, compiled by FreeBSD clang version 18.1.5, 64-bit":                      "0",
		"PostgreSQL 17.2 on x86_64-windows, compiled by msvc-19.42.34435, 64-bit":                                             "0",
	} {
		if got, err := leaseUserTimeoutOn(version); err != nil || got != want {
			t.Errorf("leaseUserTimeoutOn(%q) = (%q, %v), want %q", version, got, err, want)
		}
	}
	for _, version := range []string{
		"", "PostgreSQL 17.6 on x86_64-pc-linux-gnu", "PostgreSQL 15.4, compiled by Visual C++ build 1914, 64-bit",
		"Server 1.0 on x86_64-pc-linux-gnu, compiled by gcc, 64-bit",
	} {
		if got, err := leaseUserTimeoutOn(version); err == nil {
			t.Errorf("leaseUserTimeoutOn(%q) = %q, want an unrecognised-version error", version, got)
		}
	}
}

func TestPostgresControlPlaneLeaseStandbyAndTakeover(t *testing.T) {
	ctx, stop := context.WithTimeout(t.Context(), 20*time.Second)
	defer stop()
	pool := isolatedAppPool(t, ctx)
	active, err := openControlPlaneLease(ctx, pool)
	if err != nil {
		t.Fatal(err)
	}
	defer active.Close()
	acquired, err := active.tryAcquire(ctx)
	if err != nil || !acquired {
		t.Fatalf("first Server did not acquire Control Plane lease: %v %v", acquired, err)
	}
	private, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer private.Close()
	if err := active.attachPrivate(private); err != nil {
		t.Fatal(err)
	}
	pid := active.conn.PgConn().PID()
	if active.holderPID(ctx) != int32(pid) {
		t.Fatalf("standby could not identify lease holder PID %d", pid)
	}
	activeCtx, cancelActive := context.WithCancelCause(ctx)
	watchDone := active.watch(activeCtx, cancelActive)
	defer func() {
		cancelActive(nil)
		<-watchDone
	}()

	probe, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	standbyAddress := probe.Addr().String()
	_ = probe.Close()
	standbyCtx, cancelStandby := context.WithCancel(ctx)
	defer cancelStandby()
	type result struct {
		lease *controlPlaneLease
		err   error
	}
	standbyDone := make(chan result, 1)
	standbyReceived := false
	logger := slog.New(slog.NewTextHandler(io.Discard, nil))
	go func() {
		lease, err := awaitControlPlaneLease(standbyCtx, pool, standbyAddress, time.Second, logger)
		standbyDone <- result{lease, err}
	}()
	t.Cleanup(func() {
		cancelStandby()
		if standbyReceived {
			return
		}
		select {
		case completed := <-standbyDone:
			if completed.lease != nil {
				completed.lease.Close()
			}
		case <-time.After(2 * time.Second):
			t.Error("standby did not stop during test cleanup")
		}
	})

	client := &http.Client{Timeout: 200 * time.Millisecond}
	var ready *http.Response
	for ctx.Err() == nil {
		ready, err = client.Get("http://" + standbyAddress + "/readyz")
		if err == nil {
			break
		}
		time.Sleep(20 * time.Millisecond)
	}
	if err != nil {
		t.Fatalf("standby readiness listener did not start: %v", err)
	}
	_ = ready.Body.Close()
	if ready.StatusCode != http.StatusServiceUnavailable {
		t.Fatalf("standby readiness = %d, want 503", ready.StatusCode)
	}
	health, err := client.Get("http://" + standbyAddress + "/healthz")
	if err != nil {
		t.Fatal(err)
	}
	_ = health.Body.Close()
	if health.StatusCode != http.StatusOK {
		t.Fatalf("standby health = %d, want 200", health.StatusCode)
	}
	assertStandbyPublicAPI(t, client, "http://"+standbyAddress)
	select {
	case completed := <-standbyDone:
		standbyReceived = true
		if completed.lease != nil {
			completed.lease.Close()
		}
		t.Fatalf("standby started before active holder stopped: %v", completed.err)
	default:
	}

	if _, err := pool.Exec(ctx, `SELECT pg_terminate_backend($1)`, pid); err != nil {
		t.Fatal(err)
	}
	select {
	case <-activeCtx.Done():
		if context.Cause(activeCtx) == nil || context.Cause(activeCtx) == context.Canceled {
			t.Fatalf("active Server lost lease without failure cause: %v", context.Cause(activeCtx))
		}
	case <-ctx.Done():
		t.Fatal("active Server did not detect lease loss", ctx.Err())
	}
	if connection, err := net.DialTimeout("tcp", private.Addr().String(), 200*time.Millisecond); err == nil {
		_ = connection.Close()
		t.Fatal("active private listener stayed open after lease loss")
	}
	select {
	case completed := <-standbyDone:
		standbyReceived = true
		if completed.err != nil || completed.lease == nil {
			t.Fatalf("standby did not take over: %v", completed.err)
		}
		defer completed.lease.Close()
		rebound, err := net.Listen("tcp", standbyAddress)
		if err != nil {
			t.Fatalf("standby readiness listener remained bound after takeover: %v", err)
		}
		_ = rebound.Close()
	case <-ctx.Done():
		t.Fatal("standby did not acquire released lease", ctx.Err())
	}
}

// assertStandbyPublicAPI checks that a standby answers a /v1 request and the
// event upgrade with the retryable server_standby error envelope (PER-R3),
// never the plain-text 404 that a client reads as a missing resource.
func assertStandbyPublicAPI(t *testing.T, client *http.Client, baseURL string) {
	t.Helper()
	runs, err := http.NewRequest(http.MethodGet, baseURL+"/v1/runs", nil)
	if err != nil {
		t.Fatal(err)
	}
	runs.Header.Set("Authorization", "Bearer standby-test")
	upgrade, err := http.NewRequest(http.MethodGet, baseURL+"/v1/events/ws", nil)
	if err != nil {
		t.Fatal(err)
	}
	for name, value := range map[string]string{
		"Connection": "Upgrade", "Upgrade": "websocket", "Sec-WebSocket-Version": "13",
		"Sec-WebSocket-Key": "dGhlIHNhbXBsZSBub25jZQ==", "Sec-WebSocket-Protocol": "contractor.events.v1",
	} {
		upgrade.Header.Set(name, value)
	}
	for _, request := range []*http.Request{runs, upgrade} {
		response, err := client.Do(request)
		if err != nil {
			t.Fatalf("standby %s: %v", request.URL.Path, err)
		}
		body, err := io.ReadAll(response.Body)
		_ = response.Body.Close()
		if err != nil {
			t.Fatal(err)
		}
		var envelope struct {
			Code      string `json:"code"`
			Message   string `json:"message"`
			Retryable bool   `json:"retryable"`
			RequestID string `json:"requestId"`
		}
		decoder := json.NewDecoder(bytes.NewReader(body))
		decoder.DisallowUnknownFields()
		if err := decoder.Decode(&envelope); err != nil {
			t.Fatalf("standby %s body %q: %v", request.URL.Path, body, err)
		}
		requestID := response.Header.Get(requestid.Header)
		if response.StatusCode != http.StatusServiceUnavailable ||
			response.Header.Get("Content-Type") != "application/json" ||
			response.Header.Get("Cache-Control") != "no-store" ||
			response.Header.Get(publicapi.APIVersionHeader) != publicapi.APIVersion ||
			envelope.Code != "server_standby" || !envelope.Retryable || envelope.Message == "" ||
			!requestid.Valid(requestID) || envelope.RequestID != requestID {
			t.Fatalf("standby %s = %d headers=%v body=%s", request.URL.Path, response.StatusCode, response.Header, body)
		}
	}
}
