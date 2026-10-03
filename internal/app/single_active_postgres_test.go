package app

import (
	"context"
	"io"
	"log/slog"
	"net"
	"net/http"
	"testing"
	"time"
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
