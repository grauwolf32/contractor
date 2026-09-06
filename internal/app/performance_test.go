package app

import (
	"context"
	"errors"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/performance"
	"github.com/jackc/pgx/v5/pgxpool"
)

func TestDisabledPerformanceDoesNotConstructAnything(t *testing.T) {
	public, private := http.NewServeMux(), http.NewServeMux()
	p, q, c := instrumentPerformance(false, public, private, func() *performance.Collector { t.Fatal("disabled factory called"); return nil })
	if p != public || q != private || c != nil {
		t.Fatal("disabled mode changed handlers or allocated collector")
	}
}

func TestReadinessUsesWorkingDependencyIndependentlyOfMetrics(t *testing.T) {
	for _, enabled := range []bool{false, true} {
		checks := 0
		check := func(ctx context.Context) error {
			checks++
			deadline, ok := ctx.Deadline()
			if !ok || time.Until(deadline) > time.Second {
				t.Error("unbounded readiness")
			}
			return errors.New("secret database detail")
		}
		h, _, c := instrumentPerformance(enabled, NewReadyHandler(check, nil), http.NewServeMux(), func() *performance.Collector { return performance.New(performance.Options{}) })
		for _, path := range []string{"/healthz", "/readyz"} {
			w := httptest.NewRecorder()
			h.ServeHTTP(w, httptest.NewRequest("GET", path, nil))
			want := 200
			if path == "/readyz" {
				want = 503
			}
			if w.Code != want || strings.Contains(w.Body.String(), "secret") {
				t.Fatalf("%s: %d %s", path, w.Code, w.Body.String())
			}
		}
		if checks != 1 {
			t.Fatal("health called readiness dependency")
		}
		if c != nil {
			c.Collect()
			if c.Snapshot().Current.HTTP.Surfaces[0].Duration.Count != 0 {
				t.Fatal("health/readiness counted")
			}
		}
	}
}

func TestReadinessDeadlineAndEarlierCancellation(t *testing.T) {
	for _, duration := range []time.Duration{time.Second, 20 * time.Millisecond} {
		ctx, cancel := context.WithTimeout(context.Background(), duration)
		defer cancel()
		h := NewReadyHandler(func(ctx context.Context) error { <-ctx.Done(); return ctx.Err() }, nil)
		w := httptest.NewRecorder()
		start := time.Now()
		h.ServeHTTP(w, httptest.NewRequest("GET", "/readyz", nil).WithContext(ctx))
		if w.Code != 503 || time.Since(start) > duration+150*time.Millisecond {
			t.Fatal("readiness did not respect deadline")
		}
	}
	w := httptest.NewRecorder()
	NewReadyHandler(func(context.Context) error { return nil }, nil).ServeHTTP(w, httptest.NewRequest("GET", "/readyz", nil))
	if w.Code != 200 {
		t.Fatal("healthy DB rejected")
	}
}

func TestReadinessRealPoolStalledConnectionIsBounded(t *testing.T) {
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer listener.Close()
	accepted := make(chan net.Conn, 1)
	go func() {
		conn, err := listener.Accept()
		if err == nil {
			accepted <- conn
		}
	}()
	config, err := pgxpool.ParseConfig("postgres://test@" + listener.Addr().String() + "/test?sslmode=disable")
	if err != nil {
		t.Fatal(err)
	}
	config.MaxConns = 1
	config.MinConns = 0
	config.ConnConfig.ConnectTimeout = 2 * time.Second
	pool, err := pgxpool.NewWithConfig(context.Background(), config)
	if err != nil {
		t.Fatal(err)
	}
	defer pool.Close()
	defer func() {
		select {
		case conn := <-accepted:
			_ = conn.Close()
		case <-time.After(3 * time.Second):
			t.Error("pool did not connect to fixture")
		}
	}()
	h := NewReadyHandler(pool.Ping, nil)
	w := httptest.NewRecorder()
	start := time.Now()
	h.ServeHTTP(w, httptest.NewRequest("GET", "/readyz", nil))
	if w.Code != 503 || time.Since(start) > 1150*time.Millisecond {
		t.Fatal("stalled pgx connection exceeded readiness budget")
	}
}
