package app

import (
	"context"
	"errors"
	"fmt"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/performance"
	"github.com/jackc/pgx/v5/pgxpool"
)

func TestPerformanceAndProfilingSwitchesRemainIndependent(t *testing.T) {
	for _, metricsEnabled := range []bool{false, true} {
		for _, profilingEnabled := range []bool{false, true} {
			name := fmt.Sprintf("metrics=%t/pprof=%t", metricsEnabled, profilingEnabled)
			t.Run(name, func(t *testing.T) {
				public := http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
					w.WriteHeader(http.StatusNoContent)
				})
				private := http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
					w.WriteHeader(http.StatusNoContent)
				})
				factoryCalls := 0
				wrappedPublic, wrappedPrivate, collector := instrumentPerformance(
					metricsEnabled,
					public,
					private,
					func() *performance.Collector {
						factoryCalls++
						return performance.New(performance.Options{})
					},
				)
				if factoryCalls != boolCount(metricsEnabled) || (collector != nil) != metricsEnabled {
					t.Fatalf("metrics wiring = calls:%d collector:%v", factoryCalls, collector != nil)
				}
				for _, handler := range []http.Handler{wrappedPublic, wrappedPrivate} {
					response := httptest.NewRecorder()
					handler.ServeHTTP(response, httptest.NewRequest(http.MethodGet, "/v1/test", nil))
					if response.Code != http.StatusNoContent {
						t.Fatalf("instrumented response status = %d", response.Code)
					}
				}

				address := unusedLoopbackAddress(t)
				profileServer, err := configureProfiling(Config{
					Pprof: profilingEnabled, PprofListen: address, ShutdownTimeout: time.Second,
				})
				if err != nil || (profileServer != nil) != profilingEnabled {
					t.Fatalf("profiling wiring = (%v, %v)", profileServer != nil, err)
				}
				if profileServer != nil {
					if err := profileServer.Close(); err != nil {
						t.Fatal(err)
					}
					listener, listenErr := net.Listen("tcp", address)
					if listenErr != nil {
						t.Fatalf("profiling listener was not released: %v", listenErr)
					}
					_ = listener.Close()
				} else {
					listener, listenErr := net.Listen("tcp", address)
					if listenErr != nil {
						t.Fatalf("disabled profiling unexpectedly bound listener: %v", listenErr)
					}
					_ = listener.Close()
				}
			})
		}
	}
}

func boolCount(value bool) int {
	if value {
		return 1
	}
	return 0
}

func unusedLoopbackAddress(t *testing.T) string {
	t.Helper()
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	address := listener.Addr().String()
	if err := listener.Close(); err != nil {
		t.Fatal(err)
	}
	return address
}

func TestDisabledPerformanceDoesNotConstructAnything(t *testing.T) {
	if newPerformanceDiagnostics(false, "not a database URL") != nil {
		t.Fatal("disabled mode constructed diagnostics")
	}
	// Enabled construction is lazy too: even malformed optional diagnostics
	// cannot prevent the application from starting.
	if newPerformanceDiagnostics(true, "not a database URL") == nil {
		t.Fatal("enabled diagnostics missing")
	}
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
