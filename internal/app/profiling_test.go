package app

import (
	"context"
	"io"
	"log/slog"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

func TestApplicationListenersDoNotExposeProfilingRoutes(t *testing.T) {
	handlers := []http.Handler{
		newProcessHandler(http.HandlerFunc(writeHealthy)),
		newPrivateHandler(http.HandlerFunc(writeHealthy), http.HandlerFunc(writeHealthy)),
	}
	for index, handler := range handlers {
		for _, path := range []string{"/debug/pprof", "/debug/pprof/", "/debug/pprof/cmdline"} {
			response := httptest.NewRecorder()
			handler.ServeHTTP(response, httptest.NewRequest(http.MethodGet, path, nil))
			if response.Code != http.StatusNotFound {
				t.Fatalf("application handler %d exposed %s with status %d", index, path, response.Code)
			}
		}
	}
}

func TestConfigureProfilingDoesNotBindWhenDisabled(t *testing.T) {
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer listener.Close()
	server, err := configureProfiling(Config{
		Pprof: false, PprofListen: listener.Addr().String(), ShutdownTimeout: time.Second,
	})
	if err != nil || server != nil {
		t.Fatalf("disabled profiling = (%v, %v)", server, err)
	}
}

func TestRunCLIFailsSafelyWhenEnabledProfilingPortIsOccupied(t *testing.T) {
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer listener.Close()
	address := listener.Addr().String()
	err = RunCLI(
		context.Background(),
		[]string{"--pprof=true", "--pprof-listen=" + address},
		func(string) string { return "" },
		slog.New(slog.NewTextHandler(io.Discard, nil)),
	)
	if err == nil || !strings.Contains(err.Error(), "configure Go profiling") || strings.Contains(err.Error(), address) {
		t.Fatalf("unsafe or missing occupied-listener error: %v", err)
	}
}

func TestRunCLICleansUpEarlyProfilingListenerAfterLaterStartupFailure(t *testing.T) {
	probe, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	address := probe.Addr().String()
	if err := probe.Close(); err != nil {
		t.Fatal(err)
	}
	err = RunCLI(
		context.Background(),
		[]string{"--pprof=true", "--pprof-listen=" + address},
		func(string) string { return "" },
		slog.New(slog.NewTextHandler(io.Discard, nil)),
	)
	if err == nil || err.Error() != "database URL is required" {
		t.Fatalf("later startup error = %v", err)
	}
	reopened, err := net.Listen("tcp", address)
	if err != nil {
		t.Fatalf("profiling listener leaked after startup failure: %v", err)
	}
	reopened.Close()
}
