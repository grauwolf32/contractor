package profiling

import (
	"context"
	"net"
	"net/http"
	"testing"
	"time"
)

func TestListenRejectsUnsafeAddressesAndOccupiedPort(t *testing.T) {
	for _, address := range []string{
		"localhost:6060", "0.0.0.0:6060", "[::]:6060", "192.0.2.1:6060",
		"127.0.0.1:0", "127.0.0.1:65536", "127.0.0.1:http", "127.0.0.1",
	} {
		if server, err := Listen(Options{ListenAddress: address, ShutdownTimeout: time.Second}); err == nil {
			_ = server.Close()
			t.Fatalf("accepted profiling address %q", address)
		}
	}
	occupied, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer occupied.Close()
	if server, err := Listen(Options{ListenAddress: occupied.Addr().String(), ShutdownTimeout: time.Second}); err == nil {
		_ = server.Close()
		t.Fatal("accepted occupied profiling listener")
	}
}

func TestServerServesLoopbackAndStopsWithItsContext(t *testing.T) {
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	server, err := NewWithListener(listener, time.Second)
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() { done <- server.Run(ctx) }()

	client := &http.Client{Timeout: time.Second}
	response, err := client.Get("http://" + server.Addr().String() + "/debug/pprof/")
	if err != nil {
		cancel()
		t.Fatal(err)
	}
	response.Body.Close()
	if response.StatusCode != http.StatusOK {
		cancel()
		t.Fatalf("profile index status = %d", response.StatusCode)
	}

	cancel()
	select {
	case err := <-done:
		if err != nil {
			t.Fatal(err)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("profiling server did not stop")
	}
}

func TestServerShutdownCancelsAnActiveCapture(t *testing.T) {
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	server, err := NewWithListener(listener, time.Second)
	if err != nil {
		t.Fatal(err)
	}
	handler := server.httpServer.Handler.(*Handler)
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() { done <- server.Run(ctx) }()
	requestDone := make(chan error, 1)
	go func() {
		response, requestErr := http.Get("http://" + server.Addr().String() + "/debug/pprof/profile?seconds=60")
		if response != nil {
			response.Body.Close()
		}
		requestDone <- requestErr
	}()
	waitForSlot(t, handler.timed)
	cancel()
	select {
	case err := <-done:
		if err != nil {
			t.Fatal(err)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("active profile blocked diagnostic shutdown")
	}
	waitForReleasedSlot(t, handler.timed)
	select {
	case <-requestDone:
	case <-time.After(2 * time.Second):
		t.Fatal("active profile client did not unblock")
	}
}

func TestNewWithListenerRejectsNonLoopbackSocket(t *testing.T) {
	listener, err := net.Listen("tcp", "0.0.0.0:0")
	if err != nil {
		t.Skipf("cannot create wildcard test listener: %v", err)
	}
	defer listener.Close()
	if server, err := NewWithListener(listener, time.Second); err == nil {
		_ = server.Close()
		t.Fatal("accepted wildcard diagnostic listener")
	}
}
