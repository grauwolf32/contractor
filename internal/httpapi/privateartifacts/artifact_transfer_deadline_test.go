package privateartifacts

import (
	"bufio"
	"bytes"
	"context"
	"crypto/tls"
	"crypto/x509"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/httpapi/artifacttransfer"
)

func TestPrivateArtifactStalledUploadsReleaseTransferSlots(t *testing.T) {
	server, runtime := privateTransferServer(t, newMemoryRepository())
	for index := range 4 {
		conn, err := net.Dial("tcp", server.Listener.Addr().String())
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { _ = conn.Close() })
		_, err = fmt.Fprintf(conn, "PUT /private/v1/allocations/allocation-1/artifacts/inputs/stalled_%d HTTP/1.1\r\nHost: %s\r\nContent-Type: text/plain\r\nIf-None-Match: *\r\nContent-Length: 1\r\nConnection: close\r\n\r\n", index, server.Listener.Addr())
		if err != nil {
			t.Fatal(err)
		}
	}
	waitForPrivateTransferSaturation(t, runtime)
	client := &http.Client{Timeout: 2 * time.Second}
	requestStatus := func() int {
		t.Helper()
		request, err := http.NewRequest(http.MethodPut, server.URL+"/private/v1/allocations/allocation-1/artifacts/inputs/after_timeout", bytes.NewReader([]byte("ok")))
		if err != nil {
			t.Fatal(err)
		}
		request.Header.Set("Content-Type", "text/plain")
		request.Header.Set("If-None-Match", "*")
		response, err := client.Do(request)
		if err != nil {
			t.Fatal(err)
		}
		defer response.Body.Close()
		_, _ = io.Copy(io.Discard, response.Body)
		return response.StatusCode
	}
	if got := requestStatus(); got != http.StatusServiceUnavailable {
		t.Fatalf("saturated private upload = %d, want 503", got)
	}
	waitForPrivateTransferAvailability(t, runtime, artifacttransfer.Duration(1)+3*time.Second)
	if got := requestStatus(); got != http.StatusCreated {
		t.Fatalf("private upload after transfer deadline = %d, want 201", got)
	}
}

func TestPrivateArtifactStalledDownloadsReleaseTransferSlots(t *testing.T) {
	repository := newMemoryRepository()
	const payloadSize = 8 << 20
	repository.seed("run-a", "inputs", "large", "revision-a", bytes.Repeat([]byte("x"), payloadSize))
	server, runtime := privateTransferServer(t, repository)
	for range 4 {
		conn, err := net.Dial("tcp", server.Listener.Addr().String())
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { _ = conn.Close() })
		if tcp, ok := conn.(*net.TCPConn); ok {
			_ = tcp.SetReadBuffer(1024)
		}
		_, err = fmt.Fprintf(conn, "GET /private/v1/allocations/allocation-1/artifacts/inputs/large HTTP/1.1\r\nHost: %s\r\nConnection: close\r\n\r\n", server.Listener.Addr())
		if err != nil {
			t.Fatal(err)
		}
		_ = conn.SetReadDeadline(time.Now().Add(2 * time.Second))
		response, err := http.ReadResponse(bufio.NewReader(conn), nil)
		if err != nil || response.StatusCode != http.StatusOK {
			t.Fatalf("stalled private download response = %+v (%v)", response, err)
		}
		_ = conn.SetReadDeadline(time.Time{})
	}
	waitForPrivateTransferSaturation(t, runtime)
	client := &http.Client{Timeout: 3 * time.Second}
	requestStatus := func() int {
		t.Helper()
		response, err := client.Get(server.URL + "/private/v1/allocations/allocation-1/artifacts/inputs/large")
		if err != nil {
			t.Fatal(err)
		}
		defer response.Body.Close()
		if response.StatusCode == http.StatusOK {
			if _, err := io.Copy(io.Discard, response.Body); err != nil {
				t.Fatal(err)
			}
		}
		return response.StatusCode
	}
	if got := requestStatus(); got != http.StatusServiceUnavailable {
		t.Fatalf("saturated private download = %d, want 503", got)
	}
	waitForPrivateTransferAvailability(t, runtime, artifacttransfer.Duration(payloadSize)+3*time.Second)
	if got := requestStatus(); got != http.StatusOK {
		t.Fatalf("private download after transfer deadline = %d, want 200", got)
	}
}

func privateTransferServer(t *testing.T, repository *memoryRepository) (*httptest.Server, *artifacts.BlobRuntime) {
	t.Helper()
	inner := newTestHandler(t, &fakeRegistry{grant: testGrant("run-a")}, repository)
	trusted := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		certificate := &x509.Certificate{RawSubjectPublicKeyInfo: []byte("artifact-test-key")}
		r.TLS = &tls.ConnectionState{
			PeerCertificates: []*x509.Certificate{certificate},
			VerifiedChains:   [][]*x509.Certificate{{certificate}},
		}
		r.Header.Set(RuntimeInstanceHeader, "runtime-1")
		inner.ServeHTTP(w, r)
	})
	runtime := artifacts.NewBlobRuntime(nil, nil)
	server := httptest.NewUnstartedServer(trusted)
	server.Config.BaseContext = func(net.Listener) context.Context {
		return artifacts.WithBlobRuntime(context.Background(), runtime)
	}
	server.Start()
	t.Cleanup(server.Close)
	return server, runtime
}

func waitForPrivateTransferSaturation(t *testing.T, runtime *artifacts.BlobRuntime) {
	t.Helper()
	ctx := artifacts.WithBlobRuntime(t.Context(), runtime)
	until := time.Now().Add(2 * time.Second)
	for {
		_, release, err := artifacts.AcquireTransfer(ctx)
		if errors.Is(err, artifacts.ErrTransferCapacity) {
			return
		}
		if err != nil {
			t.Fatal(err)
		}
		release()
		if time.Now().After(until) {
			t.Fatal("four stalled private requests did not occupy transfer slots")
		}
		time.Sleep(10 * time.Millisecond)
	}
}

func waitForPrivateTransferAvailability(t *testing.T, runtime *artifacts.BlobRuntime, timeout time.Duration) {
	t.Helper()
	ctx := artifacts.WithBlobRuntime(t.Context(), runtime)
	until := time.Now().Add(timeout)
	for {
		_, release, err := artifacts.AcquireTransfer(ctx)
		if err == nil {
			release()
			return
		}
		if !errors.Is(err, artifacts.ErrTransferCapacity) {
			t.Fatal(err)
		}
		if time.Now().After(until) {
			t.Fatal("private artifact transfer slot stayed occupied after deadline")
		}
		time.Sleep(20 * time.Millisecond)
	}
}
