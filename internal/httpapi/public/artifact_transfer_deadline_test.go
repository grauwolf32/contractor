package public

import (
	"archive/zip"
	"bufio"
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/httpapi/artifacttransfer"
)

func TestPublicArtifactStalledUploadsReleaseTransferSlots(t *testing.T) {
	server, runtime := publicTransferServer(t)
	connections := make([]net.Conn, 4)
	for index := range connections {
		conn, err := net.Dial("tcp", server.Listener.Addr().String())
		if err != nil {
			t.Fatal(err)
		}
		connections[index] = conn
		t.Cleanup(func() { _ = conn.Close() })
		_, err = fmt.Fprintf(conn, "PUT /v1/artifacts/projects/stalled_%d HTTP/1.1\r\nHost: %s\r\nAuthorization: Bearer %s\r\nContent-Type: text/plain\r\nIf-None-Match: *\r\nContent-Length: 1\r\nConnection: close\r\n\r\n", index, server.Listener.Addr(), testBearerToken)
		if err != nil {
			t.Fatal(err)
		}
	}
	waitForPublicTransferSaturation(t, runtime)
	client := &http.Client{Timeout: 2 * time.Second}
	requestStatus := func() int {
		t.Helper()
		request, err := http.NewRequest(http.MethodPut, server.URL+"/v1/artifacts/projects/after_timeout", bytes.NewReader([]byte("ok")))
		if err != nil {
			t.Fatal(err)
		}
		request.Header.Set("Authorization", "Bearer "+testBearerToken)
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
		t.Fatalf("saturated upload = %d, want 503", got)
	}
	waitForPublicTransferAvailability(t, runtime, artifacttransfer.Duration(1)+3*time.Second)
	if got := requestStatus(); got != http.StatusCreated {
		t.Fatalf("upload after transfer deadline = %d, want 201", got)
	}
}

func TestPublicArtifactStalledDownloadsReleaseTransferSlots(t *testing.T) {
	fixture := newHandlerFixture(t)
	const payloadSize = 8 << 20
	store, err := fixture.artifacts.User("user-1")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Write(t.Context(), contracts.ArtifactRef{Namespace: "projects", Name: "large"},
		artifacts.Payload{MediaType: "application/octet-stream", Data: bytes.Repeat([]byte("x"), payloadSize)}, nil); err != nil {
		t.Fatal(err)
	}
	server, runtime := publicTransferServerForHandler(t, fixture.handler)
	for range 4 {
		conn, err := net.Dial("tcp", server.Listener.Addr().String())
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { _ = conn.Close() })
		if tcp, ok := conn.(*net.TCPConn); ok {
			_ = tcp.SetReadBuffer(1024)
		}
		_, err = fmt.Fprintf(conn, "GET /v1/artifacts/projects/large HTTP/1.1\r\nHost: %s\r\nAuthorization: Bearer %s\r\nConnection: close\r\n\r\n", server.Listener.Addr(), testBearerToken)
		if err != nil {
			t.Fatal(err)
		}
		_ = conn.SetReadDeadline(time.Now().Add(2 * time.Second))
		response, err := http.ReadResponse(bufio.NewReader(conn), nil)
		if err != nil || response.StatusCode != http.StatusOK {
			t.Fatalf("stalled download response = %+v (%v)", response, err)
		}
		_ = conn.SetReadDeadline(time.Time{})
	}
	waitForPublicTransferSaturation(t, runtime)
	client := &http.Client{Timeout: 3 * time.Second}
	requestStatus := func() int {
		t.Helper()
		request, err := http.NewRequest(http.MethodGet, server.URL+"/v1/artifacts/projects/large", nil)
		if err != nil {
			t.Fatal(err)
		}
		request.Header.Set("Authorization", "Bearer "+testBearerToken)
		response, err := client.Do(request)
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
		t.Fatalf("saturated download = %d, want 503", got)
	}
	waitForPublicTransferAvailability(t, runtime, artifacttransfer.Duration(payloadSize)+3*time.Second)
	if got := requestStatus(); got != http.StatusOK {
		t.Fatalf("download after transfer deadline = %d, want 200", got)
	}
}

// An archive listing reaches several MiB after JSON escaping. Clients that
// stop reading it must not keep transfer slots: the listing is built while the
// payload is held, and the slot is released before the response is written.
func TestPublicArchivePreviewStalledReadersLeaveTransferCapacity(t *testing.T) {
	fixture := newHandlerFixture(t)
	store, err := fixture.artifacts.User("user-1")
	if err != nil {
		t.Fatal(err)
	}
	written, err := store.Write(t.Context(), contracts.ArtifactRef{Namespace: "files", Name: "listing"},
		artifacts.Payload{MediaType: "application/zip", Data: largeListingArchive(t)}, nil)
	if err != nil {
		t.Fatal(err)
	}
	server, runtime := publicTransferServerForHandler(t, fixture.handler)
	target := "/v1/artifacts/files/listing/archive?revision=" + url.QueryEscape(*written.Ref.Revision)
	for range 4 {
		conn, err := net.Dial("tcp", server.Listener.Addr().String())
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { _ = conn.Close() })
		if tcp, ok := conn.(*net.TCPConn); ok {
			_ = tcp.SetReadBuffer(1024)
		}
		_, err = fmt.Fprintf(conn, "GET %s HTTP/1.1\r\nHost: %s\r\nAuthorization: Bearer %s\r\nConnection: close\r\n\r\n", target, server.Listener.Addr(), testBearerToken)
		if err != nil {
			t.Fatal(err)
		}
		_ = conn.SetReadDeadline(time.Now().Add(5 * time.Second))
		response, err := http.ReadResponse(bufio.NewReader(conn), nil)
		if err != nil || response.StatusCode != http.StatusOK {
			t.Fatalf("stalled archive preview response = %+v (%v)", response, err)
		}
		_ = conn.SetReadDeadline(time.Time{})
	}
	// Every listing is now being written to a reader that stopped reading.
	ctx := artifacts.WithBlobRuntime(t.Context(), runtime)
	releases := make([]func(), 0, 4)
	for range 4 {
		_, release, err := artifacts.AcquireTransfer(ctx)
		if err != nil {
			t.Fatalf("stalled archive previews hold transfer capacity: %v", err)
		}
		releases = append(releases, release)
	}
	for _, release := range releases {
		release()
	}
	request, err := http.NewRequest(http.MethodPut, server.URL+"/v1/artifacts/projects/during_stall", bytes.NewReader([]byte("ok")))
	if err != nil {
		t.Fatal(err)
	}
	request.Header.Set("Authorization", "Bearer "+testBearerToken)
	request.Header.Set("Content-Type", "text/plain")
	request.Header.Set("If-None-Match", "*")
	response, err := (&http.Client{Timeout: 5 * time.Second}).Do(request)
	if err != nil {
		t.Fatal(err)
	}
	_, _ = io.Copy(io.Discard, response.Body)
	_ = response.Body.Close()
	if response.StatusCode != http.StatusCreated {
		t.Fatalf("upload beside stalled archive previews = %d, want 201", response.StatusCode)
	}
}

// largeListingArchive holds 2,000 empty files whose 1,004-byte names consist
// mostly of '<', which JSON escapes to six bytes: a ~4 MiB ZIP with a ~12 MiB
// listing that cannot fit any socket buffer.
func largeListingArchive(t *testing.T) []byte {
	t.Helper()
	var data bytes.Buffer
	writer := zip.NewWriter(&data)
	prefix := strings.Repeat("<", 1000)
	for index := range 2000 {
		if _, err := writer.CreateHeader(&zip.FileHeader{Name: fmt.Sprintf("%s%04d", prefix, index), Method: zip.Store}); err != nil {
			t.Fatal(err)
		}
	}
	if err := writer.Close(); err != nil {
		t.Fatal(err)
	}
	return data.Bytes()
}

func publicTransferServer(t *testing.T) (*httptest.Server, *artifacts.BlobRuntime) {
	t.Helper()
	return publicTransferServerForHandler(t, newHandlerFixture(t).handler)
}

func publicTransferServerForHandler(t *testing.T, handler http.Handler) (*httptest.Server, *artifacts.BlobRuntime) {
	t.Helper()
	runtime := artifacts.NewBlobRuntime(nil, nil)
	server := httptest.NewUnstartedServer(handler)
	server.Config.BaseContext = func(net.Listener) context.Context {
		return artifacts.WithBlobRuntime(context.Background(), runtime)
	}
	server.Start()
	t.Cleanup(server.Close)
	return server, runtime
}

func waitForPublicTransferSaturation(t *testing.T, runtime *artifacts.BlobRuntime) {
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
			t.Fatal("four stalled requests did not occupy transfer slots")
		}
		time.Sleep(10 * time.Millisecond)
	}
}

func waitForPublicTransferAvailability(t *testing.T, runtime *artifacts.BlobRuntime, timeout time.Duration) {
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
			t.Fatal("artifact transfer slot stayed occupied after deadline")
		}
		time.Sleep(20 * time.Millisecond)
	}
}
