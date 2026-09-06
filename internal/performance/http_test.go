package performance

import (
	"bufio"
	"context"
	"errors"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/gorilla/websocket"
)

func TestHTTPAccountingAndExclusions(t *testing.T) {
	clock := newFakeClock()
	recorder := newHTTPRecorder(clock.Now)
	cases := []struct {
		name, method, path string
		handler            http.HandlerFunc
		class              int
		cancel             bool
	}{
		{"implicit", "GET", "/v1/runs", func(http.ResponseWriter, *http.Request) {}, 1, false},
		{"write", "POST", "/v1/runs?secret=query", func(w http.ResponseWriter, _ *http.Request) { _, _ = w.Write([]byte("ok")) }, 1, false},
		{"client-error", "GET", "/v1/runs", func(w http.ResponseWriter, _ *http.Request) { w.WriteHeader(404) }, 3, false},
		{"server-error", "PATCH", "/v1/runs", func(w http.ResponseWriter, _ *http.Request) { w.WriteHeader(503) }, 4, false},
		{"early-hints", "GET", "/v1/runs", func(w http.ResponseWriter, _ *http.Request) {
			w.WriteHeader(103)
			w.WriteHeader(201)
			w.WriteHeader(500)
		}, 1, false},
		{"panic", "GET", "/v1/runs", func(http.ResponseWriter, *http.Request) { panic("secret-canary") }, 5, false},
		{"cancelled", "GET", "/v1/runs", func(http.ResponseWriter, *http.Request) {}, 5, true},
		{"arbitrary-method", "SECRET-METHOD", "/private/random/path", func(w http.ResponseWriter, _ *http.Request) { w.WriteHeader(204) }, 1, false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			h := recorder.Wrap(Private, tc.handler)
			r := httptest.NewRequest(tc.method, tc.path, nil)
			if tc.cancel {
				ctx, cancel := context.WithCancel(r.Context())
				cancel()
				r = r.WithContext(ctx)
			}
			func() {
				defer func() {
					if v := recover(); v != nil && v != "secret-canary" {
						panic(v)
					}
				}()
				h.ServeHTTP(httptest.NewRecorder(), r)
			}()
			v := recorder.drain()[1]
			if v.Counts[methodIndex(tc.method)][tc.class] != 1 || v.Duration.Count != 1 || v.InFlight != 0 {
				t.Fatalf("bad accounting: %+v", v)
			}
		})
	}
	for _, path := range []string{"/healthz", "/readyz", "/v1/operations/performance?x=1", "/v1/operations/performance/history", "/v1/operations/allocation-history", "/debug/pprof/profile"} {
		recorder.Wrap(Public, http.HandlerFunc(func(http.ResponseWriter, *http.Request) {})).ServeHTTP(httptest.NewRecorder(), httptest.NewRequest("GET", path, nil))
	}
	if recorder.drain()[0].Duration.Count != 0 {
		t.Fatal("observation traffic was counted")
	}
}

func TestHTTPConcurrentDrainRetainsInFlightExactlyOnce(t *testing.T) {
	recorder := newHTTPRecorder(time.Now)
	release := make(chan struct{})
	entered := make(chan struct{}, 64)
	h := recorder.Wrap(Public, http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) { entered <- struct{}{}; <-release; w.WriteHeader(200) }))
	var wg sync.WaitGroup
	for range 64 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			h.ServeHTTP(httptest.NewRecorder(), httptest.NewRequest("GET", "/v1/arbitrary", nil))
		}()
	}
	for range 64 {
		<-entered
	}
	for range 10 {
		v := recorder.drain()[0]
		if v.InFlight != 64 || v.Duration.Count != 0 {
			t.Fatal("drain changed active requests")
		}
	}
	close(release)
	wg.Wait()
	v := recorder.drain()[0]
	if v.InFlight != 0 || v.Duration.Count != 64 || v.Counts[0][1] != 64 {
		t.Fatal("completion not accounted exactly once")
	}
}

func TestHTTPPreservesInterfacesStreamingAndReadFrom(t *testing.T) {
	recorder := newHTTPRecorder(time.Now)
	plain := &discardWriter{header: http.Header{}}
	recorder.Wrap(Public, http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		if _, ok := w.(http.Hijacker); ok {
			t.Fatal("invented Hijacker")
		}
		if _, ok := w.(http.Flusher); ok {
			t.Fatal("invented Flusher")
		}
	})).ServeHTTP(plain, httptest.NewRequest("GET", "/v1/test", nil))
	w := &interfaceWriter{ResponseRecorder: httptest.NewRecorder()}
	recorder.Wrap(Public, http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		if _, ok := w.(http.Hijacker); !ok {
			t.Fatal("lost Hijacker")
		}
		if _, ok := w.(io.ReaderFrom); !ok {
			t.Fatal("lost ReaderFrom")
		}
		if err := http.NewResponseController(w).Flush(); err != nil {
			t.Fatal(err)
		}
		_, _ = w.(io.ReaderFrom).ReadFrom(strings.NewReader("stream"))
	})).ServeHTTP(w, httptest.NewRequest("GET", "/v1/test", nil))
	if !w.Flushed || w.Body.String() != "stream" {
		t.Fatal("streaming behavior changed")
	}
	if recorder.drain()[0].Duration.Count != 2 {
		t.Fatal("streaming double-counted")
	}
}

type interfaceWriter struct{ *httptest.ResponseRecorder }

func (w *interfaceWriter) ReadFrom(r io.Reader) (int64, error) { return io.Copy(w.ResponseRecorder, r) }
func (w *interfaceWriter) Hijack() (net.Conn, *bufio.ReadWriter, error) {
	return nil, nil, errors.New("unsupported in fixture")
}

func TestHTTPWebSocketFinishesAtHandshakeNotSocketClose(t *testing.T) {
	recorder := newHTTPRecorder(time.Now)
	upgraded := make(chan struct{})
	release := make(chan struct{})
	h := recorder.Wrap(Public, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		upgrader := websocket.Upgrader{}
		conn, err := upgrader.Upgrade(w, r, nil)
		if err != nil {
			return
		}
		defer conn.Close()
		close(upgraded)
		<-release
	}))
	server := httptest.NewServer(h)
	defer server.Close()
	conn, response, err := websocket.DefaultDialer.Dial("ws"+strings.TrimPrefix(server.URL, "http"), nil)
	if response != nil && response.Body != nil {
		defer response.Body.Close()
	}
	if err != nil {
		close(release)
		t.Fatal(err)
	}
	defer conn.Close()
	<-upgraded
	v := recorder.drain()[0]
	if v.InFlight != 0 || v.Counts[0][0] != 1 || v.Duration.Count != 1 {
		close(release)
		t.Fatalf("WebSocket counted lifetime: %+v", v)
	}
	close(release)
	if recorder.drain()[0].Duration.Count != 0 {
		t.Fatal("WebSocket counted twice")
	}
}

func TestHTTPFailedWriteAndHandshakeAreNoResponse(t *testing.T) {
	recorder := newHTTPRecorder(time.Now)
	recorder.Wrap(Public, http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) { _, _ = w.Write([]byte("body")) })).ServeHTTP(&failedWriter{header: http.Header{}}, httptest.NewRequest("GET", "/v1/test", nil))
	if recorder.drain()[0].Counts[0][5] != 1 {
		t.Fatal("failed write invented success")
	}
	left, right := net.Pipe()
	_ = right.Close()
	status := 200
	conn := handshakeConn{Conn: left, finish: func(code int) { status = code }}
	_, err := conn.Write([]byte("HTTP/1.1 101 Switching Protocols\r\n\r\n"))
	_ = left.Close()
	if err == nil || status != 0 {
		t.Fatal("failed handshake invented success")
	}
}

type failedWriter struct{ header http.Header }

func (w *failedWriter) Header() http.Header       { return w.header }
func (w *failedWriter) WriteHeader(int)           {}
func (w *failedWriter) Write([]byte) (int, error) { return 0, io.ErrClosedPipe }
