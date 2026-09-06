package performance

import (
	"bufio"
	"bytes"
	"io"
	"net"
	"net/http"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/felixge/httpsnoop"
)

// HTTPRecorder retains only two fixed counter matrices and histograms. Request
// paths, identities, headers and errors never enter retained metrics.
type HTTPRecorder struct {
	mu     sync.Mutex
	values [2]HTTPSurface
	now    func() time.Time
}

func newHTTPRecorder(now func() time.Time) *HTTPRecorder {
	return &HTTPRecorder{now: now, values: [2]HTTPSurface{{Surface: Public}, {Surface: Private}}}
}

func excludedRequest(r *http.Request) bool {
	switch r.URL.Path {
	case "/healthz", "/readyz", "/v1/operations/performance", "/v1/operations/performance/history", "/v1/operations/allocation-history":
		return true
	}
	return r.URL.Path == "/debug/pprof" || strings.HasPrefix(r.URL.Path, "/debug/pprof/")
}

func methodIndex(method string) int {
	for i, value := range HTTPMethods() {
		if method == value {
			return i
		}
	}
	return 9
}

func (h *HTTPRecorder) Wrap(surface Surface, next http.Handler) http.Handler {
	index := 0
	if surface == Private {
		index = 1
	} else if surface != Public {
		panic("invalid performance HTTP surface")
	}
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if excludedRequest(r) {
			next.ServeHTTP(w, r)
			return
		}
		method := methodIndex(r.Method)
		started := h.now()
		h.mu.Lock()
		h.values[index].InFlight++
		h.mu.Unlock()
		var once sync.Once
		finish := func(status int) {
			once.Do(func() {
				elapsed := h.now().Sub(started).Seconds()
				if elapsed < 0 {
					elapsed = 0
				}
				class := 5
				if status >= 100 && status < 600 {
					class = status/100 - 1
				}
				h.mu.Lock()
				defer h.mu.Unlock()
				v := &h.values[index]
				v.InFlight--
				v.Counts[method][class]++
				v.Duration.Observe(elapsed)
			})
		}
		status, returned, hijacked := 0, false, false
		defer func() {
			if !hijacked {
				if status == 0 && returned && r.Context().Err() == nil {
					status = http.StatusOK
				}
				finish(status)
			} else {
				finish(0)
			} // unfinished handshakes never remain in flight after handler return/panic
		}()
		wrapped := httpsnoop.Wrap(w, httpsnoop.Hooks{
			WriteHeader: func(next httpsnoop.WriteHeaderFunc) httpsnoop.WriteHeaderFunc {
				return func(code int) {
					next(code) // invalid statuses still panic exactly as the underlying writer
					if status == 0 && (code == 101 || code >= 200) {
						status = code
					}
					if code == 101 {
						finish(code)
					}
				}
			},
			Write: func(next httpsnoop.WriteFunc) httpsnoop.WriteFunc {
				return func(b []byte) (int, error) {
					n, err := next(b)
					if status == 0 {
						if err != nil && n == 0 {
							finish(0)
						} else {
							status = 200
						}
					}
					return n, err
				}
			},
			ReadFrom: func(next httpsnoop.ReadFromFunc) httpsnoop.ReadFromFunc {
				return func(src io.Reader) (int64, error) {
					n, err := next(src)
					if status == 0 {
						if err != nil && n == 0 {
							finish(0)
						} else {
							status = 200
						}
					}
					return n, err
				}
			},
			Flush: func(next httpsnoop.FlushFunc) httpsnoop.FlushFunc {
				return func() {
					next()
					if status == 0 {
						status = 200
					}
				}
			},
			Hijack: func(next httpsnoop.HijackFunc) httpsnoop.HijackFunc {
				return func() (net.Conn, *bufio.ReadWriter, error) {
					conn, rw, err := next()
					if err != nil {
						return conn, rw, err
					}
					hijacked = true
					if !strings.EqualFold(r.Header.Get("Upgrade"), "websocket") || status != 0 {
						finish(status)
						return conn, rw, nil
					}
					// gorilla writes 101 directly to the hijacked connection. Observe
					// completion of that bounded header, not the socket's lifetime.
					observed := &handshakeConn{Conn: conn, finish: finish}
					if rw.Writer.Buffered() != 0 {
						// Preserve an existing custom writer rather than replacing its
						// pending bytes. Standard net/http returns an empty writer.
						finish(0)
						return conn, rw, nil
					}
					return observed, bufio.NewReadWriter(rw.Reader, bufio.NewWriterSize(observed, rw.Writer.Size())), nil
				}
			},
		})
		next.ServeHTTP(wrapped, r)
		returned = true
	})
}

// Drain swaps completed interval counters, preserving outstanding requests.
func (h *HTTPRecorder) drain() [2]HTTPSurface {
	h.mu.Lock()
	defer h.mu.Unlock()
	values := h.values
	for i := range h.values {
		h.values[i] = HTTPSurface{Surface: values[i].Surface, InFlight: values[i].InFlight}
	}
	return values
}

type handshakeConn struct {
	net.Conn
	mu     sync.Mutex
	header [4096]byte
	used   int
	done   bool
	finish func(int)
}

func (c *handshakeConn) Write(p []byte) (int, error) {
	n, err := c.Conn.Write(p)
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.done {
		return n, err
	}
	if err != nil || n != len(p) {
		c.done = true
		c.finish(0)
		return n, err
	}
	c.used += copy(c.header[c.used:], p[:n])
	if end := bytes.Index(c.header[:c.used], []byte("\r\n\r\n")); end >= 0 {
		status := 0
		lineEnd := bytes.Index(c.header[:end+2], []byte("\r\n"))
		if lineEnd >= 0 {
			fields := bytes.Fields(c.header[:lineEnd])
			if len(fields) >= 2 && bytes.HasPrefix(fields[0], []byte("HTTP/1.")) {
				status, _ = strconv.Atoi(string(fields[1]))
			}
		}
		c.done = true
		c.finish(status)
	} else if c.used == len(c.header) {
		c.done = true
		c.finish(0)
	}
	return n, err
}

func (c *handshakeConn) Close() error {
	c.finish(0)
	return c.Conn.Close()
}
