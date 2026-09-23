package publicclient

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"sync"
	"time"
)

// progressTransport bounds an exchange by inactivity rather than by total
// duration. One watchdog covers connection setup, the request body, the wait
// for response headers and the response body; every read of either body
// restarts it. A stalled exchange fails with an error matching
// context.DeadlineExceeded, like a whole-request timeout.
type progressTransport struct {
	base    http.RoundTripper
	timeout time.Duration
}

func (t *progressTransport) RoundTrip(request *http.Request) (*http.Response, error) {
	ctx, cancel := context.WithCancel(request.Context())
	watchdog := &stallWatchdog{cancel: cancel, timeout: t.timeout}
	watchdog.timer = time.AfterFunc(t.timeout, watchdog.fire)
	request = request.WithContext(ctx)
	if request.Body != nil && request.Body != http.NoBody {
		request.Body = &progressBody{ReadCloser: request.Body, watchdog: watchdog}
		if getBody := request.GetBody; getBody != nil {
			request.GetBody = func() (io.ReadCloser, error) {
				body, err := getBody()
				if err != nil {
					return nil, err
				}
				return &progressBody{ReadCloser: body, watchdog: watchdog}, nil
			}
		}
	}
	response, err := t.base.RoundTrip(request)
	if err != nil {
		watchdog.finish()
		return nil, watchdog.explain(err)
	}
	watchdog.progress()
	response.Body = &progressBody{ReadCloser: response.Body, watchdog: watchdog, response: true}
	return response, nil
}

type stallWatchdog struct {
	mu      sync.Mutex
	timer   *time.Timer
	cancel  context.CancelFunc
	timeout time.Duration
	stalled bool
	done    bool
}

func (w *stallWatchdog) fire() {
	w.mu.Lock()
	defer w.mu.Unlock()
	if !w.done {
		w.stalled = true
		w.cancel()
	}
}

func (w *stallWatchdog) progress() {
	w.mu.Lock()
	defer w.mu.Unlock()
	if !w.done && !w.stalled {
		w.timer.Reset(w.timeout)
	}
}

func (w *stallWatchdog) finish() {
	w.mu.Lock()
	defer w.mu.Unlock()
	w.done = true
	w.timer.Stop()
	w.cancel()
}

func (w *stallWatchdog) explain(err error) error {
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.stalled {
		return fmt.Errorf("public API transfer made no progress for %s: %w", w.timeout, context.DeadlineExceeded)
	}
	return err
}

type progressBody struct {
	io.ReadCloser
	watchdog *stallWatchdog
	response bool
}

func (b *progressBody) Read(p []byte) (int, error) {
	n, err := b.ReadCloser.Read(p)
	if n > 0 {
		b.watchdog.progress()
	}
	if err != nil && err != io.EOF {
		err = b.watchdog.explain(err)
	}
	return n, err
}

func (b *progressBody) Close() error {
	err := b.ReadCloser.Close()
	if b.response {
		b.watchdog.finish()
	}
	return err
}
