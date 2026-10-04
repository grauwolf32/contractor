// Package artifacttransfer admits full-payload Artifact HTTP exchanges into the
// Server process's four transfer slots and bounds the client socket I/O that
// happens while a slot is held.
package artifacttransfer

import (
	"context"
	"net/http"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
)

// Transfer is one admitted exchange. A context deadline alone does not
// interrupt a blocked request read or response write, so socket deadlines
// bound client I/O while the slot is held, and a response that no longer needs
// the payload is written only after ReleaseBeforeWrite.
type Transfer struct {
	controller *http.ResponseController
	cancel     context.CancelFunc
	release    func()
}

// Acquire takes one transfer slot without waiting. Call it after method, route
// and ownership checks, so rejected requests never consume capacity, and
// before buffering a request body or reading a payload. The returned request
// carries the slot lease, so nested Artifact operations share it, and a total
// transfer deadline starting now. A missing or unknown body length uses the
// 64 MiB maximum; known small bodies get a much shorter deadline.
func Acquire(w http.ResponseWriter, r *http.Request, size int64) (*http.Request, *Transfer, error) {
	ctx, release, err := artifacts.AcquireTransfer(r.Context())
	if err != nil {
		return r, nil, err
	}
	deadline := time.Now().Add(Duration(size))
	ctx, cancel := context.WithDeadline(ctx, deadline)
	controller := http.NewResponseController(w)
	_ = controller.SetReadDeadline(deadline)
	_ = controller.SetWriteDeadline(deadline)
	return r.WithContext(ctx), &Transfer{controller: controller, cancel: cancel, release: release}, nil
}

// BoundWrite gives the payload response of size bytes, written while the slot
// stays held, its own socket deadline measured from the start of the write.
func (t *Transfer) BoundWrite(size int64) {
	_ = t.controller.SetWriteDeadline(time.Now().Add(Duration(size)))
}

// ReleaseBeforeWrite frees the slot once the payload is no longer referenced
// and bounds the response of size bytes that follows, so a client that stops
// reading cannot hold transfer capacity. It is idempotent.
func (t *Transfer) ReleaseBeforeWrite(size int64) {
	t.release()
	t.BoundWrite(size)
}

// Close releases a slot that is still held. Socket deadlines stay armed: the
// response may still be buffered, and net/http flushes it after the handler
// returns, then resets the deadlines before reusing the connection.
func (t *Transfer) Close() {
	t.release()
	t.cancel()
}
