// Package artifacttransfer admits full-payload Artifact HTTP exchanges into the
// Server process's four transfer slots and bounds the client socket I/O that
// happens while a slot is held.
package artifacttransfer

import (
	"context"
	"errors"
	"io"
	"net/http"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
)

// Transfer is one admitted exchange. A context deadline alone does not
// interrupt a blocked request read or response write, so each client socket
// operation performed while the slot is held gets its own deadline. Storage
// work is bounded separately by StorageBudget, and a response that no longer
// needs the payload is written only after ReleaseBeforeWrite.
type Transfer struct {
	controller *http.ResponseController
	release    func()
}

// Acquire takes one transfer slot without waiting. Call it after method, route
// and ownership checks, so rejected requests never consume capacity, and
// before buffering a request body or reading a payload. The returned request
// carries the slot lease, so nested Artifact operations share it, and no
// deadline: client I/O deadlines must not cancel storage work.
func Acquire(w http.ResponseWriter, r *http.Request) (*http.Request, *Transfer, error) {
	ctx, release, err := artifacts.AcquireTransfer(r.Context())
	if err != nil {
		return r, nil, err
	}
	return r.WithContext(ctx), &Transfer{controller: http.NewResponseController(w), release: release}, nil
}

// ReadBody reads the complete request body of at most limit bytes. Its socket
// deadline is sized from the declared length, or the 64 MiB maximum when the
// length is unknown, and measured from the start of the read. A declared
// length is allocated once; an unknown one grows geometrically up to limit.
//
// Once the body is complete the read deadline is cleared. net/http reads the
// idle connection to notice a disconnect (for an empty body even before the
// handler runs), and a deadline expiring there would cancel the request
// context during storage work. After a failed read the deadline stays armed,
// so the server's drain of an unread body cannot block.
func (t *Transfer) ReadBody(r *http.Request, limit int64) ([]byte, error) {
	if r.ContentLength > limit {
		return nil, artifacts.ErrPayloadTooLarge
	}
	deadline := time.Now().Add(Duration(r.ContentLength))
	_ = t.controller.SetReadDeadline(deadline)
	// The first read writes an Expect: 100-continue interim response.
	_ = t.controller.SetWriteDeadline(deadline)
	data, err := readBounded(r.Body, r.ContentLength, limit)
	if err != nil {
		return nil, err
	}
	_ = t.controller.SetReadDeadline(time.Time{})
	return data, nil
}

func readBounded(body io.Reader, declared, limit int64) ([]byte, error) {
	if declared >= 0 {
		data := make([]byte, declared)
		if _, err := io.ReadFull(body, data); err != nil {
			return nil, err
		}
		return data, nil
	}
	data := make([]byte, 0, min(limit, 512))
	for {
		if len(data) == cap(data) {
			if int64(len(data)) == limit {
				return data, expectEOF(body)
			}
			grown := make([]byte, len(data), min(2*int64(cap(data)), limit))
			copy(grown, data)
			data = grown
		}
		n, err := body.Read(data[len(data):cap(data)])
		data = data[:len(data)+n]
		if errors.Is(err, io.EOF) {
			return data, nil
		}
		if err != nil {
			return nil, err
		}
	}
}

// expectEOF confirms that a body of unknown length ends at its limit.
func expectEOF(body io.Reader) error {
	var probe [1]byte
	for {
		n, err := body.Read(probe[:])
		if n > 0 {
			return artifacts.ErrPayloadTooLarge
		}
		if errors.Is(err, io.EOF) {
			return nil
		}
		if err != nil {
			return err
		}
	}
}

// StorageContext bounds blob preparation and registry publication or reads by
// StorageBudget, measured from the start of that work. Socket deadlines never
// cancel it, so a client whose I/O has finished gets the storage outcome.
func StorageContext(ctx context.Context) (context.Context, context.CancelFunc) {
	return context.WithTimeout(ctx, StorageBudget)
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
}
