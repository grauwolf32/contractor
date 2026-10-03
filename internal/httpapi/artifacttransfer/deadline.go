// Package artifacttransfer bounds client socket I/O while an Artifact handler
// holds one of the Server process's four full-payload transfer slots.
package artifacttransfer

import (
	"context"
	"net/http"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
)

const (
	// MinimumThroughput is the documented floor for a full-size transfer.
	MinimumThroughput = 1 << 20 // 1 MiB/s
	TransferGrace     = 5 * time.Second
)

type Deadline struct {
	controller *http.ResponseController
	started    time.Time
	cancel     context.CancelFunc
}

// Bound starts a total transfer deadline at slot acquisition. A missing or
// unknown body length uses the 64 MiB maximum; known small bodies get a much
// shorter deadline. Socket deadlines are needed because a Context deadline
// alone does not interrupt a blocked request read or response write.
func Bound(w http.ResponseWriter, r *http.Request, size int64) (*http.Request, *Deadline) {
	started := time.Now()
	deadline := started.Add(Duration(size))
	ctx, cancel := context.WithDeadline(r.Context(), deadline)
	controller := http.NewResponseController(w)
	_ = controller.SetReadDeadline(deadline)
	_ = controller.SetWriteDeadline(deadline)
	return r.WithContext(ctx), &Deadline{controller: controller, started: started, cancel: cancel}
}

// LimitWrite tightens a GET deadline once the exact payload size is known.
func (d *Deadline) LimitWrite(size int64) {
	_ = d.controller.SetWriteDeadline(d.started.Add(Duration(size)))
}

func (d *Deadline) Close() {
	_ = d.controller.SetReadDeadline(time.Time{})
	_ = d.controller.SetWriteDeadline(time.Time{})
	d.cancel()
}

func Duration(size int64) time.Duration {
	if size < 0 || size > artifacts.MaxPayloadSize {
		size = artifacts.MaxPayloadSize
	}
	seconds := (size + MinimumThroughput - 1) / MinimumThroughput
	return TransferGrace + time.Duration(seconds)*time.Second
}
