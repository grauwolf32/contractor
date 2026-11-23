package artifacttransfer

import (
	"net/http"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
)

const (
	// MinimumThroughput is the documented floor for a full-size transfer.
	MinimumThroughput = 1 << 20 // 1 MiB/s
	TransferGrace     = 5 * time.Second
	// StorageBudget bounds blob preparation and registry work for one transfer,
	// independent of payload size and client throughput. It exceeds the default
	// database budgets (2 s acquire, 20 s per query), which still bound each
	// operation inside it.
	StorageBudget = time.Minute
)

// Duration is the socket budget for size bytes at MinimumThroughput. A missing
// or unknown length uses the 64 MiB maximum.
func Duration(size int64) time.Duration {
	if size < 0 || size > artifacts.MaxPayloadSize {
		size = artifacts.MaxPayloadSize
	}
	seconds := (size + MinimumThroughput - 1) / MinimumThroughput
	return TransferGrace + time.Duration(seconds)*time.Second
}

// BoundWrite gives the next response write of size bytes its own socket
// deadline, measured from now. Zero suits a small metadata or error response.
func BoundWrite(w http.ResponseWriter, size int64) {
	_ = http.NewResponseController(w).SetWriteDeadline(time.Now().Add(Duration(size)))
}
