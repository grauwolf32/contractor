//go:build !race

package stateview

import (
	"runtime"
	"testing"
)

// The cost of one changed-revision read at the contract maximum, measured with
// go1.26.9 on darwin/arm64 (2026-10-30): 40,193 allocations and 10.4 MB here,
// and 40,203-40,228 allocations and 11.6-13.2 MB per call in
// BenchmarkWorkerStateChangedRevision, whose collections also empty the
// decoders' buffer pools. The bounds leave 19 percent headroom over the count
// and 27 percent over the benchmark's bytes for toolchain and library changes.
// A deep copy of the snapshot (marshal and strictly decode it again, as before
// PLN-R115) or a second strict decode of the body doubles both (80,329
// allocations, 20.6 MB) and fails here, as does added per-path work of the
// same order. The race detector pads every allocation (18.2 MB for the same
// read), so this file builds without it, in the normal unit pass.
const (
	maximumChangedRevisionAllocs = 48_000
	maximumChangedRevisionBytes  = 16 << 20
)

// TestWorkerStateChangedRevisionReadCostIsBounded is the deterministic guard
// of PLN-R115: allocation counts and bytes do not depend on the host's load,
// unlike BenchmarkWorkerStateChangedRevision's wall time.
func TestWorkerStateChangedRevisionReadCostIsBounded(t *testing.T) {
	read, _ := changedRevisionRead(t)
	read()
	const runs = 5
	defer runtime.GOMAXPROCS(runtime.GOMAXPROCS(1))
	var before, after runtime.MemStats
	runtime.GC()
	runtime.ReadMemStats(&before)
	for range runs {
		read()
	}
	runtime.ReadMemStats(&after)
	allocs := (after.Mallocs - before.Mallocs) / runs
	bytes := (after.TotalAlloc - before.TotalAlloc) / runs
	t.Logf("one changed-revision read: %d allocations, %d bytes", allocs, bytes)
	if allocs > maximumChangedRevisionAllocs || bytes > maximumChangedRevisionBytes {
		t.Fatalf("one changed-revision read made %d allocations of %d bytes; the bound is %d allocations and %d bytes",
			allocs, bytes, maximumChangedRevisionAllocs, maximumChangedRevisionBytes)
	}
}

// A complete 50-page traversal of an unchanged 10,000-path workspace must
// not rebuild its full path set on every page (about 30 MB before caching).
func TestWorkerStateCachedPaginationAllocationsAreBounded(t *testing.T) {
	read := cachedPagination(t)
	read()
	defer runtime.GOMAXPROCS(runtime.GOMAXPROCS(1))
	runtime.GC()
	var before, after runtime.MemStats
	runtime.ReadMemStats(&before)
	read()
	runtime.ReadMemStats(&after)
	if allocated := after.TotalAlloc - before.TotalAlloc; allocated > 2<<20 {
		t.Fatalf("cached traversal allocated %d bytes, want at most 2 MiB", allocated)
	}
}
