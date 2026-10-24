package scanplan

import (
	"sync/atomic"
	"testing"
)

// SourceWork counts the whole-source work of preparation: source document
// parses and source content digests. Option parsing and preparation or
// request digests are operation work and are not counted.
type SourceWork struct {
	parses  atomic.Int64
	digests atomic.Int64
}

func (w *SourceWork) Parses() int64  { return w.parses.Load() }
func (w *SourceWork) Digests() int64 { return w.digests.Load() }

// CountSourceWork counts source work for the rest of one test. Tests that
// call it must not run in parallel.
func CountSourceWork(t testing.TB) *SourceWork {
	t.Helper()
	work := &SourceWork{}
	parse, digest := parseSourceDocument, sourceContentDigest
	parseSourceDocument = func(data []byte, mediaType string) (map[string]any, error) {
		work.parses.Add(1)
		return parse(data, mediaType)
	}
	sourceContentDigest = func(data []byte) string {
		work.digests.Add(1)
		return digest(data)
	}
	t.Cleanup(func() { parseSourceDocument, sourceContentDigest = parse, digest })
	return work
}
