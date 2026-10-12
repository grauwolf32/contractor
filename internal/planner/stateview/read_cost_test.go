package stateview

import (
	"context"
	"encoding/json"
	"fmt"
	"syscall"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/reporting"
)

// maximumStatePaths is the Worker State workspace bound: 10,000 scope paths
// and 10,000 interactions.
const maximumStatePaths = 10_000

// decodingReader returns a freshly decoded snapshot on every read, as the
// Runtime client does with each response body.
type decodingReader struct {
	body []byte
	etag string
}

func (r *decodingReader) ReadWorkerState(
	context.Context, contracts.WorkerHandle, string,
) (reporting.WorkerStateReadResult, error) {
	snapshot, err := contracts.DecodePrivateStrict[reporting.AgentStateSnapshot](r.body)
	if err != nil {
		return reporting.WorkerStateReadResult{}, err
	}
	return reporting.WorkerStateReadResult{Snapshot: &snapshot, ETag: r.etag}, nil
}

// maximumStateBody encodes the fixture State with the workspace at its path
// bounds, about 2 MB.
func maximumStateBody(t testing.TB) []byte {
	t.Helper()
	snapshot := stateFixture(t)
	workspace := snapshot.State.LastCompletedInvocation.Workspace
	workspace.ScopePaths = make([]string, 0, maximumStatePaths)
	workspace.Interactions = make([]reporting.WorkerStateWorkspaceInteraction, 0, maximumStatePaths)
	for index := range maximumStatePaths {
		path := fmt.Sprintf("src/component%03d/module%04d/source.go", index/100, index)
		workspace.ScopePaths = append(workspace.ScopePaths, path)
		workspace.Interactions = append(workspace.Interactions, reporting.WorkerStateWorkspaceInteraction{
			Path: path, FirstOrdinal: uint64(index + 1), LastOrdinal: uint64(index + 2),
			DiscoveryCalls: 1, ReadCalls: uint64(index % 2), MatchCalls: uint64(index % 3),
		})
	}
	body, err := json.Marshal(snapshot)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := contracts.DecodePrivateStrict[reporting.AgentStateSnapshot](body); err != nil {
		t.Fatalf("maximum State is invalid: %v", err)
	}
	return body
}

// processCPU returns the CPU time this process has used, which a shared
// machine inflates less than wall time.
func processCPU(b *testing.B) time.Duration {
	var usage syscall.Rusage
	if err := syscall.Getrusage(syscall.RUSAGE_SELF, &usage); err != nil {
		b.Fatal(err)
	}
	return time.Duration(usage.Utime.Nano() + usage.Stime.Nano())
}

// BenchmarkWorkerStateChangedRevision measures one Planner State tool call
// that finds a changed revision at the contract maximum: the Runtime client's
// strict decode and validation of the body, then the projection. Besides
// wall time it reports process CPU per call, garbage collection included.
func BenchmarkWorkerStateChangedRevision(b *testing.B) {
	reader := &decodingReader{body: maximumStateBody(b), etag: stateETag(7)}
	service := testService(b, reader, Binding{
		LogicalName: "builder", Handle: stateHandle("allocation-builder"), WorkspaceEligible: true,
	})
	completion := stateCompletion("worker-completed", "1.1", 7)
	b.SetBytes(int64(len(reader.body)))
	b.ReportAllocs()
	started := processCPU(b)
	calls := 0
	for b.Loop() {
		calls++
		// Recording the completion drops the cached revision, so every call
		// reads and decodes the whole State.
		if err := service.RecordCompletion("builder", "1.1", completion); err != nil {
			b.Fatal(err)
		}
		coverage, err := service.GetWorkspaceCoverage(b.Context(), "builder")
		if err != nil || coverage.ScopedFiles != maximumStatePaths {
			b.Fatalf("coverage = (%+v, %v)", coverage, err)
		}
	}
	b.ReportMetric(float64(processCPU(b)-started)/float64(time.Millisecond)/float64(calls), "cpu-ms/op")
}
