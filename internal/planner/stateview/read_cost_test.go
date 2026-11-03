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

// maximumStateBody encodes a valid State whose last completed invocation has
// the workspace at its path bounds, about 1.9 MB. It is built here rather
// than from the shared agent-state-snapshot.json fixture, which only a
// TestShared* test may read, so the benchmark runs on its own.
func maximumStateBody(t testing.TB) []byte {
	t.Helper()
	outcome := "succeeded"
	workspace := &reporting.WorkerStateWorkspace{
		ScopePaths:     make([]string, 0, maximumStatePaths),
		ScopeComplete:  true,
		Interactions:   make([]reporting.WorkerStateWorkspaceInteraction, 0, maximumStatePaths),
		DetailComplete: true,
	}
	for index := range maximumStatePaths {
		path := fmt.Sprintf("src/component%03d/module%04d/source.go", index/100, index)
		workspace.ScopePaths = append(workspace.ScopePaths, path)
		workspace.Interactions = append(workspace.Interactions, reporting.WorkerStateWorkspaceInteraction{
			Path: path, FirstOrdinal: uint64(index + 1), LastOrdinal: uint64(index + 2),
			DiscoveryCalls: 1, ReadCalls: uint64(index % 2), MatchCalls: uint64(index % 3),
		})
	}
	snapshot := reporting.AgentStateSnapshot{
		APIVersion: contracts.APIVersion,
		State: reporting.ContractorWorkerState{
			SchemaVersion: 2,
			StateRevision: 7,
			Metrics: reporting.WorkerAllocationState{
				Counters:     map[string]uint64{"llm_calls": 1, "tool_calls": 1, "tool_calls.read_file": 1},
				ToolCalls:    []reporting.WorkerStateToolCall{},
				Errors:       []reporting.ExecutionError{},
				FinalOutcome: &outcome,
			},
			LastCompletedInvocation: &reporting.WorkerInvocationState{
				InvocationID: "worker-completed",
				SubtaskID:    "1.1",
				Phase:        "succeeded",
				Metrics: reporting.WorkerStateInvocationMetric{
					ModelCalls: 1, InputTokens: 16, OutputTokens: 4, TotalTokens: 20, ToolCalls: 1,
					Tools: map[string]reporting.WorkerStateInvocationToolMetric{"read_file": {Calls: 1}},
				},
				Summarizer: reporting.WorkerStateSummarizer{Phase: "disabled"},
				Workspace:  workspace,
			},
		},
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

// changedRevisionRead returns one Planner State tool call that finds a
// changed revision at the contract maximum, and the size of the State it
// reads: recording the completion drops the cached revision, so the call
// reads, strictly decodes and validates the whole State, then projects it.
func changedRevisionRead(t testing.TB) (func(), int) {
	t.Helper()
	reader := &decodingReader{body: maximumStateBody(t), etag: stateETag(7)}
	service := testService(t, reader, Binding{
		LogicalName: "builder", Handle: stateHandle("allocation-builder"), WorkspaceEligible: true,
	})
	completion := stateCompletion("worker-completed", "1.1", 7)
	return func() {
		if err := service.RecordCompletion("builder", "1.1", completion); err != nil {
			t.Fatal(err)
		}
		coverage, err := service.GetWorkspaceCoverage(t.Context(), "builder")
		if err != nil || coverage.ScopedFiles != maximumStatePaths {
			t.Fatalf("coverage = (%+v, %v)", coverage, err)
		}
	}, len(reader.body)
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
	read, size := changedRevisionRead(b)
	b.SetBytes(int64(size))
	b.ReportAllocs()
	started := processCPU(b)
	calls := 0
	for b.Loop() {
		calls++
		read()
	}
	b.ReportMetric(float64(processCPU(b)-started)/float64(time.Millisecond)/float64(calls), "cpu-ms/op")
}

// cachedPaginationReader models the Runtime's strong-ETag response without
// retaining a growing call log in the benchmark.
type cachedPaginationReader struct {
	snapshot reporting.AgentStateSnapshot
}

func (r *cachedPaginationReader) ReadWorkerState(
	_ context.Context, _ contracts.WorkerHandle, etag string,
) (reporting.WorkerStateReadResult, error) {
	current := stateETag(r.snapshot.State.StateRevision)
	if etag == current {
		return reporting.WorkerStateReadResult{ETag: current, NotModified: true}, nil
	}
	return reporting.WorkerStateReadResult{ETag: current, Snapshot: &r.snapshot}, nil
}

func cachedPagination(t testing.TB) func() {
	t.Helper()
	snapshot, err := contracts.DecodePrivateStrict[reporting.AgentStateSnapshot](maximumStateBody(t))
	if err != nil {
		t.Fatal(err)
	}
	service := testService(t, &cachedPaginationReader{snapshot: snapshot}, Binding{
		LogicalName: "builder", Handle: stateHandle("allocation-builder"), WorkspaceEligible: true,
	})
	if err := service.RecordCompletion("builder", "1.1", stateCompletion("worker-completed", "1.1", 7)); err != nil {
		t.Fatal(err)
	}
	return func() {
		cursor, total := "", 0
		for {
			page, err := service.ListUnreadFiles(t.Context(), "builder", cursor, 100)
			if err != nil || !page.Complete || len(page.Files) != 100 {
				t.Fatalf("unread page = (%+v, %v)", page, err)
			}
			for _, path := range page.Files {
				index := total * 2
				want := fmt.Sprintf("src/component%03d/module%04d/source.go", index/100, index)
				if path != want {
					t.Fatalf("unread path %d = %q, want %q", total, path, want)
				}
				total++
			}
			if page.NextCursor == "" {
				break
			}
			cursor = page.NextCursor
		}
		if total != maximumStatePaths/2 {
			t.Fatalf("unread files = %d", total)
		}
	}
}

func TestWorkerStateCachedPaginationAtMaximumPaths(t *testing.T) {
	read := cachedPagination(t)
	read()
	read()
}

func BenchmarkWorkerStateCachedPagination(b *testing.B) {
	read := cachedPagination(b)
	read()
	b.ReportAllocs()
	for b.Loop() {
		read()
	}
}
