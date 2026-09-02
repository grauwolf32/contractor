package stateview

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"sync"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/planner"
)

func TestServiceProjectsCorrelatedStateAndRevalidatesCache(t *testing.T) {
	snapshot := stateFixture(t)
	workspace := snapshot.State.LastCompletedInvocation.Workspace
	workspace.ScopePaths = []string{"README.md", "docs/unread.md", "src/main.py"}
	workspace.Interactions = append(workspace.Interactions,
		contracts.WorkerStateWorkspaceInteraction{
			Path: "README.md", FirstOrdinal: 2, LastOrdinal: 3,
			DiscoveryCalls: 1, ReadCalls: 1,
		},
		contracts.WorkerStateWorkspaceInteraction{
			Path: "generated.txt", FirstOrdinal: 4, LastOrdinal: 4, MutationCalls: 1,
		},
	)
	metrics := &snapshot.State.LastCompletedInvocation.Metrics
	metrics.ToolCalls = 3
	metrics.Tools["grep"] = contracts.WorkerStateInvocationToolMetric{Calls: 2, Failures: 1}
	metrics.ToolErrors = 1
	reader := &recordingReader{states: map[string]contracts.AgentStateSnapshot{"allocation-builder": snapshot}}
	service := testService(t, reader, Binding{
		LogicalName: "builder", Handle: stateHandle("allocation-builder"), WorkspaceEligible: true,
	})
	service.RecordCompletion("builder", "1.1", stateCompletion("worker-completed", "1.1", 7))

	coverage, err := service.GetWorkspaceCoverage(t.Context(), "builder")
	if err != nil || coverage.ScopedFiles != 3 || coverage.DiscoveredFiles != 1 ||
		coverage.ReadFiles != 2 || coverage.MatchedFiles != 0 || coverage.ModifiedFiles != 1 ||
		coverage.UnreadFiles == nil || *coverage.UnreadFiles != 1 ||
		!coverage.ScopeComplete || !coverage.DetailComplete {
		t.Fatalf("workspace coverage = (%+v, %v)", coverage, err)
	}
	first, err := service.ListReadFiles(t.Context(), "builder", "", 1)
	if err != nil || !reflect.DeepEqual(first.Files, []string{"src/main.py"}) ||
		first.NextCursor == "" || !first.Complete {
		t.Fatalf("first read page = (%+v, %v)", first, err)
	}
	for _, forbidden := range []string{"builder", "allocation-builder", "worker-completed"} {
		if strings.Contains(first.NextCursor, forbidden) {
			t.Fatalf("opaque cursor exposed %q: %q", forbidden, first.NextCursor)
		}
	}
	second, err := service.ListReadFiles(t.Context(), "builder", first.NextCursor, 1)
	if err != nil || !reflect.DeepEqual(second.Files, []string{"README.md"}) || second.NextCursor != "" {
		t.Fatalf("second read page = (%+v, %v)", second, err)
	}
	unread, err := service.ListUnreadFiles(t.Context(), "builder", "", 100)
	if err != nil || !reflect.DeepEqual(unread.Files, []string{"docs/unread.md"}) || !unread.Complete {
		t.Fatalf("unread page = (%+v, %v)", unread, err)
	}
	usage, err := service.GetWorkerToolUsage(t.Context(), "builder")
	if err != nil || usage.ModelCalls != 1 || usage.ToolCalls != 3 || usage.ToolErrors != 1 ||
		!reflect.DeepEqual(usage.Tools, []ToolUsageItem{
			{Name: "grep", Calls: 2, Failures: 1},
			{Name: "read_file", Calls: 1, Failures: 0},
		}) {
		t.Fatalf("Worker tool usage = (%+v, %v)", usage, err)
	}
	if calls, conditionals := reader.observed(); calls != 5 ||
		!reflect.DeepEqual(conditionals, []string{"", stateETag(7), stateETag(7), stateETag(7), stateETag(7)}) {
		t.Fatalf("State reads = %d, conditionals = %v", calls, conditionals)
	}

	service.Close()
	if _, err := service.GetWorkerToolUsage(t.Context(), "builder"); errorCode(err) != CodeUnavailable {
		t.Fatalf("closed State service error = %v", err)
	}
	if len(service.bindings) != 0 || len(service.selectors) != 0 || len(service.cache) != 0 ||
		len(service.cursors) != 0 || service.cursorKey != nil {
		t.Fatalf("closed State service retained private data: %+v", service)
	}
}

func TestServiceRejectsStaleStateIncompleteCoverageAndCursorMisuse(t *testing.T) {
	snapshot := stateFixture(t)
	reader := &recordingReader{states: map[string]contracts.AgentStateSnapshot{"allocation-builder": snapshot}}
	service := testService(t, reader, Binding{
		LogicalName: "builder", Handle: stateHandle("allocation-builder"), WorkspaceEligible: true,
	})
	if _, err := service.GetWorkerToolUsage(t.Context(), "builder"); errorCode(err) != CodeUnavailable {
		t.Fatalf("State without completion error = %v", err)
	}
	if calls, _ := reader.observed(); calls != 0 {
		t.Fatalf("State reader called before completion %d times", calls)
	}
	if err := service.RecordCompletion(
		"builder", "1.1", stateCompletion("worker-completed", "1.1", 7),
	); err != nil {
		t.Fatal(err)
	}
	first, err := service.ListReadFiles(t.Context(), "builder", "", 1)
	if err != nil {
		t.Fatal(err)
	}
	if first.NextCursor != "" {
		t.Fatal("one read path unexpectedly required pagination")
	}
	if _, err := service.ListReadFiles(t.Context(), "builder", "not-a-cursor", 10); errorCode(err) != CodeCursorInvalid {
		t.Fatalf("malformed cursor error = %v", err)
	}

	snapshot.State.StateRevision = 8
	snapshot.State.LastCompletedInvocation.InvocationID = "worker-newer"
	snapshot.State.LastCompletedInvocation.SubtaskID = "2"
	reader.set("allocation-builder", snapshot)
	if _, err := service.GetWorkerToolUsage(t.Context(), "builder"); errorCode(err) != CodeChanged {
		t.Fatalf("newer invocation error = %v", err)
	}

	service.RecordCompletion("builder", "2", stateCompletion("worker-newer", "2", 8))
	snapshot.State.LastCompletedInvocation.Workspace.ScopeComplete = false
	reader.set("allocation-builder", snapshot)
	if _, err := service.ListUnreadFiles(t.Context(), "builder", "", 10); errorCode(err) != CodeWorkspaceIncomplete {
		t.Fatalf("incomplete unread error = %v", err)
	}
	if _, err := service.ListReadFiles(t.Context(), "builder", "", 101); errorCode(err) != CodeRequestInvalid {
		t.Fatalf("oversized page error = %v", err)
	}
}

func TestServiceCursorIsBoundToProjectionWorkerAndRevision(t *testing.T) {
	builder := stateFixture(t)
	builder.State.LastCompletedInvocation.Workspace.Interactions = append(
		builder.State.LastCompletedInvocation.Workspace.Interactions,
		contracts.WorkerStateWorkspaceInteraction{
			Path: "README.md", FirstOrdinal: 2, LastOrdinal: 2, ReadCalls: 1,
		},
	)
	reviewer := stateFixture(t)
	reviewer.State.LastCompletedInvocation.InvocationID = "reviewer-completed"
	reader := &recordingReader{states: map[string]contracts.AgentStateSnapshot{
		"allocation-builder": builder, "allocation-reviewer": reviewer,
	}}
	service := testService(t, reader,
		Binding{LogicalName: "builder", Handle: stateHandle("allocation-builder"), WorkspaceEligible: true},
		Binding{LogicalName: "reviewer", Handle: stateHandle("allocation-reviewer"), WorkspaceEligible: true},
	)
	service.RecordCompletion("builder", "1.1", stateCompletion("worker-completed", "1.1", 7))
	service.RecordCompletion("reviewer", "1.1", stateCompletion("reviewer-completed", "1.1", 7))
	first, err := service.ListReadFiles(t.Context(), "builder", "", 1)
	if err != nil || first.NextCursor == "" {
		t.Fatalf("builder cursor = (%+v, %v)", first, err)
	}
	if _, err := service.ListUnreadFiles(t.Context(), "builder", first.NextCursor, 1); errorCode(err) != CodeChanged {
		t.Fatalf("cross-projection cursor error = %v", err)
	}
	if _, err := service.ListReadFiles(t.Context(), "reviewer", first.NextCursor, 1); errorCode(err) != CodeChanged {
		t.Fatalf("cross-Worker cursor error = %v", err)
	}

	// Even a broken Runtime that reuses a revision cannot replay a cursor into
	// another invocation: the full hidden completion selector is bound.
	builder.State.LastCompletedInvocation.InvocationID = "worker-latest-same-revision"
	builder.State.LastCompletedInvocation.SubtaskID = "2"
	reader.set("allocation-builder", builder)
	service.RecordCompletion(
		"builder", "2", stateCompletion("worker-latest-same-revision", "2", 7),
	)
	if _, err := service.ListReadFiles(t.Context(), "builder", first.NextCursor, 1); errorCode(err) != CodeChanged {
		t.Fatalf("cross-invocation cursor error = %v", err)
	}

	updated := stateCompletion("worker-latest", "3", 8)
	if err := service.RecordCompletion("builder", "3", updated); err != nil {
		t.Fatal(err)
	}
	if _, err := service.ListReadFiles(t.Context(), "builder", first.NextCursor, 1); errorCode(err) != CodeChanged {
		t.Fatalf("cross-revision cursor error = %v", err)
	}
}

func TestServiceNormalizesReaderFailureWithoutRetainingCause(t *testing.T) {
	const secret = "https://runtime.internal/state?token=recognizable-secret"
	reader := &recordingReader{err: errors.New(secret)}
	service := testService(t, reader, Binding{
		LogicalName: "builder", Handle: stateHandle("allocation-builder"), WorkspaceEligible: false,
	})
	service.RecordCompletion("builder", "1.1", stateCompletion("worker-completed", "1.1", 7))
	_, err := service.GetWorkerToolUsage(t.Context(), "builder")
	if errorCode(err) != CodeUnavailable || strings.Contains(err.Error(), secret) {
		t.Fatalf("unsafe normalized reader error = %v", err)
	}
	if _, err := service.GetWorkspaceCoverage(t.Context(), "builder"); errorCode(err) != CodeWorkspaceUnavailable {
		t.Fatalf("ineligible workspace error = %v", err)
	}
}

type recordingReader struct {
	mu           sync.Mutex
	states       map[string]contracts.AgentStateSnapshot
	err          error
	conditionals []string
}

func (r *recordingReader) ReadWorkerState(
	_ context.Context,
	handle contracts.WorkerHandle,
	ifNoneMatch string,
) (planner.WorkerStateReadResult, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.conditionals = append(r.conditionals, ifNoneMatch)
	if r.err != nil {
		return planner.WorkerStateReadResult{}, r.err
	}
	snapshot, exists := r.states[handle.AllocationID]
	if !exists {
		return planner.WorkerStateReadResult{}, &planner.WorkerStateReadError{
			Code: CodeUnavailable, Retryable: true,
		}
	}
	etag := stateETag(snapshot.State.StateRevision)
	if ifNoneMatch == etag {
		return planner.WorkerStateReadResult{ETag: etag, NotModified: true}, nil
	}
	return planner.WorkerStateReadResult{Snapshot: &snapshot, ETag: etag}, nil
}

func (r *recordingReader) set(allocationID string, snapshot contracts.AgentStateSnapshot) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.states[allocationID] = snapshot
}

func (r *recordingReader) observed() (int, []string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	return len(r.conditionals), append([]string(nil), r.conditionals...)
}

func testService(t *testing.T, reader planner.WorkerStateReader, bindings ...Binding) *Service {
	t.Helper()
	service, err := New(reader, bindings, Options{CursorKey: []byte(strings.Repeat("k", 32))})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(service.Close)
	return service
}

func stateFixture(t *testing.T) contracts.AgentStateSnapshot {
	t.Helper()
	data, err := os.ReadFile(filepath.Join(
		"..", "..", "..", "api", "testdata", "v1alpha1", "valid", "agent-state-snapshot.json",
	))
	if err != nil {
		t.Fatal(err)
	}
	result, err := contracts.DecodeStrict[contracts.AgentStateSnapshot](data)
	if err != nil {
		t.Fatal(err)
	}
	result.State.CurrentInvocation = nil
	return result
}

func stateCompletion(invocationID, subtaskID string, revision uint64) contracts.WorkerCompletion {
	return contracts.WorkerCompletion{
		APIVersion: contracts.APIVersion,
		Result: &contracts.WorkerResult{
			SubtaskID: subtaskID, Result: "complete",
			Observations: contracts.WorkerObservations{
				Profile: contracts.WorkerObservationProfileLeanV1,
				Tools:   map[string]contracts.ToolObservationCount{},
			},
			Artifacts: map[string]contracts.ArtifactRef{},
		},
		InvocationID: invocationID, StateRevision: revision,
	}
}

func stateHandle(allocationID string) contracts.WorkerHandle {
	return contracts.WorkerHandle{AllocationID: allocationID, AgentCard: map[string]any{"safe": true}}
}

func stateETag(revision uint64) string {
	return fmt.Sprintf("\"contractor-agent-state-v1-%d\"", revision)
}

func errorCode(err error) string {
	var typed *Error
	if errors.As(err, &typed) {
		return typed.Code
	}
	return ""
}
