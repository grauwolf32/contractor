package contracts

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestWorkerSubtaskIDContract(t *testing.T) {
	t.Parallel()

	for _, value := range []string{"0", "1.1", "review:2_retry-1", strings.Repeat("a", 128)} {
		if err := validateWorkerSubtaskID(value); err != nil {
			t.Errorf("valid subtask ID %q rejected: %v", value, err)
		}
	}
	for _, value := range []string{"", ".1", "with space", "сложная", strings.Repeat("a", 129)} {
		if err := validateWorkerSubtaskID(value); err == nil {
			t.Errorf("invalid subtask ID %q accepted", value)
		}
	}
}

func TestWorkerCompletionRejectsEveryInvalidBoundary(t *testing.T) {
	t.Parallel()

	tests := map[string]func(*WorkerCompletion){
		"both variants": func(value *WorkerCompletion) {
			value.Failure = &WorkerFailure{Code: "worker_failed", Message: "failed", Retryable: true}
		},
		"neither variant": func(value *WorkerCompletion) {
			value.Result = nil
		},
		"zero revision": func(value *WorkerCompletion) {
			value.StateRevision = 0
		},
		"invalid invocation": func(value *WorkerCompletion) {
			value.InvocationID = " worker-1 "
		},
		"unversioned artifact": func(value *WorkerCompletion) {
			ref := value.Result.Artifacts["report"]
			ref.Revision = nil
			value.Result.Artifacts["report"] = ref
		},
		"purpose-reserved artifact": func(value *WorkerCompletion) {
			ref := value.Result.Artifacts["report"]
			ref.Namespace = "inputs"
			value.Result.Artifacts["report"] = ref
		},
		"Memory artifact": func(value *WorkerCompletion) {
			ref := value.Result.Artifacts["report"]
			ref.Name = "memory.note"
			value.Result.Artifacts["report"] = ref
		},
		"tool failures exceed calls": func(value *WorkerCompletion) {
			value.Result.Observations.Tools["read_file"] = ToolObservationCount{Calls: 1, Failures: 2}
		},
		"missing exact unread count": func(value *WorkerCompletion) {
			value.Result.Observations.Workspace.UnreadFiles = nil
		},
		"read detail mismatch": func(value *WorkerCompletion) {
			value.Result.Observations.Workspace.ReadFiles = 3
		},
		"nil read detail": func(value *WorkerCompletion) {
			value.Result.Observations.Workspace.FilesRead = nil
		},
		"false positive read truncation": func(value *WorkerCompletion) {
			value.Result.Observations.Workspace.FilesReadTruncated = true
			value.Result.Observations.Truncated = true
		},
		"unreported truncation": func(value *WorkerCompletion) {
			value.Result.Observations.Workspace.DetailComplete = false
			value.Result.Observations.Workspace.UnreadFiles = nil
		},
		"oversized result": func(value *WorkerCompletion) {
			value.Result.Result = strings.Repeat("x", MaxWorkerResultBytes+1)
		},
	}

	for name, mutate := range tests {
		name, mutate := name, mutate
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			value := validWorkerCompletion()
			mutate(&value)
			if err := value.Validate(); err == nil {
				t.Fatal("invalid WorkerCompletion was accepted")
			}
		})
	}
}

func TestWorkerCompletionStrictDecodeRejectsUnknownNestedField(t *testing.T) {
	t.Parallel()

	value := validWorkerCompletion()
	encoded, err := json.Marshal(value)
	if err != nil {
		t.Fatal(err)
	}
	var document map[string]any
	if err := json.Unmarshal(encoded, &document); err != nil {
		t.Fatal(err)
	}
	result := document["result"].(map[string]any)
	result["outcome"] = "succeeded"
	encoded, _ = json.Marshal(document)
	if _, err := DecodeStrict[WorkerCompletion](encoded); err == nil {
		t.Fatal("model-authored outcome was accepted")
	}
}

func validWorkerCompletion() WorkerCompletion {
	revision := "report-r1"
	unread := uint64(1)
	return WorkerCompletion{
		APIVersion: APIVersion,
		Result: &WorkerResult{
			SubtaskID: "1.1",
			Result:    "completed",
			Observations: WorkerObservations{
				Profile: WorkerObservationProfileLeanV1,
				Tools: map[string]ToolObservationCount{
					"read_file": {Calls: 2, Failures: 0},
				},
				Workspace: &WorkspaceObservationSummary{
					ScopedFiles: 2, ScopeComplete: true, DiscoveredFiles: 1,
					ReadFiles: 1, MatchedFiles: 0, ModifiedFiles: 0,
					DetailComplete: true, UnreadFiles: &unread,
					FilesRead: []string{"src/main.go"}, FilesReadTruncated: false,
				},
				Truncated: false,
			},
			Artifacts: map[string]ArtifactRef{
				"report": {Namespace: "builder", Name: "report", Revision: &revision},
			},
			Summarized: false,
		},
		InvocationID:  "worker-0123456789abcdef",
		StateRevision: 7,
	}
}
