package planner

import (
	"reflect"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/planner/stageresult"
)

func TestPassthroughSummarizedCompletionIsRetainedForContinuation(t *testing.T) {
	const summary = "Inspected the source; the report still needs publication."
	completion := workerCompletionFromCandidate(stageresult.StageContentResult{
		Summary: summary, Artifacts: map[string]contracts.ArtifactRef{},
	})
	completion.Result.Summarized = true
	sessions := &memorySessions{}
	worker := &recordingWorker{result: completion}
	factory, err := NewPassthroughFactory(sessions, worker, &fakeInspector{})
	if err != nil {
		t.Fatal(err)
	}
	invocation := testInvocation()
	instance, err := factory.Create(invocation)
	if err != nil {
		t.Fatal(err)
	}
	result, err := instance.Run(t.Context())
	if err != nil {
		t.Fatalf("summarized intermediate completion was discarded: %v", err)
	}
	if result.Outcome != stageresult.StageFailed || result.Error == nil ||
		result.Error.Code != "worker_summarized" || result.Summary != summary {
		t.Fatalf("continuation candidate = %+v", result)
	}
	if sessions.completion == nil || sessions.completion.Result == nil ||
		sessions.completion.Failure != nil || !reflect.DeepEqual(*sessions.completion.Result, result) {
		t.Fatalf("summary was not retained in the durable completion: %+v", sessions.completion)
	}
	recovered, err := factory.Create(invocation)
	if err != nil {
		t.Fatal(err)
	}
	replayed, err := recovered.Run(t.Context())
	if err != nil || !reflect.DeepEqual(replayed, result) || worker.calls != 1 {
		t.Fatalf("summary recovery replayed work or lost progress: result=%+v err=%v calls=%d", replayed, err, worker.calls)
	}
}
