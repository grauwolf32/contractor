package streamline

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/planner"
	"github.com/grauwolf32/contractor/internal/planner/stageresult"
	"google.golang.org/adk/model"
)

func continuationInvocation() planner.Invocation {
	invocation := testInvocation("builder")
	invocation.Context.Continuation = &stageresult.SummaryContinuation{
		SourceStageExecutionID: "stage-source", Summary: "Analysis complete; publish the remaining report.",
		SummaryArtifact: exactRef("builder", "summary.stage-source", "summary-r1"),
		Artifacts:       map[string]contracts.ArtifactRef{"saved": exactRef("builder", "analysis", "analysis-r1")},
	}
	return invocation
}

func TestContinuationPromptPreservesSeparateTaskAndSavedProgress(t *testing.T) {
	invocation := continuationInvocation()
	factory := mustFactory(t, newFakeSessions(), &fakeWorkerInvoker{}, &fakeInspector{}, &scriptedModel{}, Limits{})
	instance, err := factory.Create(invocation)
	if err != nil {
		t.Fatal(err)
	}
	invocation.Context.Continuation.Summary = "caller mutation"
	p := instance.(*streamlinePlanner)
	message, err := p.initialMessage()
	if err != nil {
		t.Fatal(err)
	}
	var prompt promptContext
	if err := json.Unmarshal([]byte(message.Parts[0].Text), &prompt); err != nil {
		t.Fatal(err)
	}
	if prompt.Objective != invocation.Stage.Objective || prompt.Instructions != invocation.Stage.Instructions.Text ||
		prompt.Parameters["mode"] != "strict" || *prompt.Artifacts["source"].Revision != "source-r1" ||
		prompt.Continuation.Summary != "Analysis complete; publish the remaining report." || *prompt.Continuation.Artifacts["saved"].Revision != "analysis-r1" {
		t.Fatalf("task/summary changed: %+v", prompt)
	}
	if strings.Contains(message.Parts[0].Text, "sourceStageExecutionId") || strings.Contains(message.Parts[0].Text, "deadline") {
		t.Fatal("control metadata leaked into model context")
	}
	if strings.Contains(p.systemInstruction(), "Memory") {
		t.Fatal("continuation instructs a Worker without selected Memory tools to call them")
	}
}

func TestContinuationInputLimitIsCheckedBeforeModelFactory(t *testing.T) {
	invocation := continuationInvocation()
	invocation.Context.Continuation.Summary = strings.Repeat("ж", stageresult.MaxStageResultSummaryBytes/2)
	invocation.Context.Parameters["padding"] = ""
	sessions := newFakeSessions()
	calls := 0
	factory, err := NewConfiguredFactory(sessions, sessions, &fakeWorkerInvoker{}, &fakeInspector{}, unavailableWorkerStateReader{},
		func(planner.ModelAccess) (model.LLM, error) { calls++; return &scriptedModel{}, nil }, Limits{})
	if err != nil {
		t.Fatal(err)
	}
	initial, err := factory.Create(invocation)
	if err != nil {
		t.Fatal(err)
	}
	message, err := initial.(*streamlinePlanner).initialMessage()
	if err != nil {
		t.Fatal(err)
	}
	padding := maxPlannerContextBytes - len(message.Parts[0].Text)
	invocation.Context.Parameters["padding"] = strings.Repeat("x", padding)
	exact, err := factory.Create(invocation)
	if err != nil {
		t.Fatalf("exact input limit rejected: %v", err)
	}
	message, err = exact.(*streamlinePlanner).initialMessage()
	if err != nil || len(message.Parts[0].Text) != maxPlannerContextBytes {
		t.Fatalf("exact input size: %v", err)
	}
	before := calls
	invocation.Context.Parameters["padding"] += "x"
	if _, err := factory.Create(invocation); err == nil {
		t.Fatal("oversized combined original task and summary accepted")
	}
	if calls != before {
		t.Fatal("model factory called before context validation")
	}
}
