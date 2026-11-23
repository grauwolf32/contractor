package streamline

import (
	"context"
	"encoding/json"
	"errors"
	"iter"
	"reflect"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/planner"
	"github.com/grauwolf32/contractor/internal/stageresult"
	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/runner"
	adksession "google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"
	"google.golang.org/genai"
)

const migrationHistoryText = "fixture useful history text"

// These cases keep SDK projection observations separate from the raw history
// and no-replay guarantees. The migration harness compares the safe marker
// against explicit source-specific dispositions; no SDK version branch or
// permissive production reader is introduced here.
func TestGoADKMigrationHistory(t *testing.T) {
	call := func(id string) *genai.Part {
		return &genai.Part{FunctionCall: &genai.FunctionCall{Name: "echo", ID: id, Args: map[string]any{"value": id}}}
	}
	response := func(id, value string) *genai.Part {
		return &genai.Part{FunctionResponse: &genai.FunctionResponse{Name: "echo", ID: id, Response: map[string]any{"value": value}}}
	}
	for _, test := range []struct {
		name  string
		parts [][]*genai.Part
		roles []string
	}{
		{"complete_pair", [][]*genai.Part{{call("first")}, {response("first", "one")}}, []string{"model", "user"}},
		{"call_only", [][]*genai.Part{{call("first")}}, []string{"model"}},
		{"response_only", [][]*genai.Part{{response("orphan", "one")}}, []string{"user"}},
		{"interrupted_parallel_pair", [][]*genai.Part{{call("first"), call("second")}, {response("first", "one")}}, []string{"model", "user"}},
		{"repeated_name_distinct_ids", [][]*genai.Part{{call("first"), call("second")}, {response("first", "one")}, {response("second", "two")}}, []string{"model", "user", "user"}},
		{"empty_ids", [][]*genai.Part{{call("")}, {response("", "one")}}, []string{"model", "user"}},
		{"response_only_with_text", [][]*genai.Part{{response("orphan", "one"), {Text: migrationHistoryText}}}, []string{"user"}},
		{"mixed_complete_orphan_response_with_text", [][]*genai.Part{{call("first")}, {response("first", "one"), response("orphan", "two"), {Text: migrationHistoryText}}}, []string{"model", "user"}},
	} {
		t.Run(test.name, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(t.Context(), 10*time.Second)
			defer cancel()
			service := adksession.InMemoryService()
			created, err := service.Create(ctx, &adksession.CreateRequest{AppName: "migration_history", UserID: "fixture-user", SessionID: "fixture-session"})
			if err != nil {
				t.Fatal(err)
			}
			for index, parts := range test.parts {
				event := migrationHistoryEvent(ctx, index, test.roles[index], parts)
				if err := service.AppendEvent(ctx, created.Session, event); err != nil {
					t.Fatal(err)
				}
			}
			before := migrationRawPrefix(t, created.Session, len(test.parts))
			var tools atomic.Int64
			echo, err := functiontool.New(functiontool.Config{Name: "echo", Description: "Return a bounded fixture value."}, func(_ agent.Context, args struct {
				Value string `json:"value"`
			}) (map[string]any, error) {
				tools.Add(1)
				return map[string]any{"value": args.Value}, nil
			})
			if err != nil {
				t.Fatal(err)
			}
			llm := &migrationHistoryModel{}
			run := migrationRunner(t, service, llm, []tool.Tool{echo})
			for _, err := range run.Run(ctx, "fixture-user", "fixture-session", genai.NewContentFromText("Continue the fixture", genai.RoleUser), agent.RunConfig{}) {
				if err != nil {
					t.Fatal(err)
				}
			}
			if llm.calls != 1 || tools.Load() != 0 {
				t.Fatalf("history model/tool calls=%d/%d, want1/0", llm.calls, tools.Load())
			}
			stored, err := service.Get(ctx, &adksession.GetRequest{AppName: "migration_history", UserID: "fixture-user", SessionID: "fixture-session"})
			if err != nil {
				t.Fatal(err)
			}
			rawPreserved := reflect.DeepEqual(before, migrationRawPrefix(t, stored.Session, len(test.parts)))
			if !rawPreserved {
				t.Fatal("model projection mutated raw retained seed events")
			}
			projection := migrationHistoryProjection(llm.contents)
			if test.name == "complete_pair" && (!reflect.DeepEqual(projection.calls, []string{"first"}) || !reflect.DeepEqual(projection.responses, []string{"first"}) || !reflect.DeepEqual(projection.values, []string{"one"})) {
				t.Fatal("complete group changed identity or actual result")
			}
			if test.name == "repeated_name_distinct_ids" && (!reflect.DeepEqual(projection.calls, []string{"first", "second"}) || !reflect.DeepEqual(projection.responses, []string{"first", "second"}) || !reflect.DeepEqual(projection.values, []string{"one", "two"})) {
				t.Fatal("same-name completed siblings lost distinct IDs or results")
			}
			migrationObservation(t, map[string]any{"group": "history", "case": test.name, "calls": projection.calls, "responses": projection.responses, "values": projection.values, "useful_text": projection.usefulText, "raw_preserved": rawPreserved, "model_calls": llm.calls, "tool_calls": tools.Load()})
		})
	}
}

func migrationHistoryEvent(ctx context.Context, index int, role string, parts []*genai.Part) *adksession.Event {
	event := adksession.NewEvent(ctx, "fixture-seed")
	event.Timestamp = time.Unix(100+int64(index), 0)
	event.Author = "user"
	if role == "model" {
		event.Author = "history_agent"
	}
	event.LLMResponse = model.LLMResponse{Content: &genai.Content{Role: role, Parts: parts}}
	return event
}

func migrationRawPrefix(t *testing.T, session adksession.Session, count int) []string {
	t.Helper()
	result := make([]string, 0, count)
	for event := range session.Events().All() {
		if len(result) == count {
			break
		}
		encoded, err := json.Marshal(event)
		if err != nil {
			t.Fatal(err)
		}
		result = append(result, string(encoded))
	}
	if len(result) != count {
		t.Fatal("raw retained history lost a seed event")
	}
	return result
}

type migrationProjection struct {
	calls, responses, values []string
	usefulText               bool
}

func migrationHistoryProjection(contents []*genai.Content) migrationProjection {
	result := migrationProjection{calls: []string{}, responses: []string{}, values: []string{}}
	for _, content := range contents {
		for _, part := range content.Parts {
			if part == nil {
				continue
			}
			if call := part.FunctionCall; call != nil {
				result.calls = append(result.calls, call.ID)
			}
			if response := part.FunctionResponse; response != nil {
				result.responses = append(result.responses, response.ID)
				value, _ := response.Response["value"].(string)
				result.values = append(result.values, value)
			}
			result.usefulText = result.usefulText || strings.Contains(part.Text, migrationHistoryText)
		}
	}
	return result
}

type migrationHistoryModel struct {
	calls    int
	contents []*genai.Content
}

func (*migrationHistoryModel) Name() string { return "migration-history-model" }

func (m *migrationHistoryModel) GenerateContent(_ context.Context, request *model.LLMRequest, stream bool) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		m.calls++
		if stream || m.calls != 1 {
			yield(nil, errors.New("unexpected fixture model admission"))
			return
		}
		encoded, err := json.Marshal(request.Contents)
		if err != nil {
			yield(nil, err)
			return
		}
		if err := json.Unmarshal(encoded, &m.contents); err != nil {
			yield(nil, err)
			return
		}
		response, _ := textStep("History fixture completed")(nil)
		yield(response, nil)
	}
}

func migrationRunner(t *testing.T, service adksession.Service, llm model.LLM, tools []tool.Tool) *runner.Runner {
	t.Helper()
	root, err := llmagent.New(llmagent.Config{Name: "history_agent", Model: llm, Tools: tools})
	if err != nil {
		t.Fatal(err)
	}
	run, err := runner.New(runner.Config{AppName: "migration_history", Agent: root, SessionService: service})
	if err != nil {
		t.Fatal(err)
	}
	return run
}

func migrationObservation(t *testing.T, fields map[string]any) {
	t.Helper()
	encoded, err := json.Marshal(fields)
	if err != nil {
		t.Fatal(err)
	}
	t.Log("GO_ADK_OBSERVATION " + string(encoded))
}

func TestGoADKMigrationHistoryPlannerAdmission(t *testing.T) {
	for _, profile := range []string{"streamline", "router"} {
		for _, scenario := range []string{"unknown_call", "multiple_calls", "finish_at_budget"} {
			t.Run(profile+"/"+scenario, func(t *testing.T) {
				sessions := newFakeSessions()
				workers := &fakeWorkerInvoker{results: map[string]stageresult.StageContentResult{"builder": stageResult("Completed fixture", map[string]contracts.ArtifactRef{})}}
				invocation := testInvocation("builder")
				invocation.Stage.Planner = resolved.PlannerRef{PlannerID: profile, Version: "1"}
				invocation.Stage.Result.Artifacts = map[string]resolved.ArtifactSlot{}
				finish := functionStep(finishToolName, map[string]any{"outcome": "failed", "summary": "Bounded fixture decision", "artifacts": map[string]any{}, "error": map[string]any{"code": "not_completed", "message": "Bounded fixture decision", "retryable": false}})
				first := functionStep("fixture_unknown_name", map[string]any{})
				if scenario == "multiple_calls" {
					first = func(*model.LLMRequest) (*model.LLMResponse, error) {
						return &model.LLMResponse{Content: &genai.Content{Role: "model", Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{Name: addSubtaskToolName, ID: "first", Args: map[string]any{"objective": "Bounded", "instructions": "Bounded"}}}, {FunctionCall: &genai.FunctionCall{Name: addSubtaskToolName, ID: "second", Args: map[string]any{"objective": "Bounded", "instructions": "Bounded"}}}}}, UsageMetadata: &genai.GenerateContentResponseUsageMetadata{PromptTokenCount: 10, CandidatesTokenCount: 10, TotalTokenCount: 20}}, nil
					}
				}
				steps, wantModels, wantWorkers := []modelStep{first, finish}, 2, 0
				invocation.ModelAccess.ModelPolicy.MaxModelCalls = 2
				invocation.ModelAccess.ModelPolicy.MaxTotalTokens = 40
				if scenario == "finish_at_budget" {
					execute := map[string]any{"subtask_id": "0"}
					if profile == "router" {
						execute["worker_name"] = "builder"
					}
					steps = []modelStep{addSubtaskStep("Bounded", "Bounded"), functionStep(executeCurrentSubtaskToolName, execute), functionStep(finishToolName, map[string]any{"outcome": "succeeded", "summary": "Completed fixture", "artifacts": map[string]any{}})}
					wantModels, wantWorkers = 3, 1
					invocation.ModelAccess.ModelPolicy.MaxModelCalls = 3
					invocation.ModelAccess.ModelPolicy.MaxWorkerCalls = 1
					invocation.ModelAccess.ModelPolicy.MaxTotalTokens = 60
				}
				llm := &scriptedModel{steps: steps}
				create := NewConfiguredFactory
				if profile == "router" {
					create = NewConfiguredRouterDelegate
				}
				factory, err := create(sessions, sessions, workers, &fakeInspector{}, unavailableWorkerStateReader{}, testModelFactory(llm), Limits{MaxWallTime: time.Minute})
				if err != nil {
					t.Fatal(err)
				}
				instance, err := factory.Create(invocation)
				if err != nil {
					t.Fatal(err)
				}
				result, err := instance.Run(t.Context())
				if err != nil || llm.callCount() != wantModels || len(workers.bindingCalls()) != wantWorkers {
					code := "none"
					if err != nil {
						code = planner.FailureFrom(err).Code
					}
					t.Fatalf("admission outcome code=%s model/Worker=%d/%d", code, llm.callCount(), len(workers.bindingCalls()))
				}
				if scenario == "finish_at_budget" && result.Outcome != stageresult.StageSucceeded || scenario != "finish_at_budget" && (result.Outcome != stageresult.StageFailed || len(sessions.transitions) != 0) {
					t.Fatal("finish boundary or rejected-call zero-effect contract changed")
				}
				report, _ := instance.(planner.ReportProvider).ExecutionReport()
				encoded, _ := json.Marshal(report)
				if strings.Contains(string(encoded), "fixture_unknown_name") {
					t.Fatal("unknown provider function name reached the report")
				}
				migrationObservation(t, map[string]any{"group": "planner_admission", "case": profile + "/" + scenario, "model_calls": llm.callCount(), "worker_calls": len(workers.bindingCalls()), "outcome": result.Outcome, "plan_transitions": len(sessions.transitions)})
			})
		}
	}
}

func TestGoADKMigrationLifecycle(t *testing.T) {
	for _, stop := range []string{"parent_cancel", "consumer_break"} {
		t.Run(stop, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(t.Context(), 10*time.Second)
			defer cancel()
			service := adksession.InMemoryService()
			if _, err := service.Create(ctx, &adksession.CreateRequest{AppName: "migration_history", UserID: "fixture-user", SessionID: "fixture-session"}); err != nil {
				t.Fatal(err)
			}
			llm := &migrationLifecycleModel{started: make(chan struct{}), cancelling: make(chan struct{}), release: make(chan struct{}), joined: make(chan struct{})}
			run := migrationRunner(t, service, llm, nil)
			runContext, stopRun := context.WithCancel(ctx)
			yielded, done := make(chan struct{}), make(chan error, 1)
			var release sync.Once
			allowJoin := func() { release.Do(func() { close(llm.release) }) }
			go func() {
				defer close(done)
				var iterationErr error
				for _, err := range run.Run(runContext, "fixture-user", "fixture-session", genai.NewContentFromText("Exercise one lifecycle", genai.RoleUser), agent.RunConfig{}) {
					if err != nil {
						iterationErr = err
						break
					}
					close(yielded)
					if stop == "consumer_break" {
						break
					}
				}
				done <- iterationErr
			}()
			defer func() {
				stopRun()
				allowJoin()
				select {
				case <-done:
				case <-time.After(5 * time.Second):
					t.Error("fixture could not join Runner owner")
				}
			}()
			migrationWait(t, ctx, llm.started)
			migrationWait(t, ctx, yielded)
			if stop == "parent_cancel" {
				stopRun()
			}
			migrationWait(t, ctx, llm.cancelling)
			select {
			case <-done:
				t.Fatal("Runner returned while admitted model cleanup was blocked")
			default:
			}
			allowJoin()
			var iterationErr error
			select {
			case iterationErr = <-done:
			case <-ctx.Done():
				t.Fatal("Runner did not join admitted model cleanup")
			}
			migrationWait(t, ctx, llm.joined)
			if llm.active.Load() != 0 || llm.calls.Load() != 1 {
				t.Fatal("model work survived iterator stop or was admitted twice")
			}
			migrationObservation(t, map[string]any{"group": "lifecycle", "case": stop, "model_calls": llm.calls.Load(), "active_after_return": llm.active.Load(), "iteration_cancelled": errors.Is(iterationErr, context.Canceled), "caller_cancelled": runContext.Err() != nil, "joined": true})
		})
	}
}

func migrationWait(t *testing.T, ctx context.Context, channel <-chan struct{}) {
	t.Helper()
	select {
	case <-channel:
	case <-ctx.Done():
		t.Fatal("deterministic fixture boundary was not reached")
	}
}

type migrationLifecycleModel struct {
	started, cancelling, release, joined chan struct{}
	active, calls                        atomic.Int64
}

func (*migrationLifecycleModel) Name() string { return "migration-history-model" }

func (m *migrationLifecycleModel) GenerateContent(ctx context.Context, _ *model.LLMRequest, _ bool) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		if m.calls.Add(1) != 1 {
			yield(nil, errors.New("unexpected lifecycle model retry"))
			return
		}
		child, cancel := context.WithCancel(ctx)
		m.active.Add(1)
		go func() {
			close(m.started)
			<-child.Done()
			close(m.cancelling)
			<-m.release
			m.active.Add(-1)
			close(m.joined)
		}()
		defer func() { cancel(); <-m.joined }()
		response, _ := textStep("Partial fixture response")(nil)
		response.Partial = true
		if !yield(response, nil) {
			return
		}
		<-ctx.Done()
		yield(nil, ctx.Err())
	}
}
