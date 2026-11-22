package streamline

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"iter"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/planner"
	"github.com/grauwolf32/contractor/internal/stageresult"
	"google.golang.org/adk/v2/model"
	adksession "google.golang.org/adk/v2/session"
	"google.golang.org/genai"
)

const compactorPolicyID = "compactor-test"

func compactionInvocation(profile string) planner.Invocation {
	invocation := testInvocation("builder")
	invocation.Stage.Planner = resolved.PlannerRef{PlannerID: profile, Version: "1"}
	main := invocation.ModelAccess.ModelPolicy
	main.ContextWindowTokens, main.MaxOutputTokens = 4096, 512
	invocation.ModelAccess.ModelPolicy = main
	compactor := contracts.ResolvedModelPolicy{
		Ref: contracts.ModelPolicyRef{
			PolicyID: compactorPolicyID, Version: "1", Digest: "sha256:" + strings.Repeat("e", 64),
		},
		Model: "fake-compactor", ContextWindowTokens: 131_072, MaxOutputTokens: 1024,
		MaxModelCalls: 1, MaxTotalTokens: 140_000,
	}
	invocation.Stage.PlannerContextManagement = &resolved.PlannerContextManagement{
		Strategy: resolved.PlannerContextCompact,
		Instructions: contracts.ResolvedInstructions{
			Ref: "instructions/compact.md", Text: "Keep decisions and open work.",
			Digest: "sha256:" + strings.Repeat("d", 64),
		},
		ContextWindowRatio: 0.75, KeepRecentGroups: 1, MaxCompactions: 8,
	}
	invocation.Stage.ExecutionConfig.Planner = &resolved.ResolvedConsumerExecutionConfig{
		ModelPolicy: main, CompactorModelPolicy: &compactor,
		Origins: resolved.ExecutionConfigOrigins{CompactorModelPolicy: "workflow.executionConfig.planner"},
	}
	return invocation
}

type compactorStep func(*model.LLMRequest) (compactorReply, error)

// scriptedCompactor is the test compactor client. Its ordinary
// GenerateContent path is never used by the Planner.
type scriptedCompactor struct {
	mu       sync.Mutex
	steps    []compactorStep
	requests []string
}

func (*scriptedCompactor) Name() string { return "fake-compactor" }

func (*scriptedCompactor) GenerateContent(
	context.Context, *model.LLMRequest, bool,
) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		yield(nil, errors.New("compactor must use the compaction path"))
	}
}

func (c *scriptedCompactor) compact(_ context.Context, request *model.LLMRequest) (compactorReply, error) {
	encoded, _ := encodeChatRequest(request, "fake-compactor", 1024)
	c.mu.Lock()
	index := len(c.requests)
	c.requests = append(c.requests, string(encoded))
	var step compactorStep
	if index < len(c.steps) {
		step = c.steps[index]
	}
	c.mu.Unlock()
	if step == nil {
		return compactorReply{}, errors.New("unexpected compactor call")
	}
	return step(request)
}

func (c *scriptedCompactor) calls() []string {
	c.mu.Lock()
	defer c.mu.Unlock()
	return append([]string(nil), c.requests...)
}

func compactorText(text, finish string) compactorStep {
	return func(*model.LLMRequest) (compactorReply, error) {
		return compactorReply{
			text: text, finishReason: finish,
			usage: &genai.GenerateContentResponseUsageMetadata{
				PromptTokenCount: 50, CandidatesTokenCount: 10, TotalTokenCount: 60,
			},
		}, nil
	}
}

func summaryStep(summary string) compactorStep {
	return compactorText(`{"summary":"`+summary+`"}`, "stop")
}

func compactionModelFactory(main model.LLM, compactor model.LLM) InvocationModelFactory {
	return func(access planner.ModelAccess) (model.LLM, error) {
		if access.ModelPolicy.Ref.PolicyID == compactorPolicyID {
			if access.Recovery != nil {
				return nil, errors.New("compactor received a Gateway recovery participant")
			}
			return compactor, nil
		}
		return main, nil
	}
}

func withPrompt(step modelStep, prompt int32) modelStep {
	return func(request *model.LLMRequest) (*model.LLMResponse, error) {
		response, err := step(request)
		if response != nil {
			response.UsageMetadata = &genai.GenerateContentResponseUsageMetadata{
				PromptTokenCount: prompt, CandidatesTokenCount: 10, TotalTokenCount: prompt + 10,
			}
		}
		return response, err
	}
}

type requestLog struct {
	mu       sync.Mutex
	requests []string
}

func (l *requestLog) capture(step modelStep) modelStep {
	return func(request *model.LLMRequest) (*model.LLMResponse, error) {
		encoded, _ := json.Marshal(request.Contents)
		l.mu.Lock()
		l.requests = append(l.requests, string(encoded))
		l.mu.Unlock()
		return step(request)
	}
}

func (l *requestLog) at(index int) string {
	l.mu.Lock()
	defer l.mu.Unlock()
	if index >= len(l.requests) {
		return ""
	}
	return l.requests[index]
}

// sequenceWorkers returns one exact report revision per dispatch.
type sequenceWorkers struct {
	mu    sync.Mutex
	calls []string
}

func (w *sequenceWorkers) Invoke(
	_ context.Context, _ string, _ contracts.WorkerHandle, request contracts.StageContentRequest,
) (contracts.WorkerCompletion, error) {
	w.mu.Lock()
	defer w.mu.Unlock()
	w.calls = append(w.calls, request.SubtaskID)
	revision := fmt.Sprintf("report-r%d", len(w.calls))
	return workerCompletion(request.SubtaskID, "validated result "+revision, map[string]contracts.ArtifactRef{
		"review": exactRef("builder", "review", revision),
	}), nil
}

func (w *sequenceWorkers) count() int {
	w.mu.Lock()
	defer w.mu.Unlock()
	return len(w.calls)
}

func compactionInspector() *fakeInspector {
	return &fakeInspector{mediaTypes: map[string]string{
		refKey(exactRef("builder", "review", "report-r1")): "application/json",
		refKey(exactRef("builder", "review", "report-r2")): "application/json",
	}}
}

func executeArgs(profile, subtaskID string) map[string]any {
	args := map[string]any{"subtask_id": subtaskID}
	if profile == "router" {
		args["worker_name"] = "builder"
	}
	return args
}

func historyArgsFor(profile string) map[string]any {
	args := map[string]any{}
	if profile == "router" {
		args["worker_name"] = "builder"
	}
	return args
}

func finishSelectedMaterial() modelStep {
	return functionStep(finishToolName, map[string]any{
		"outcome": "succeeded", "summary": "Reviewed report",
		"artifacts": map[string]any{"report": materialArgs("builder", "review")},
	})
}

// twoCompactionSteps drives two checkpoints: three main turns reach the
// threshold, the Planner reads the compacted exact result back, dispatches
// once more, compacts again and finishes with the old exact reference.
func twoCompactionSteps(profile string, log *requestLog) []modelStep {
	return []modelStep{
		log.capture(withPrompt(addSubtaskStep("first-objective", "first-instructions"), 100)),
		log.capture(withPrompt(functionStep(executeCurrentSubtaskToolName, executeArgs(profile, "0")), 200)),
		log.capture(withPrompt(addSubtaskStep("second-objective", "second-instructions"), 3500)),
		log.capture(withPrompt(functionStep(readContextHistoryToolName, historyArgsFor(profile)), 500)),
		log.capture(withPrompt(functionStep(executeCurrentSubtaskToolName, executeArgs(profile, "1")), 3600)),
		log.capture(withPrompt(finishSelectedMaterial(), 400)),
	}
}

func compactionFactory(
	t *testing.T, profile string, sessions *fakeSessions, workers planner.WorkerInvoker,
	main model.LLM, compactor model.LLM,
) *Factory {
	t.Helper()
	create := NewConfiguredFactory
	if profile == "router" {
		create = NewConfiguredRouterDelegate
	}
	factory, err := create(
		sessions, sessions, workers, compactionInspector(), unavailableWorkerStateReader{},
		compactionModelFactory(main, compactor), Limits{MaxWallTime: time.Minute},
	)
	if err != nil {
		t.Fatal(err)
	}
	return factory
}

func assertCompactionReleased(t *testing.T, instance planner.Planner) {
	t.Helper()
	if instance.(*streamlinePlanner).compaction != nil {
		t.Fatal("Planner retained its raw compaction history and index after Run returned")
	}
}

func TestCompactionContinuesSameInvocationAcrossTwoCheckpoints(t *testing.T) {
	for _, profile := range []string{"streamline", "router"} {
		t.Run(profile, func(t *testing.T) {
			log := &requestLog{}
			llm := &scriptedModel{steps: twoCompactionSteps(profile, log)}
			compactor := &scriptedCompactor{steps: []compactorStep{summaryStep("summary-one"), summaryStep("summary-two")}}
			workers := &sequenceWorkers{}
			sessions := newFakeSessions()
			instance, err := compactionFactory(t, profile, sessions, workers, llm, compactor).Create(compactionInvocation(profile))
			if err != nil {
				t.Fatal(err)
			}
			result, err := instance.Run(t.Context())
			if err != nil {
				t.Fatal(err)
			}
			assertCompactionReleased(t, instance)
			if result.Outcome != stageresult.StageSucceeded || refKey(result.Artifacts["report"]) != "builder/review@report-r2" {
				t.Fatalf("result = %+v", result)
			}
			if llm.callCount() != 6 || workers.count() != 2 || len(compactor.calls()) != 2 || sessions.adkCalls != 1 {
				t.Fatalf("main/workers/compactor/sessions = %d/%d/%d/%d", llm.callCount(), workers.count(), len(compactor.calls()), sessions.adkCalls)
			}
			first, second := compactor.calls()[0], compactor.calls()[1]
			if !strings.Contains(first, "first-objective") || !strings.Contains(first, "call-execute_current_subtask") ||
				strings.Contains(first, `"tools"`) || !strings.Contains(first, "Keep decisions and open work.") {
				t.Fatal("first compactor request lacks its source groups or carries tools")
			}
			// The second cycle summarizes the previous summary and the uncovered
			// groups only, never the earlier raw prefix or the recent tail.
			if !strings.Contains(second, "summary-one") || !strings.Contains(second, "call-read_context_history") ||
				strings.Contains(second, "call-execute_current_subtask") {
				t.Fatal("second compactor request does not replace only the uncovered prefix")
			}
			// After the first checkpoint the old exact result left the prompt;
			// the history reader returned it unchanged in the next request.
			afterFirst, afterRead, afterSecond := log.at(3), log.at(4), log.at(5)
			for _, wanted := range []string{"Produce a reviewed report", "summary-one", "second-objective"} {
				if !strings.Contains(afterFirst, wanted) {
					t.Fatalf("projection lacks %q", wanted)
				}
			}
			if strings.Contains(afterFirst, "call-execute_current_subtask") || strings.Contains(afterFirst, "report-r1") {
				t.Fatal("covered groups remained in the projected prompt")
			}
			if !strings.Contains(afterRead, `report-r1`) || !strings.Contains(afterRead, "validated result report-r1") {
				t.Fatal("history reader did not return the exact old result")
			}
			if !strings.Contains(afterSecond, "summary-two") || strings.Contains(afterSecond, "summary-one") ||
				!strings.Contains(afterSecond, "Produce a reviewed report") || !strings.Contains(afterSecond, "report-r2") {
				t.Fatal("second projection lost its protected context, rolling summary or recent tail")
			}
			report, ok := instance.(planner.ReportProvider).ExecutionReport()
			if !ok || *report.Metrics.ModelCalls != 8 ||
				*report.Metrics.InputTokens != 100+200+3500+500+3600+400+2*50 {
				t.Fatalf("report model calls/input tokens = %d/%d", *report.Metrics.ModelCalls, *report.Metrics.InputTokens)
			}
			if sessions.completion == nil || sessions.completion.Result == nil {
				t.Fatal("candidate completion was not recorded")
			}
			if sessions.plan == nil || sessions.plan.Revision == 0 || len(sessions.plan.Subtasks) != 2 {
				t.Fatal("plan authority changed across checkpoints")
			}
		})
	}
}

func TestCompactionFailuresStopBeforeFurtherEffects(t *testing.T) {
	for _, test := range []struct {
		name      string
		compactor []compactorStep
		calls     int
		mutate    func(*planner.Invocation)
		steps     func(profile string, log *requestLog) []modelStep
		code      string
		retryable bool
		main      int
		workers   int
		compacts  int
		tokens    int64
	}{
		{
			name: "malformed", compactor: []compactorStep{compactorText(`{"summary":"x","extra":true}`, "stop")},
			code: "planner_compaction_failed", main: 3, workers: 1, compacts: 1, tokens: 3800 + 50,
		},
		{
			name: "truncated", compactor: []compactorStep{compactorText(`{"summary":"x"}`, "length")},
			code: "planner_compaction_failed", main: 3, workers: 1, compacts: 1, tokens: 3800 + 50,
		},
		{
			name: "missing-usage",
			compactor: []compactorStep{func(*model.LLMRequest) (compactorReply, error) {
				return compactorReply{}, &gatewayResponseRejected{reason: "invalid token usage"}
			}},
			code: "planner_gateway_invalid_response", retryable: true, main: 3, workers: 1, compacts: 1, tokens: 3800,
		},
		{
			name: "model-call-limit", compactor: []compactorStep{summaryStep("unused")},
			mutate: func(invocation *planner.Invocation) { invocation.ModelAccess.ModelPolicy.MaxModelCalls = 3 },
			code:   "planner_model_call_limit", retryable: true, main: 3, workers: 1, compacts: 0, tokens: 3800,
		},
		{
			name: "input-capacity", compactor: []compactorStep{summaryStep("unused")},
			mutate: func(invocation *planner.Invocation) {
				compactor := invocation.Stage.ExecutionConfig.Planner.CompactorModelPolicy
				compactor.ContextWindowTokens, compactor.MaxOutputTokens, compactor.MaxTotalTokens = 2048, 1000, 2048
			},
			code: "planner_compaction_failed", main: 3, workers: 1, compacts: 0, tokens: 3800,
		},
		{
			name: "postcommit-ineffective", compactor: []compactorStep{summaryStep("summary-one")},
			steps: func(profile string, log *requestLog) []modelStep {
				steps := twoCompactionSteps(profile, log)[:3]
				return append(steps, withPrompt(functionStep(executeCurrentSubtaskToolName, executeArgs(profile, "1")), 3500))
			},
			code: "planner_compaction_failed", main: 4, workers: 1, compacts: 1, tokens: 3800 + 50 + 3500,
		},
		{
			name: "post-checkpoint-tail", compactor: []compactorStep{summaryStep("summary-one")},
			steps: func(profile string, log *requestLog) []modelStep {
				steps := twoCompactionSteps(profile, log)[:3]
				return append(steps, withPrompt(textStep(strings.Repeat("x", maxCompactionTailBytes)), 500))
			},
			code: "planner_compaction_limit_exceeded", main: 4, workers: 1, compacts: 1, tokens: 3800 + 50 + 500,
		},
		{
			name: "cycle-limit", compactor: []compactorStep{summaryStep("summary-one")},
			mutate: func(invocation *planner.Invocation) { invocation.Stage.PlannerContextManagement.MaxCompactions = 1 },
			code:   "planner_compaction_limit_exceeded", main: 5, workers: 2, compacts: 1, tokens: 7900 + 50,
		},
	} {
		for _, profile := range []string{"streamline", "router"} {
			t.Run(test.name+"/"+profile, func(t *testing.T) {
				log := &requestLog{}
				steps := twoCompactionSteps(profile, log)
				if test.steps != nil {
					steps = test.steps(profile, log)
				}
				llm := &scriptedModel{steps: steps}
				compactor := &scriptedCompactor{steps: test.compactor}
				workers := &sequenceWorkers{}
				sessions := newFakeSessions()
				invocation := compactionInvocation(profile)
				if test.mutate != nil {
					test.mutate(&invocation)
				}
				instance, err := compactionFactory(t, profile, sessions, workers, llm, compactor).Create(invocation)
				if err != nil {
					t.Fatal(err)
				}
				result, err := instance.Run(t.Context())
				assertPlannerCode(t, err, test.code)
				assertCompactionReleased(t, instance)
				if planner.FailureFrom(err).Retryable != test.retryable || result.Outcome != "" {
					t.Fatalf("retryable=%t outcome=%q", planner.FailureFrom(err).Retryable, result.Outcome)
				}
				if llm.callCount() != test.main || workers.count() != test.workers || len(compactor.calls()) != test.compacts {
					t.Fatalf("main/workers/compactor = %d/%d/%d", llm.callCount(), workers.count(), len(compactor.calls()))
				}
				report, _ := instance.(planner.ReportProvider).ExecutionReport()
				if got := *report.Metrics.InputTokens; got != test.tokens {
					t.Fatalf("charged input tokens = %d, want %d", got, test.tokens)
				}
				wantCalls := int64(test.main + test.compacts)
				if *report.Metrics.ModelCalls != wantCalls {
					t.Fatalf("charged model calls = %d, want %d", *report.Metrics.ModelCalls, wantCalls)
				}
				if sessions.completion == nil || sessions.completion.Failure == nil ||
					sessions.completion.Failure.Code != test.code || sessions.completion.Result != nil {
					t.Fatal("failure completion was not recorded")
				}
			})
		}
	}
}

func TestCompactionCancellationJoinsCompactorBeforeReturn(t *testing.T) {
	log := &requestLog{}
	blocking := &blockingCompactor{
		scriptedCompactor: &scriptedCompactor{}, waiting: make(chan struct{}), exited: make(chan struct{}),
	}
	llm := &scriptedModel{steps: twoCompactionSteps("streamline", log)}
	workers := &sequenceWorkers{}
	instance, err := compactionFactory(t, "streamline", newFakeSessions(), workers, llm, blocking).Create(compactionInvocation("streamline"))
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	done := make(chan error, 1)
	go func() {
		_, runErr := instance.Run(ctx)
		done <- runErr
	}()
	<-blocking.waiting
	cancel()
	runErr := <-done
	select {
	case <-blocking.exited:
	default:
		t.Fatal("Planner returned before its compactor call joined")
	}
	assertPlannerCode(t, runErr, "planner_cancelled")
	assertCompactionReleased(t, instance)
	if llm.callCount() != 3 || workers.count() != 1 {
		t.Fatalf("main/workers after cancellation = %d/%d", llm.callCount(), workers.count())
	}
}

// blockingCompactor blocks its first call until the caller cancels.
type blockingCompactor struct {
	*scriptedCompactor
	waiting chan struct{}
	exited  chan struct{}
}

func (c *blockingCompactor) compact(ctx context.Context, _ *model.LLMRequest) (compactorReply, error) {
	close(c.waiting)
	<-ctx.Done()
	close(c.exited)
	return compactorReply{}, ctx.Err()
}

func TestCompactionOffKeepsToolsAndModelClients(t *testing.T) {
	var models int
	var declared []string
	llm := &scriptedModel{steps: []modelStep{func(request *model.LLMRequest) (*model.LLMResponse, error) {
		for _, group := range request.Config.Tools {
			for _, declaration := range group.FunctionDeclarations {
				declared = append(declared, declaration.Name)
			}
		}
		return functionStep(finishToolName, map[string]any{
			"outcome": "failed", "summary": "stop", "artifacts": map[string]any{},
			"error": map[string]any{"code": "stopped", "message": "stopped", "retryable": false},
		})(request)
	}}}
	factory, err := NewConfiguredFactory(
		newFakeSessions(), newFakeSessions(), &sequenceWorkers{}, compactionInspector(), unavailableWorkerStateReader{},
		func(planner.ModelAccess) (model.LLM, error) { models++; return llm, nil }, Limits{MaxWallTime: time.Minute},
	)
	if err != nil {
		t.Fatal(err)
	}
	invocation := testInvocation("builder")
	invocation.Stage.Result.Artifacts = map[string]resolved.ArtifactSlot{}
	instance, err := factory.Create(invocation)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := instance.Run(t.Context()); err != nil {
		t.Fatal(err)
	}
	if models != 1 || strings.Contains(strings.Join(declared, ","), readContextHistoryToolName) {
		t.Fatalf("none created %d model clients and declared %v", models, declared)
	}
}

func TestCompactionCreateRejectsIncompleteSelection(t *testing.T) {
	compactor := &scriptedCompactor{}
	for _, test := range []struct {
		name   string
		mutate func(*planner.Invocation)
		client model.LLM
	}{
		{name: "missing-window", mutate: func(invocation *planner.Invocation) {
			invocation.ModelAccess.ModelPolicy.ContextWindowTokens = 0
		}},
		{name: "missing-compactor", mutate: func(invocation *planner.Invocation) {
			invocation.Stage.ExecutionConfig.Planner.CompactorModelPolicy = nil
		}},
		{name: "compactor-tools", mutate: func(invocation *planner.Invocation) {
			invocation.Stage.ExecutionConfig.Planner.CompactorModelPolicy.MaxToolCalls = 1
		}},
		{name: "two-calls", mutate: func(invocation *planner.Invocation) {
			invocation.Stage.ExecutionConfig.Planner.CompactorModelPolicy.MaxModelCalls = 2
		}},
		{name: "ratio", mutate: func(invocation *planner.Invocation) {
			invocation.Stage.PlannerContextManagement.ContextWindowRatio = 1
		}},
		{name: "unsupported-client", client: &scriptedModel{}},
	} {
		t.Run(test.name, func(t *testing.T) {
			invocation := compactionInvocation("streamline")
			if test.mutate != nil {
				test.mutate(&invocation)
			}
			client := model.LLM(compactor)
			if test.client != nil {
				client = test.client
			}
			factory := compactionFactory(t, "streamline", newFakeSessions(), &sequenceWorkers{}, &scriptedModel{}, client)
			if _, err := factory.Create(invocation); err == nil {
				t.Fatal("incomplete compact@1 selection created a Planner")
			}
		})
	}
}

func TestCompactorReplyValidation(t *testing.T) {
	policy := compactionInvocation("streamline").Stage.ExecutionConfig.Planner.CompactorModelPolicy
	usage := &genai.GenerateContentResponseUsageMetadata{PromptTokenCount: 5, CandidatesTokenCount: 5, TotalTokenCount: 10}
	for _, test := range []struct {
		name   string
		reply  compactorReply
		reason string
	}{
		{"valid", compactorReply{text: `{"summary":"memory"}`, finishReason: "stop"}, ""},
		{"length", compactorReply{text: `{"summary":"memory"}`, finishReason: "length"}, compactionReasonOutputLimit},
		{"content-filter", compactorReply{text: "", finishReason: "content_filter"}, compactionReasonEmpty},
		{"missing-finish", compactorReply{text: `{"summary":"memory"}`}, compactionReasonInvalid},
		{"unknown-finish", compactorReply{text: `{"summary":"memory"}`, finishReason: "paused"}, compactionReasonInvalid},
		{"tool", compactorReply{text: `{"summary":"memory"}`, finishReason: "stop", toolCalls: 1}, compactionReasonToolOutput},
		{"tool-finish", compactorReply{finishReason: "tool_calls"}, compactionReasonToolOutput},
		{"empty", compactorReply{text: " ", finishReason: "stop"}, compactionReasonEmpty},
		{"blank-summary", compactorReply{text: `{"summary":"  "}`, finishReason: "stop"}, compactionReasonEmpty},
		{"prose", compactorReply{text: "memory", finishReason: "stop"}, compactionReasonInvalid},
		{"extra", compactorReply{text: `{"summary":"memory","extra":1}`, finishReason: "stop"}, compactionReasonInvalid},
		{"duplicate", compactorReply{text: `{"summary":"a","summary":"b"}`, finishReason: "stop"}, compactionReasonInvalid},
		{"missing-member", compactorReply{text: `{}`, finishReason: "stop"}, compactionReasonInvalid},
		{"oversized", compactorReply{text: `{"summary":"` + strings.Repeat("x", maxCompactionSummaryBytes+1) + `"}`, finishReason: "stop"}, compactionReasonOversized},
		{"at-limit", compactorReply{text: `{"summary":"` + strings.Repeat("x", maxCompactionSummaryBytes) + `"}`, finishReason: "stop"}, ""},
		{"ineffective", compactorReply{text: `{"summary":"` + strings.Repeat("x", 100) + `"}`, finishReason: "stop"}, compactionReasonIneffective},
		{"output-tokens", compactorReply{text: `{"summary":"memory"}`, finishReason: "stop", usage: &genai.GenerateContentResponseUsageMetadata{
			PromptTokenCount: 5, CandidatesTokenCount: 1025, TotalTokenCount: 1030,
		}}, compactionReasonOutputLimit},
	} {
		t.Run(test.name, func(t *testing.T) {
			if test.reply.usage == nil {
				test.reply.usage = usage
			}
			source := int64(100)
			if strings.HasSuffix(test.name, "limit") {
				source = 1 << 20
			}
			summary, failure := validateCompactorReply(test.reply, *policy, source)
			if test.reason == "" {
				if failure != nil || summary == "" {
					t.Fatalf("valid reply rejected: %v", failure)
				}
				return
			}
			if failure == nil || failure.Code != "planner_compaction_failed" || failure.Retryable ||
				!strings.Contains(failure.Message, "("+test.reason+")") || summary != "" {
				t.Fatalf("failure = %v, want reason %s", failure, test.reason)
			}
		})
	}
}

func TestCompactorAdmissionCountsSerializedRequestAndFraming(t *testing.T) {
	policy := contracts.ResolvedModelPolicy{
		Model: "fake-compactor", ContextWindowTokens: 4096, MaxOutputTokens: 1024, MaxModelCalls: 1, MaxTotalTokens: 8192,
	}
	request := func(messages, bytes int) *model.LLMRequest {
		contents := make([]*genai.Content, 0, messages)
		for range messages {
			contents = append(contents, genai.NewContentFromText(strings.Repeat("x", bytes), genai.RoleUser))
		}
		return &model.LLMRequest{Contents: contents}
	}
	if failure := admitCompactorRequest(request(1, 1500), policy, 0); failure != nil {
		t.Fatalf("fitting request rejected: %v", failure)
	}
	// 1024 framing + 1024 output reserve leave 2048 bytes for the request.
	if failure := admitCompactorRequest(request(1, 2048), policy, 0); failure == nil ||
		!strings.Contains(failure.Message, compactionReasonInputCapacity) {
		t.Fatalf("over-window request admitted: %v", failure)
	}
	// Nine messages add 128 framing tokens each.
	if failure := admitCompactorRequest(request(9, 100), policy, 0); failure == nil {
		t.Fatal("per-message framing was not reserved")
	}
	limited := policy
	limited.MaxTotalTokens = 2500
	if failure := admitCompactorRequest(request(1, 1000), limited, 0); failure == nil {
		t.Fatal("request above the compactor total was admitted")
	}
	if failure := admitCompactorRequest(request(1, 100), policy, maxCompactionPeakBytes); failure == nil ||
		failure.Code != "planner_compaction_limit_exceeded" {
		t.Fatalf("peak ledger was not enforced: %v", failure)
	}
}

func TestCompactionIngestionRejectsUnsafeScope(t *testing.T) {
	call := func(id string) *genai.Content {
		return &genai.Content{Role: genai.RoleModel, Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{ID: id, Name: "execute"}}}}
	}
	reply := func(id string) *genai.Content {
		return &genai.Content{Role: genai.RoleUser, Parts: []*genai.Part{{FunctionResponse: &genai.FunctionResponse{ID: id, Name: "execute"}}}}
	}
	text := genai.NewContentFromText("text", genai.RoleModel)
	for _, test := range []struct {
		name   string
		events []*adksession.Event
		valid  bool
	}{
		{name: "complete", valid: true, events: []*adksession.Event{
			{ID: "1", LLMResponse: model.LLMResponse{Content: call("a")}},
			{ID: "2", LLMResponse: model.LLMResponse{Content: reply("a")}},
			{ID: "3", LLMResponse: model.LLMResponse{Content: text}},
		}},
		{name: "interleaved", events: []*adksession.Event{
			{ID: "1", LLMResponse: model.LLMResponse{Content: call("a")}},
			{ID: "2", LLMResponse: model.LLMResponse{Content: call("b")}},
		}},
		{name: "orphan", events: []*adksession.Event{{ID: "1", LLMResponse: model.LLMResponse{Content: reply("a")}}}},
		{name: "incomplete", events: []*adksession.Event{
			{ID: "1", LLMResponse: model.LLMResponse{Content: call("a")}},
			{ID: "2", LLMResponse: model.LLMResponse{Content: text}},
		}},
		{name: "duplicate", events: []*adksession.Event{
			{ID: "1", LLMResponse: model.LLMResponse{Content: text}},
			{ID: "1", LLMResponse: model.LLMResponse{Content: text}},
		}},
		{name: "error", events: []*adksession.Event{{ID: "1", LLMResponse: model.LLMResponse{Content: text, ErrorCode: "x"}}}},
		{name: "branch", events: []*adksession.Event{{ID: "1", Branch: "other", LLMResponse: model.LLMResponse{Content: text}}}},
		{name: "isolation", events: []*adksession.Event{{ID: "1", IsolationScope: "other", LLMResponse: model.LLMResponse{Content: text}}}},
		{name: "native", events: []*adksession.Event{{ID: "1", LLMResponse: model.LLMResponse{Content: text}, Actions: adksession.EventActions{
			Compaction: &adksession.EventCompaction{CompactedContent: text},
		}}}},
	} {
		t.Run(test.name, func(t *testing.T) {
			compaction := newContextCompaction(compactionSettings{keepRecent: 1, maxCompactions: 1, threshold: 1})
			events := append([]*adksession.Event{{ID: "protected", Author: "user", LLMResponse: model.LLMResponse{
				Content: genai.NewContentFromText("protected", genai.RoleUser),
			}}}, test.events...)
			failure := compaction.ingestLocked(fixedSession(events))
			if test.valid {
				if failure != nil || len(compaction.groups) != 2 || compaction.openStart != -1 {
					t.Fatalf("complete groups rejected: %v groups=%d", failure, len(compaction.groups))
				}
				return
			}
			if failure == nil || !strings.Contains(failure.Message, compactionReasonScope) {
				t.Fatalf("unsafe scope accepted: %v", failure)
			}
		})
	}
}

type fixedEvents []*adksession.Event

func (e fixedEvents) All() iter.Seq[*adksession.Event] {
	return func(yield func(*adksession.Event) bool) {
		for _, event := range e {
			if !yield(event) {
				return
			}
		}
	}
}
func (e fixedEvents) Len() int                       { return len(e) }
func (e fixedEvents) At(index int) *adksession.Event { return e[index] }

type fixedSessionValue struct {
	adksession.Session
	events fixedEvents
}

func (s fixedSessionValue) Events() adksession.Events { return s.events }

func fixedSession(events []*adksession.Event) adksession.Session {
	return fixedSessionValue{events: events}
}

func TestCompactionHistoryPagesExactResultsWithinBounds(t *testing.T) {
	compaction := newContextCompaction(compactionSettings{keepRecent: 1, maxCompactions: 1, threshold: 1})
	large := contracts.WorkerResult{
		SubtaskID: "0", Result: strings.Repeat("я", contracts.MaxWorkerResultBytes/2-1),
		Artifacts: map[string]contracts.ArtifactRef{"report": exactRef("builder", "review", "report-r1")},
	}
	if failure := compaction.recordResult("builder", "0", large); failure != nil {
		t.Fatal(failure)
	}
	for index := 1; index <= 20; index++ {
		small := contracts.WorkerResult{SubtaskID: fmt.Sprint(index), Result: fmt.Sprintf("result-%d", index)}
		if failure := compaction.recordResult("builder", fmt.Sprint(index), small); failure != nil {
			t.Fatal(failure)
		}
	}
	if failure := compaction.recordResult("other", "21", contracts.WorkerResult{Result: "foreign"}); failure != nil {
		t.Fatal(failure)
	}
	recovered := map[int][]byte{}
	cursor, pages := "", 0
	for {
		page := compaction.page("session", "builder", cursor)
		if !page.OK || page.Error != nil || len(page.Items) == 0 || len(page.Items) > maxHistoryPageItems {
			t.Fatalf("page %d = %+v", pages, page.Error)
		}
		encoded, _ := json.Marshal(page)
		if len(encoded) > maxHistoryPageBytes {
			t.Fatalf("page encoded %d bytes", len(encoded))
		}
		for _, item := range page.Items {
			if len(item.Text) > maxHistoryFragmentBytes || item.Offset != len(recovered[item.Ordinal]) {
				t.Fatal("fragment exceeds its bound or skips bytes")
			}
			recovered[item.Ordinal] = append(recovered[item.Ordinal], item.Text...)
		}
		pages++
		if page.Next == "" {
			break
		}
		cursor = page.Next
	}
	if len(recovered) != 21 || pages < 3 {
		t.Fatalf("recovered %d entries in %d pages", len(recovered), pages)
	}
	for ordinal, data := range recovered {
		if string(data) != string(compaction.history.entries[ordinal-1].data) {
			t.Fatalf("entry %d was not recovered exactly", ordinal)
		}
	}
	if !strings.Contains(string(recovered[1]), `"revision":"report-r1"`) {
		t.Fatal("exact artifact reference lost")
	}
	if strings.Contains(string(recovered[1]), "foreign") {
		t.Fatal("another Worker's result leaked")
	}
}

func TestCompactionHistoryCursorsFailClosed(t *testing.T) {
	compaction := newContextCompaction(compactionSettings{keepRecent: 1, maxCompactions: 1, threshold: 1})
	for index := range 40 {
		result := contracts.WorkerResult{SubtaskID: fmt.Sprint(index), Result: strings.Repeat("r", 8*1024)}
		if failure := compaction.recordResult("builder", fmt.Sprint(index), result); failure != nil {
			t.Fatal(failure)
		}
	}
	first := compaction.page("session", "builder", "")
	if first.Next == "" {
		t.Fatal("first page issued no cursor")
	}
	invalid := func(page historyOutput) bool {
		return !page.OK && page.Error != nil && page.Error.Code == "history_cursor_invalid" && len(page.Items) == 0
	}
	for _, page := range []historyOutput{
		compaction.page("session", "other", first.Next),
		compaction.page("foreign", "builder", first.Next),
		compaction.page("session", "builder", "not-a-cursor"),
	} {
		if !invalid(page) {
			t.Fatal("foreign or invalid cursor returned data")
		}
	}
	second := compaction.page("session", "builder", first.Next)
	if !second.OK || len(second.Items) == 0 {
		t.Fatal("valid cursor rejected")
	}
	if !invalid(compaction.page("session", "builder", first.Next)) {
		t.Fatal("consumed cursor was accepted twice")
	}
	// A new index revision invalidates every outstanding cursor.
	if failure := compaction.recordResult("builder", "late", contracts.WorkerResult{Result: "late"}); failure != nil {
		t.Fatal(failure)
	}
	if !invalid(compaction.page("session", "builder", second.Next)) {
		t.Fatal("stale index cursor was accepted")
	}
	// A checkpoint generation invalidates every outstanding cursor.
	current := compaction.page("session", "builder", "")
	compaction.generation++
	if !invalid(compaction.page("session", "builder", current.Next)) {
		t.Fatal("stale generation cursor was accepted")
	}
	// At most eight chains stay outstanding; the oldest is evicted.
	var chains []string
	for range maxHistoryCursors + 1 {
		chains = append(chains, compaction.page("session", "builder", "").Next)
	}
	if len(compaction.history.cursors) != maxHistoryCursors || !invalid(compaction.page("session", "builder", chains[0])) {
		t.Fatal("cursor chains are unbounded")
	}
	if page := compaction.page("session", "builder", chains[len(chains)-1]); !page.OK {
		t.Fatal("newest chain was evicted")
	}
}

func TestCompactionHistoryEvictsOnlyLiveCursors(t *testing.T) {
	compaction := newContextCompaction(compactionSettings{})
	for index := range 40 {
		if failure := compaction.recordResult("builder", fmt.Sprint(index), contracts.WorkerResult{
			SubtaskID: fmt.Sprint(index), Result: strings.Repeat("r", 8*1024),
		}); failure != nil {
			t.Fatal(failure)
		}
	}
	first := compaction.page("session", "builder", "")
	advancing := compaction.page("session", "builder", "")
	for range maxHistoryCursors + 1 {
		if advancing.Next == "" {
			t.Fatal("fixture ended before enough cursor advances")
		}
		advancing = compaction.page("session", "builder", advancing.Next)
		if !advancing.OK {
			t.Fatalf("advancing chain rejected: %+v", advancing.Error)
		}
	}
	// Rejected foreign use must not consume a valid outstanding cursor.
	if page := compaction.page("foreign", "builder", first.Next); page.OK {
		t.Fatal("foreign cursor accepted")
	}
	first = compaction.page("session", "builder", first.Next)
	if !first.OK || first.Next == "" {
		t.Fatalf("another chain evicted an unconsumed cursor below capacity: %+v", first.Error)
	}
	for range maxHistoryCursors - 2 {
		if page := compaction.page("session", "builder", ""); !page.OK || page.Next == "" {
			t.Fatal("could not fill live cursor capacity")
		}
	}
	if page := compaction.page("session", "builder", ""); !page.OK || page.Next == "" {
		t.Fatal("could not issue the ninth live cursor")
	}
	if page := compaction.page("session", "builder", advancing.Next); page.Error == nil || page.Error.Code != "history_cursor_invalid" {
		t.Fatal("ninth live cursor did not evict the oldest")
	}
	if page := compaction.page("session", "builder", first.Next); !page.OK {
		t.Fatalf("newer live cursor was evicted: %+v", page.Error)
	}
}

func TestCompactionRetainedLedgerStopsBeforeCeiling(t *testing.T) {
	compaction := newContextCompaction(compactionSettings{keepRecent: 1, maxCompactions: 1, threshold: 1})
	result := contracts.WorkerResult{Result: strings.Repeat("r", contracts.MaxWorkerResultBytes)}
	var failure *planner.Error
	recorded := 0
	for range 1024 {
		if failure = compaction.recordResult("builder", "0", result); failure != nil {
			break
		}
		recorded++
	}
	if failure == nil || failure.Code != "planner_compaction_limit_exceeded" || failure.Retryable {
		t.Fatalf("retained ceiling not enforced: %v", failure)
	}
	if retained := compaction.retainedLocked(); retained > maxCompactionRetainedBytes || recorded < 400 {
		t.Fatalf("retained=%d recorded=%d", retained, recorded)
	}
}
