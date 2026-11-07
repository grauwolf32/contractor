package streamline

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/grauwolf32/contractor/internal/planner"
	"github.com/grauwolf32/contractor/internal/stageresult"
	"google.golang.org/adk/v2/model"
)

// This runs the real main and auxiliary ADK Runners and the real Gateway
// adapter. Only the remote model and Worker results are controlled fixtures.
func TestCompactorAgentStructuredOutputHTTP(t *testing.T) {
	for _, profile := range []string{"streamline", "router"} {
		for _, outcome := range []string{"valid", "invalid", "tool-call", "truncated", "schema-rejected"} {
			t.Run(profile+"/"+outcome, func(t *testing.T) {
				var calls atomic.Int32
				server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					call := calls.Add(1)
					var payload chatRequest
					if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
						t.Error(err)
						http.Error(w, "invalid request", http.StatusBadRequest)
						return
					}
					want := `{"type":"json_schema","json_schema":{"name":"response","strict":true,"schema":{"additionalProperties":false,"properties":{"summary":{"type":"string"}},"required":["summary"],"type":"object"}}}`
					encoded, _ := json.Marshal(payload.ResponseFormat)
					if string(encoded) != want || len(payload.Tools) != 0 || payload.ToolChoice != nil {
						t.Errorf("auxiliary ADK request: format=%s tools=%+v choice=%v", encoded, payload.Tools, payload.ToolChoice)
					}
					if payload.Model != "fake-compactor" || payload.MaxTokens != 1024 || len(payload.Messages) < 2 ||
						!strings.Contains(payload.Messages[0].Content, "Keep decisions and open work.") {
						t.Errorf("auxiliary model/config/instruction = %+v", payload)
					}
					if outcome == "schema-rejected" {
						http.Error(w, `{"error":{"message":"unsupported response format"}}`, http.StatusBadRequest)
						return
					}
					text := `{"summary":"summary-one"}`
					if call == 2 {
						text = `{"summary":"summary-two"}`
					}
					finish := "stop"
					message := map[string]any{"content": text}
					switch outcome {
					case "invalid":
						message["content"] = "Here is the summary: " + text
					case "truncated":
						finish = "length"
					case "tool-call":
						finish = "tool_calls"
						message["tool_calls"] = []any{map[string]any{
							"id": "forbidden", "type": "function", "function": map[string]any{"name": "execute_current_subtask", "arguments": "{}"},
						}}
					}
					w.Header().Set("Content-Type", "application/json")
					_ = json.NewEncoder(w).Encode(map[string]any{
						"choices": []any{map[string]any{"finish_reason": finish, "message": message}},
						"usage":   map[string]any{"prompt_tokens": 50, "completion_tokens": 10, "total_tokens": 60},
					})
				}))
				defer server.Close()
				compactor, err := NewOpenAICompatibleModel(GatewaySettings{
					URL: server.URL, Model: "fake-compactor", MaxOutputTokens: 1024, HTTPClient: server.Client(),
				})
				if err != nil {
					t.Fatal(err)
				}
				log := &requestLog{}
				main := &scriptedModel{steps: twoCompactionSteps(profile, log)}
				workers, sessions := &sequenceWorkers{}, newFakeSessions()
				instance, err := compactionFactory(t, profile, sessions, workers, main, compactor).Create(compactionInvocation(profile))
				if err != nil {
					t.Fatal(err)
				}
				result, err := instance.Run(t.Context())
				report, _ := instance.(planner.ReportProvider).ExecutionReport()
				generation := instance.(*streamlinePlanner).compaction.generation
				if outcome == "valid" {
					if err != nil || result.Outcome != stageresult.StageSucceeded || calls.Load() != 2 || generation != 2 ||
						main.callCount() != 6 || workers.count() != 2 || *report.Metrics.ModelCalls != 8 {
						t.Fatalf("result=%+v err=%v calls=%d generation=%d main=%d workers=%d report=%+v", result, err, calls.Load(), generation, main.callCount(), workers.count(), report.Metrics)
					}
				} else {
					code, inputTokens := "planner_compaction_failed", int64(3850)
					if outcome == "schema-rejected" {
						code, inputTokens = "planner_gateway_rejected", 3800
					}
					assertPlannerCode(t, err, code)
					if calls.Load() != 1 || generation != 0 || main.callCount() != 3 || workers.count() != 1 ||
						*report.Metrics.ModelCalls != 4 || *report.Metrics.InputTokens != inputTokens {
						t.Fatalf("failure retried, published or mischarged: calls=%d generation=%d main=%d workers=%d metrics=%+v", calls.Load(), generation, main.callCount(), workers.count(), report.Metrics)
					}
				}
			})
		}
	}
}

func TestCompactorAgentAdmitsActualADKRequest(t *testing.T) {
	invocation := compactionInvocation("streamline")
	compactor := &scriptedCompactor{steps: []compactorStep{summaryStep("unused")}}
	instance, err := compactionFactory(t, "streamline", newFakeSessions(), &sequenceWorkers{}, &scriptedModel{}, compactor).Create(invocation)
	if err != nil {
		t.Fatal(err)
	}
	p := instance.(*streamlinePlanner)
	p.compaction = newContextCompaction(*p.compactionSettings)
	var checked bool
	_, err = p.runCompactor(t.Context(), compactionSelection{}, func(request *model.LLMRequest) error {
		checked = true
		if request.Config == nil || request.Config.ResponseSchema == nil || request.Config.ResponseMIMEType != "application/json" {
			t.Fatal("ADK did not populate the output schema before admission")
		}
		policy := p.compaction.settings.compactor
		withoutSchema := *request.Config
		withoutSchema.ResponseSchema, withoutSchema.ResponseMIMEType = nil, ""
		plain := *request
		plain.Config = &withoutSchema
		payload, err := encodeChatRequest(&plain, policy.Model, policy.MaxOutputTokens)
		if err != nil {
			t.Fatal(err)
		}
		policy.ContextWindowTokens = len(payload) + compactorFramingTokens + policy.MaxOutputTokens
		if failure := admitCompactorRequest(&plain, policy, 0); failure != nil {
			t.Fatalf("schema-free boundary should fit: %v", failure)
		}
		failure := admitCompactorRequest(request, policy, 0)
		if failure == nil || !strings.Contains(failure.Message, compactionReasonInputCapacity) {
			t.Fatalf("schema bytes were not admitted: %v", failure)
		}
		return failure
	})
	if !checked || len(compactor.calls()) != 0 {
		t.Fatalf("admission checked=%v provider calls=%d err=%v", checked, len(compactor.calls()), err)
	}
	assertPlannerCode(t, err, "planner_compaction_failed")
}

func TestOneShotCompactorPreventsSecondProviderCall(t *testing.T) {
	delegate := &scriptedCompactor{steps: []compactorStep{summaryStep("one")}}
	admissions := 0
	llm := &oneShotCompactor{delegate: delegate, before: func(*model.LLMRequest) error { admissions++; return nil }}
	for call := range 2 {
		var failure error
		for _, err := range llm.GenerateContent(t.Context(), &model.LLMRequest{}, false) {
			failure = err
		}
		if call == 0 && failure != nil {
			t.Fatal(failure)
		}
		if call == 1 {
			assertPlannerCode(t, failure, "planner_compaction_failed")
		}
	}
	if admissions != 1 || len(delegate.calls()) != 1 {
		t.Fatalf("admissions=%d provider calls=%d", admissions, len(delegate.calls()))
	}
}
