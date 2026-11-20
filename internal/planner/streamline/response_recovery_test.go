package streamline

import (
	"context"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/planner"
	"github.com/grauwolf32/contractor/internal/stageresult"
	"github.com/grauwolf32/contractor/internal/telemetry"
	collectortracev1 "go.opentelemetry.io/proto/otlp/collector/trace/v1"
	"google.golang.org/adk/v2/model"
	"google.golang.org/genai"
	"google.golang.org/protobuf/proto"
)

func TestDocumentPlannerRecoveryEnforcesFinish(t *testing.T) {
	for _, first := range []string{"empty", "length"} {
		for _, follow := range []string{"finish", "dispatch", "dispatch-over-budget", "mixed", "empty", "incomplete", "model-budget", "token-budget", "default", "missing-binding", "invalid-finish", "length", "cancelled"} {
			t.Run(first+"/"+follow, func(t *testing.T) {
				ctx, cancel := context.WithCancel(t.Context())
				defer cancel()
				invocation := testInvocation("builder")
				binding := invocation.Stage.Agents["builder"]
				binding.Template.ResponseRecovery = "document@1"
				if follow == "default" {
					binding.Template.ResponseRecovery = ""
				}
				invocation.Stage.Agents["builder"] = binding
				ref := exactRef("builder", "report", "report-r1")
				workers := &fakeWorkerInvoker{results: map[string]stageresult.StageContentResult{"builder": stageResult("done", map[string]contracts.ArtifactRef{"report": ref})}}
				finish := functionStep(finishToolName, map[string]any{"outcome": "succeeded", "summary": "complete", "artifacts": map[string]any{"report": artifactArgs("builder", "report", "report-r1")}})
				followup := finish
				switch follow {
				case "invalid-finish":
					followup = functionStep(finishToolName, map[string]any{"outcome": "succeeded", "summary": "complete", "artifacts": map[string]any{}})
				case "dispatch", "dispatch-over-budget":
					followup = functionStep(executeCurrentSubtaskToolName, map[string]any{"subtask_id": "0"})
				case "mixed":
					followup = func(r *model.LLMRequest) (*model.LLMResponse, error) {
						v, e := finish(r)
						v.Content.Parts = append(v.Content.Parts, genai.NewPartFromFunctionCall(executeCurrentSubtaskToolName, map[string]any{"subtask_id": "0"}))
						return v, e
					}
				case "empty", "length":
					followup = func(*model.LLMRequest) (*model.LLMResponse, error) {
						return nil, &gatewayResponseRejected{emptyStop: follow == "empty", outputLimited: follow == "length", usage: &genai.GenerateContentResponseUsageMetadata{PromptTokenCount: 7, CandidatesTokenCount: 3, TotalTokenCount: 10}}
					}
				}
				scripted := &scriptedModel{steps: []modelStep{
					addSubtaskStep("Review", "Read documents"), functionStep(executeCurrentSubtaskToolName, map[string]any{"subtask_id": "0"}),
					func(*model.LLMRequest) (*model.LLMResponse, error) {
						if follow == "cancelled" {
							cancel()
						}
						return nil, &gatewayResponseRejected{emptyStop: first == "empty", outputLimited: first == "length", usage: &genai.GenerateContentResponseUsageMetadata{PromptTokenCount: 7, CandidatesTokenCount: 3, TotalTokenCount: 10}}
					},
					followup,
				}}
				if follow == "incomplete" {
					scripted.steps = []modelStep{addSubtaskStep("Pending", "Must finish"), scripted.steps[2]}
				}
				if follow == "model-budget" || follow == "token-budget" {
					invocation.ModelAccess.ModelPolicy.MaxModelCalls = 3
					if follow == "token-budget" {
						invocation.ModelAccess.ModelPolicy.MaxTotalTokens = 45
					}
				}
				if follow == "missing-binding" {
					workers.results["builder"] = stageResult("done", map[string]contracts.ArtifactRef{})
				}
				var adapter telemetry.PlannerTelemetry
				var payloads <-chan []byte
				if follow == "dispatch-over-budget" {
					invocation.ModelAccess.ModelPolicy.MaxTotalTokens = 60
					adapter, payloads = streamlineTestTelemetry(t, invocation)
					invocation.Instrumentation = adapter.Instrumentation()
				}
				inspector := &fakeInspector{mediaTypes: map[string]string{refKey(ref): "application/json"}}
				instance, err := mustFactory(t, newFakeSessions(), workers, inspector, scripted, Limits{}).Create(invocation)
				if err != nil {
					t.Fatal(err)
				}
				_, err = instance.Run(ctx)
				wantDispatches, wantCalls := 1, 4
				if follow == "incomplete" {
					wantDispatches, wantCalls = 0, 2
				}
				if follow == "model-budget" || follow == "token-budget" || follow == "default" || follow == "missing-binding" || follow == "cancelled" {
					wantCalls = 3
				}
				if (err == nil) != (follow == "finish") || len(workers.calls) != wantDispatches || scripted.callCount() != wantCalls {
					t.Fatalf("error=%v dispatches=%d calls=%d", err, len(workers.calls), scripted.callCount())
				}
				if follow == "cancelled" && planner.FailureFrom(err).Code != "planner_cancelled" {
					t.Fatalf("cancellation=%v", err)
				}
				if follow == "dispatch-over-budget" {
					assertPlannerCode(t, err, "planner_token_limit")
					checkRecoveryBudgetTrace(t, adapter, payloads)
				}
				report, _ := instance.(planner.ReportProvider).ExecutionReport()
				wantTokens := int64(70)
				switch follow {
				case "empty", "length":
					wantTokens = 60
				case "incomplete":
					wantTokens = 30
				case "model-budget", "token-budget", "default", "missing-binding", "cancelled":
					wantTokens = 50
				}
				if report.Metrics.TotalTokens == nil || *report.Metrics.TotalTokens != wantTokens {
					t.Fatalf("tokens=%+v, want %d", report.Metrics, wantTokens)
				}
				if report.Metrics.ModelCalls == nil || *report.Metrics.ModelCalls != int64(wantCalls) {
					t.Fatalf("usage=%+v", report.Metrics)
				}
			})
		}
	}
}

func checkRecoveryBudgetTrace(t *testing.T, adapter telemetry.PlannerTelemetry, payloads <-chan []byte) {
	t.Helper()
	if export := adapter.Flush(t.Context()); !export.Succeeded {
		t.Fatalf("flush: %+v", export)
	}
	var trace collectortracev1.ExportTraceServiceRequest
	if err := proto.Unmarshal(<-payloads, &trace); err != nil {
		t.Fatal(err)
	}
	modelSpans, budgetFailures := 0, 0
	for _, span := range trace.ResourceSpans[0].ScopeSpans[0].Spans {
		if span.Name != string(telemetry.PlannerSpanModel) {
			continue
		}
		modelSpans++
		var code string
		var input, output int64
		for _, attr := range span.Attributes {
			switch attr.Key {
			case "error.type":
				code = attr.Value.GetStringValue()
			case "gen_ai.usage.input_tokens":
				input = attr.Value.GetIntValue()
			case "gen_ai.usage.output_tokens":
				output = attr.Value.GetIntValue()
			}
		}
		if code == "planner_token_limit" && input == 10 && output == 10 {
			budgetFailures++
		}
	}
	if modelSpans != 4 || budgetFailures != 1 {
		t.Fatalf("exported %d model spans, %d charged budget failures", modelSpans, budgetFailures)
	}
}
