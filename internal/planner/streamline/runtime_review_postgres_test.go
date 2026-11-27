package streamline

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"
	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/planner"
	"github.com/grauwolf32/contractor/internal/runstore"
)

func TestPostgresRuntimePlannerTerminalPrecedence(t *testing.T) {
	for _, profile := range []plannerProfile{streamlineProfile, routerProfile} {
		for _, outcome := range []string{"succeeded", "failed"} {
			for _, late := range []string{"none", "cancel", "transport"} {
				for _, reject := range []bool{false, true} {
					t.Run(fmt.Sprintf("%s/%s/%s/reject=%t", profile.ref, outcome, late, reject), func(t *testing.T) {
						ctx, cancel := context.WithCancel(t.Context())
						defer cancel()
						pool := isolatedStreamlinePool(t, t.Context())
						store := runstore.NewPostgresStore(pool, artifactpgtest.NewStorage())
						createPreparingStreamlineStage(t, t.Context(), store)
						if reject {
							installRequiredEventGuard(t, t.Context(), pool, "finish", "invalid")
						}
						sessions := migrationBoundarySession(t, store)
						workers := appendProbeWorkers()
						args := map[string]any{"subtask_id": "0"}
						if profile.routesWorkers {
							args["worker_name"] = "builder"
						}
						finish := map[string]any{"outcome": outcome, "summary": "Complete", "artifacts": map[string]any{}}
						if outcome == "failed" {
							finish["error"] = map[string]any{"code": "final_failure", "message": "Final failed candidate", "retryable": false}
						}
						script := &operationScript{steps: []operationStep{addSubtaskStep("Inspect", "Read source"), functionStep(executeCurrentSubtaskToolName, args), functionStep(finishToolName, finish)}}
						f, err := runtimeScriptFactory(profile, sessions, workers, &fakeInspector{}, unavailableWorkerStateReader{}, script, Limits{})
						if err != nil {
							t.Fatal(err)
						}
						original := f.invoker
						f.invoker = runtimeInvokerFunc(func(call context.Context, name string, h contracts.WorkerHandle, r contracts.StageContentRequest) (contracts.WorkerCompletion, error) {
							result, err := original.Invoke(call, name, h, r)
							if h.WorkerRuntimeRef.IsPlanner() {
								if late == "cancel" {
									cancel()
								}
								if late == "transport" {
									return contracts.WorkerCompletion{}, errors.New("controlled final poll failure")
								}
							}
							return result, err
						})
						instance, err := f.Create(appendProbeInvocation(strings.TrimSuffix(profile.ref, "@1")))
						if err != nil {
							t.Fatal(err)
						}
						result, err := instance.Run(ctx)
						if reject {
							want := "planner_session_rejected"
							if late == "cancel" {
								want = "planner_cancelled"
							}
							assertPlannerCode(t, err, want)
							if planner.FailureFrom(err).Retryable {
								t.Fatal("required store rejection became retryable")
							}
						} else if err != nil || string(result.Outcome) != outcome {
							t.Fatalf("confirmed terminal outcome lost: %+v, %v", result, err)
						}
						// A fresh owner reads the durable completion without invoking any work.
						recovered, err := migrationBoundarySession(t, store).Begin(t.Context(), "stage-1")
						if err != nil || recovered.Invoke || recovered.Completion == nil {
							t.Fatalf("session not settled: %+v %v", recovered, err)
						}
						if reject && recovered.Completion.Result != nil {
							t.Fatal("rejected required append published candidate")
						}
						if !reject && (recovered.Completion.Result == nil || string(recovered.Completion.Result.Outcome) != outcome) {
							t.Fatalf("durable terminal outcome lost: %+v", recovered.Completion)
						}
						if got := len(workers.bindingCalls()); got != 1 {
							t.Fatalf("Worker calls=%d", got)
						}
					})
				}
			}
		}
	}
}

func TestPostgresRuntimePlannerOversizeContextCompletesSession(t *testing.T) {
	for _, profile := range []plannerProfile{streamlineProfile, routerProfile} {
		for _, kind := range []string{"input", "envelope"} {
			t.Run(profile.ref+"/"+kind, func(t *testing.T) {
				pool := isolatedStreamlinePool(t, t.Context())
				store := runstore.NewPostgresStore(pool, artifactpgtest.NewStorage())
				createPreparingStreamlineStage(t, t.Context(), store)
				inv := appendProbeInvocation(strings.TrimSuffix(profile.ref, "@1"))
				if kind == "input" {
					inv.Context.Parameters["large"] = strings.Repeat("x", maxPlannerContextBytes-600)
					b := inv.Stage.Agents["builder"]
					b.Template.Description = strings.Repeat("d", 512)
					inv.Stage.Agents["builder"] = b
				} else {
					// Stage instructions and protected prompt each fit their own bounds, but
					// JSON escaping and declarations make the final allocation envelope too big.
					inv = compactionInvocation(strings.TrimSuffix(profile.ref, "@1"))
					inv.Stage.Result.Artifacts = map[string]resolved.ArtifactSlot{}
					inv.Context.Parameters["large"] = strings.Repeat("<", 42000)
					inv.Stage.Instructions.Text = strings.Repeat("i", 210000)
					inv.Stage.PlannerContextManagement.Instructions.Text = strings.Repeat("🙂", 8000)
				}
				calls := 0
				f := runtimeTestFactory(t, strings.TrimSuffix(profile.ref, "@1"), migrationBoundarySession(t, store), runtimeInvokerFunc(func(context.Context, string, contracts.WorkerHandle, contracts.StageContentRequest) (contracts.WorkerCompletion, error) {
					calls++
					return contracts.WorkerCompletion{}, nil
				}))
				instance, err := f.Create(inv)
				if err != nil {
					t.Fatal(err)
				}
				if kind == "input" {
					if _, err := instance.(*streamlinePlanner).initialMessage(); err == nil {
						t.Fatal("fixture must exceed initial prompt bound")
					}
				}
				_, err = instance.Run(t.Context())
				want := "planner_context_invalid"
				if kind == "envelope" {
					want = "invalid_stage_content"
				}
				assertPlannerCode(t, err, want)
				if planner.FailureFrom(err).Retryable || calls != 0 {
					t.Fatalf("invalid context retried/dispatched: %v, calls=%d", err, calls)
				}
				recovered, err := migrationBoundarySession(t, store).Begin(t.Context(), "stage-1")
				if err != nil || recovered.Invoke || recovered.Completion == nil || recovered.Completion.Failure == nil || recovered.Completion.Failure.Code != want {
					t.Fatalf("context rejection left session open: %+v %v", recovered, err)
				}
			})
		}
	}
}

func TestPostgresRuntimePlannerParallelSelectionAccountsUsage(t *testing.T) {
	for _, profile := range []plannerProfile{streamlineProfile, routerProfile} {
		for _, count := range []int{2, 65, 1000} {
			t.Run(fmt.Sprintf("%s/%d", profile.ref, count), func(t *testing.T) {
				pool := isolatedStreamlinePool(t, t.Context())
				store := runstore.NewPostgresStore(pool, artifactpgtest.NewStorage())
				createPreparingStreamlineStage(t, t.Context(), store)
				var f *Factory
				invoker := runtimeInvokerFunc(func(ctx context.Context, _ string, h contracts.WorkerHandle, _ contracts.StageContentRequest) (contracts.WorkerCompletion, error) {
					sequence := 0
					op := func(name string, args any) contracts.PlannerOperationResult {
						t.Helper()
						sequence++
						data, _ := json.Marshal(args)
						reply, err := f.ExecuteOperation(ctx, "stage-1", h.AllocationID, contracts.PlannerOperation{APIVersion: contracts.APIVersion, Sequence: sequence, Name: name, Arguments: data})
						if err != nil || reply.Failure != nil {
							t.Fatalf("%s failed: %+v, %v", name, reply, err)
						}
						return reply
					}
					op("before_model", map[string]any{"retainedBytes": 0})
					calls := make([]string, count)
					for i := range calls {
						calls[i] = listSubtasksToolName
					}
					reply := op("after_model", map[string]any{"inputTokens": 7, "outputTokens": 3, "cachedTokens": 0, "calls": calls, "cost": map[string]any{"currency": "USD", "calls": 1, "groups": []any{map[string]any{"model": "fake-streamline-model", "phase": "main", "source": "gateway", "calls": 1, "amount": "0.01"}}}})
					if string(reply.Result) != `{"rejected":true}` {
						t.Fatalf("parallel selection admitted: %s", reply.Result)
					}
					// Continue with a single permitted finish; rejected calls caused no effects.
					op("before_model", map[string]any{"retainedBytes": 0})
					op("after_model", map[string]any{"inputTokens": 7, "outputTokens": 3, "cachedTokens": 0, "calls": []string{finishToolName}})
					op(finishToolName, map[string]any{"outcome": "failed", "summary": "Stop", "artifacts": map[string]any{}, "error": map[string]any{"code": "done", "message": "No dispatch", "retryable": false}})
					return workerCompletion("0", "Complete", nil), nil
				})
				f = runtimeTestFactory(t, strings.TrimSuffix(profile.ref, "@1"), migrationBoundarySession(t, store), invoker)
				p, err := f.Create(appendProbeInvocation(strings.TrimSuffix(profile.ref, "@1")))
				if err != nil {
					t.Fatal(err)
				}
				if _, err = p.Run(t.Context()); err != nil {
					t.Fatal(err)
				}
				report, _ := p.(planner.ReportProvider).ExecutionReport()
				if report.Metrics.TotalTokens == nil || *report.Metrics.TotalTokens != 20 || report.Metrics.LLMCost == nil || report.Metrics.LLMCost.Calls != 2 || len(report.Metrics.LLMCost.Groups) != 1 || report.Metrics.LLMCost.Groups[0].Amount != "0.01" {
					t.Fatalf("selection usage/cost lost: %+v", report.Metrics)
				}
			})
		}
	}
}

func TestPostgresRuntimePlannerCompactorOutcomeAccounting(t *testing.T) {
	for _, profile := range []string{"streamline", "router"} {
		for _, test := range []struct {
			reason     string
			tools      int
			text, want string
		}{
			{"stop", 0, `{"summary":"Keep completed work."}`, ""},
			{"tool_calls", 0, `{"summary":"Keep completed work."}`, "tool_output"},
			{"function_call", 0, `{"summary":"Keep completed work."}`, "tool_output"},
			{"stop", 1, "", "tool_output"},
			{"stop", 0, "", "empty"},
			{"content_filter", 0, "", "empty"},
			{"length", 0, `{"summary":"Keep completed work."}`, "output_limit"},
			{"", 0, `{"summary":"Keep completed work."}`, "invalid"},
		} {
			t.Run(fmt.Sprintf("%s/%s/tools=%d", profile, test.reason, test.tools), func(t *testing.T) {
				pool := isolatedStreamlinePool(t, t.Context())
				store := runstore.NewPostgresStore(pool, artifactpgtest.NewStorage())
				createPreparingStreamlineStage(t, t.Context(), store)
				var f *Factory
				invoker := runtimeInvokerFunc(func(ctx context.Context, _ string, h contracts.WorkerHandle, _ contracts.StageContentRequest) (contracts.WorkerCompletion, error) {
					sequence := 0
					op := func(name string, args any) contracts.PlannerOperationResult {
						t.Helper()
						sequence++
						data, _ := json.Marshal(args)
						reply, err := f.ExecuteOperation(ctx, "stage-1", h.AllocationID, contracts.PlannerOperation{APIVersion: contracts.APIVersion, Sequence: sequence, Name: name, Arguments: data})
						if err != nil {
							t.Fatal(err)
						}
						return reply
					}
					reply := op("context", map[string]any{})
					var ctxResult struct{ Objective string }
					if err := json.Unmarshal(reply.Result, &ctxResult); err != nil || ctxResult.Objective != "Produce a reviewed report" {
						t.Fatalf("canonical objective lost: %s %v", reply.Result, err)
					}
					op("before_compactor", map[string]any{"retainedBytes": 0, "inputBytes": 1000, "messages": 2})
					reply = op("after_compactor", map[string]any{"inputTokens": 13, "outputTokens": 4, "cachedTokens": 0, "text": test.text, "finishReason": test.reason, "toolCalls": test.tools, "sourceBytes": 1000})
					if test.want != "" {
						if reply.Failure == nil || reply.Failure.Code != "planner_compaction_failed" || reply.Failure.Retryable || !strings.Contains(reply.Failure.Message, "("+test.want+")") {
							t.Fatalf("wrong compactor failure: %+v", reply)
						}
					} else if reply.Failure != nil {
						t.Fatalf("valid summary rejected: %+v", reply)
					}
					return contracts.WorkerCompletion{}, nil
				})
				f = runtimeTestFactory(t, profile, migrationBoundarySession(t, store), invoker)
				p, err := f.Create(compactionInvocation(profile))
				if err != nil {
					t.Fatal(err)
				}
				_, err = p.Run(t.Context())
				if test.want != "" {
					assertPlannerCode(t, err, "planner_compaction_failed")
				}
				report, _ := p.(planner.ReportProvider).ExecutionReport()
				if report.Metrics.TotalTokens == nil || *report.Metrics.TotalTokens != 17 || report.Metrics.ModelCalls == nil || *report.Metrics.ModelCalls != 1 {
					t.Fatalf("compactor usage lost: %+v", report.Metrics)
				}
			})
		}
	}
}
