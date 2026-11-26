package streamline

import (
	"testing"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/runner"
	adksession "google.golang.org/adk/v2/session"
	"google.golang.org/genai"
)

// Check real SDK function-response events, rather than synthetic action mocks:
// the adapter must terminate only when its Server operation authorizes it.
func TestPlannerADKToolActions(t *testing.T) {
	for _, profile := range []string{"streamline", "router"} {
		for _, outcome := range []string{"accepted-finish", "rejected-finish", "worker-completed", "worker-budget"} {
			t.Run(profile+"/"+outcome, func(t *testing.T) {
				p, _ := migrationPlanner(t, profile, "core", &scriptedModel{})
				defer p.stateViews.Close()
				p.resultContract = nil
				p.invoker = appendProbeWorkers()
				p.deadline = p.invocation.Deadline
				state := newExecutionState(p.limits)
				identity := p.sessions.(*fakeSessions).identity
				name := finishToolName
				args := map[string]any{"outcome": "succeeded", "summary": "Complete", "artifacts": map[string]any{}}
				wantSkip, wantEscalate := false, false
				switch outcome {
				case "accepted-finish":
					args["outcome"] = "failed"
					args["error"] = map[string]any{"code": "insufficient_evidence", "message": "Evidence unavailable", "retryable": false}
					wantSkip, wantEscalate = true, true
				case "worker-completed", "worker-budget":
					if added := p.addSubtask(t.Context(), state, identity, addSubtaskArgs{Objective: "Inspect", Instructions: "Return evidence"}); !added.OK {
						t.Fatalf("add subtask = %+v", added)
					}
					name, args = executeCurrentSubtaskToolName, map[string]any{"subtask_id": "0"}
					if profile == "router" {
						args["worker_name"] = "builder"
					}
					if outcome == "worker-budget" {
						state.workerCalls = int64(state.limits.MaxWorkerCalls)
						wantSkip = true
					}
				}
				tools, _, err := p.buildTools(state, identity)
				if err != nil {
					t.Fatal(err)
				}
				llm := &scriptedModel{steps: []modelStep{functionStep(name, args), textStep("Continue after nonterminal operation")}}
				root, err := llmagent.New(llmagent.Config{Name: "actions", Model: llm, Tools: tools})
				if err != nil {
					t.Fatal(err)
				}
				service := adksession.InMemoryService()
				if _, err := service.Create(t.Context(), &adksession.CreateRequest{AppName: "actions", UserID: "owner", SessionID: "session"}); err != nil {
					t.Fatal(err)
				}
				actual, err := runner.New(runner.Config{AppName: "actions", Agent: root, SessionService: service})
				if err != nil {
					t.Fatal(err)
				}
				responses := 0
				for event, err := range actual.Run(t.Context(), "owner", "session", genai.NewContentFromText("Inspect bounded input", genai.RoleUser), agent.RunConfig{}) {
					if err != nil {
						t.Fatal(err)
					}
					if event.Content == nil {
						continue
					}
					for _, part := range event.Content.Parts {
						if part.FunctionResponse != nil && part.FunctionResponse.Name == name {
							responses++
							if event.Actions.SkipSummarization != wantSkip || event.Actions.Escalate != wantEscalate {
								t.Fatalf("%s actions = %+v, want skip=%v escalate=%v", outcome, event.Actions, wantSkip, wantEscalate)
							}
						}
					}
				}
				if responses != 1 {
					t.Fatalf("observed %d function responses, want one", responses)
				}
				wantCalls := 2
				if wantSkip {
					wantCalls = 1
				}
				if llm.callCount() != wantCalls {
					t.Fatalf("model calls=%d, want %d", llm.callCount(), wantCalls)
				}
			})
		}
	}
}
