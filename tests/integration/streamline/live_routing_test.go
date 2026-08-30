package streamline_test

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"os"
	"strings"
	"sync"
	"testing"
	"time"

	workflowconfig "github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/planner"
	"github.com/grauwolf32/contractor/internal/planner/router"
	plannersession "github.com/grauwolf32/contractor/internal/planner/session"
	"github.com/grauwolf32/contractor/internal/planner/streamline"
	"github.com/grauwolf32/contractor/internal/runstore"
	adksession "google.golang.org/adk/session"
)

// TestLiveRouterWorkflow is an opt-in model-contract evaluation. Deterministic
// E2E tests cover the production transport boundaries; this test keeps those
// semantics fixed and asks the deployed model to drive the real Router tools.
func TestLiveRouterWorkflow(t *testing.T) {
	gatewayURL := strings.TrimSpace(os.Getenv("CONTRACTOR_LIVE_LLM_URL"))
	modelName := strings.TrimSpace(os.Getenv("CONTRACTOR_LIVE_LLM_MODEL"))
	if gatewayURL == "" || modelName == "" {
		t.Skip("live routing Gateway settings are not set")
	}
	token := os.Getenv("CONTRACTOR_LIVE_LLM_TOKEN")
	if token == "" {
		token = "unused"
	}
	llm, err := streamline.NewOpenAICompatibleModel(streamline.GatewaySettings{
		URL: gatewayURL, Token: contracts.NewSecretString(token), Model: modelName,
	})
	if err != nil {
		t.Fatal("live routing Gateway settings are invalid")
	}
	sessions := newLiveRouterSessions()
	worker := &liveRouterWorker{}
	factory, err := router.NewFactory(
		sessions, sessions, worker, liveRouterInspector{}, llm,
		router.Limits{
			MaxModelCalls: 8, MaxTokens: 32_768, MaxWorkerCalls: 2,
			MaxWallTime: 90 * time.Second,
		},
	)
	if err != nil {
		t.Fatal("construct live Router")
	}
	instance, err := factory.Create(liveRouterInvocation())
	if err != nil {
		t.Fatal("construct live Router invocation")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()
	result, err := instance.Run(ctx)
	if err != nil {
		failure := planner.FailureFrom(err)
		t.Fatalf("live Router contract failed: code=%s retryable=%t", failure.Code, failure.Retryable)
	}
	if result.Outcome != contracts.StageSucceeded {
		code := "none"
		if result.Error != nil {
			code = result.Error.Code
		}
		t.Fatalf("live Router returned %s: code=%s", result.Outcome, code)
	}
	calls := worker.callsSnapshot()
	if len(calls) != 1 || calls[0] != "reviewer" {
		t.Fatalf("live Router selected logical Workers %v, want [reviewer]", calls)
	}
	selected := result.Artifacts["reviewed"]
	if selected.Namespace != "review" || selected.Name != "copied" ||
		selected.Revision == nil || *selected.Revision != "live-review-r1" {
		t.Fatalf("live Router selected an inexact result artifact")
	}
	report, ok := instance.(planner.ReportProvider).ExecutionReport()
	if !ok || !report.Complete || report.Metrics.ModelCalls == nil ||
		*report.Metrics.ModelCalls < 3 || *report.Metrics.ModelCalls > 8 {
		t.Fatalf("live Router returned incomplete bounded metrics")
	}
	digest := sha256.Sum256([]byte(modelName))
	t.Logf("live routing evidence: model_sha256=%s model_calls=%d selected_worker=reviewer outcome=succeeded",
		hex.EncodeToString(digest[:]), *report.Metrics.ModelCalls)
}

type liveRouterSessions struct {
	mu         sync.Mutex
	identity   planner.SessionIdentity
	plan       *planner.PlannerPlanProjection
	completion *planner.Completion
}

func newLiveRouterSessions() *liveRouterSessions {
	return &liveRouterSessions{identity: planner.SessionIdentity{
		SessionID: "live-router-session", StageExecutionID: "live-router-stage",
		InvocationID: "live-router-invocation",
	}}
}

func (s *liveRouterSessions) Begin(context.Context, string) (planner.SessionStart, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.completion != nil {
		completion := *s.completion
		return planner.SessionStart{Identity: s.identity, Completion: &completion}, nil
	}
	return planner.SessionStart{Identity: s.identity, Invoke: true}, nil
}

func (*liveRouterSessions) RecordRequest(
	context.Context, planner.SessionIdentity, planner.RequestFacts,
) error {
	return nil
}

func (s *liveRouterSessions) Complete(
	_ context.Context, _ planner.SessionIdentity, completion planner.Completion,
) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	value := completion
	s.completion = &value
	return nil
}

func (s *liveRouterSessions) RecordPlan(
	_ context.Context,
	_ planner.SessionIdentity,
	transition planner.PlannerPlanTransition,
) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	var previous *planner.PlannerPlanProjection
	if s.plan != nil {
		value := cloneLivePlan(*s.plan)
		previous = &value
	}
	currentRevision := uint64(0)
	if previous != nil {
		currentRevision = previous.Revision
	}
	if transition.ExpectedRevision != currentRevision {
		return runstore.ErrConflict
	}
	if err := planner.ValidatePlannerPlanTransition(previous, transition.Plan, transition.Kind); err != nil {
		return err
	}
	value := cloneLivePlan(transition.Plan)
	s.plan = &value
	return nil
}

func (s *liveRouterSessions) RecordFact(
	_ context.Context, _ planner.SessionIdentity, fact planner.PlannerFact,
) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	revision := uint64(0)
	if s.plan != nil {
		revision = s.plan.Revision
	}
	if fact.PlanRevision != revision {
		return runstore.ErrConflict
	}
	return nil
}

func (s *liveRouterSessions) LoadPlan(
	context.Context, planner.SessionIdentity,
) (planner.PlannerPlanProjection, bool, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.plan == nil {
		return planner.PlannerPlanProjection{}, false, nil
	}
	return cloneLivePlan(*s.plan), true, nil
}

func (*liveRouterSessions) NewADKSession(
	context.Context, planner.SessionIdentity, plannersession.ADKOptions,
) (adksession.Service, error) {
	return adksession.InMemoryService(), nil
}

func cloneLivePlan(value planner.PlannerPlanProjection) planner.PlannerPlanProjection {
	value.Subtasks = append([]planner.PlannerSubtask(nil), value.Subtasks...)
	if value.ActiveDispatch != nil {
		dispatch := *value.ActiveDispatch
		value.ActiveDispatch = &dispatch
	}
	return value
}

type liveRouterWorker struct {
	mu    sync.Mutex
	calls []string
}

func (w *liveRouterWorker) Invoke(
	_ context.Context,
	binding string,
	_ contracts.WorkerHandle,
	request contracts.StageContentRequest,
) (contracts.StageContentResult, error) {
	w.mu.Lock()
	w.calls = append(w.calls, binding)
	w.mu.Unlock()
	if binding != "reviewer" {
		return contracts.StageContentResult{}, planner.NewError(
			"live_router_misroute", "Router selected a Worker that cannot perform review", false, nil,
		)
	}
	input := request.Artifacts["source"]
	if request.Parameters["mode"] != "live-routing" || input.Revision == nil ||
		*input.Revision != "live-source-r1" {
		return contracts.StageContentResult{}, errors.New("live Router omitted immutable Stage context")
	}
	revision := "live-review-r1"
	return contracts.StageContentResult{
		APIVersion: contracts.APIVersion, Outcome: contracts.StageSucceeded,
		Summary: "reviewer produced the exact reviewed copy",
		Artifacts: map[string]contracts.ArtifactRef{
			"reviewed": {Namespace: "review", Name: "copied", Revision: &revision},
		},
	}, nil
}

func (w *liveRouterWorker) callsSnapshot() []string {
	w.mu.Lock()
	defer w.mu.Unlock()
	return append([]string(nil), w.calls...)
}

type liveRouterInspector struct{}

func (liveRouterInspector) Inspect(
	_ context.Context, runID string, ref contracts.ArtifactRef,
) (planner.ArtifactMetadata, error) {
	if runID != "live-router-run" || ref.Revision == nil {
		return planner.ArtifactMetadata{}, errors.New("artifact is outside the live Run")
	}
	key := ref.Namespace + "/" + ref.Name + "@" + *ref.Revision
	if key != "inputs/source@live-source-r1" && key != "review/copied@live-review-r1" {
		return planner.ArtifactMetadata{}, errors.New("artifact does not exist")
	}
	return planner.ArtifactMetadata{MediaType: "text/plain"}, nil
}

func liveRouterInvocation() planner.Invocation {
	digest := "sha256:" + strings.Repeat("a", 64)
	runtimeRef := contracts.WorkerRuntimeRef{RuntimeID: "adk", Version: "1"}
	agents := map[string]workflowconfig.ResolvedAgentBinding{}
	workers := map[string]contracts.WorkerHandle{}
	for _, binding := range []struct {
		name        string
		description string
		namespace   string
	}{
		{name: "builder", description: "Builds new documents but never reviews them", namespace: "builder"},
		{name: "reviewer", description: "Reviews the exact input and produces the approved result", namespace: "review"},
	} {
		templateRef := contracts.AgentTemplateRef{
			TemplateID: binding.name, Version: "1", Digest: digest,
		}
		agents[binding.name] = workflowconfig.ResolvedAgentBinding{
			Namespace: binding.namespace,
			Template: contracts.ResolvedAgentTemplate{
				Ref: templateRef, Runtime: runtimeRef, Description: binding.description,
			},
		}
		workers[binding.name] = contracts.WorkerHandle{
			AllocationID:     "live-allocation-" + binding.name,
			AgentTemplateRef: templateRef, WorkerRuntimeRef: runtimeRef,
			AgentCard:      map[string]any{"name": binding.name, "url": "https://placement.invalid"},
			LeaseExpiresAt: time.Now().Add(3 * time.Minute),
		}
	}
	inputRevision := "live-source-r1"
	return planner.Invocation{
		StageExecutionID: "live-router-stage", RunID: "live-router-run",
		Deadline: time.Now().Add(90 * time.Second),
		Stage: workflowconfig.ResolvedStage{
			Objective:    "Route one review of the exact source to reviewer and return its reviewed artifact.",
			Instructions: contracts.ResolvedInstructions{Text: "Create exactly one subtask, execute it with reviewer, then finish succeeded using the exact artifact returned by reviewer. Never select builder."},
			Planner:      workflowconfig.PlannerRef{PlannerID: "router", Version: "1"},
			Agents:       agents,
			Context: workflowconfig.StageContext{Artifacts: map[string]workflowconfig.ContextArtifact{
				"source": {Namespace: "inputs", Name: "source", Required: true},
			}},
			Result: workflowconfig.StageResultContract{Artifacts: map[string]workflowconfig.ArtifactSlot{
				"reviewed": {Required: true, MediaTypes: []string{"text/plain"}},
			}},
		},
		Context: planner.StageContext{
			Parameters: map[string]string{"mode": "live-routing"},
			Artifacts: map[string]*contracts.ArtifactRef{
				"source": {Namespace: "inputs", Name: "source", Revision: &inputRevision},
			},
		},
		Workers: workers,
	}
}
