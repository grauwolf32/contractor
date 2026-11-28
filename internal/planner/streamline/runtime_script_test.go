package streamline

import (
	"context"
	"encoding/json"
	"fmt"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/planner"
)

// Scripts drive the real Server operation protocol, without a model or Go SDK.
// Model behavior itself is exercised by Python and process Gateway tests.
type operationStep struct {
	name string
	args map[string]any
}
type operationScript struct {
	steps   []operationStep
	content string
}

func functionStep(name string, args map[string]any) operationStep { return operationStep{name, args} }
func addSubtaskStep(objective, instructions string) operationStep {
	return functionStep(addSubtaskToolName, map[string]any{"objective": objective, "instructions": instructions})
}
func runtimeScriptFactory(profile plannerProfile, sessions planner.PlanSessionService, workers planner.WorkerInvoker, inspector planner.ArtifactInspector, reader planner.WorkerStateReader, script *operationScript, limits Limits) (*Factory, error) {
	var factory *Factory
	invoke := runtimeInvokerFunc(func(ctx context.Context, name string, handle contracts.WorkerHandle, request contracts.StageContentRequest) (contracts.WorkerCompletion, error) {
		if !handle.WorkerRuntimeRef.IsPlanner() {
			return workers.Invoke(ctx, name, handle, request)
		}
		seq := 0
		operation := func(name string, args any) (contracts.PlannerOperationResult, error) {
			seq++
			encoded, err := json.Marshal(args)
			if err != nil {
				return contracts.PlannerOperationResult{}, err
			}
			reply, err := factory.ExecuteOperation(ctx, "stage-1", handle.AllocationID, contracts.PlannerOperation{APIVersion: contracts.APIVersion, Sequence: seq, Name: name, Arguments: encoded})
			if err == nil && reply.Failure != nil {
				err = fmt.Errorf("operation stopped: %s", reply.Failure.Code)
			}
			return reply, err
		}
		for _, step := range script.steps {
			if _, err := operation("before_model", map[string]any{"retainedBytes": 0}); err != nil {
				return contracts.WorkerCompletion{}, err
			}
			if _, err := operation("after_model", map[string]any{"inputTokens": 7, "outputTokens": 3, "cachedTokens": 0, "calls": []string{step.name}, "content": script.content}); err != nil {
				return contracts.WorkerCompletion{}, err
			}
			reply, err := operation(step.name, step.args)
			if err != nil {
				return contracts.WorkerCompletion{}, err
			}
			if reply.Stop {
				break
			}
		}
		return workerCompletion("0", "Planner finished", map[string]contracts.ArtifactRef{}), nil
	})
	var err error
	factory, err = newFactory(profile, sessions, invoke, inspector, reader, nil, limits)
	return factory, err
}
func runtimeTestFactory(t *testing.T, profile string, sessions planner.PlanSessionService, workers planner.WorkerInvoker) *Factory {
	t.Helper()
	selected := streamlineProfile
	if profile == "router" {
		selected = routerProfile
	}
	f, e := newFactory(selected, sessions, workers, &fakeInspector{}, unavailableWorkerStateReader{}, nil, Limits{})
	if e != nil {
		t.Fatal(e)
	}
	return f
}
func (s *fakeSessions) RecordActivity(_ context.Context, _ planner.SessionIdentity, a planner.Activity) error {
	return a.Validate()
}
