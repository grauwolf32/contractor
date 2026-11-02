package session

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"

	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/planner"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/telemetry"
)

// The real session service on PostgreSQL refuses the completion of a
// non-retryable failed candidate with a non-transient error. Passthrough then
// returns non-retryable planner_session_unavailable after one write, and the
// session stays running (PLN-R106). Starting that Stage execution again finds
// the running session: the error and the session span carry
// planner_invocation_in_progress, retryable (PLN-R107).
func TestPostgresUnwrittenFailedCandidateKeepsItsRetryability(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 45*time.Second)
	defer cancel()
	pool := isolatedPlannerPool(t, ctx)
	store := runstore.NewPostgresStore(pool, artifactpgtest.NewStorage())
	createPlannerStage(t, ctx, store)
	if _, err := pool.Exec(ctx, `
CREATE FUNCTION reject_planner_completion() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    RAISE EXCEPTION 'test rejects the Planner session completion';
END
$$;
CREATE TRIGGER reject_planner_completion BEFORE UPDATE ON planner_sessions FOR EACH ROW
WHEN (NEW.state->>'status' = 'completed') EXECUTE FUNCTION reject_planner_completion();`); err != nil {
		t.Fatal(err)
	}
	service, err := New(store, Options{})
	if err != nil {
		t.Fatal(err)
	}
	worker := &failingWorker{failure: contracts.WorkerFailure{
		Code: "tool_input_invalid", Message: "Worker input was invalid", Retryable: false,
	}}
	factory, err := planner.NewPassthroughFactory(service, worker, noArtifacts{})
	if err != nil {
		t.Fatal(err)
	}
	instance, err := factory.Create(passthroughInvocation())
	if err != nil {
		t.Fatal(err)
	}

	result, err := instance.Run(ctx)
	var plannerError *planner.Error
	if !errors.As(err, &plannerError) || plannerError.Code != "planner_session_unavailable" ||
		plannerError.Retryable || result.Outcome != "" || worker.calls != 1 {
		t.Fatalf("Run = (%+v, %v), Worker calls = %d", result, err, worker.calls)
	}
	execution, err := store.GetStageExecution(ctx, "stage-planner")
	if err != nil || execution.PlannerSessionID == nil {
		t.Fatalf("StageExecution = (%+v, %v)", execution, err)
	}
	session, err := store.GetPlannerSession(ctx, *execution.PlannerSessionID)
	if err != nil {
		t.Fatal(err)
	}
	state, err := decodeState(session.State)
	if err != nil || state.Status != statusRunning || state.Completion != nil || !state.RequestRecorded {
		t.Fatalf("Planner session state = %s (%v), want running without a completion", session.State, err)
	}

	spans := &recordingSpans{}
	_, err = planner.StartSession(ctx, service, spans, "stage-planner")
	if !errors.As(err, &plannerError) || plannerError.Code != "planner_invocation_in_progress" ||
		!plannerError.Retryable {
		t.Fatalf("StartSession on a running session = %v", err)
	}
	if len(spans.codes) != 1 || spans.codes[0] != "planner_invocation_in_progress" {
		t.Fatalf("session span codes = %v", spans.codes)
	}
}

func passthroughInvocation() planner.Invocation {
	templateRef := contracts.AgentTemplateRef{
		TemplateID: "artifact_builder", Version: "1", Digest: "sha256:" + strings.Repeat("a", 64),
	}
	runtimeRef := contracts.WorkerRuntimeRef{RuntimeID: "adk", Version: "1"}
	return planner.Invocation{
		StageExecutionID: "stage-planner",
		RunID:            "run-planner",
		Stage: resolved.ResolvedStage{
			Objective: "build",
			Instructions: contracts.ResolvedInstructions{
				Ref: "instructions/build.md", Digest: "sha256:" + strings.Repeat("b", 64), Text: "Build the report.",
			},
			Planner: resolved.PlannerRef{PlannerID: "passthrough", Version: "1"},
			Agents: map[string]resolved.ResolvedAgentBinding{
				"builder": {
					Template:  contracts.ResolvedAgentTemplate{Ref: templateRef, Runtime: runtimeRef},
					Namespace: "builder",
				},
			},
			Context: resolved.StageContext{Artifacts: map[string]resolved.ContextArtifact{}},
			Result: resolved.StageResultContract{Artifacts: map[string]resolved.ArtifactSlot{
				"report": {Required: true, MediaTypes: []string{"application/json"}},
			}},
		},
		Context: planner.StageContext{
			Parameters: map[string]string{"mode": "strict"}, Artifacts: map[string]*contracts.ArtifactRef{},
		},
		Workers: map[string]contracts.WorkerHandle{
			"builder": {
				AllocationID: "allocation-1", AgentTemplateRef: templateRef, WorkerRuntimeRef: runtimeRef,
				AgentCard: map[string]any{"name": "builder"}, LeaseExpiresAt: time.Now().Add(time.Minute),
			},
		},
		Deadline: time.Now().Add(30 * time.Second),
	}
}

type failingWorker struct {
	failure contracts.WorkerFailure
	calls   int
}

func (w *failingWorker) Invoke(
	context.Context, string, contracts.WorkerHandle, contracts.StageContentRequest,
) (contracts.WorkerCompletion, error) {
	w.calls++
	failure := w.failure
	return contracts.WorkerCompletion{
		APIVersion: contracts.APIVersion, Failure: &failure,
		InvocationID: "worker-invocation-failed", StateRevision: 1,
	}, nil
}

// noArtifacts verifies no artifact: a failed candidate carries none.
type noArtifacts struct{}

func (noArtifacts) Inspect(context.Context, string, contracts.ArtifactRef) (planner.ArtifactMetadata, error) {
	return planner.ArtifactMetadata{}, errors.New("unexpected artifact verification")
}

type recordingSpans struct{ codes []string }

func (r *recordingSpans) StartSpan(
	telemetry.PlannerSpanName, telemetry.PlannerSpanAttributes,
) telemetry.PlannerSpan {
	return recordingSpan{spans: r}
}

type recordingSpan struct{ spans *recordingSpans }

func (s recordingSpan) End(_ string, attributes telemetry.PlannerSpanAttributes) {
	s.spans.codes = append(s.spans.codes, attributes.ErrorCode)
}
