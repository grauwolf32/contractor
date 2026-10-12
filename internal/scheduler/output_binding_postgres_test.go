package scheduler

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contracts"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/jackc/pgx/v5/pgxpool"
)

// recordingResultCommits counts the result progressions the Scheduler asks
// the store to commit and keeps the store's answer to the last one.
type recordingResultCommits struct {
	AtomicPersistence
	commits int
	last    error
}

func (p *recordingResultCommits) CommitResultProgression(ctx context.Context, value ResultProgression) error {
	p.commits++
	p.last = p.AtomicPersistence.CommitResultProgression(ctx, value)
	return p.last
}

// newOutputBindingHarness is the PostgreSQL Scheduler harness of run-1 with
// its result commits recorded.
func newOutputBindingHarness(
	t *testing.T, ctx context.Context,
) (deferredPlacementHarness, *recordingResultCommits, *artifacts.Service, artifacts.ScopedStore) {
	t.Helper()
	recording := &recordingResultCommits{}
	h := newDeferredPlacementHarness(t, ctx, func(persistence AtomicPersistence) AtomicPersistence {
		recording.AtomicPersistence = persistence
		return recording
	})
	service := artifacts.NewService(artifacts.NewPostgresRepository(h.pool))
	runArtifacts, err := service.Run("run-1")
	if err != nil {
		t.Fatal(err)
	}
	return h, recording, service, runArtifacts
}

// A Run creates each Workflow output binding once (ART-R90). A finalizing
// Stage whose mapping meets outputs/<slot> already bound at another revision
// can never commit its acceptance, so the Scheduler ends the Run as durable
// state it cannot progress (SCH-R102) instead of replaying the same
// rolled-back transaction on every pass.
func TestPostgresOutputMappingOntoAnotherRevisionEndsRunOnce(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	h, commits, service, runArtifacts := newOutputBindingHarness(t, ctx)
	earlier, err := runArtifacts.Write(
		ctx, contracts.ArtifactRef{Namespace: "builder", Name: "earlier"},
		artifacts.Payload{MediaType: "text/plain", Data: []byte("earlier\n")}, nil,
	)
	if err != nil {
		t.Fatal(err)
	}
	existing, err := bindOutput(ctx, service, "run-1", "result", earlier.Ref)
	if err != nil {
		t.Fatal(err)
	}

	worked, err := h.scheduler.RunOnce(ctx)
	run, runErr := h.store.GetRun(ctx, "run-1")
	if err != nil || !worked || runErr != nil || run.State != runstore.RunFailed {
		t.Errorf("first pass = (%v, %v), Run %s (%v); want the Run failed", worked, err, run.State, runErr)
	}
	// The second pass finds the Run terminal and replays nothing.
	if _, err := h.scheduler.RunOnce(ctx); err != nil {
		t.Errorf("second pass: %v", err)
	}
	if commits.commits != 1 || !errors.Is(commits.last, errInvalidRunState) ||
		!errors.Is(commits.last, artifacts.ErrArtifactConflict) {
		t.Fatalf("result commits = %d, last = %v; want one, refused as invalid output state", commits.commits, commits.last)
	}
	requireRunEndedAsInvalidState(t, ctx, h)
	requireOutputBinding(t, ctx, h.pool, runArtifacts, existing.TargetRef, "earlier\n")
	// Terminal release recovery released what the ended Stage held.
	executions, err := h.store.ListStageExecutions(ctx, "run-1")
	if err != nil {
		t.Fatal(err)
	}
	allocations, err := h.store.ListStageAllocations(ctx, executions[0].StageExecutionID)
	if err != nil || len(allocations) != 1 || allocations[0].ReleaseCompletedAt == nil {
		t.Fatalf("Stage allocations after the second pass = (%+v, %v), want released", allocations, err)
	}
}

// The binding a lost acceptance would have committed, forked by output_bind
// from the candidate's exact source revision, is the same binding: replaying
// the acceptance onto it succeeds without a second revision (ART-R90).
func TestPostgresOutputMappingReplaysIdenticalSourceBinding(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	h, commits, service, runArtifacts := newOutputBindingHarness(t, ctx)
	existing, err := bindOutput(ctx, service, "run-1", "result", h.invoker.result.Artifacts["copied"])
	if err != nil {
		t.Fatal(err)
	}

	if worked, err := h.scheduler.RunOnce(ctx); err != nil || !worked {
		t.Fatalf("RunOnce = (%v, %v)", worked, err)
	}
	if commits.commits != 1 || commits.last != nil {
		t.Fatalf("result commits = %d, last = %v; want one successful commit", commits.commits, commits.last)
	}
	h.requireSingleSucceededStage(t, ctx)
	requireOutputBinding(t, ctx, h.pool, runArtifacts, existing.TargetRef, "copied\n")
	metadata, err := runArtifacts.Metadata(ctx, contracts.ArtifactRef{Namespace: "outputs", Name: "result"})
	if err != nil || !metadata.Frozen {
		t.Fatalf("output after Run success = (%+v, %v), want frozen", metadata, err)
	}
}

// A transient database failure while binding an output says nothing about the
// binding: it is no invalid output state, the acceptance rolls back, and the
// next pass commits it.
func TestPostgresTransientOutputBindingFailureIsRetried(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	h, commits, _, runArtifacts := newOutputBindingHarness(t, ctx)
	if _, err := h.pool.Exec(ctx, `
CREATE SEQUENCE test_output_bind_attempts;
CREATE FUNCTION fault_test_output_bind()
RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    IF NEW.lineage_kind = 'output_bind' AND nextval('test_output_bind_attempts') = 1 THEN
        RAISE EXCEPTION 'injected output binding fault' USING ERRCODE = '`+persistencepostgres.SQLStateDeadlockDetected+`';
    END IF;
    RETURN NEW;
END;
$$;
CREATE TRIGGER fault_test_output_bind
BEFORE INSERT ON artifact_lineage
FOR EACH ROW EXECUTE FUNCTION fault_test_output_bind()`); err != nil {
		t.Fatal(err)
	}

	worked, err := h.scheduler.RunOnce(ctx)
	if !worked || persistencepostgres.SQLState(err) != persistencepostgres.SQLStateDeadlockDetected ||
		errors.Is(err, errInvalidRunState) {
		t.Fatalf("RunOnce with an injected deadlock = (%v, %v), want the retryable database error", worked, err)
	}
	run, err := h.store.GetRun(ctx, "run-1")
	if err != nil || run.State != runstore.RunRunning {
		t.Fatalf("Run after the deadlock = (%s, %v), want running", run.State, err)
	}
	executions, err := h.store.ListStageExecutions(ctx, "run-1")
	if err != nil || len(executions) != 1 || executions[0].State != runstore.StageFinalizing {
		t.Fatalf("StageExecutions after the deadlock = (%v, %v), want finalizing", stageExecutionStates(executions), err)
	}
	if worked, err := h.scheduler.RunOnce(ctx); err != nil || !worked {
		t.Fatalf("replayed RunOnce = (%v, %v)", worked, err)
	}
	if commits.commits != 2 || commits.last != nil {
		t.Fatalf("result commits = %d, last = %v; want the replay committed", commits.commits, commits.last)
	}
	h.requireSingleSucceededStage(t, ctx)
	bound, err := runArtifacts.Metadata(ctx, contracts.ArtifactRef{Namespace: "outputs", Name: "result"})
	if err != nil {
		t.Fatal(err)
	}
	requireOutputBinding(t, ctx, h.pool, runArtifacts, bound.Ref, "copied\n")
}

// Every Workflow output check of a result progression reads only durable
// state: the candidate, the Run's pinned Workflow and immutable artifact
// versions. A check that fails therefore fails every replay alike: it is
// classified as invalid output state, which ends the Run (SCH-R102) instead
// of being retried, and the acceptance rolls back.
func TestPostgresDeterministicOutputChecksAreInvalidOutputState(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	pool := isolatedSchedulerPool(t, ctx)
	fixture := createFinalizingFixture(t, ctx, pool)
	valid := ResultProgression{
		RunID: "run-1", StageExecutionID: fixture.executionID, Result: fixture.result,
		WorkflowOutputs: fixture.workflow.Stages["copy"].WorkflowOutputs,
		OutputContracts: fixture.workflow.Outputs,
		Progression:     terminalSuccessProgression("run-1", fixture.executionID),
	}
	text := []string{"text/plain"}
	for _, test := range []struct {
		name   string
		change func(*ResultProgression)
		want   string
	}{
		{"required output without Stage result", func(value *ResultProgression) {
			value.Result.Artifacts = map[string]contracts.ArtifactRef{}
		}, `required Workflow output "result" has no Stage result`},
		{"undeclared output mapped", func(value *ResultProgression) {
			value.WorkflowOutputs = map[string]string{"result": "copied", "extra": "copied"}
		}, `undeclared Workflow output "extra" is mapped`},
		{"mapped media type rejected", func(value *ResultProgression) {
			value.OutputContracts = map[string]resolved.ArtifactSlot{
				"result": {Required: true, MediaTypes: []string{"application/json"}},
			}
		}, `Workflow output "result" has incompatible media type`},
		{"required output unbound at success", func(value *ResultProgression) {
			value.OutputContracts = map[string]resolved.ArtifactSlot{
				"result": {Required: true, MediaTypes: text}, "summary": {Required: true, MediaTypes: text},
			}
		}, `required Workflow output "summary" is not bound`},
	} {
		t.Run(test.name, func(t *testing.T) {
			value := valid
			test.change(&value)
			err := fixture.persistence.CommitResultProgression(ctx, value)
			if !errors.Is(err, errInvalidRunState) || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("result commit = %v, want invalid output state: %s", err, test.want)
			}
		})
	}

	// Nothing above committed: the Stage still finalizes, no output is bound,
	// and the unchanged progression commits.
	execution, err := fixture.store.GetStageExecution(ctx, fixture.executionID)
	if err != nil || execution.State != runstore.StageFinalizing || execution.AcceptedResult != nil {
		t.Fatalf("Stage after refused commits = (%+v, %v)", execution, err)
	}
	runArtifacts, err := fixture.artifacts.Run("run-1")
	if err != nil {
		t.Fatal(err)
	}
	outputs := "outputs"
	if page, err := runArtifacts.ListRefs(ctx, artifacts.RefPageQuery{Namespace: &outputs, Limit: 10}); err != nil ||
		len(page.Refs) != 0 {
		t.Fatalf("outputs after refused commits = (%+v, %v)", page.Refs, err)
	}
	if err := fixture.persistence.CommitResultProgression(ctx, valid); err != nil {
		t.Fatalf("valid result commit: %v", err)
	}
	if run, err := fixture.store.GetRun(ctx, "run-1"); err != nil || run.State != runstore.RunSucceeded {
		t.Fatalf("Run after the valid commit = (%+v, %v)", run, err)
	}
}

// requireRunEndedAsInvalidState checks that run-1 failed as state the
// Scheduler cannot progress and that its finalizing Stage accepted the
// candidate without Workflow progression (SCH-R102).
func requireRunEndedAsInvalidState(t *testing.T, ctx context.Context, h deferredPlacementHarness) {
	t.Helper()
	run, err := h.store.GetRun(ctx, "run-1")
	if err != nil || run.State != runstore.RunFailed || run.StateReason.Code != "scheduler_state_invalid" {
		t.Fatalf("Run = (%s %+v, %v), want failed as scheduler_state_invalid", run.State, run.StateReason, err)
	}
	executions, err := h.store.ListStageExecutions(ctx, "run-1")
	if err != nil || len(executions) != 1 || executions[0].State != runstore.StageSucceeded ||
		executions[0].Termination != nil || executions[0].AcceptedResult == nil ||
		!executions[0].AcceptedResult.Artifacts["copied"].SameExact(h.invoker.result.Artifacts["copied"]) {
		t.Fatalf("StageExecutions = (%v, %v), want the candidate accepted", stageExecutionStates(executions), err)
	}
}

// requireOutputBinding checks that run-1's outputs/result is still the one
// revision want, holding payload, with one output_bind lineage edge.
func requireOutputBinding(
	t *testing.T, ctx context.Context, pool *pgxpool.Pool, runArtifacts artifacts.ScopedStore,
	want contracts.ArtifactRef, payload string,
) {
	t.Helper()
	output, err := runArtifacts.Read(ctx, contracts.ArtifactRef{Namespace: "outputs", Name: "result"})
	if err != nil || !output.Ref.SameExact(want) || string(output.Payload.Data) != payload {
		t.Fatalf("outputs/result = (%+v %q, %v), want %+v holding %q", output.Ref, output.Payload.Data, err, want, payload)
	}
	var revisions, edges int
	if err := pool.QueryRow(ctx, `
SELECT
    (SELECT count(*) FROM artifact_binding_revisions
     WHERE scope_kind = 'run' AND scope_id = 'run-1' AND namespace = 'outputs' AND name = 'result'),
    (SELECT count(*) FROM artifact_lineage
     WHERE target_scope_kind = 'run' AND target_scope_id = 'run-1'
       AND target_namespace = 'outputs' AND target_name = 'result' AND lineage_kind = 'output_bind')`,
	).Scan(&revisions, &edges); err != nil || revisions != 1 || edges != 1 {
		t.Fatalf("outputs/result revisions = %d, output_bind edges = %d (%v), want one each", revisions, edges, err)
	}
}
