//go:build integration

package findingintake

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"sync"
	"testing"
	"time"

	workflowconfig "github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/controlplane"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

// Two distinct submissions of one Run race for its last quota slot. The
// per-Run lock must serialize their counts, so exactly one is admitted.
func TestPostgresRunProposalQuotaSerializesDistinctSubmissions(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	pool := isolatedFindingPool(t, ctx)
	const ownerID, projectID, runID = "quota-owner", "quota-project", "quota-run"
	if _, _, err := projectstore.NewPostgresStore(pool).Create(ctx, projectstore.CreateParams{
		ProjectID: projectID, OwnerID: ownerID, Kind: projectstore.KindProject,
		Name: "Quota project", IdempotencyKey: "quota-project",
		RequestDigest: digestBytes([]byte("quota-project")),
	}); err != nil {
		t.Fatal(err)
	}
	grant := quotaSubmissionGrant(t, ctx, pool, ownerID, projectID, runID)
	if _, err := pool.Exec(ctx, `
INSERT INTO finding_proposal_receipts (
    receipt_id, proposal_id, allocation_id, runtime_agent_id,
    runtime_instance_id, stage_execution_id, logical_agent_name,
    invocation_id, submission_id, client_key, request_digest,
    run_id, owner_id, project_id,
    workflow_name, workflow_version, workflow_schema_version,
    workflow_configuration_ref, workflow_closure_digest,
    proposal_ref, proposal_digest, proposal_media_type,
    proposal_size_bytes, evidence
)
SELECT 'filler-' || n, 'filler-proposal-' || n, 'filler-allocation', 'runtime',
       'instance', 'stage', 'worker', 'filler-invocation', 'filler-' || n,
       'filler-' || n, 'sha256:' || repeat('a', 64), $1, $2, $3,
       'filler', '1', 'contractor/v1alpha1', '{"name":"filler","version":"1"}'::jsonb,
       'sha256:' || repeat('b', 64),
       jsonb_build_object('namespace', 'filler', 'name', 'filler-' || n, 'revision', 'r1'),
       'sha256:' || repeat('c', 64), 'application/json', 1, '[]'::jsonb
  FROM generate_series(1, $4::integer) AS n`,
		runID, ownerID, projectID, maxProposalsPerRun-1); err != nil {
		t.Fatal(err)
	}

	gate := &quotaCountGate{held: make(chan struct{}), resume: make(chan struct{})}
	defer gate.release()
	first, firstPID := quotaActorPool(t, ctx, pool, gate)
	second, _ := quotaActorPool(t, ctx, pool, nil)
	firstService, err := New(first)
	if err != nil {
		t.Fatal(err)
	}
	secondService, err := New(second)
	if err != nil {
		t.Fatal(err)
	}
	results := make(chan error, 2)
	go func() {
		_, _, err := firstService.Submit(ctx, grant, testSubmission("invocation-a", "candidate-a", []contracts.ArtifactRef{}))
		results <- err
	}()
	select {
	case <-gate.held:
	case <-ctx.Done():
		t.Fatal("first submission never counted its Run proposals")
	}
	secondDone := make(chan error, 1)
	go func() {
		_, _, err := secondService.Submit(ctx, grant, testSubmission("invocation-b", "candidate-b", []contracts.ArtifactRef{}))
		secondDone <- err
	}()
	// The first submission holds its transaction after counting. The second
	// either waits on that transaction or, without a Run lock, finishes.
	var secondErr error
	secondFinished := false
wait:
	for {
		select {
		case secondErr = <-secondDone:
			secondFinished = true
			break wait
		case <-ctx.Done():
			t.Fatal("second submission neither finished nor waited")
		default:
		}
		var blocked bool
		if err := pool.QueryRow(ctx, `
SELECT EXISTS (
    SELECT 1 FROM pg_stat_activity
     WHERE $1::integer = ANY(pg_blocking_pids(pid))
)`, firstPID).Scan(&blocked); err != nil {
			t.Fatal(err)
		}
		if blocked {
			break
		}
	}
	gate.release()
	firstErr := <-results
	if !secondFinished {
		secondErr = <-secondDone
	}
	admitted, rejected := 0, 0
	for _, err := range []error{firstErr, secondErr} {
		switch {
		case err == nil:
			admitted++
		case errors.Is(err, ErrInvalid) && strings.Contains(err.Error(), "quota"):
			rejected++
		default:
			t.Fatalf("unexpected submission error: %v", err)
		}
	}
	var count int
	if err := pool.QueryRow(ctx, `SELECT count(*) FROM finding_proposal_receipts WHERE run_id = $1`, runID).Scan(&count); err != nil {
		t.Fatal(err)
	}
	if admitted != 1 || rejected != 1 || count != maxProposalsPerRun {
		t.Fatalf("admitted=%d rejected=%d count=%d", admitted, rejected, count)
	}
}

func quotaSubmissionGrant(
	t *testing.T, ctx context.Context, pool *pgxpool.Pool, ownerID, projectID, runID string,
) controlplane.AllocationGrant {
	t.Helper()
	snapshot, err := workflowconfig.Load("../config/testdata/valid", workflowconfig.MVPDescriptors())
	if err != nil {
		t.Fatal(err)
	}
	workflow, err := snapshot.Workflow("artifact-copy@1")
	if err != nil {
		t.Fatal(err)
	}
	stage := workflow.Stages[workflow.EntryStage]
	binding := stage.Agents["builder"]
	binding.Template.Toolsets = append(binding.Template.Toolsets, contracts.ToolsetSelection{
		Ref:   contracts.ToolsetRef{ToolsetID: "security-findings", Version: "1"},
		Tools: []string{"finding"},
	})
	stage.Agents["builder"] = binding
	workflow.Stages[workflow.EntryStage] = stage
	workflowJSON, err := json.Marshal(workflow)
	if err != nil {
		t.Fatal(err)
	}
	stageJSON, err := json.Marshal(stage)
	if err != nil {
		t.Fatal(err)
	}
	runs := runstore.NewPostgresStore(pool)
	project := projectID
	if _, err := runs.CreateRun(ctx, runstore.CreateRunParams{
		RunID: runID, OwnerID: ownerID, ProjectID: &project,
		WorkflowName: workflow.Ref.Name, WorkflowVersion: workflow.Ref.Version,
		WorkflowSchemaVersion: contracts.APIVersion, WorkflowSnapshot: workflowJSON,
		Parameters: map[string]string{}, RuntimeConfig: runtimeconfig.BuiltInRunSnapshot(),
	}); err != nil {
		t.Fatal(err)
	}
	if _, err := runs.TransitionRun(ctx, runID, runstore.RunInitializing, runstore.RunRunning,
		runstore.Reason{Code: "quota_test_started"}); err != nil {
		t.Fatal(err)
	}
	stageID, allocationID := runID+"-stage", runID+"-allocation"
	if _, err := runs.CreateStageExecution(ctx, runstore.CreateStageExecutionParams{
		StageExecutionID: stageID, RunID: runID, StageName: workflow.EntryStage, Attempt: 1,
		StageSpecSchemaVersion: contracts.APIVersion, StageSpecSnapshot: stageJSON,
		StageContextSchemaVersion: contracts.APIVersion,
		StageContext: runstore.StageContextSnapshot{
			Parameters: map[string]string{}, Artifacts: map[string]runstore.PinnedContextArtifact{},
		},
	}); err != nil {
		t.Fatal(err)
	}
	runtimeAgentID := strings.Repeat("a", 64)
	if err := runs.RecordStageAllocation(ctx, runstore.StageAllocation{
		AllocationID: allocationID, StageExecutionID: stageID,
		LogicalAgentName: "builder", Namespace: binding.Namespace,
		AgentTemplateRef: binding.Template.Ref, WorkerRuntimeRef: binding.Template.Runtime,
		RuntimeAgentID: runtimeAgentID, RuntimeAgentInstanceID: "quota-instance",
		RuntimeAgentLabelRevision:         1,
		RuntimeConfigurationSchemaVersion: runstore.AllocationRuntimeConfigurationSchemaVersion,
		RuntimeConfiguration:              findingRuntimeConfiguration(),
		PerformanceCollectionPolicy:       contracts.PerformanceCollectionDisabled,
	}); err != nil {
		t.Fatal(err)
	}
	return controlplane.AllocationGrant{
		AllocationID: allocationID, RuntimeAgentID: runtimeAgentID,
		RuntimeInstanceID: "quota-instance", RunID: runID, StageExecutionID: stageID,
		LogicalAgentName: "builder", Namespace: binding.Namespace,
	}
}

// quotaCountGate pauses the first transaction that counts Run proposals until
// released, while that transaction still owns every lock it has taken.
type quotaCountGate struct {
	held, resume chan struct{}
	once         sync.Once
	releaseOnce  sync.Once
}

type quotaCountContextKey struct{}

func (*quotaCountGate) TraceQueryStart(ctx context.Context, _ *pgx.Conn, data pgx.TraceQueryStartData) context.Context {
	return context.WithValue(ctx, quotaCountContextKey{}, strings.Contains(data.SQL, "count(*) FROM finding_proposal_receipts"))
}

func (gate *quotaCountGate) TraceQueryEnd(ctx context.Context, _ *pgx.Conn, _ pgx.TraceQueryEndData) {
	if matched, _ := ctx.Value(quotaCountContextKey{}).(bool); matched {
		gate.once.Do(func() {
			close(gate.held)
			select {
			case <-gate.resume:
			case <-ctx.Done():
			}
		})
	}
}

func (gate *quotaCountGate) release() { gate.releaseOnce.Do(func() { close(gate.resume) }) }

func quotaActorPool(
	t *testing.T, ctx context.Context, pool *pgxpool.Pool, trace pgx.QueryTracer,
) (*pgxpool.Pool, uint32) {
	t.Helper()
	configuration := pool.Config()
	configuration.MaxConns, configuration.MinConns = 1, 0
	if trace != nil {
		configuration.ConnConfig.Tracer = trace
	}
	actor, err := pgxpool.NewWithConfig(ctx, configuration)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(actor.Close)
	connection, err := actor.Acquire(ctx)
	if err != nil {
		t.Fatal(err)
	}
	pid := connection.Conn().PgConn().PID()
	connection.Release()
	return actor, pid
}
