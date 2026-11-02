//go:build integration

package findingintake

import (
	"context"
	"encoding/json"
	"errors"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"

	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/configload"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/reporting"
	"github.com/grauwolf32/contractor/internal/controlplane"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/runstore/runstoretest"
	"github.com/grauwolf32/contractor/internal/runtimeconfig/runtimeconfigtest"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

const quotaRunID = "quota-run"

// seedQuotaRun creates a running Run with maxProposalsPerRun-1 receipts, one
// slot below its proposal quota, and returns the grant of its allocation.
func seedQuotaRun(t *testing.T, ctx context.Context, pool *pgxpool.Pool) controlplane.AllocationGrant {
	t.Helper()
	const ownerID, projectID, runID = "quota-owner", "quota-project", quotaRunID
	if _, _, err := projectstore.NewPostgresStore(pool).Create(ctx, projectstore.CreateParams{
		ProjectID: projectID, OwnerID: ownerID, Kind: projectstore.KindProject,
		Name: "Quota project", IdempotencyKey: "quota-project",
		RequestDigest: auditdomain.DigestBytes([]byte("quota-project")),
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
    proposal_size_bytes, evidence, report_projection
, checks_projection)
SELECT 'filler-' || n, 'filler-proposal-' || n, 'filler-allocation', 'runtime',
       'instance', 'stage', 'worker', 'filler-invocation', 'filler-' || n,
       'filler-' || n, 'sha256:' || repeat('a', 64), $1, $2, $3,
       'filler', '1', 'contractor/v1alpha1', '{"name":"filler","version":"1"}'::jsonb,
       'sha256:' || repeat('b', 64),
       jsonb_build_object('namespace', 'filler', 'name', 'filler-' || n, 'revision', 'r1'),
       'sha256:' || repeat('c', 64), 'application/json', 1, '[]'::jsonb,
       '{"description":"Test finding.","limitations":[],"preconditions":[],"severity_suggestion":"","standard_refs":[],"subject":null,"title":"Test finding."}', json_build_object('schema','contractor.audit.finding-checks-projection.v1','proposal_digest',('sha256:' || repeat('c', 64))::text,'size_bytes',(1)::bigint,'discarded',false,'subject',NULL,'limitations',NULL,'checks','[]'::json)
  FROM generate_series(1, $4::integer) AS n`,
		runID, ownerID, projectID, maxProposalsPerRun-1); err != nil {
		t.Fatal(err)
	}
	return grant
}

// quotaReceiptCount counts the quota Run's receipts.
func quotaReceiptCount(t *testing.T, ctx context.Context, pool *pgxpool.Pool) int {
	t.Helper()
	var count int
	if err := pool.QueryRow(ctx, `SELECT count(*) FROM finding_proposal_receipts WHERE run_id = $1`, quotaRunID).Scan(&count); err != nil {
		t.Fatal(err)
	}
	return count
}

// Two distinct submissions of one Run race for its last quota slot. The
// per-Run lock must serialize their counts, so exactly one is admitted. At
// capacity a replay of the admitted submission returns its receipt without a
// quota check, and a third distinct submission is the rejected 4097th.
func TestPostgresRunProposalQuotaSerializesDistinctSubmissions(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	pool := isolatedFindingPool(t, ctx)
	grant := seedQuotaRun(t, ctx, pool)

	gate := &quotaCountGate{match: "count(*) FROM finding_proposal_receipts", held: make(chan struct{}), resume: make(chan struct{})}
	defer gate.release()
	first, firstPID := quotaActorPool(t, ctx, pool, gate)
	second, _ := quotaActorPool(t, ctx, pool, nil)
	firstService, err := New(first, artifactpgtest.NewStorage())
	if err != nil {
		t.Fatal(err)
	}
	secondService, err := New(second, artifactpgtest.NewStorage())
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
	if count := quotaReceiptCount(t, ctx, pool); admitted != 1 || rejected != 1 || count != maxProposalsPerRun {
		t.Fatalf("admitted=%d rejected=%d count=%d", admitted, rejected, count)
	}
	winner := testSubmission("invocation-a", "candidate-a", []contracts.ArtifactRef{})
	if firstErr != nil {
		winner = testSubmission("invocation-b", "candidate-b", []contracts.ArtifactRef{})
	}
	stored, err := readReceiptBySubmission(ctx, pool, grant.AllocationID, winner.InvocationID, winner.SubmissionID)
	if err != nil {
		t.Fatal(err)
	}
	replay, replayed, err := secondService.Submit(ctx, grant, winner)
	if err != nil || !replayed || replay.ReceiptID != stored.ReceiptID {
		t.Fatalf("replay at capacity = (%s, replayed=%t, %v), want %s", replay.ReceiptID, replayed, err, stored.ReceiptID)
	}
	_, _, err = secondService.Submit(ctx, grant, testSubmission("invocation-c", "candidate-c", []contracts.ArtifactRef{}))
	if !errors.Is(err, ErrInvalid) || !strings.Contains(err.Error(), "quota") {
		t.Fatalf("4097th distinct submission = %v, want the quota rejection", err)
	}
	if count := quotaReceiptCount(t, ctx, pool); count != maxProposalsPerRun {
		t.Fatalf("Run has %d receipts after replay and rejection, want %d", count, maxProposalsPerRun)
	}
}

// A submission for the last quota slot whose transaction rolls back after it
// wrote the proposal Artifact and the receipt admits neither, and leaves the
// slot to the next distinct submission.
func TestPostgresRunProposalQuotaRollbackLeavesSlot(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	pool := isolatedFindingPool(t, ctx)
	grant := seedQuotaRun(t, ctx, pool)
	gate := &quotaCountGate{match: "INSERT INTO finding_proposal_receipts", held: make(chan struct{}), resume: make(chan struct{})}
	defer gate.release()
	actor, _ := quotaActorPool(t, ctx, pool, gate)
	actorService, err := New(actor, artifactpgtest.NewStorage())
	if err != nil {
		t.Fatal(err)
	}
	submitCtx, cancelSubmit := context.WithCancel(ctx)
	defer cancelSubmit()
	done := make(chan error, 1)
	go func() {
		_, _, err := actorService.Submit(submitCtx, grant, testSubmission("invocation-rollback", "candidate-rollback", []contracts.ArtifactRef{}))
		done <- err
	}()
	select {
	case <-gate.held:
	case <-ctx.Done():
		t.Fatal("submission never inserted its receipt")
	}
	// Cancel while the transaction holds the uncommitted Artifact and receipt.
	cancelSubmit()
	if err := <-done; err == nil {
		t.Fatal("cancelled submission committed")
	}
	var bindings, revisions int
	if err := pool.QueryRow(ctx, `
SELECT (SELECT count(*) FROM artifact_bindings
         WHERE scope_kind = 'run' AND scope_id = $1 AND namespace = 'finding-proposals'),
       (SELECT count(*) FROM artifact_binding_revisions
         WHERE scope_kind = 'run' AND scope_id = $1 AND namespace = 'finding-proposals')`,
		quotaRunID).Scan(&bindings, &revisions); err != nil {
		t.Fatal(err)
	}
	if count := quotaReceiptCount(t, ctx, pool); count != maxProposalsPerRun-1 || bindings != 0 || revisions != 0 {
		t.Fatalf("rolled-back submission left %d receipts, %d bindings, %d revisions", count, bindings, revisions)
	}
	service, err := New(pool, artifactpgtest.NewStorage())
	if err != nil {
		t.Fatal(err)
	}
	if _, replayed, err := service.Submit(ctx, grant, testSubmission("invocation-next", "candidate-next", []contracts.ArtifactRef{})); err != nil || replayed {
		t.Fatalf("submission for the freed slot = (replayed=%t, %v)", replayed, err)
	}
	if count := quotaReceiptCount(t, ctx, pool); count != maxProposalsPerRun {
		t.Fatalf("Run has %d receipts, want %d", count, maxProposalsPerRun)
	}
}

func quotaSubmissionGrant(
	t *testing.T, ctx context.Context, pool *pgxpool.Pool, ownerID, projectID, runID string,
) controlplane.AllocationGrant {
	return quotaSubmissionGrantWithInputs(t, ctx, pool, ownerID, projectID, runID, nil)
}

func quotaSubmissionGrantWithInputs(
	t *testing.T, ctx context.Context, pool *pgxpool.Pool, ownerID, projectID, runID string,
	prepare func() map[string]runstore.PinnedContextArtifact,
) controlplane.AllocationGrant {
	t.Helper()
	snapshot, _, err := configload.LoadUnionReadOnly("../config/testdata/valid", filepath.Join(t.TempDir(), "managed"))
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
	runs := runstore.NewPostgresStore(pool, artifactpgtest.NewStorage())
	project := projectID
	if _, err := runstoretest.CreateRun(ctx, runs, runstore.CreateRunParams{
		LaneClass: resolved.LaneClassModel,
		RunID:     runID, OwnerID: ownerID, ProjectID: &project,
		WorkflowName: workflow.Ref.Name, WorkflowVersion: workflow.Ref.Version,
		WorkflowSchemaVersion: contracts.APIVersion, WorkflowSnapshot: workflowJSON,
		Parameters: map[string]string{}, RuntimeConfig: runtimeconfigtest.BuiltInRunSnapshot(),
	}); err != nil {
		t.Fatal(err)
	}
	if _, err := runs.TransitionRun(
		ctx, runID, runstore.RunInitializing, runstore.RunPending, runstore.Reason{Code: "quota_test_started"},
	); err != nil {
		t.Fatal(err)
	}
	if _, err := runs.TransitionRun(
		ctx, runID, runstore.RunPending, runstore.RunRunning, runstore.Reason{Code: "admitted"},
	); err != nil {
		t.Fatal(err)
	}
	inputs := map[string]runstore.PinnedContextArtifact{}
	if prepare != nil {
		inputs = prepare()
	}
	stageID, allocationID := runID+"-stage", runID+"-allocation"
	if _, err := runs.CreateStageExecution(ctx, runstore.CreateStageExecutionParams{
		LaneClass:        resolved.LaneClassModel,
		StageExecutionID: stageID, RunID: runID, StageName: workflow.EntryStage, Attempt: 1,
		StageSpecSchemaVersion: contracts.APIVersion, StageSpecSnapshot: stageJSON,
		StageContextSchemaVersion: contracts.APIVersion,
		StageContext: runstore.StageContextSnapshot{
			Parameters: map[string]string{}, Artifacts: inputs,
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
		PerformanceCollectionPolicy:       reporting.PerformanceCollectionDisabled,
	}); err != nil {
		t.Fatal(err)
	}
	return controlplane.AllocationGrant{
		AllocationID: allocationID, RuntimeAgentID: runtimeAgentID,
		RuntimeInstanceID: "quota-instance", RunID: runID, StageExecutionID: stageID,
		LogicalAgentName: "builder", Namespace: binding.Namespace,
	}
}

// quotaCountGate pauses the first transaction that runs a statement
// containing match until released or cancelled, while that transaction still
// owns every lock it has taken.
type quotaCountGate struct {
	match        string
	held, resume chan struct{}
	once         sync.Once
	releaseOnce  sync.Once
}

type quotaCountContextKey struct{}

func (gate *quotaCountGate) TraceQueryStart(ctx context.Context, _ *pgx.Conn, data pgx.TraceQueryStartData) context.Context {
	return context.WithValue(ctx, quotaCountContextKey{}, strings.Contains(data.SQL, gate.match))
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
