//go:build integration

package findingintake

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/artifacts/artifactstest"
	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore/querystore"
	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/configload"
	"github.com/grauwolf32/contractor/internal/contracts"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/runstore/runstoretest"
	"github.com/grauwolf32/contractor/internal/runtimeconfig/runtimeconfigtest"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

func TestPostgresCandidateOperationReplayAfterSourcePurgeAndNewRun(t *testing.T) {
	for _, qualified := range []bool{true, false} {
		t.Run(fmt.Sprintf("qualified=%t", qualified), func(t *testing.T) { candidateOperationReplayAfterPurge(t, qualified) })
	}
}
func candidateOperationReplayAfterPurge(t *testing.T, qualified bool) {
	ctx := t.Context()
	pool := isolatedFindingPool(t, ctx)
	const owner, project = "candidate-owner", "candidate-project"
	if _, _, err := projectstore.NewPostgresStore(pool).Create(ctx, projectstore.CreateParams{ProjectID: project, OwnerID: owner, Kind: projectstore.KindProject, Name: "Candidates", IdempotencyKey: "create", RequestDigest: auditdomain.DigestBytes([]byte("create"))}); err != nil {
		t.Fatal(err)
	}
	storage := artifactpgtest.NewStorage()
	service, err := New(pool, storage)
	if err != nil {
		t.Fatal(err)
	}
	source := quotaSubmissionGrant(t, ctx, pool, owner, project, "original-scanner")
	sourceOrigin := &artifacts.ExecutionOrigin{RunID: source.RunID, StageExecutionID: source.StageExecutionID, AllocationID: source.AllocationID, LogicalAgentName: source.LogicalAgentName}
	sourceStore, _ := artifacts.NewService(storage.Repository(pool)).Run(source.RunID)
	candidate := NormalizedCandidate{Key: strings.Repeat("a", 64), Producer: "test-scanner", Rule: "problem-rule", Title: "Observed problem", Assertion: "The selected scanner matched an exposed configuration.", Subject: "https://target.test/.git/config", Method: "GET", Severity: "medium", QualifiedKey: []string{"test-scanner", "rule-v1", "GET", "https://target.test/.git/config"}}
	if !qualified {
		candidate.QualifiedKey = nil
	}
	proof, _ := json.Marshal(map[string]any{"schema": "contractor.finding.scanner-evidence.v1", "records": []map[string]any{{"key": candidate.Key, "matched": true}}})
	savedProof, err := sourceStore.Write(ctx, contracts.ArtifactRef{Namespace: source.Namespace, Name: "evidence"}, artifacts.Payload{MediaType: CandidateEvidenceMediaType, Data: proof, ExecutionOrigin: sourceOrigin}, nil)
	if err != nil {
		t.Fatal(err)
	}
	proofMeta, err := sourceStore.Metadata(ctx, savedProof.Ref)
	if err != nil {
		t.Fatal(err)
	}
	batch := CandidateBatch{Schema: CandidateSchema, Complete: false, Candidates: []NormalizedCandidate{candidate}, Skipped: map[string]int{"informational": 1}, Evidence: proofMeta.Exact()}
	raw, _ := json.Marshal(batch)
	savedBatch, err := sourceStore.Write(ctx, contracts.ArtifactRef{Namespace: source.Namespace, Name: "candidates"}, artifacts.Payload{MediaType: CandidateMediaType, Data: raw, ExecutionOrigin: sourceOrigin}, nil)
	if err != nil {
		t.Fatal(err)
	}
	batchMeta, err := sourceStore.Metadata(ctx, savedBatch.Ref)
	if err != nil {
		t.Fatal(err)
	}
	selected := map[string]contracts.ExactArtifact{}
	for name, exact := range map[string]contracts.ExactArtifact{"candidates": batchMeta.Exact(), "evidence": proofMeta.Exact()} {
		retained, err := retainFindingArtifact(ctx, artifacts.NewService(storage.Repository(pool)), source.RunID, project, "scanner-inputs", name, exact)
		if err != nil {
			t.Fatal(err)
		}
		selected[name] = retained
	}
	// Uploading byte-identical candidate JSON cannot impersonate the scan.
	projectStore, _ := artifacts.NewService(storage.Repository(pool)).Project(project)
	forged, err := projectStore.Write(ctx, contracts.ArtifactRef{Namespace: "scanner-inputs", Name: "uploaded-candidates"}, artifacts.Payload{MediaType: CandidateMediaType, Data: raw}, nil)
	if err != nil {
		t.Fatal(err)
	}
	forgedMeta, err := projectStore.Metadata(ctx, forged.Ref)
	if err != nil {
		t.Fatal(err)
	}
	forgedRun := candidateOperationRun(t, ctx, pool, owner, project, "publish-untrusted", map[string]contracts.ExactArtifact{"candidates": forgedMeta.Exact(), "evidence": selected["evidence"]})
	if _, err := service.PublishCandidates(ctx, forgedRun); !errors.Is(err, ErrAccessDenied) {
		t.Fatalf("uploaded copy impersonated a scanner: %v", err)
	}
	var before int
	if err := pool.QueryRow(ctx, `SELECT count(*) FROM project_finding_sources WHERE project_id=$1`, project).Scan(&before); err != nil || before != 0 {
		t.Fatalf("invalid batch mutated domain: %d %v", before, err)
	}
	first := candidateOperationRun(t, ctx, pool, owner, project, "publish-first", selected)
	published, err := service.PublishCandidates(ctx, first)
	if err != nil {
		t.Fatal(err)
	}
	if published.Published != 1 || published.Replayed != 0 || published.Skipped != 1 || published.Complete || len(published.Sources) != 1 || published.Sources[0].FindingID == nil {
		t.Fatalf("publication: %+v", published)
	}
	finding, err := querystore.New(pool, storage).GetProjectFinding(ctx, owner, project, *published.Sources[0].FindingID)
	if err != nil || finding.State != nil || finding.AnalystVerdict != nil || finding.AnalystSeverity != nil {
		t.Fatalf("analyst state: %+v %v", finding, err)
	}
	again, err := service.PublishCandidates(ctx, first)
	if err != nil || again.Published != 0 || again.Replayed != 1 {
		t.Fatalf("lost acknowledgement replay: %+v %v", again, err)
	}
	runs := runstore.NewPostgresStore(pool, storage)
	if err := runs.MarkStageAllocationReleased(ctx, source.AllocationID); err != nil {
		t.Fatal(err)
	}
	if _, err := runs.TransitionRun(ctx, source.RunID, runstore.RunRunning, runstore.RunFailed, runstore.Reason{Code: "fixture_finished"}); err != nil {
		t.Fatal(err)
	}
	if err := deleteRun(ctx, pool, owner, source.RunID); err != nil {
		t.Fatal(err)
	}
	second := candidateOperationRun(t, ctx, pool, owner, project, "publish-second", selected)
	replayed, err := service.PublishCandidates(ctx, second)
	if err != nil || replayed.Replayed != 1 || replayed.Sources[0].ReceiptID != published.Sources[0].ReceiptID {
		t.Fatalf("new Run replay after original purge: %+v %v", replayed, err)
	}
	retained, err := service.GetProjectReceipt(ctx, owner, project, published.Sources[0].ReceiptID)
	if err != nil || retained.Origin.Operation != contracts.PublishFindingsOperation || retained.Origin.AllocationID != "" || retained.Origin.LogicalAgentName != "" || retained.Origin.Scan == nil || !sameScanExecution(retained.Origin.Scan, sourceOrigin) || !retained.Origin.Scan.Artifact.SameExact(savedBatch.Ref) || len(retained.ProjectEvidence) != 2 {
		t.Fatalf("source provenance: %+v %v", retained.Origin, err)
	}
	var count int
	if err := pool.QueryRow(ctx, `SELECT count(*) FROM project_finding_sources WHERE project_id=$1`, project).Scan(&count); err != nil || count != 1 {
		t.Fatalf("sources=%d %v", count, err)
	}
	// A new actual scan with identical semantics has a distinct immutable
	// source identity. Qualified equality groups it; missing equality cannot.
	independent := quotaSubmissionGrant(t, ctx, pool, owner, project, "independent-scanner")
	independentOrigin := &artifacts.ExecutionOrigin{RunID: independent.RunID, StageExecutionID: independent.StageExecutionID, AllocationID: independent.AllocationID, LogicalAgentName: independent.LogicalAgentName}
	independentStore, _ := artifacts.NewService(storage.Repository(pool)).Run(independent.RunID)
	newProof, err := independentStore.Write(ctx, contracts.ArtifactRef{Namespace: independent.Namespace, Name: "evidence"}, artifacts.Payload{MediaType: CandidateEvidenceMediaType, Data: proof, ExecutionOrigin: independentOrigin}, nil)
	if err != nil {
		t.Fatal(err)
	}
	newProofMeta, err := independentStore.Metadata(ctx, newProof.Ref)
	if err != nil {
		t.Fatal(err)
	}
	batch.Evidence = newProofMeta.Exact()
	newRaw, _ := json.Marshal(batch)
	newBatch, err := independentStore.Write(ctx, contracts.ArtifactRef{Namespace: independent.Namespace, Name: "candidates"}, artifacts.Payload{MediaType: CandidateMediaType, Data: newRaw, ExecutionOrigin: independentOrigin}, nil)
	if err != nil {
		t.Fatal(err)
	}
	newBatchMeta, err := independentStore.Metadata(ctx, newBatch.Ref)
	if err != nil {
		t.Fatal(err)
	}
	independentSelected := map[string]contracts.ExactArtifact{}
	for name, exact := range map[string]contracts.ExactArtifact{"candidates": newBatchMeta.Exact(), "evidence": newProofMeta.Exact()} {
		retained, err := retainFindingArtifact(ctx, artifacts.NewService(storage.Repository(pool)), independent.RunID, project, "scanner-inputs", "independent-"+name, exact)
		if err != nil {
			t.Fatal(err)
		}
		independentSelected[name] = retained
	}
	third := candidateOperationRun(t, ctx, pool, owner, project, "publish-independent", independentSelected)
	extra, err := service.PublishCandidates(ctx, third)
	if err != nil || extra.Published != 1 || extra.Sources[0].ReceiptID == published.Sources[0].ReceiptID {
		t.Fatalf("independent scan lost its source: %+v %v", extra, err)
	}
	sameFinding := *extra.Sources[0].FindingID == *published.Sources[0].FindingID
	if sameFinding != qualified {
		t.Fatalf("qualified equality=%t expected=%t", sameFinding, qualified)
	}
	// Claim loss must reject even an otherwise read-only replay.
	stale := second
	stale.ClaimID = "not-the-live-claim"
	if _, err := service.PublishCandidates(ctx, stale); err == nil {
		t.Fatal("stale claim replay was authorized")
	}
	if err := pool.QueryRow(ctx, `SELECT count(*) FROM stage_allocations WHERE stage_execution_id=$1 OR stage_execution_id=$2`, first.StageID, second.StageID).Scan(&count); err != nil || count != 0 {
		t.Fatalf("operation allocated a Runtime: %d %v", count, err)
	}
}

func candidateOperationRun(t *testing.T, ctx context.Context, pool *pgxpool.Pool, owner, project, runID string, selected map[string]contracts.ExactArtifact) CandidatePublication {
	snapshot, _, err := configload.LoadUnionReadOnly("../config/testdata/valid", filepath.Join(t.TempDir(), "managed"))
	if err != nil {
		t.Fatal(err)
	}
	workflow, err := snapshot.Workflow("artifact-copy@1")
	if err != nil {
		t.Fatal(err)
	}
	stage := workflow.Stages[workflow.EntryStage]
	stage.Planner = resolved.PlannerRef{}
	stage.Operation = &resolved.ServerOperationConfig{Ref: contracts.PublishFindingsOperation, Arguments: map[string]contracts.ToolArgumentBinding{}}
	stage.Instructions = contracts.ResolvedInstructions{}
	stage.Agents = map[string]resolved.ResolvedAgentBinding{}
	stage.ExecutionConfig = resolved.ResolvedStageExecutionConfig{Agents: map[string]resolved.ResolvedConsumerExecutionConfig{}}
	stage.Context = resolved.StageContext{Artifacts: map[string]resolved.ContextArtifact{"candidates": {Namespace: "inputs", Name: "candidates", Required: true}, "evidence": {Namespace: "inputs", Name: "evidence", Required: true}}}
	stage.Result.Artifacts = map[string]resolved.ArtifactSlot{}
	stage.WorkflowOutputs = map[string]string{}
	workflow.Stages = map[string]resolved.ResolvedStage{workflow.EntryStage: stage}
	workflow.Inputs = map[string]resolved.ArtifactSlot{"candidates": {Required: true, MediaTypes: []string{CandidateMediaType}}, "evidence": {Required: true, MediaTypes: []string{CandidateEvidenceMediaType}}}
	workflow.Outputs = map[string]resolved.ArtifactSlot{}
	if err := resolved.ValidateWorkflowGraph(workflow); err != nil {
		t.Fatalf("invalid operation workflow: %v", err)
	}
	encoded, _ := json.Marshal(workflow)
	stageJSON, _ := json.Marshal(stage)
	storage := artifactpgtest.NewStorage()
	runs := runstore.NewPostgresStore(pool, storage)
	if _, err := runstoretest.CreateRun(ctx, runs, runstore.CreateRunParams{LaneClass: resolved.LaneClassTool, RunID: runID, OwnerID: owner, ProjectID: &project, WorkflowName: workflow.Ref.Name, WorkflowVersion: workflow.Ref.Version, WorkflowSchemaVersion: contracts.APIVersion, WorkflowSnapshot: encoded, Parameters: map[string]string{}, RuntimeConfig: runtimeconfigtest.BuiltInRunSnapshot()}); err != nil {
		t.Fatal(err)
	}
	inputs := map[string]runstore.PinnedContextArtifact{}
	for name, exact := range selected {
		fork, err := artifactstest.ForkProjectInput(ctx, artifacts.NewService(storage.Repository(pool)), project, exact.Ref, runID, name)
		if err != nil {
			t.Fatal(err)
		}
		ref := fork.TargetRef
		inputs[name] = runstore.PinnedContextArtifact{Required: true, Artifact: &ref}
	}
	if _, err := runs.TransitionRun(ctx, runID, runstore.RunInitializing, runstore.RunPending, runstore.Reason{Code: "ready"}); err != nil {
		t.Fatal(err)
	}
	claim := runID + "-claim"
	if _, err := runs.ClaimRunnableRun(ctx, claim, 10*time.Minute, nil, resolved.LaneClassTool); err != nil {
		t.Fatal(err)
	}
	if _, err := runs.TransitionRun(ctx, runID, runstore.RunPending, runstore.RunRunning, runstore.Reason{Code: "admitted"}); err != nil {
		t.Fatal(err)
	}
	stageID := runID + "-stage"
	if _, err := runs.CreateStageExecution(ctx, runstore.CreateStageExecutionParams{LaneClass: resolved.LaneClassTool, StageExecutionID: stageID, RunID: runID, StageName: workflow.EntryStage, Attempt: 1, StageSpecSchemaVersion: contracts.APIVersion, StageSpecSnapshot: stageJSON, StageContextSchemaVersion: contracts.APIVersion, StageContext: runstore.StageContextSnapshot{Parameters: map[string]string{}, Artifacts: inputs}}); err != nil {
		t.Fatal(err)
	}
	if _, err := runs.MarkStageAdmitted(ctx, stageID); err != nil {
		t.Fatal(err)
	}
	if err := persistencepostgres.InTx(ctx, pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		_, err := runstore.NewPostgresTxStore(tx, storage).StartStageOperation(ctx, runID, stageID, claim)
		return err
	}); err != nil {
		t.Fatal(err)
	}
	return CandidatePublication{RunID: runID, StageID: stageID, ClaimID: claim, Candidates: *inputs["candidates"].Artifact, Evidence: *inputs["evidence"].Artifact}
}
