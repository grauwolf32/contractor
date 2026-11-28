//go:build integration

package findingintake

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/auditcontract"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/jackc/pgx/v5"
	"testing"
	"time"
)

func TestPostgresCandidateAuditBatchPartialCommitMembershipAndHistoricalReplay(t *testing.T) {
	for _, selection := range []string{"pair", "materials"} {
		t.Run(selection, func(t *testing.T) { candidateAuditBatchPartialCommit(t, selection == "materials") })
	}
}

func candidateAuditBatchPartialCommit(t *testing.T, materialSet bool) {
	ctx := t.Context()
	pool := isolatedFindingPool(t, ctx)
	const owner, project, run = "candidate-audit-owner", "candidate-audit-project", "candidate-audit-run"
	if _, _, err := projectstore.NewPostgresStore(pool).Create(ctx, projectstore.CreateParams{ProjectID: project, OwnerID: owner, Kind: projectstore.KindProject, Name: "Scan batch", IdempotencyKey: "project", RequestDigest: auditdomain.DigestBytes([]byte("project"))}); err != nil {
		t.Fatal(err)
	}
	var operation resolved.ResolvedStage
	grant, origin := auditSubmissionGrantWithWorkflow(t, ctx, pool, owner, project, run, nil, func(workflow *resolved.ResolvedWorkflow) {
		operation = workflow.Stages[workflow.EntryStage]
		operation.Planner = resolved.PlannerRef{}
		operation.Operation = &resolved.ServerOperationConfig{Ref: contracts.PublishFindingsOperation, Arguments: map[string]contracts.ToolArgumentBinding{}}
		operation.Instructions = contracts.ResolvedInstructions{}
		operation.Agents = map[string]resolved.ResolvedAgentBinding{}
		operation.ExecutionConfig = resolved.ResolvedStageExecutionConfig{Agents: map[string]resolved.ResolvedConsumerExecutionConfig{}}
		operation.Context = resolved.StageContext{Artifacts: map[string]resolved.ContextArtifact{"candidates": {Namespace: "builder", Name: "candidates", Required: true}, "evidence": {Namespace: "builder", Name: "evidence", Required: true}}}
		operation.Result.Artifacts = map[string]resolved.ArtifactSlot{}
		operation.WorkflowOutputs = map[string]string{}
		workflow.Stages["publish"] = operation
		entry := workflow.Stages[workflow.EntryStage]
		entry.On.Succeeded = resolved.TransitionAction{Kind: resolved.TransitionNext, NextStage: "publish"}
		workflow.Stages[workflow.EntryStage] = entry
	}, 2)
	storage := artifactpgtest.NewStorage()
	service, err := New(pool, storage)
	if err != nil {
		t.Fatal(err)
	}
	store, _ := artifacts.NewService(storage.Repository(pool)).Run(run)
	scan := &artifacts.ExecutionOrigin{RunID: run, StageExecutionID: grant.StageExecutionID, AllocationID: grant.AllocationID, LogicalAgentName: grant.LogicalAgentName}
	candidates := []NormalizedCandidate{}
	records := []map[string]any{}
	for index := 0; index < 3; index++ {
		item := auditcontract.ExecutionItemKey(min(index, 1))
		candidate := NormalizedCandidate{Key: fmt.Sprintf("%064x", index), Producer: "nuclei", Rule: fmt.Sprintf("rule-%d", index), Title: "Observed problem", Assertion: "The selected scanner matched the selected endpoint.", Location: auditdomain.FindingLocation{URL: fmt.Sprintf("https://target.test/%d", index), Method: "GET"}, Severity: "medium", QualifiedKey: []string{"nuclei", fmt.Sprint(index)}, ItemKey: &item, Incidental: index == 2}
		candidates = append(candidates, candidate)
		records = append(records, map[string]any{"key": candidate.Key, "matched": true})
	}
	proof, _ := json.Marshal(map[string]any{"schema": "contractor.finding.scanner-evidence.v1", "records": records})
	saved, err := store.Write(ctx, contracts.ArtifactRef{Namespace: grant.Namespace, Name: "evidence"}, artifacts.Payload{MediaType: CandidateEvidenceMediaType, Data: proof, ExecutionOrigin: scan}, nil)
	if err != nil {
		t.Fatal(err)
	}
	proofMeta, err := store.Metadata(ctx, saved.Ref)
	if err != nil {
		t.Fatal(err)
	}
	raw, _ := json.Marshal(CandidateBatch{Schema: CandidateSchema, Complete: false, Candidates: candidates, Skipped: map[string]int{}, Evidence: proofMeta.Exact()})
	batch, err := store.Write(ctx, contracts.ArtifactRef{Namespace: grant.Namespace, Name: "candidates"}, artifacts.Payload{MediaType: CandidateMediaType, Data: raw, ExecutionOrigin: scan}, nil)
	if err != nil {
		t.Fatal(err)
	}
	batchMeta, err := store.Metadata(ctx, batch.Ref)
	if err != nil {
		t.Fatal(err)
	}
	inputs := map[string]runstore.PinnedContextArtifact{"candidates": {Required: true, Artifact: &batch.Ref}, "evidence": {Required: true, Artifact: &saved.Ref}}
	var selection contracts.ArtifactRef
	if materialSet {
		materials := []contracts.MaterialDescriptor{}
		for _, exact := range []contracts.ExactArtifact{batchMeta.Exact(), proofMeta.Exact()} {
			materials = append(materials, contracts.MaterialDescriptor{ScopeKind: "run", ScopeID: run, Ref: exact.Ref, Digest: exact.Digest, MediaType: exact.MediaType, SizeBytes: exact.SizeBytes})
		}
		data, err := json.Marshal(contracts.MaterialSet{Schema: contracts.MaterialSetSchema, Materials: materials})
		if err != nil {
			t.Fatal(err)
		}
		selected, err := store.Write(ctx, contracts.ArtifactRef{Namespace: grant.Namespace, Name: "materials"}, artifacts.Payload{MediaType: "application/json", Data: data}, nil)
		if err != nil {
			t.Fatal(err)
		}
		selection = selected.Ref
		operation.Context.Artifacts = map[string]resolved.ContextArtifact{"materials": {Namespace: "builder", Name: "materials", Required: true}}
		inputs = map[string]runstore.PinnedContextArtifact{"materials": {Required: true, Artifact: &selection}}
	}
	runs := runstore.NewPostgresStore(pool, storage)
	claim := run + "-scheduler"
	if _, err := runs.ClaimRunnableRun(ctx, claim, 10*time.Minute, nil, resolved.LaneClassModel); err != nil {
		t.Fatal(err)
	}
	stage := run + "-publication"
	spec, _ := json.Marshal(operation)
	if _, err := runs.CreateStageExecution(ctx, runstore.CreateStageExecutionParams{LaneClass: resolved.LaneClassTool, StageExecutionID: stage, RunID: run, StageName: "publish", Attempt: 1, StageSpecSchemaVersion: contracts.APIVersion, StageSpecSnapshot: spec, StageContextSchemaVersion: contracts.APIVersion, StageContext: runstore.StageContextSnapshot{Parameters: map[string]string{}, Artifacts: inputs}}); err != nil {
		t.Fatal(err)
	}
	if _, err := runs.MarkStageAdmitted(ctx, stage); err != nil {
		t.Fatal(err)
	}
	if err := persistencepostgres.InTx(ctx, pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		_, err := runstore.NewPostgresTxStore(tx, storage).StartStageOperation(ctx, run, stage, claim)
		return err
	}); err != nil {
		t.Fatal(err)
	}
	request := CandidatePublication{RunID: run, StageID: stage, ClaimID: claim, Candidates: batch.Ref, Evidence: saved.Ref}
	publish := func(ctx context.Context) (CandidatePublicationSummary, error) {
		if materialSet {
			return service.PublishCandidateSet(ctx, run, stage, claim, selection)
		}
		return service.PublishCandidates(ctx, request)
	}
	interrupted, cancel := context.WithCancel(ctx)
	defer cancel()
	attempts := 0
	service.afterPrepare = func(context.Context) {
		attempts++
		if attempts == 2 {
			cancel()
		}
	}
	partial, err := publish(interrupted)
	service.afterPrepare = nil
	if !errors.Is(err, context.Canceled) || partial.Published != 1 || len(partial.Sources) != 1 {
		t.Fatalf("partial=%+v err=%v", partial, err)
	}
	committed, err := service.GetProjectReceipt(ctx, owner, project, partial.Sources[0].ReceiptID)
	if err != nil || committed.Origin.Audit == nil || committed.Origin.Audit.TaskID != origin.TaskID {
		t.Fatalf("committed first member=%+v %v", committed.Origin, err)
	}
	completed, err := publish(ctx)
	if err != nil || completed.Replayed != 1 || completed.Published != 2 {
		t.Fatalf("reconciliation=%+v %v", completed, err)
	}
	imports := []ImportRequest{}
	for index, source := range completed.Sources {
		receipt, err := service.GetProjectReceipt(ctx, owner, project, source.ReceiptID)
		if err != nil {
			t.Fatal(err)
		}
		want := origin.TaskID
		if index > 0 {
			want = run + "-task-1"
		}
		if receipt.Origin.Audit == nil || receipt.Origin.Audit.TaskID != want || (receipt.Origin.Audit.Hypothesis != nil) != (index == 1) {
			t.Fatalf("member %d origin=%+v", index, receipt.Origin)
		}
		imports = append(imports, ImportRequest{OwnerID: owner, AuditID: origin.AuditID, RunID: run, Proposal: receipt.Proposal.Ref})
	}
	if err := service.RetainAuditCollectionBatch(ctx, imports); err != nil {
		t.Fatal(err)
	}
	refs, err := service.AuditExecutionReceiptReferences(ctx, owner, origin.AuditID, origin.ExecutionID, run)
	if err != nil || len(refs[origin.TaskID].FindingIDs) != 1 || len(refs[run+"-task-1"].FindingIDs) != 2 {
		t.Fatalf("Task results=%+v %v", refs, err)
	}
	retained := map[string]contracts.ExactArtifact{}
	for name, exact := range map[string]contracts.ExactArtifact{"candidates": batchMeta.Exact(), "evidence": proofMeta.Exact()} {
		retained[name], err = retainFindingArtifact(ctx, artifacts.NewService(storage.Repository(pool)), run, project, "scanner-inputs", name, exact)
		if err != nil {
			t.Fatal(err)
		}
	}
	replay := candidateOperationRun(t, ctx, pool, owner, project, "candidate-audit-historical-replay", retained)
	replayed, err := service.PublishCandidates(ctx, replay)
	if err != nil || replayed.Replayed != 3 || replayed.Published != 0 {
		t.Fatalf("historical replay=%+v %v", replayed, err)
	}
	var newMembers int
	if err := pool.QueryRow(ctx, `SELECT count(*) FROM finding_proposal_receipts WHERE run_id=$1`, replay.RunID).Scan(&newMembers); err != nil || newMembers != 0 {
		t.Fatalf("replay reassigned historical membership: %d %v", newMembers, err)
	}
}
