package auditservice

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/auditstore/auditstoretest"
	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/auditcontract"
	"github.com/grauwolf32/contractor/internal/controlplane"
	"github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/runtimeconfig/runtimeconfigtest"
	"github.com/jackc/pgx/v5"
)

func TestSharedPostgresIncrementalEvidenceExactReplayFencesAndBounds(t *testing.T) {
	f := newCommonTaskServiceFixture(t, func(program *auditcontract.Program) {
		program.Limits.MaxEvidenceBytes = 16 << 20
		binding := program.Workflows["inspect"]
		binding.Ref = "audit-contract-check@1"
		binding.Inputs["context"] = auditcontract.InputMapping{Source: "task-context"}
		binding.Outputs = map[string]string{"result": "result"}
		program.Workflows["inspect"] = binding
	}, func(workflow *resolved.ResolvedWorkflow) {
		stage := workflow.Stages[workflow.EntryStage]
		agent := stage.Agents["builder"]
		agent.Template.Summarizer = nil
		agent.Template.Toolsets = append(agent.Template.Toolsets, contracts.ToolsetSelection{Ref: contracts.ToolsetRef{ToolsetID: "audit-results", Version: "2"}, Tools: []string{"read_audit_task", "submit_check_result", "submit_check_results", "evidence"}})
		stage.Agents["builder"] = agent
		workflow.Stages[workflow.EntryStage] = stage
	})
	ctx := t.Context()
	storage := artifactpgtest.NewStorage()
	store := auditstoretest.NewStore(f.pool, storage)
	if _, err := f.pool.Exec(ctx, `UPDATE audits SET state='active',dispatch_state='open' WHERE audit_id=$1`, f.audit.AuditID); err != nil {
		t.Fatal(err)
	}
	claims, err := store.Claim(ctx, auditstore.ClaimParams{HolderID: "evidence-controller", Lease: time.Minute, Limit: 1})
	if err != nil || len(claims) != 1 {
		t.Fatalf("claim: %v", err)
	}
	claim := claims[0]
	snapshot, err := store.GetReconcileSnapshot(ctx, claim)
	if err != nil {
		t.Fatal(err)
	}
	if result, err := f.service.AcceptInitialTasks(ctx, claim, snapshot); err != nil || !result.Complete {
		t.Fatalf("initial: %+v %v", result, err)
	}
	page, err := f.service.Tasks(ctx, auditstore.ListTasksParams{OwnerID: "owner", AuditID: f.audit.AuditID, Limit: 2})
	if err != nil || len(page.Tasks) != 1 {
		t.Fatalf("tasks: %v", err)
	}
	execution, _, err := store.SelectTasks(ctx, auditstore.SelectTasksParams{Claim: claim, TaskIDs: []string{page.Tasks[0].Document.TaskID}})
	if err != nil {
		t.Fatal(err)
	}
	workflow := f.pinned.Workflows["inspect"]
	raw, _ := resolved.EncodeResolvedWorkflowSnapshot(workflow)
	const runID = "evidence-run"
	err = postgres.InTx(ctx, f.pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		_, err := runstore.NewPostgresStore(tx, storage).CreateAuditRun(ctx, runstore.CreateAuditRunParams{CreateRunParams: runstore.CreateRunParams{RunID: runID, OwnerID: "owner", ProjectID: &f.audit.ProjectID, WorkflowName: workflow.Ref.Name, WorkflowVersion: workflow.Ref.Version, WorkflowSchemaVersion: contracts.APIVersion, WorkflowSnapshot: raw, Parameters: map[string]string{}, RuntimeConfig: runtimeconfigtest.BuiltInRunSnapshot(), LaneClass: resolved.EntryLaneClass(workflow)}, AuditExecutionID: execution.ExecutionID, AuditSubmissionKey: execution.SubmissionKey})
		if err != nil {
			return err
		}
		_, err = auditstoretest.NewStore(tx, storage).BindRun(ctx, auditstore.BindRunParams{Claim: claim, ExecutionID: execution.ExecutionID, RunID: runID})
		return err
	})
	if err != nil {
		t.Fatal(err)
	}
	runs := runstore.NewPostgresStore(f.pool, storage)
	for _, states := range [][2]runstore.WorkflowRunState{{runstore.RunInitializing, runstore.RunPending}, {runstore.RunPending, runstore.RunRunning}} {
		if _, err := runs.TransitionRun(ctx, runID, states[0], states[1], runstore.Reason{Code: "evidence-fixture"}); err != nil {
			t.Fatal(err)
		}
	}
	spec := workflow.Stages[workflow.EntryStage]
	stageRaw, _ := json.Marshal(spec)
	stage, err := runs.CreateStageExecution(ctx, runstore.CreateStageExecutionParams{LaneClass: resolved.LaneClassModel, StageExecutionID: "evidence-stage", RunID: runID, StageName: workflow.EntryStage, Attempt: 1, StageSpecSchemaVersion: contracts.APIVersion, StageSpecSnapshot: stageRaw, StageContextSchemaVersion: contracts.APIVersion, StageContext: runstore.StageContextSnapshot{}})
	if err != nil {
		t.Fatal(err)
	}
	grant := controlplane.AllocationGrant{AllocationID: "evidence-allocation", RunID: runID, StageExecutionID: stage.StageExecutionID, LogicalAgentName: "builder", Namespace: spec.Agents["builder"].Namespace, ReadPolicy: controlplane.ReadCurrentRun, WritePolicy: controlplane.WriteInputsAndIntermediates}
	art := artifacts.NewService(storage.Repository(f.pool))
	run, _ := art.Run(runID)
	written, err := run.Write(ctx, contracts.ArtifactRef{Namespace: grant.Namespace, Name: "observation"}, artifacts.Payload{MediaType: "text/plain", Data: []byte("original selected observation")}, nil)
	if err != nil {
		t.Fatal(err)
	}
	submission := auditcontract.EvidenceSubmission{APIVersion: contracts.APIVersion, InvocationID: "invocation", SubmissionID: "source-000", ItemKey: auditcontract.ExecutionItemKey(0), Sources: []contracts.ArtifactRef{written.Ref}, Interpretation: "This source supports a bounded observation, not universal verification.", TestedConditions: []string{"source inspected"}, Limitations: []string{}}
	receipt, err := f.service.RecordEvidence(ctx, grant, submission)
	if err != nil || receipt.Replayed || len(receipt.SelectedSources) != 1 {
		t.Fatalf("accept: %+v %v", receipt, err)
	}
	if receipt.Material.ScopeID != f.audit.AuditID || receipt.Sources[0].ScopeKind != "audit" || receipt.SelectedSources[0].ScopeID != runID {
		t.Fatal("wrong retained origin")
	}
	links, err := store.ListExecutionEvidence(ctx, f.audit.AuditID, execution.ExecutionID)
	if err != nil || len(links) != 1 {
		t.Fatalf("accepted prefix: %d %v", len(links), err)
	}
	if _, err := run.Write(ctx, contracts.ArtifactRef{Namespace: grant.Namespace, Name: "observation"}, artifacts.Payload{MediaType: "text/plain", Data: []byte("later unrelated observation")}, written.Ref.Revision); err != nil {
		t.Fatal(err)
	}
	selected, err := run.ReadExact(ctx, receipt.SelectedSources[0].Exact())
	if err != nil || string(selected.Payload.Data) != "original selected observation" {
		t.Fatalf("exact source changed: %v", err)
	}
	replay, err := f.service.RecordEvidence(ctx, grant, submission)
	if err != nil || !replay.Replayed || !replay.Material.Ref.SameExact(receipt.Material.Ref) {
		t.Fatalf("replay: %+v %v", replay, err)
	}
	changed := submission
	changed.Interpretation = "different intent"
	if _, err := f.service.RecordEvidence(ctx, grant, changed); !errors.Is(err, auditstore.ErrConflict) {
		t.Fatalf("changed identity: %v", err)
	}
	invalid := submission
	invalid.SubmissionID = "foreign-item"
	invalid.ItemKey = auditcontract.ExecutionItemKey(1)
	if _, err := f.service.RecordEvidence(ctx, grant, invalid); err == nil {
		t.Fatal("foreign item accepted")
	}
	fenced := grant
	fenced.WriteFenced = true
	if replay, err := f.service.RecordEvidence(ctx, fenced, submission); err != nil || !replay.Replayed {
		t.Fatalf("lost acknowledgement after fence: %v", err)
	}
	invalid.SubmissionID = "new-after-fence"
	invalid.ItemKey = auditcontract.ExecutionItemKey(0)
	if _, err := f.service.RecordEvidence(ctx, fenced, invalid); err == nil {
		t.Fatal("new write passed fence")
	}

	before, err := f.service.Get(ctx, "owner", f.audit.AuditID)
	if err != nil {
		t.Fatal(err)
	}
	large, err := run.Write(ctx, contracts.ArtifactRef{Namespace: grant.Namespace, Name: "large-source"}, artifacts.Payload{MediaType: "application/octet-stream", Data: bytes.Repeat([]byte{'x'}, 7<<20)}, nil)
	if err != nil {
		t.Fatal(err)
	}
	oversized := submission
	oversized.SubmissionID = "oversized"
	oversized.Sources = []contracts.ArtifactRef{large.Ref}
	if _, err := f.service.RecordEvidence(ctx, grant, oversized); !errors.Is(err, auditstore.ErrEvidenceBudgetExhausted) {
		t.Fatalf("reserved result/stop capacity: %v", err)
	}
	after, err := f.service.Get(ctx, "owner", f.audit.AuditID)
	if err != nil || after.Revision != before.Revision || after.RetainedEvidenceBytes != before.RetainedEvidenceBytes {
		t.Fatalf("failed evidence changed the Audit: %v", err)
	}
	// Real persisted receipts, not a counter-only fake, exercise the collection cap.
	for index := 1; index < 128; index++ {
		next := submission
		next.SubmissionID = fmt.Sprintf("source-%03d", index)
		if _, err := f.service.RecordEvidence(ctx, grant, next); err != nil {
			t.Fatalf("record %d: %v", index, err)
		}
	}
	overflow := submission
	overflow.SubmissionID = "source-overflow"
	if _, err := f.service.RecordEvidence(ctx, grant, overflow); !errors.Is(err, auditstore.ErrEvidenceBudgetExhausted) {
		t.Fatalf("collection cap: %v", err)
	}
	links, err = store.ListExecutionEvidence(ctx, f.audit.AuditID, execution.ExecutionID)
	if err != nil || len(links) != 128 {
		t.Fatalf("retained prefix: %d %v", len(links), err)
	}
}
