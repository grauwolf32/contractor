package auditservice

import (
	"encoding/json"
	"errors"
	"fmt"
	"sync"
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

type judgmentFixture struct {
	commonTaskServiceFixture
	grant     controlplane.AllocationGrant
	execution auditstore.Execution
	tasks     []auditstore.Task
	claim     auditstore.ControllerClaim
}

func newJudgmentFixture(t *testing.T, count int, checking bool) judgmentFixture {
	t.Helper()
	f := newCommonTaskServiceFixture(t, func(p *auditcontract.Program) {
		p.Limits.MaxEvidenceBytes = 16 << 20
		binding := p.Workflows["inspect"]
		binding.Ref = "audit-contract-check@1"
		binding.Inputs["context"] = auditcontract.InputMapping{Source: "task-context"}
		binding.Outputs = map[string]string{"result": "result"}
		p.Workflows["inspect"] = binding
		initial := p.InitialWork[0]
		p.InitialWork = nil
		if checking {
			p.Hypotheses["claim"] = auditcontract.HypothesisSeed{Statement: "Selected source suggests a scoped boundary", Scope: map[string]string{}, Grounds: "Fixture ground, not accepted check evidence"}
		}
		for index := range count {
			item := initial
			item.Key = fmt.Sprintf("item-%02d", index)
			if checking {
				item.Hypothesis = "claim"
			}
			p.InitialWork = append(p.InitialWork, item)
		}
	}, func(workflow *resolved.ResolvedWorkflow) {
		stage := workflow.Stages[workflow.EntryStage]
		agent := stage.Agents["builder"]
		agent.Template.Summarizer = nil
		agent.Template.Toolsets = append(agent.Template.Toolsets, contracts.ToolsetSelection{Ref: contracts.ToolsetRef{ToolsetID: "audit-results", Version: "2"}, Tools: []string{"read_audit_task", "submit_result", "submit_results", "evidence"}})
		stage.Agents["builder"] = agent
		workflow.Stages[workflow.EntryStage] = stage
	})
	ctx := t.Context()
	storage := artifactpgtest.NewStorage()
	store := auditstoretest.NewStore(f.pool, storage)
	if _, err := f.pool.Exec(ctx, `UPDATE audits SET state='active',dispatch_state='open' WHERE audit_id=$1`, f.audit.AuditID); err != nil {
		t.Fatal(err)
	}
	claims, err := store.Claim(ctx, auditstore.ClaimParams{HolderID: "judgment-controller", Lease: time.Minute, Limit: 1})
	if err != nil || len(claims) != 1 {
		t.Fatalf("claim: %v", err)
	}
	claim := claims[0]
	snapshot, err := store.GetReconcileSnapshot(ctx, claim)
	if err != nil {
		t.Fatal(err)
	}
	complete := false
	for range 128 {
		result, err := f.service.AcceptInitialTasks(ctx, claim, snapshot)
		if err != nil {
			t.Fatal(err)
		}
		if result.Complete {
			complete = true
			break
		}
		if !result.Changed {
			t.Fatal("initial intake made no progress")
		}
		snapshot, err = store.GetReconcileSnapshot(ctx, claim)
		if err != nil {
			t.Fatal(err)
		}
	}
	if !complete {
		t.Fatal("bounded initial intake did not finish")
	}
	page, err := f.service.Tasks(ctx, auditstore.ListTasksParams{OwnerID: "owner", AuditID: f.audit.AuditID, Limit: 200})
	if err != nil || len(page.Tasks) != count {
		t.Fatalf("tasks=%d error=%v", len(page.Tasks), err)
	}
	ids := make([]string, 0, count)
	for _, task := range page.Tasks {
		ids = append(ids, task.Document.TaskID)
	}
	execution, _, err := store.SelectTasks(ctx, auditstore.SelectTasksParams{Claim: claim, TaskIDs: ids})
	if err != nil {
		t.Fatal(err)
	}
	workflow := f.pinned.Workflows["inspect"]
	raw, _ := resolved.EncodeResolvedWorkflowSnapshot(workflow)
	const runID = "judgment-run"
	err = postgres.InTx(ctx, f.pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		runs := runstore.NewPostgresStore(tx, storage)
		_, err := runs.CreateAuditRun(ctx, runstore.CreateAuditRunParams{CreateRunParams: runstore.CreateRunParams{RunID: runID, OwnerID: "owner", ProjectID: &f.audit.ProjectID, WorkflowName: workflow.Ref.Name, WorkflowVersion: workflow.Ref.Version, WorkflowSchemaVersion: contracts.APIVersion, WorkflowSnapshot: raw, Parameters: map[string]string{}, RuntimeConfig: runtimeconfigtest.BuiltInRunSnapshot(), LaneClass: resolved.EntryLaneClass(workflow)}, AuditExecutionID: execution.ExecutionID, AuditSubmissionKey: execution.SubmissionKey})
		if err != nil {
			return err
		}
		revision := "context-r1"
		if err := runs.SetAuditCompletion(ctx, runID, runstore.AuditCompletionSnapshot{Stage: workflow.EntryStage, Agent: "builder", Contract: contracts.WorkerCompletionContract{Kind: contracts.AuditCheckResultsV1, Context: contracts.ArtifactRef{Namespace: "inputs", Name: "context", Revision: &revision}, ResultArtifact: contracts.ArtifactRef{Namespace: workflow.Stages[workflow.EntryStage].Agents["builder"].Namespace, Name: "report"}}}); err != nil {
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
		if _, err := runs.TransitionRun(ctx, runID, states[0], states[1], runstore.Reason{Code: "fixture"}); err != nil {
			t.Fatal(err)
		}
	}
	spec := workflow.Stages[workflow.EntryStage]
	stageRaw, _ := json.Marshal(spec)
	stage, err := runs.CreateStageExecution(ctx, runstore.CreateStageExecutionParams{LaneClass: resolved.LaneClassModel, StageExecutionID: "judgment-stage", RunID: runID, StageName: workflow.EntryStage, Attempt: 1, StageSpecSchemaVersion: contracts.APIVersion, StageSpecSnapshot: stageRaw, StageContextSchemaVersion: contracts.APIVersion, StageContext: runstore.StageContextSnapshot{}})
	if err != nil {
		t.Fatal(err)
	}
	grant := controlplane.AllocationGrant{AllocationID: "judgment-allocation", RunID: runID, StageExecutionID: stage.StageExecutionID, LogicalAgentName: "builder", Namespace: spec.Agents["builder"].Namespace, ReadPolicy: controlplane.ReadCurrentRun, WritePolicy: controlplane.WriteInputsAndIntermediates}
	return judgmentFixture{commonTaskServiceFixture: f, grant: grant, execution: execution, tasks: page.Tasks, claim: claim}
}

func judgmentInput(key, text, identity string, base uint64) auditcontract.JudgmentSubmission {
	return auditcontract.JudgmentSubmission{APIVersion: contracts.APIVersion, InvocationID: "invocation", SubmissionID: identity, Mode: "single", Items: []auditcontract.JudgmentUpdate{{BaseGeneration: base, Conclusion: auditcontract.Conclusion{ItemKey: key, Rationale: text, TestedConditions: []string{}, Limitations: []string{}, Evidence: []auditcontract.Material{}}}}}
}

func TestSharedPostgresJudgmentCorrectionReplayAndFences(t *testing.T) {
	f := newJudgmentFixture(t, 1, false)
	ctx := t.Context()
	a := judgmentInput(auditcontract.ExecutionItemKey(0), "Accepted exploratory A", "a", 0)
	first, err := f.service.RecordJudgments(ctx, f.grant, a)
	if err != nil || first.Generations[0].Generation != 1 || first.Replayed {
		t.Fatalf("first: %+v %v", first, err)
	}
	b := judgmentInput(a.Items[0].Conclusion.ItemKey, "Accepted correction B", "b", 1)
	second, err := f.service.RecordJudgments(ctx, f.grant, b)
	if err != nil || second.Generations[0].Generation != 2 {
		t.Fatalf("correction: %+v %v", second, err)
	}
	old, err := f.service.RecordJudgments(ctx, f.grant, a)
	if err != nil || !old.Replayed || !old.Material.Ref.SameExact(first.Material.Ref) {
		t.Fatalf("old receipt: %+v %v", old, err)
	}
	links, err := auditstoretest.NewStore(f.pool, artifactpgtest.NewStorage()).ListExecutionJudgments(ctx, f.audit.AuditID, f.execution.ExecutionID)
	if err != nil || len(links) != 2 {
		t.Fatalf("receipts=%d error=%v", len(links), err)
	}
	stale := judgmentInput(a.Items[0].Conclusion.ItemKey, "Unaccepted stale C", "stale", 1)
	var conflict *JudgmentStaleError
	if _, err := f.service.RecordJudgments(ctx, f.grant, stale); !errors.As(err, &conflict) || conflict.Generations[0].Generation != 2 {
		t.Fatalf("stale: %v", err)
	}
	changed := a
	changed.Items = append([]auditcontract.JudgmentUpdate{}, a.Items...)
	changed.Items[0].Conclusion.Rationale = "Changed receipt identity"
	if _, err := f.service.RecordJudgments(ctx, f.grant, changed); !errors.Is(err, auditstore.ErrConflict) {
		t.Fatalf("changed key: %v", err)
	}
	fenced := f.grant
	fenced.WriteFenced = true
	if replay, err := f.service.RecordJudgments(ctx, fenced, b); err != nil || !replay.Replayed {
		t.Fatalf("fenced replay: %v", err)
	}
	if _, err := f.service.RecordJudgments(ctx, fenced, judgmentInput(a.Items[0].Conclusion.ItemKey, "Late", "late", 2)); !errors.Is(err, auditstore.ErrPrecondition) {
		t.Fatalf("late: %v", err)
	}
	scope, _ := artifacts.AuditScope(f.audit.AuditID)
	owned, _ := artifacts.NewService(artifactpgtest.NewStorage().Repository(f.pool)).Scoped(scope)
	read, err := owned.ReadExact(ctx, second.Material.Exact())
	if err != nil {
		t.Fatal(err)
	}
	var doc auditcontract.ResultDocument
	if json.Unmarshal(read.Payload.Data, &doc) != nil || len(doc.Items) != 1 || doc.Items[0].Rationale != "Accepted correction B" || doc.Items[0].Author != "model" || doc.Items[0].Origin.TaskID != f.tasks[0].Document.TaskID {
		t.Fatal("stored latest judgment/origin changed")
	}
}

func TestSharedPostgresJudgmentSiblingConflictAndAtomicBatch(t *testing.T) {
	f := newJudgmentFixture(t, 2, false)
	ctx := t.Context()
	key := auditcontract.ExecutionItemKey(0)
	if _, err := f.service.RecordJudgments(ctx, f.grant, judgmentInput(key, "Initial", "initial", 0)); err != nil {
		t.Fatal(err)
	}
	start := make(chan struct{})
	errorsCh := make(chan error, 2)
	var group sync.WaitGroup
	for _, name := range []string{"left", "right"} {
		group.Add(1)
		go func() {
			defer group.Done()
			<-start
			_, err := f.service.RecordJudgments(ctx, f.grant, judgmentInput(key, name, name, 1))
			errorsCh <- err
		}()
	}
	close(start)
	group.Wait()
	close(errorsCh)
	accepted, rejected := 0, 0
	for err := range errorsCh {
		var stale *JudgmentStaleError
		if err == nil {
			accepted++
		} else if errors.As(err, &stale) {
			rejected++
		} else {
			t.Fatal(err)
		}
	}
	if accepted != 1 || rejected != 1 {
		t.Fatalf("siblings accepted=%d rejected=%d", accepted, rejected)
	}
	batch := judgmentInput(key, "Attempted batch replacement", "batch", 2)
	batch.Mode = "batch"
	batch.Items = append(batch.Items, judgmentInput(auditcontract.ExecutionItemKey(1), "New member", "unused", 0).Items[0])
	if _, err := f.service.RecordJudgments(ctx, f.grant, batch); !errors.Is(err, auditstore.ErrConflict) {
		t.Fatalf("batch replacement: %v", err)
	}
	links, err := auditstoretest.NewStore(f.pool, artifactpgtest.NewStorage()).ListExecutionJudgments(ctx, f.audit.AuditID, f.execution.ExecutionID)
	if err != nil || len(links) != 2 {
		t.Fatalf("partial batch committed: %d %v", len(links), err)
	}
}

func TestSharedPostgresJudgmentBatchAndEvidenceMembership(t *testing.T) {
	t.Run("byte-bounded-batch", func(t *testing.T) {
		// The existing owner reserves 48 KiB of metadata per Task inside its
		// 1 MiB selection budget. Exercise a realistic full page without
		// weakening that independent bound to manufacture a 64-row execution.
		const count = 20
		f := newJudgmentFixture(t, count, false)
		input := judgmentInput(auditcontract.ExecutionItemKey(0), "Exploratory record", "batch-20", 0)
		input.Mode = "batch"
		input.Items = nil
		for index := range count {
			input.Items = append(input.Items, judgmentInput(auditcontract.ExecutionItemKey(index), "Exploratory record", "unused", 0).Items[0])
		}
		receipt, err := f.service.RecordJudgments(t.Context(), f.grant, input)
		if err != nil || len(receipt.Generations) != count {
			t.Fatalf("maximum batch: %d %v", len(receipt.Generations), err)
		}
	})
	t.Run("explicit-check-association", func(t *testing.T) {
		f := newJudgmentFixture(t, 1, true)
		ctx := t.Context()
		key := auditcontract.ExecutionItemKey(0)
		input := judgmentInput(key, "Supported scoped claim", "supported", 0)
		verdict := "supported"
		input.Items[0].Conclusion.Verdict = &verdict
		input.Items[0].Conclusion.TestedConditions = []string{"source inspected"}
		if _, err := f.service.RecordJudgments(ctx, f.grant, input); !errors.Is(err, ErrInvalid) {
			t.Fatalf("ground-only conclusion: %v", err)
		}
		run, _ := artifacts.NewService(artifactpgtest.NewStorage().Repository(f.pool)).Run(f.grant.RunID)
		written, err := run.Write(ctx, contracts.ArtifactRef{Namespace: f.grant.Namespace, Name: "observation"}, artifacts.Payload{MediaType: "text/plain", Data: []byte("selected source")}, nil)
		if err != nil {
			t.Fatal(err)
		}
		note, err := f.service.RecordEvidence(ctx, f.grant, auditcontract.EvidenceSubmission{APIVersion: contracts.APIVersion, InvocationID: "invocation", SubmissionID: "note", ItemKey: key, Sources: []contracts.ArtifactRef{written.Ref}, Interpretation: "Supports the scoped claim only", TestedConditions: []string{"source inspected"}, Limitations: []string{}})
		if err != nil {
			t.Fatal(err)
		}
		input.Items[0].Conclusion.Evidence = append(note.Sources, note.Material)
		if _, err := f.service.RecordJudgments(ctx, f.grant, input); err != nil {
			t.Fatal(err)
		}
	})
}
