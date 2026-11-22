package auditservice

import (
	"bytes"
	"context"
	"crypto/rand"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/artifacts/blob"
	inline "github.com/grauwolf32/contractor/internal/artifacts/blob/backends/postgres"
	artifactpostgres "github.com/grauwolf32/contractor/internal/artifacts/postgres"
	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"
	"github.com/grauwolf32/contractor/internal/auditimport"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/auditstore/auditstoretest"
	"github.com/grauwolf32/contractor/internal/auditstore/hookstore"
	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/auditcontract"
	"github.com/grauwolf32/contractor/internal/controlplane"
	"github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/runtimeconfig/runtimeconfigtest"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
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
	return newJudgmentSelectionFixture(t, count, count, checking)
}

func newJudgmentSelectionFixture(t *testing.T, count, selected int, checking bool) judgmentFixture {
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
	for _, task := range page.Tasks[:selected] {
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
	var index string
	if err := f.pool.QueryRow(t.Context(), `SELECT pg_get_indexdef('audit_artifact_links_receipt_order'::regclass)`).Scan(&index); err != nil || !strings.Contains(index, `logical_key COLLATE "C"`) {
		t.Fatalf("bytewise receipt index missing: %v", err)
	}

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

func TestSharedPostgresPartialMixedJudgmentsKeepUnassignedWorkUnchanged(t *testing.T) {
	f := newJudgmentSelectionFixture(t, 5, 4, true)
	ctx := t.Context()
	storage := artifactpgtest.NewStorage()
	store := auditstoretest.NewStore(f.pool, storage)
	art := artifacts.NewService(storage.Repository(f.pool))
	runScope, _ := art.Run(f.grant.RunID)
	written, err := runScope.Write(ctx, contracts.ArtifactRef{Namespace: f.grant.Namespace, Name: "mixed-observation"}, artifacts.Payload{MediaType: "text/plain", Data: []byte("an explicitly selected source")}, nil)
	if err != nil {
		t.Fatal(err)
	}
	var supportedNote auditcontract.EvidenceReceipt
	for _, index := range []int{0, 2} {
		note, err := f.service.RecordEvidence(ctx, f.grant, auditcontract.EvidenceSubmission{APIVersion: contracts.APIVersion, InvocationID: "invocation", SubmissionID: fmt.Sprintf("note-%d", index), ItemKey: auditcontract.ExecutionItemKey(index), Sources: []contracts.ArtifactRef{written.Ref}, Interpretation: "Attributed reading of selected source", TestedConditions: []string{"this source only"}, Limitations: []string{}})
		if err != nil {
			t.Fatal(err)
		}
		if index == 0 {
			supportedNote = note
		}
	}
	for index, verdict := range []string{"supported", "inconclusive"} {
		input := judgmentInput(auditcontract.ExecutionItemKey(index), "An accepted model judgment", fmt.Sprintf("judgment-%d", index), 0)
		input.Items[0].Conclusion.Verdict = &verdict
		if index == 0 {
			input.Items[0].Conclusion.TestedConditions = []string{"this source only"}
			input.Items[0].Conclusion.Evidence = append(supportedNote.Sources, supportedNote.Material)
		} else {
			input.Items[0].Conclusion.Limitations = []string{"No scoped judgment can be made from available data"}
		}
		if _, err := f.service.RecordJudgments(ctx, f.grant, input); err != nil {
			t.Fatal(err)
		}
	}
	runs := runstore.NewPostgresStore(f.pool, storage)
	if _, err := runs.TransitionRun(ctx, f.grant.RunID, runstore.RunRunning, runstore.RunFailed, runstore.Reason{Code: "worker_model_calls_exceeded", Message: "Configured call limit reached"}); err != nil {
		t.Fatal(err)
	}
	cursor, err := runs.GetRunEventCursor(ctx, f.grant.RunID)
	if err != nil {
		t.Fatal(err)
	}
	execution, err := store.ObserveTerminal(ctx, auditstore.ObserveTerminalParams{Claim: f.claim, ExecutionID: f.execution.ExecutionID, RunID: f.grant.RunID, Generation: cursor.Generation, Sequence: uint64(cursor.Sequence)})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := f.pool.Exec(ctx, `UPDATE audits SET state='cancelling' WHERE audit_id=$1`, f.audit.AuditID); err != nil {
		t.Fatal(err)
	}
	audit, err := store.Get(ctx, "owner", f.audit.AuditID)
	if err != nil {
		t.Fatal(err)
	}
	access, err := auditimport.NewArtifactAccess(art, hookstore.NewClaimedArtifacts(f.pool, storage))
	if err != nil {
		t.Fatal(err)
	}
	collector, err := auditimport.New(store, store, runs, access)
	if err != nil {
		t.Fatal(err)
	}
	if changed, err := collector.Collect(ctx, f.claim, auditstore.ReconcileSnapshot{Audit: audit}, execution); err != nil || !changed {
		t.Fatalf("collect mixed results: %t %v", changed, err)
	}
	for index, original := range f.tasks {
		task, err := store.GetTask(ctx, "owner", audit.AuditID, original.Document.TaskID)
		if err != nil {
			t.Fatal(err)
		}
		switch index {
		case 0, 1:
			if task.State != auditstore.TaskSettled || task.Result == nil || task.Stop != nil {
				t.Fatalf("accepted model result was replaced: %+v", task)
			}
		case 2, 3:
			if task.State != auditstore.TaskSettled || task.Result != nil || task.Stop == nil || task.Stop.Author != "system" || task.Stop.Meaning != "no_accepted_conclusion" || task.Stop.Reason.Source != "run" || task.Stop.Reason.Code != "worker_model_calls_exceeded" {
				t.Fatalf("missing system disposition: %+v", task)
			}
			expected := 0
			if index == 2 {
				expected = 2
			}
			if len(task.Stop.Evidence) != expected {
				t.Fatalf("recorded activity manufactured or lost: %+v", task.Stop)
			}
		case 4:
			if task.State != auditstore.TaskReady || task.Attempts != 0 || task.LastExecution != nil || task.Result != nil || task.Stop != nil {
				t.Fatalf("unassigned work acquired a fabricated attempt: %+v", task)
			}
		}
	}
}

func TestSharedPostgresPartialJudgmentAndSystemStopSurviveExhaustedEvidenceBudget(t *testing.T) {
	for _, checking := range []bool{false, true} {
		for _, faultKind := range []string{"none", "judgment-missing", "judgment-corrupt", "note-missing", "note-corrupt", "temporary", "paused-corrupt", "active-missing"} {
			t.Run(fmt.Sprintf("hypothesis=%t/%s", checking, faultKind), func(t *testing.T) {
				f := newJudgmentFixture(t, 2, checking)
				ctx := t.Context()
				storage := artifactpgtest.NewStorage()
				store := auditstoretest.NewStore(f.pool, storage)
				art := artifacts.NewService(storage.Repository(f.pool))
				input := judgmentInput(auditcontract.ExecutionItemKey(0), "Accepted A before interruption", "accepted-a", 0)
				if checking {
					verdict := "inconclusive"
					input.Items[0].Conclusion.Verdict = &verdict
					input.Items[0].Conclusion.Limitations = []string{"The selected boundary remains uncertain"}
				}
				if _, err := f.service.RecordJudgments(ctx, f.grant, input); err != nil {
					t.Fatal(err)
				}
				input.SubmissionID = "accepted-b"
				input.Items[0].BaseGeneration = 1
				input.Items[0].Conclusion.Rationale = "Latest accepted B before interruption"
				latest, err := f.service.RecordJudgments(ctx, f.grant, input)
				if err != nil {
					t.Fatal(err)
				}
				runScope, _ := art.Run(f.grant.RunID)
				written, err := runScope.Write(ctx, contracts.ArtifactRef{Namespace: f.grant.Namespace, Name: "unfinished-observation"}, artifacts.Payload{MediaType: "text/plain", Data: []byte("retained observation without an accepted judgment")}, nil)
				if err != nil {
					t.Fatal(err)
				}
				note, err := f.service.RecordEvidence(ctx, f.grant, auditcontract.EvidenceSubmission{APIVersion: contracts.APIVersion, InvocationID: "invocation", SubmissionID: "unfinished-note", ItemKey: auditcontract.ExecutionItemKey(1), Sources: []contracts.ArtifactRef{written.Ref}, Interpretation: "The observation alone does not complete the check", TestedConditions: []string{"Only this source was inspected"}, Limitations: []string{"No conclusion submitted"}})
				if err != nil {
					t.Fatal(err)
				}
				runs := runstore.NewPostgresStore(f.pool, storage)
				if _, err := runs.TransitionRun(ctx, f.grant.RunID, runstore.RunRunning, runstore.RunFailed, runstore.Reason{Code: "worker_model_calls_exceeded", Message: "The configured model call limit was reached"}); err != nil {
					t.Fatal(err)
				}
				cursor, err := runs.GetRunEventCursor(ctx, f.grant.RunID)
				if err != nil {
					t.Fatal(err)
				}
				execution, err := store.ObserveTerminal(ctx, auditstore.ObserveTerminalParams{Claim: f.claim, ExecutionID: f.execution.ExecutionID, RunID: f.grant.RunID, Generation: cursor.Generation, Sequence: uint64(cursor.Sequence)})
				if err != nil {
					t.Fatal(err)
				}
				// Exercise zero remaining selected-material bytes. Stop disposition is
				// owner metadata, and already held source/result links cost zero.
				state := auditstore.AuditCancelling
				if strings.HasPrefix(faultKind, "paused-") {
					state = auditstore.AuditPaused
				}
				if strings.HasPrefix(faultKind, "active-") {
					state = auditstore.AuditActive
				}
				if _, err := f.pool.Exec(ctx, `UPDATE audits SET state=$2,max_evidence_bytes=retained_evidence_bytes WHERE audit_id=$1`, f.audit.AuditID, state); err != nil {
					t.Fatal(err)
				}
				audit, err := store.Get(ctx, "owner", f.audit.AuditID)
				if err != nil {
					t.Fatal(err)
				}
				access, err := auditimport.NewArtifactAccess(art, hookstore.NewClaimedArtifacts(f.pool, storage))
				if err != nil {
					t.Fatal(err)
				}
				collector, err := auditimport.New(store, store, runs, access)
				if err != nil {
					t.Fatal(err)
				}
				snapshot := auditstore.ReconcileSnapshot{Audit: audit}
				if faultKind != "none" {
					schema := "contractor.audit.result.v1"
					if strings.HasPrefix(faultKind, "note-") {
						schema = "contractor.audit.evidence.v1"
					}
					fault := blob.ErrIntegrity
					if strings.HasSuffix(faultKind, "missing") {
						fault = blob.ErrMissing
					}
					temporary := faultKind == "temporary"
					if temporary {
						fault = errors.New("temporary storage failure")
					}
					observed := &observedReceiptStore{fault: fault, faultSchema: schema}
					broken := artifacts.NewService(observedReceiptStorage(observed).Repository(f.pool))
					brokenAccess, err := auditimport.NewArtifactAccess(broken, hookstore.NewClaimedArtifacts(f.pool, storage))
					if err != nil {
						t.Fatal(err)
					}
					brokenCollector, err := auditimport.New(store, store, runs, brokenAccess)
					if err != nil {
						t.Fatal(err)
					}
					changed, err := brokenCollector.Collect(ctx, f.claim, snapshot, execution)
					if temporary {
						if changed || !errors.Is(err, fault) {
							t.Fatalf("transient read changed collection: %t %v", changed, err)
						}
						if _, err := store.CollectionReceipt(ctx, audit.AuditID, execution.ExecutionID); !errors.Is(err, auditstore.ErrNotFound) {
							t.Fatalf("transient read receipt: %v", err)
						}
					} else {
						if !changed || err != nil {
							t.Fatalf("permanent read not settled: %t %v", changed, err)
						}
						receipt, err := store.CollectionReceipt(ctx, audit.AuditID, execution.ExecutionID)
						if err != nil || receipt.Disposition != auditstore.CollectionContractInvalid {
							t.Fatalf("invalid receipt: %+v %v", receipt, err)
						}
						for _, task := range f.tasks {
							item, err := store.GetTask(ctx, "owner", audit.AuditID, task.Document.TaskID)
							if err != nil || item.State != auditstore.TaskSettled || item.Result != nil || item.Stop != nil || item.GapCode == nil || *item.GapCode != "collection-contract-invalid" {
								t.Fatalf("untrusted projection: %+v %v", item, err)
							}
						}
						reads := observed.reads
						if changed, err := brokenCollector.Collect(ctx, f.claim, snapshot, execution); changed || err != nil || observed.reads != reads {
							t.Fatalf("receipt replay reread corrupt data: %t %v", changed, err)
						}
						scope, _ := artifacts.AuditScope(audit.AuditID)
						owned, _ := art.Scoped(scope)
						if _, err := owned.ReadExact(ctx, latest.Material.Exact()); err != nil {
							t.Fatalf("accepted bytes removed: %v", err)
						}
						return
					}
				}
				if changed, err := collector.Collect(ctx, f.claim, snapshot, execution); err != nil || !changed {
					t.Fatalf("partial collection: %t %v", changed, err)
				}
				accepted, err := store.GetTask(ctx, "owner", audit.AuditID, f.tasks[0].Document.TaskID)
				if err != nil || accepted.State != auditstore.TaskSettled || accepted.Result == nil || !accepted.Result.Ref.SameExact(latest.Material.Ref) || accepted.Stop != nil {
					t.Fatalf("accepted result lost: %+v %v", accepted, err)
				}
				missing, err := store.GetTask(ctx, "owner", audit.AuditID, f.tasks[1].Document.TaskID)
				if err != nil || missing.State != auditstore.TaskSettled || missing.Result != nil || missing.Stop == nil || missing.Stop.Author != "system" || missing.Stop.Meaning != "no_accepted_conclusion" || missing.Stop.Reason.Code != "worker_model_calls_exceeded" || missing.Stop.Reason.Source != "run" || checking != (missing.Stop.Verdict != nil) || len(missing.Stop.Evidence) != 2 {
					t.Fatalf("missing judgment disposition: %+v %v", missing, err)
				}
				for _, table := range []string{"audit_tasks", "audit_execution_items"} {
					tx, err := f.pool.Begin(ctx)
					if err != nil {
						t.Fatal(err)
					}
					_, err = tx.Exec(ctx, "UPDATE "+table+" SET stop_record=jsonb_set(stop_record,'{reason,message}',to_jsonb('changed'::text)) WHERE state='settled' AND stop_record IS NOT NULL")
					_ = tx.Rollback(ctx)
					if err == nil {
						t.Fatalf("settled %s stop record changed", table)
					}
				}
				for _, task := range []auditstore.Task{accepted, missing} {
					attempts, err := store.ListTaskAttempts(ctx, auditstore.TaskAttemptListParams{OwnerID: "owner", AuditID: audit.AuditID, TaskID: task.Document.TaskID, Limit: 10})
					if err != nil || len(attempts.Items) != 1 {
						t.Fatalf("attempts: %+v %v", attempts, err)
					}
					attempt := attempts.Items[0]
					if attempt.Outcome == nil || *attempt.Outcome != auditstore.TerminalFailed {
						t.Fatalf("technical failure changed: %+v", attempt)
					}
					if task.Result != nil && (attempt.Result == nil || attempt.ResultKey == nil || !attempt.Result.Ref.SameExact(task.Result.Ref)) {
						t.Fatal("accepted attempt result not retained")
					}
					if task.Stop != nil && (attempt.Stop == nil || attempt.Stop.Meaning != "no_accepted_conclusion") {
						t.Fatal("system attempt disposition not retained")
					}
				}
				ownedScope, _ := artifacts.AuditScope(audit.AuditID)
				owned, _ := art.Scoped(ownedScope)
				if _, err := owned.ReadExact(ctx, note.Sources[0].Exact()); err != nil {
					t.Fatal(err)
				}
				if changed, err := collector.Collect(ctx, f.claim, snapshot, execution); err != nil || changed {
					t.Fatalf("terminal replay: %t %v", changed, err)
				}
				current, err := runs.GetRun(ctx, f.grant.RunID)
				if err != nil || current.State != runstore.RunFailed {
					t.Fatal("partial records changed the Run outcome")
				}
			})
		}
	}
}

func TestSharedPostgresJudgmentCanUseReservedCapacity(t *testing.T) {
	for _, boundary := range []string{"records", "execution-bytes", "audit-bytes"} {
		t.Run(boundary, func(t *testing.T) {
			memberCount := 1
			if boundary == "records" {
				memberCount = 20
			}
			f := newJudgmentFixture(t, memberCount, false)
			ctx := t.Context()
			art := artifacts.NewService(artifactpgtest.NewStorage().Repository(f.pool))
			run, _ := art.Run(f.grant.RunID)
			sourceBytes, noteCount := 16, 1
			if boundary == "records" {
				noteCount = (256 - memberCount) / 2
			}
			if boundary == "execution-bytes" {
				sourceBytes = (7 << 20) - 8192
			}
			source, err := run.Write(ctx, contracts.ArtifactRef{Namespace: f.grant.Namespace, Name: "reserve-source"}, artifacts.Payload{MediaType: "text/plain", Data: bytes.Repeat([]byte{'x'}, sourceBytes)}, nil)
			if err != nil {
				t.Fatal(err)
			}
			input := auditcontract.EvidenceSubmission{APIVersion: contracts.APIVersion, InvocationID: "invocation", ItemKey: auditcontract.ExecutionItemKey(0), Sources: []contracts.ArtifactRef{source.Ref}, Interpretation: "Selected observation", TestedConditions: []string{"source inspected"}, Limitations: []string{}}
			var note auditcontract.EvidenceReceipt
			for index := range noteCount {
				input.SubmissionID = fmt.Sprintf("note-%03d", index)
				note, err = f.service.RecordEvidence(ctx, f.grant, input)
				if err != nil {
					t.Fatalf("note %d: %v", index, err)
				}
			}
			if boundary == "records" {
				input.SubmissionID = "overflow-note"
				if _, err := f.service.RecordEvidence(ctx, f.grant, input); !errors.Is(err, auditstore.ErrEvidenceBudgetExhausted) {
					t.Fatalf("evidence consumed result slots: %v", err)
				}
			}
			if boundary == "audit-bytes" {
				// Evidence reserve is capped at 1 MiB; leave just enough for the result.
				if _, err := f.pool.Exec(ctx, `UPDATE audits SET max_evidence_bytes=retained_evidence_bytes+70000 WHERE audit_id=$1`, f.audit.AuditID); err != nil {
					t.Fatal(err)
				}
			}
			observed := &observedReceiptStore{}
			f.service.storage = observedReceiptStorage(observed)
			judgment := judgmentInput(input.ItemKey, strings.Repeat("r", 65536), "reserved-result", 0)
			judgment.Items[0].Conclusion.Evidence = append(note.Sources, note.Material)
			judgment.Items[0].Conclusion.TestedConditions = []string{"source inspected"}
			result, err := f.service.RecordJudgments(ctx, f.grant, judgment)
			if err != nil || len(result.Generations) != 1 {
				t.Fatalf("reserved conclusion rejected: %v", err)
			}
			replay, err := f.service.RecordJudgments(ctx, f.grant, judgment)
			if err != nil || !replay.Replayed || !replay.Material.Ref.SameExact(result.Material.Ref) {
				t.Fatalf("result replay: %v", err)
			}
			if observed.reads != noteCount+1 {
				t.Fatalf("initial judgment and replay read %d payloads, want %d", observed.reads, noteCount+1)
			}
			for index := 1; index < memberCount; index++ {
				next := judgmentInput(auditcontract.ExecutionItemKey(index), "Accepted remaining assigned member", fmt.Sprintf("result-%02d", index), 0)
				if _, err := f.service.RecordJudgments(ctx, f.grant, next); err != nil {
					t.Fatalf("reserved member %d: %v", index, err)
				}
			}
			if observed.reads != noteCount+1 {
				t.Fatalf("unrelated members reread payloads: %d, want %d", observed.reads, noteCount+1)
			}
		})
	}
}

type receiptInlineStore = inline.Store

type observedReceiptStore struct {
	receiptInlineStore
	reads       int
	fault       error
	faultSchema string
}

func (s *observedReceiptStore) Read(ctx context.Context, object blob.Object) ([]byte, error) {
	s.reads++
	if s.fault != nil && (s.faultSchema == "" || bytes.Contains(object.Inline, []byte(s.faultSchema))) {
		return nil, s.fault
	}
	return s.receiptInlineStore.Read(ctx, object)
}
func observedReceiptStorage(s *observedReceiptStore) *artifactpostgres.Storage {
	return artifactpgtest.NewConfigured(blob.NewRuntime(s, nil), artifactpostgres.DefaultHistoryPolicy())
}

// Use actual database locales; an expression on a caller parameter cannot test
// the ordering of the stored indexed receipt keys.
func TestSharedPostgresAuditReceiptsAcrossCollations(t *testing.T) {
	dsn := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if dsn == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	for _, locale := range []string{"C", "en-US"} {
		t.Run(locale, func(t *testing.T) {
			config, err := pgxpool.ParseConfig(dsn)
			if err != nil {
				t.Fatal(err)
			}
			admin, err := pgx.ConnectConfig(t.Context(), config.ConnConfig.Copy())
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() { _ = admin.Close(context.Background()) })
			name := "audit_receipts_" + strings.ToLower(rand.Text())
			quoted := pgx.Identifier{name}.Sanitize()
			options := ` TEMPLATE template0 ENCODING 'UTF8' LOCALE_PROVIDER libc LC_COLLATE 'C' LC_CTYPE 'C'`
			if locale == "en-US" {
				options = ` TEMPLATE template0 ENCODING 'UTF8' LOCALE_PROVIDER icu ICU_LOCALE 'en-US'`
			}
			t.Cleanup(func() {
				ctx, cancel := context.WithTimeout(context.Background(), time.Minute)
				defer cancel()
				if _, err := admin.Exec(ctx, "DROP DATABASE IF EXISTS "+quoted+" WITH (FORCE)"); err != nil {
					t.Error(err)
				}
			})
			if _, err := admin.Exec(t.Context(), "CREATE DATABASE "+quoted+options); err != nil {
				t.Fatal(err)
			}
			config.ConnConfig.Database = name
			t.Setenv("CONTRACTOR_TEST_DATABASE_URL", config.ConnConfig.ConnString())
			t.Run("corrections", TestSharedPostgresJudgmentCorrectionReplayAndFences)
			t.Run("membership", TestSharedPostgresJudgmentBatchAndEvidenceMembership)
			t.Run("capacity", TestSharedPostgresJudgmentCanUseReservedCapacity)
		})
	}
}
