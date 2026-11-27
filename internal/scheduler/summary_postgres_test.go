package scheduler

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"reflect"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/artifacts/blob"
	"github.com/grauwolf32/contractor/internal/artifacts/blob/backends"
	artifactpostgres "github.com/grauwolf32/contractor/internal/artifacts/postgres"
	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"
	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/reporting"
	"github.com/grauwolf32/contractor/internal/controlplane"
	plannermemory "github.com/grauwolf32/contractor/internal/memory"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/planner"
	plannersession "github.com/grauwolf32/contractor/internal/planner/session"
	"github.com/grauwolf32/contractor/internal/planner/streamline"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/jackc/pgx/v5/pgconn"
)

type summaryLostAcknowledgement struct {
	ArtifactResolver
	lost   bool
	before bool
	loss   func(string)
}

func (r *summaryLostAcknowledgement) SaveSummary(ctx context.Context, runID, namespace, source, summary string) (contracts.ArtifactRef, error) {
	if r.loss != nil && !r.lost {
		r.lost = true
		r.loss(source)
		return contracts.ArtifactRef{}, context.Canceled
	}
	if r.before && !r.lost {
		r.lost = true
		return contracts.ArtifactRef{}, blob.ErrIO
	}
	ref, err := r.ArtifactResolver.SaveSummary(ctx, runID, namespace, source, summary)
	if err == nil && !r.lost {
		r.lost = true
		return contracts.ArtifactRef{}, blob.ErrIO
	}
	return ref, err
}

type summaryProgressionFault struct {
	AtomicPersistence
	mode string
	done bool
}

func (p *summaryProgressionFault) CommitResultProgression(ctx context.Context, value ResultProgression) error {
	if p.done || value.Progression.Decision.Action != runstore.StageTransitionEscalate {
		return p.AtomicPersistence.CommitResultProgression(ctx, value)
	}
	p.done = true
	if p.mode == "transaction-conflict" {
		return &pgconn.PgError{Code: persistencepostgres.SQLStateSerializationFailure}
	}
	if err := p.AtomicPersistence.CommitResultProgression(ctx, value); err != nil {
		return err
	}
	return io.ErrUnexpectedEOF
}

// The continuation exercises the actual Runtime host protocol and PostgreSQL
// domain operations. Python model dispatch is covered by process tests.
type summaryScript struct {
	t              *testing.T
	inputRevision  string
	report         contracts.ArtifactRef
	calls          int
	failBeforeWork bool
}
type summaryRuntimeInvoker func(context.Context, string, contracts.WorkerHandle, contracts.StageContentRequest) (contracts.WorkerCompletion, error)

func (f summaryRuntimeInvoker) Invoke(ctx context.Context, n string, h contracts.WorkerHandle, r contracts.StageContentRequest) (contracts.WorkerCompletion, error) {
	return f(ctx, n, h, r)
}
func (m *summaryScript) run(ctx context.Context, f *streamline.Factory, stage string, handle contracts.WorkerHandle, request contracts.StageContentRequest) (contracts.WorkerCompletion, error) {
	var initial struct {
		Objective    string
		Instructions string
		Parameters   map[string]string
		Artifacts    map[string]*contracts.ArtifactRef
		Continuation *struct {
			Summary         string
			SummaryArtifact contracts.ArtifactRef
		}
	}
	if request.Planner == nil || json.Unmarshal([]byte(request.Planner.Input), &initial) != nil || initial.Continuation == nil || initial.Objective == "" || initial.Instructions == "" || initial.Parameters["objective"] != "copy exactly" || initial.Artifacts["mutable"] == nil || *initial.Artifacts["mutable"].Revision != m.inputRevision || initial.Continuation.Summary != summarizedCandidate().Summary || initial.Continuation.SummaryArtifact.Namespace != "builder" {
		return contracts.WorkerCompletion{}, errors.New("Streamline lost original task or separate summary")
	}
	seq := 0
	op := func(name string, args any) (contracts.PlannerOperationResult, error) {
		seq++
		data, _ := json.Marshal(args)
		r, e := f.ExecuteOperation(ctx, stage, handle.AllocationID, contracts.PlannerOperation{APIVersion: contracts.APIVersion, Sequence: seq, Name: name, Arguments: data})
		if e == nil && r.Failure != nil {
			e = errors.New(r.Failure.Code)
		}
		return r, e
	}
	step := func(name string, args any) (contracts.PlannerOperationResult, error) {
		m.calls++
		if _, e := op("before_model", map[string]any{"retainedBytes": 0}); e != nil {
			return contracts.PlannerOperationResult{}, e
		}
		if _, e := op("after_model", map[string]any{"inputTokens": 1, "outputTokens": 1, "cachedTokens": 0, "calls": []string{name}}); e != nil {
			return contracts.PlannerOperationResult{}, e
		}
		return op(name, args)
	}
	note, e := step("read_memory", map[string]any{"name": "progress"})
	if e != nil {
		return contracts.WorkerCompletion{}, e
	}
	if !m.failBeforeWork && !containsSummaryNote(note.Result) {
		return contracts.WorkerCompletion{}, errors.New("same-namespace Memory note lost")
	}
	if m.failBeforeWork {
		_, e = step("finish", map[string]any{"outcome": "failed", "summary": "Temporary blocker", "artifacts": map[string]any{}, "error": map[string]any{"code": "blocked", "message": "Requires a manual continuation", "retryable": false}})
	} else {
		if _, e = step("add_subtask", map[string]any{"objective": "Publish remaining report", "instructions": "Reuse saved analysis; verify the final artifact."}); e != nil {
			return contracts.WorkerCompletion{}, e
		}
		if _, e = step("execute_current_subtask", map[string]any{"subtask_id": "0"}); e != nil {
			return contracts.WorkerCompletion{}, e
		}
		_, e = step("finish", map[string]any{"outcome": "succeeded", "summary": "Report complete", "artifacts": map[string]any{"copied": map[string]any{"namespace": m.report.Namespace, "name": m.report.Name}}})
	}
	return contracts.WorkerCompletion{}, e
}

func containsSummaryNote(data []byte) bool {
	var value any
	if json.Unmarshal(data, &value) != nil {
		return false
	}
	var search func(any) bool
	search = func(v any) bool {
		switch x := v.(type) {
		case string:
			return x == "analysis already recorded"
		case map[string]any:
			for _, child := range x {
				if search(child) {
					return true
				}
			}
		case []any:
			for _, child := range x {
				if search(child) {
					return true
				}
			}
		}
		return false
	}
	return search(value)
}

func TestPostgresSummaryContinuationReconcilesAndRetainsArtifactsAndMemory(t *testing.T) {
	artifactRuntime := artifactpgtest.NewRuntime()
	artifactHistory := artifactpostgres.DefaultHistoryPolicy()
	artifactStorage := artifactpgtest.NewConfigured(artifactRuntime, artifactHistory)

	for _, backend := range []blob.Backend{blob.Postgres, blob.Filesystem} {
		for _, fault := range []string{"none", "summary-response-loss", "process-loss-before-summary", "allocation-loss-after-summary", "transition-response-loss", "transaction-conflict", "pause", "cancel", "manual-resume"} {
			t.Run(string(backend)+"/"+fault, func(t *testing.T) {
				ctx, cancel := context.WithTimeout(t.Context(), 45*time.Second)
				defer cancel()
				_, blobs, err := backends.Open(ctx, backend, t.TempDir(), 0)
				if err != nil {
					t.Fatal(err)
				}
				t.Cleanup(func() {
					if c, ok := blobs.(io.Closer); ok {
						_ = c.Close()
					}
				})
				artifactRuntime = blob.NewRuntime(blobs, nil)
				artifactStorage = artifactpgtest.NewConfigured(artifactRuntime, artifactHistory)
				h := newDeferredPlacementHarness(t, ctx, nil, func(workflow *resolved.ResolvedWorkflow) {
					configureSummaryWorkflow(t, workflow)
					stage := workflow.Stages[workflow.EntryStage]
					stage.Context.Artifacts["mutable"] = resolved.ContextArtifact{Namespace: "builder", Name: "material", Required: true}
					binding := stage.Agents["builder"]
					binding.Template.Toolsets = append(binding.Template.Toolsets, contracts.ToolsetSelection{Ref: contracts.ToolsetRef{ToolsetID: "memory-tools", Version: "1"}, Tools: []string{"read_memory", "write_memory"}})
					stage.Agents["builder"] = binding
					workflow.Stages[workflow.EntryStage] = stage
					if err := resolved.ResolveEscalationVariants(workflow); err != nil {
						t.Fatal(err)
					}
				})
				service := artifacts.NewService(artifactStorage.Repository(h.pool))
				store, _ := service.Run("run-1")
				material, err := store.Write(ctx, contracts.ArtifactRef{Namespace: "builder", Name: "material"}, artifacts.Payload{MediaType: "text/plain", Data: []byte("original")}, nil)
				if err != nil {
					t.Fatal(err)
				}
				sessions, err := plannersession.New(h.store, plannersession.Options{})
				if err != nil {
					t.Fatal(err)
				}
				inspector, _ := planner.NewArtifactServiceInspector(service)
				memory, err := plannermemory.NewPostgresStore(h.pool, artifactStorage)
				if err != nil {
					t.Fatal(err)
				}
				report := h.invoker.result.Clone()
				script := &summaryScript{t: t, inputRevision: *material.Ref.Revision, report: report.Artifacts["copied"]}
				script.failBeforeWork = fault == "manual-resume"
				var continuation *streamline.Factory
				runtimeInvoker := summaryRuntimeInvoker(func(call context.Context, n string, handle contracts.WorkerHandle, request contracts.StageContentRequest) (contracts.WorkerCompletion, error) {
					if request.Planner == nil {
						return h.invoker.Invoke(call, n, handle, request)
					}
					active, err := h.store.GetActiveStageExecution(call, "run-1")
					if err != nil {
						return contracts.WorkerCompletion{}, err
					}
					return script.run(call, continuation, active.StageExecutionID, handle, request)
				})
				continuation, err = streamline.NewRuntimeFactoryWithMemory(sessions, runtimeInvoker, inspector, summaryUnavailableState{}, memory, streamline.Limits{})
				if err != nil {
					t.Fatal(err)
				}
				passthrough, _ := planner.NewPassthroughFactory(sessions, h.invoker, inspector)
				registry, _ := planner.NewRegistry(passthrough, continuation)
				h.scheduler.planners = registry
				h.scheduler.options.PlannerSessions = sessions
				historyReads := &stageHistoryReads{Store: h.store}
				h.scheduler.store = historyReads
				h.invoker.result = summarizedCandidate()
				h.invoker.before = func(ctx context.Context, _ contracts.WorkerHandle) error {
					execution, err := h.store.GetActiveStageExecution(ctx, "run-1")
					if err != nil {
						return err
					}
					notebook, err := plannermemory.NewNamespace(memory, plannermemory.Binding{RunID: "run-1", StageExecutionID: execution.StageExecutionID, Namespace: "builder"})
					if err != nil {
						return err
					}
					if h.invoker.calls == 1 {
						if _, err := notebook.WriteMemory(ctx, "progress", "analysis already recorded", "", nil); err != nil {
							return err
						}
						_, err := store.Write(ctx, contracts.ArtifactRef{Namespace: "builder", Name: "material"}, artifacts.Payload{MediaType: "text/plain", Data: []byte("advanced")}, material.Ref.Revision)
						return err
					}
					if _, err := notebook.ReadMemory(ctx, "progress"); err != nil {
						return err
					}
					h.invoker.result = report
					return nil
				}
				switch fault {
				case "summary-response-loss":
					h.scheduler.artifacts = &summaryLostAcknowledgement{ArtifactResolver: h.scheduler.artifacts}
				case "process-loss-before-summary":
					h.scheduler.artifacts = &summaryLostAcknowledgement{ArtifactResolver: h.scheduler.artifacts, before: true}
				case "allocation-loss-after-summary":
					h.scheduler.artifacts = &summaryLostAcknowledgement{ArtifactResolver: h.scheduler.artifacts, loss: func(source string) {
						for id, grant := range h.workers.allocator.grants {
							if grant.StageExecutionID == source {
								grant.Lost, grant.WriteFenced = true, true
								h.workers.allocator.grants[id] = grant
								if !h.scheduler.interruptStageAllocationLoss(controlplane.AllocationLoss{
									RunID: "run-1", StageExecutionID: source, AllocationID: id, Reason: controlplane.LossControlLeaseExpired,
								}) {
									t.Error("allocation loss did not interrupt the source claim")
								}
							}
						}
					}}
				case "transition-response-loss", "transaction-conflict":
					h.scheduler.persistence = &summaryProgressionFault{AtomicPersistence: h.scheduler.persistence, mode: fault}
				case "pause":
					h.scheduler.persistence = &pauseBeforeResultProgression{AtomicPersistence: h.scheduler.persistence, store: h.store}
				}
				worked, err := h.scheduler.RunOnce(ctx)
				if !worked || ((fault == "none" || fault == "cancel") && err != nil) {
					t.Fatalf("source pass: %v %v", worked, err)
				}
				if fault == "pause" {
					if !errors.Is(err, ErrDeferred) {
						t.Fatalf("paused source progression: %v", err)
					}
					waiting, loadErr := h.store.ListStageExecutions(ctx, "run-1")
					if loadErr != nil || len(waiting) != 1 || waiting[0].CandidateResult.SummaryArtifact == nil || h.invoker.calls != 1 {
						t.Fatalf("pause lost summary or admitted continuation: %v", loadErr)
					}
					if _, err := h.store.UpdateOwnerQueueControl(ctx, runstore.UpdateOwnerQueueControlParams{OwnerID: "user-1", ExpectedRevision: 1, Paused: false}); err != nil {
						t.Fatal(err)
					}
				}
				if (fault == "summary-response-loss" || fault == "process-loss-before-summary" || fault == "transition-response-loss" || fault == "transaction-conflict") && err == nil {
					t.Fatal("injected failure was not observed")
				}
				if fault == "process-loss-before-summary" || fault == "summary-response-loss" {
					// The new Control Plane has no live source grants.
					h.workers.allocator.grants = map[string]controlplane.AllocationGrant{}
					h.workers.allocator.cached = nil
				}
				// A new Scheduler reconstructs the pinned strategy and a new
				// Planner instance from the durable session/transition receipts.
				restarted, err := New(historyReads, h.scheduler.persistence, h.scheduler.artifacts, h.scheduler.allocator, h.workers, registry, h.scheduler.options)
				if err != nil {
					t.Fatal(err)
				}
				for i := 0; i < 2; i++ {
					executions, err := h.store.ListStageExecutions(ctx, "run-1")
					if err != nil {
						t.Fatal(err)
					}
					if len(executions) == 2 {
						break
					}
					if _, err := restarted.RunOnce(ctx); err != nil {
						t.Fatal(err)
					}
				}
				executions, _ := h.store.ListStageExecutions(ctx, "run-1")
				if len(executions) != 2 || executions[0].AcceptedResult == nil {
					t.Fatalf("source/target missing: %+v", executions)
				}
				summary := executions[0].AcceptedResult.SummaryArtifact
				if exists, err := h.store.HasSummaryEscalation(ctx, "run-1", executions[1].StageName); err != nil || !exists {
					t.Fatalf("durable strategy root lookup: %v %v", exists, err)
				}
				if summary == nil || !reflect.DeepEqual(executions[0].StageContext.Artifacts, executions[1].StageContext.Artifacts) {
					t.Fatal("summary or exact original context lost")
				}
				saved, err := store.Read(ctx, *summary)
				if err != nil || string(saved.Payload.Data) != summarizedCandidate().Summary {
					t.Fatalf("saved summary: %v", err)
				}
				if again, err := h.scheduler.artifacts.SaveSummary(ctx, "run-1", "builder", executions[0].StageExecutionID, summarizedCandidate().Summary); err != nil || !again.SameExact(*summary) {
					t.Fatalf("summary replay appended: %+v %v", again, err)
				}
				want := runstore.RunSucceeded
				if fault == "cancel" {
					want = runstore.RunCancelled
					if _, err := h.store.RequestRunCancellation(ctx, "run-1", runstore.WorkflowRunCancellation{Code: runstore.CancellationUserRequested, RequestedAt: time.Now().UTC()}); err != nil {
						t.Fatal(err)
					}
				}
				if _, err := restarted.RunOnce(ctx); err != nil {
					t.Fatal(err)
				}
				if fault == "manual-resume" {
					failed, err := h.store.GetRun(ctx, "run-1")
					if err != nil || failed.State != runstore.RunFailed {
						t.Fatalf("continuation should fail before manual resume: %v", err)
					}
					resumable, err := h.store.ResumableStage(ctx, "user-1", "run-1")
					if err != nil || resumable == nil {
						t.Fatalf("summary continuation cannot resume: %v", err)
					}
					resumed, err := h.store.ResumeFailedRun(ctx, "user-1", "run-1", *resumable, "manual-summary-resume")
					if err != nil {
						t.Fatal(err)
					}
					replayed, err := h.store.ResumeFailedRun(ctx, "user-1", "run-1", *resumable, "unused-replay-id")
					if err != nil || replayed.StageExecutionID != resumed.StageExecutionID {
						t.Fatalf("manual receipt failed: %v", err)
					}
					script = &summaryScript{t: t, inputRevision: *material.Ref.Revision, report: report.Artifacts["copied"]}
					next, err := h.store.GetStageExecution(ctx, resumed.StageExecutionID)
					if err != nil || next.AdmittedAt != nil ||
						next.ExecutionConfigVariant != runstore.StageExecutionConfigSummaryEscalation || !reflect.DeepEqual(next.StageContext, executions[1].StageContext) {
						t.Fatalf("manual resume copied deadline or changed context: %v", err)
					}
					if _, err := restarted.RunOnce(ctx); err != nil {
						t.Fatal(err)
					}
				}
				run, err := h.store.GetRun(ctx, "run-1")
				if err != nil || run.State != want {
					ended, _ := h.store.ListStageExecutions(ctx, "run-1")
					for _, e := range ended {
						t.Logf("attempt %d %s termination=%+v candidate=%+v", e.Attempt, e.State, e.Termination, e.CandidateResult)
					}
					t.Fatalf("continued Run: %s reason=%+v err=%v model calls=%d", run.State, run.StateReason, err, script.calls)
				}
				if fault != "cancel" && (h.invoker.calls != 2 || script.calls != 4) {
					t.Fatalf("source replayed or progress rerun: Worker=%d Planner=%d", h.invoker.calls, script.calls)
				}
				if fault == "cancel" && h.invoker.calls != 1 {
					t.Fatal("cancelled target dispatched")
				}
				if historyReads.reads != 1 {
					t.Fatalf("summary progression scanned full history after initial empty-Run check: %d reads", historyReads.reads)
				}
				attempts, err := h.store.ListStageExecutions(ctx, "run-1")
				if err != nil {
					t.Fatal(err)
				}
				for _, attempt := range attempts {
					stage, err := resolved.DecodeResolvedStageSnapshot(attempt.StageSpecSnapshot)
					if err != nil || stage.MaxSubtasks.Effective() != 64 {
						t.Fatalf("attempt %d lost the pinned subtask limit: %v", attempt.Attempt, err)
					}
				}
				decisions, err := h.store.ListStageTransitionDecisions(ctx, "run-1")
				if err != nil || len(decisions) < 1 || decisions[0].Action != runstore.StageTransitionEscalate {
					t.Fatalf("transition receipt: %+v %v", decisions, err)
				}
			})
		}
	}
}

type summaryUnavailableState struct{}

func (summaryUnavailableState) ReadWorkerState(context.Context, contracts.WorkerHandle, string) (reporting.WorkerStateReadResult, error) {
	return reporting.WorkerStateReadResult{}, &reporting.WorkerStateReadError{Code: "runtime_unavailable", Retryable: true}
}
