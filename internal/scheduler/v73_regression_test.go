package scheduler

import (
	"context"
	"errors"
	"fmt"
	"testing"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/reporting"
	"github.com/grauwolf32/contractor/internal/planner"
	"github.com/grauwolf32/contractor/internal/runstore"
)

type metadataOnlyRepository struct {
	artifacts.Repository
	artifacts.QueryRepository
	readCalls int
	ref       contracts.ArtifactRef
}

func (r *metadataOnlyRepository) Read(context.Context, artifacts.Scope, contracts.ArtifactRef) (artifacts.ReadResult, error) {
	r.readCalls++
	return artifacts.ReadResult{}, artifacts.ErrTransferCapacity
}

func (r *metadataOnlyRepository) Metadata(_ context.Context, scope artifacts.Scope, ref contracts.ArtifactRef) (artifacts.Metadata, error) {
	if scope.Kind() != artifacts.ScopeRun || scope.ID() != "run-1" || ref.Namespace != r.ref.Namespace || ref.Name != r.ref.Name {
		return artifacts.Metadata{}, artifacts.ErrArtifactNotFound
	}
	return artifacts.Metadata{Ref: r.ref, MediaType: "text/plain"}, nil
}

func TestArtifactResolverUsesMetadataWithAllTransferSlotsOccupied(t *testing.T) {
	ctx := artifacts.WithBlobRuntime(t.Context(), artifacts.NewBlobRuntime(artifacts.PostgresBlobStore{}, nil))
	for range 4 {
		_, release, err := artifacts.AcquireTransfer(ctx)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(release)
	}
	if _, _, err := artifacts.AcquireTransfer(ctx); !errors.Is(err, artifacts.ErrTransferCapacity) {
		t.Fatalf("transfer budget was not exhausted: %v", err)
	}
	repo := &metadataOnlyRepository{ref: exactRef("builder", "copied", "result-r1")}
	resolver, err := NewArtifactServiceResolver(artifacts.NewService(repo))
	if err != nil {
		t.Fatal(err)
	}
	resolved, err := resolver.Resolve(ctx, "run-1", repo.ref)
	if err != nil {
		t.Fatal(err)
	}
	if !resolved.Ref.SameExact(repo.ref) || resolved.MediaType != "text/plain" || repo.readCalls != 0 {
		t.Fatalf("metadata resolution = %+v, content reads = %d", resolved, repo.readCalls)
	}
}

type resultResolverError struct {
	ArtifactResolver
	err error
}

func (r resultResolverError) Resolve(ctx context.Context, runID string, ref contracts.ArtifactRef) (ResolvedArtifact, error) {
	if ref.Namespace == "builder" {
		return ResolvedArtifact{}, r.err
	}
	return r.ArtifactResolver.Resolve(ctx, runID, ref)
}

func TestCandidateArtifactLookupErrorsKeepTheirMeaning(t *testing.T) {
	for _, test := range []struct {
		name      string
		err       error
		code      string
		retryable bool
	}{
		{"transfer capacity", artifacts.ErrTransferCapacity, "result_verification_unavailable", true},
		{"database unavailable", errors.New("temporary database failure"), "result_verification_unavailable", true},
		{"missing artifact", artifacts.ErrArtifactNotFound, "result_contract_violation", false},
	} {
		t.Run(test.name, func(t *testing.T) {
			h := newSchedulerHarness(t)
			h.scheduler.artifacts = resultResolverError{ArtifactResolver: h.artifacts, err: test.err}
			worked, err := h.scheduler.RunOnce(t.Context())
			if err != nil || !worked {
				t.Fatalf("RunOnce = (%v, %v)", worked, err)
			}
			execution := h.store.stages[0]
			if execution.State != runstore.StageInterrupted || execution.Termination == nil ||
				execution.Termination.Code != test.code || execution.Termination.Retryable != test.retryable {
				t.Fatalf("result resolution termination = %+v", execution)
			}
			if _, finalizing := eventIndex(h.events.values, "enter_finalizing"); finalizing {
				t.Fatal("failed candidate resolution entered finalizing")
			}
		})
	}
}

type outputCheckingPersistence struct {
	AtomicPersistence
	ArtifactResolver
	commitCalls int
}

func (p *outputCheckingPersistence) CommitResultProgression(ctx context.Context, value ResultProgression) error {
	p.commitCalls++
	for outputName, resultName := range value.WorkflowOutputs {
		ref, present := value.Result.Artifacts[resultName]
		if !present {
			continue
		}
		resolved, err := p.ArtifactResolver.Resolve(ctx, value.RunID, ref)
		if err != nil {
			return err
		}
		if !contracts.AcceptsMediaType(value.OutputContracts[outputName].MediaTypes, resolved.MediaType) {
			return fmt.Errorf("Workflow output %q has incompatible media type", outputName)
		}
	}
	return p.AtomicPersistence.CommitResultProgression(ctx, value)
}

func TestMappedWorkflowOutputRejectsIncompatibleResultBeforeFinalizing(t *testing.T) {
	h := newSchedulerHarness(t)
	output := h.workflow.Outputs["result"]
	output.MediaTypes = []string{"application/json"}
	h.workflow.Outputs["result"] = output
	stage := h.workflow.Stages[h.workflow.EntryStage]
	resultSlot := stage.Result.Artifacts["copied"]
	resultSlot.MediaTypes = []string{"text/plain", "application/json"}
	stage.Result.Artifacts["copied"] = resultSlot
	h.workflow.Stages[h.workflow.EntryStage] = stage
	installHarnessWorkflow(t, h)
	checking := &outputCheckingPersistence{AtomicPersistence: h.persistence, ArtifactResolver: h.artifacts}
	h.scheduler.persistence = checking
	for range 3 {
		if _, err := h.scheduler.RunOnce(t.Context()); err != nil {
			t.Fatal(err)
		}
	}
	execution := h.store.stages[0]
	if execution.State != runstore.StageInterrupted || execution.Termination == nil ||
		execution.Termination.Code != "result_contract_violation" || execution.Termination.Retryable ||
		h.store.run.State != runstore.RunFailed || h.workers.releaseCalls == 0 || checking.commitCalls != 0 {
		t.Fatalf("mapped media mismatch: run=%s stage=%+v releases=%d commits=%d", h.store.run.State, execution, h.workers.releaseCalls, checking.commitCalls)
	}
	if _, finalizing := eventIndex(h.events.values, "enter_finalizing"); finalizing {
		t.Fatal("invalid mapped media entered finalizing")
	}
}

type reportingPlannerRegistry struct {
	PlannerRegistry
	report reporting.ExecutionReport
}

func (r reportingPlannerRegistry) Create(ref string, invocation planner.Invocation) (planner.Planner, error) {
	created, err := r.PlannerRegistry.Create(ref, invocation)
	if err != nil {
		return nil, err
	}
	return reportingPlanner{Planner: created, report: r.report}, nil
}

type reportingPlanner struct {
	planner.Planner
	report reporting.ExecutionReport
}

func (p reportingPlanner) ExecutionReport() (reporting.ExecutionReport, bool) { return p.report, true }

type cancellationAwareStageStore struct {
	Store
	memory            *memorySchedulerStore
	completeAtRebuild bool
}

func (s *cancellationAwareStageStore) GetStageExecution(ctx context.Context, id string) (runstore.StageExecution, error) {
	if err := ctx.Err(); err != nil {
		return runstore.StageExecution{}, err
	}
	return s.Store.GetStageExecution(ctx, id)
}

func (s *cancellationAwareStageStore) RebuildStageMetrics(ctx context.Context, runID, stageID string) error {
	for _, recorded := range s.memory.plannerReports {
		if recorded.Report.Complete && recorded.Report.Metrics.TotalTokens != nil {
			s.completeAtRebuild = true
		}
	}
	return s.Store.RebuildStageMetrics(ctx, runID, stageID)
}

func TestInterruptedPlannerKeepsCompleteUsageUnlessClaimIsLost(t *testing.T) {
	for _, reason := range []string{"user cancellation", "allocation lease loss", "shutdown", "claim loss"} {
		t.Run(reason, func(t *testing.T) {
			h := newSchedulerHarness(t)
			awareStore := &cancellationAwareStageStore{Store: h.store, memory: h.store}
			h.scheduler.store = awareStore
			modelCalls, inputTokens, outputTokens, totalTokens := int64(1), int64(12345), int64(55), int64(12400)
			h.scheduler.planners = reportingPlannerRegistry{
				PlannerRegistry: h.planners,
				report: reporting.ExecutionReport{
					ReportID: "planner-real", Complete: true,
					Metrics: reporting.ExecutionMetrics{
						ModelCalls: &modelCalls, InputTokens: &inputTokens,
						OutputTokens: &outputTokens, TotalTokens: &totalTokens,
						Tools: map[string]reporting.ToolMetrics{},
					},
					ToolCalls: []reporting.ToolCallRecord{}, Errors: []reporting.ExecutionError{},
				},
			}
			ctx, cancel := context.WithCancelCause(t.Context())
			defer cancel(nil)
			originalOnRun := h.planners.onRun
			h.planners.onRun = func() {
				originalOnRun()
				switch reason {
				case "user cancellation":
					h.requestCancellation("stop")
					h.scheduler.Cancel(h.store.run.RunID)
				case "allocation lease loss":
					h.scheduler.interruptRun(h.store.run.RunID, &AllocationLeaseLossError{})
				case "shutdown":
					cancel(context.Canceled)
				case "claim loss":
					cancel(ErrClaimLost)
				}
			}
			_, _ = h.scheduler.RunOnce(ctx)
			complete := 0
			for _, recorded := range h.store.plannerReports {
				if recorded.Report.Complete && recorded.Report.ReportID == "planner-real" {
					complete++
					if recorded.Report.Metrics.TotalTokens == nil || *recorded.Report.Metrics.TotalTokens != totalTokens ||
						recorded.Report.Metrics.InputTokens == nil || *recorded.Report.Metrics.InputTokens != inputTokens {
						t.Fatalf("Planner usage was lost: %+v", recorded.Report)
					}
				}
			}
			if reason == "claim loss" {
				if len(h.store.plannerReports) != 0 || awareStore.completeAtRebuild {
					t.Fatalf("lost claim persisted a report: %+v", h.store.plannerReports)
				}
			} else if complete != 1 || !awareStore.completeAtRebuild {
				t.Fatalf("interrupted Planner report or metrics rebuild = %+v, rebuilt=%v", h.store.plannerReports, awareStore.completeAtRebuild)
			}
		})
	}
}
