package runservice

import (
	"context"
	"fmt"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/gatewayrecovery"
	"github.com/grauwolf32/contractor/internal/pagelimit"
	"github.com/grauwolf32/contractor/internal/planner"
	"github.com/grauwolf32/contractor/internal/rundeletion"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/telemetry"
)

// RunDetail is an owner-visible Run with everything its public detail
// shows, read from one database snapshot. EventCursor is therefore the
// cursor of exactly this state (API-R48): a change committed after the
// snapshot has a later sequence, so a subscription from EventCursor delivers
// it instead of skipping it.
type RunDetail struct {
	Run                    runstore.WorkflowRun
	DeletionBlocker        *rundeletion.Reason
	Executions             []runstore.StageExecution
	Transitions            []runstore.StageTransitionDecision
	Inputs                 map[string]contracts.ArtifactRef
	Outputs                map[string]contracts.ArtifactRef
	OutputPublications     []runstore.RunOutputPublication
	EventCursor            runstore.WorkflowRunEventCursor
	ResumeStageExecutionID *string
	Recovery               *gatewayrecovery.Status
	// The records below are keyed by StageExecution ID. Metrics are optional
	// diagnostics: a failed read leaves the map empty.
	Allocations map[string][]runstore.StageAllocation
	Metrics     map[string]telemetry.StageMetricsRecord
	Plans       map[string]planner.PlannerPlanProjection
	Resources   map[string][]telemetry.AllocationResourceSummary
}

// RunDetailRunReader reads the Run-owned records of a Run detail.
type RunDetailRunReader interface {
	GetRun(context.Context, string) (runstore.WorkflowRun, error)
	GetRunEventCursor(context.Context, string) (runstore.WorkflowRunEventCursor, error)
	ListStageExecutions(context.Context, string) ([]runstore.StageExecution, error)
	ListStageTransitionDecisions(context.Context, string) ([]runstore.StageTransitionDecision, error)
	ListRunOutputPublications(context.Context, string) ([]runstore.RunOutputPublication, error)
	ListStageAllocationsBatch(context.Context, []string) (map[string][]runstore.StageAllocation, error)
	ResumableStage(context.Context, string, string) (*string, error)
}

type RunDeletionReader interface {
	RunDeletionBlocker(context.Context, string, string) (*rundeletion.Reason, error)
}

type PlannerPlanReader interface {
	LoadPlans(context.Context, []planner.SessionIdentity) (map[string]planner.PlannerPlanProjection, error)
}

type StageMetricsReader interface {
	GetStageMetricsBatch(context.Context, []string) (map[string]telemetry.StageMetricsRecord, error)
}

type AllocationResourceReader interface {
	ListStageAllocationResources(context.Context, string, []string) (map[string][]telemetry.AllocationResourceSummary, error)
}

// RunDetailStores are the readers of one Run detail. A RunDetailTransaction
// binds all of them to the same snapshot.
type RunDetailStores struct {
	Runs      RunDetailRunReader
	Deletion  RunDeletionReader
	Artifacts *artifacts.Service
	Plans     PlannerPlanReader
	// Metrics must leave the snapshot usable when it fails; the other reads
	// still complete.
	Metrics   StageMetricsReader
	Resources AllocationResourceReader
	Recovery  func(context.Context, string) (*gatewayrecovery.Status, error)
}

// RunDetailTransaction calls fn once with readers that all observe one
// database snapshot. PostgresRunDetailTransaction is the production
// implementation.
type RunDetailTransaction func(context.Context, func(RunDetailStores) error) error

// RunDetailReader reads public Run details.
type RunDetailReader struct {
	transaction RunDetailTransaction
}

func NewRunDetailReader(transaction RunDetailTransaction) (*RunDetailReader, error) {
	if transaction == nil {
		return nil, fmt.Errorf("%w: Run detail transaction is missing", ErrNotConfigured)
	}
	return &RunDetailReader{transaction: transaction}, nil
}

// Read returns the detail of runID if ownerID owns it. Another owner's Run is
// runstore.ErrNotFound, reported before any related record is read. The
// number of queries does not grow with the number of StageExecutions: their
// related records are read in one batch per collection.
func (r *RunDetailReader) Read(ctx context.Context, ownerID, runID string) (RunDetail, error) {
	var detail RunDetail
	err := r.transaction(ctx, func(stores RunDetailStores) error {
		var err error
		detail, err = readRunDetail(ctx, stores, ownerID, runID)
		return err
	})
	if err != nil {
		return RunDetail{}, err
	}
	return detail, nil
}

func readRunDetail(ctx context.Context, stores RunDetailStores, ownerID, runID string) (RunDetail, error) {
	if stores.Runs == nil || stores.Deletion == nil || stores.Artifacts == nil || stores.Plans == nil ||
		stores.Metrics == nil || stores.Resources == nil || stores.Recovery == nil {
		return RunDetail{}, fmt.Errorf("%w: Run detail readers are incomplete", ErrNotConfigured)
	}
	run, err := stores.Runs.GetRun(ctx, runID)
	if err != nil {
		return RunDetail{}, err
	}
	if run.OwnerID != ownerID {
		return RunDetail{}, runstore.ErrNotFound
	}
	detail := RunDetail{Run: run}
	if detail.DeletionBlocker, err = stores.Deletion.RunDeletionBlocker(ctx, ownerID, runID); err != nil {
		return RunDetail{}, err
	}
	if detail.Executions, err = stores.Runs.ListStageExecutions(ctx, runID); err != nil {
		return RunDetail{}, err
	}
	if detail.Transitions, err = stores.Runs.ListStageTransitionDecisions(ctx, runID); err != nil {
		return RunDetail{}, err
	}
	if detail.Outputs, err = runArtifactsByNamespace(ctx, stores.Artifacts, runID, "outputs"); err != nil {
		return RunDetail{}, err
	}
	if detail.OutputPublications, err = stores.Runs.ListRunOutputPublications(ctx, runID); err != nil {
		return RunDetail{}, err
	}
	if detail.Inputs, err = runArtifactsByNamespace(ctx, stores.Artifacts, runID, "inputs"); err != nil {
		return RunDetail{}, err
	}
	if detail.EventCursor, err = stores.Runs.GetRunEventCursor(ctx, runID); err != nil {
		return RunDetail{}, err
	}
	if run.State == runstore.RunFailed {
		if detail.ResumeStageExecutionID, err = stores.Runs.ResumableStage(ctx, ownerID, runID); err != nil {
			return RunDetail{}, err
		}
	}
	if detail.Recovery, err = stores.Recovery(ctx, runID); err != nil {
		return RunDetail{}, err
	}
	if err := readStageRecords(ctx, stores, ownerID, &detail); err != nil {
		return RunDetail{}, err
	}
	return detail, nil
}

// readStageRecords reads the records of every StageExecution with at most one
// query per collection, including Stages without allocations or metrics.
func readStageRecords(ctx context.Context, stores RunDetailStores, ownerID string, detail *RunDetail) error {
	detail.Allocations = make(map[string][]runstore.StageAllocation)
	detail.Metrics = make(map[string]telemetry.StageMetricsRecord)
	detail.Plans = make(map[string]planner.PlannerPlanProjection)
	detail.Resources = make(map[string][]telemetry.AllocationResourceSummary)
	if len(detail.Executions) == 0 {
		return nil
	}
	ids := make([]string, len(detail.Executions))
	var identities []planner.SessionIdentity
	for index, execution := range detail.Executions {
		ids[index] = execution.StageExecutionID
		if execution.PlannerSessionID != nil && execution.PlannerInvocationID != nil {
			identities = append(identities, planner.SessionIdentity{
				SessionID: *execution.PlannerSessionID, StageExecutionID: execution.StageExecutionID,
				InvocationID: *execution.PlannerInvocationID,
			})
		}
	}
	var err error
	if detail.Allocations, err = stores.Runs.ListStageAllocationsBatch(ctx, ids); err != nil {
		return err
	}
	if detail.Plans, err = stores.Plans.LoadPlans(ctx, identities); err != nil {
		return err
	}
	if detail.Resources, err = stores.Resources.ListStageAllocationResources(ctx, ownerID, ids); err != nil {
		return err
	}
	// Optional diagnostics must not prevent reading the durable Run state.
	if metrics, metricsErr := stores.Metrics.GetStageMetricsBatch(ctx, ids); metricsErr == nil {
		detail.Metrics = metrics
	}
	return nil
}

// runArtifactsByNamespace maps binding names in one Run namespace to their
// current exact refs, reading metadata in keyset pages instead of one lookup
// per binding.
func runArtifactsByNamespace(
	ctx context.Context, service *artifacts.Service, runID, namespace string,
) (map[string]contracts.ArtifactRef, error) {
	store, err := service.Run(runID)
	if err != nil {
		return nil, err
	}
	result := make(map[string]contracts.ArtifactRef)
	query := artifacts.BindingPageQuery{Namespace: &namespace, Limit: pagelimit.MaxRows}
	for {
		page, err := store.ListMetadata(ctx, query)
		if err != nil {
			return nil, err
		}
		for _, metadata := range page {
			result[metadata.Ref.Name] = metadata.Ref
		}
		if len(page) < query.Limit {
			return result, nil
		}
		last := page[len(page)-1].Ref
		query.AfterNamespace, query.AfterName = last.Namespace, last.Name
	}
}
