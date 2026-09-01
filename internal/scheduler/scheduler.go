package scheduler

import (
	"bytes"
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/url"
	"reflect"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/grauwolf32/contractor/internal/artifactpolicy"
	"github.com/grauwolf32/contractor/internal/artifacts"
	workflowconfig "github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/controlplane"
	"github.com/grauwolf32/contractor/internal/planner"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
	"github.com/grauwolf32/contractor/internal/telemetry"
)

const (
	defaultPollInterval           = time.Second
	defaultClaimDuration          = 2 * time.Minute
	defaultOperationTimeout       = 15 * time.Second
	defaultPlannerTimeout         = 45 * time.Second
	defaultFinalizationTimeout    = 10 * time.Second
	defaultAbortTimeout           = 10 * time.Second
	defaultLeaseScanInterval      = time.Second
	defaultMetricsCleanupInterval = 24 * time.Hour
	defaultMetricsCleanupBatch    = 500
	maxCandidateBytes             = 256 * 1024
	maxCandidateSummaryBytes      = 64 * 1024
	maxCandidateArtifacts         = 128
)

type Scheduler struct {
	store       Store
	persistence AtomicPersistence
	artifacts   ArtifactResolver
	allocator   Allocator
	workers     WorkerController
	planners    PlannerRegistry
	options     Options
	wake        chan struct{}
	activeMu    sync.Mutex
	active      map[string]context.CancelCauseFunc
}

func New(
	store Store,
	persistence AtomicPersistence,
	artifactResolver ArtifactResolver,
	allocator Allocator,
	workers WorkerController,
	planners PlannerRegistry,
	options Options,
) (*Scheduler, error) {
	if store == nil || persistence == nil || artifactResolver == nil || allocator == nil ||
		workers == nil || planners == nil {
		return nil, fmt.Errorf("Scheduler dependencies are incomplete")
	}
	applyOptionDefaults(&options)
	if options.PollInterval <= 0 || options.ClaimDuration <= 0 || options.OperationTimeout <= 0 ||
		options.PlannerTimeout <= 0 || options.FinalizationTimeout <= 0 || options.AbortTimeout <= 0 ||
		options.LeaseScanInterval <= 0 || options.MetricsCleanupInterval <= 0 ||
		options.MetricsCleanupBatch <= 0 || options.MetricsCleanupBatch > 10_000 {
		return nil, fmt.Errorf("Scheduler durations must be positive and cleanup batch must be at most 10000")
	}
	if err := validateRuntimeSettings(options.RuntimeSettings); err != nil {
		return nil, err
	}
	return &Scheduler{
		store: store, persistence: persistence, artifacts: artifactResolver,
		allocator: allocator, workers: workers, planners: planners, options: options,
		wake: make(chan struct{}, 1), active: make(map[string]context.CancelCauseFunc),
	}, nil
}

func applyOptionDefaults(options *Options) {
	if options.PollInterval == 0 {
		options.PollInterval = defaultPollInterval
	}
	if options.ClaimDuration == 0 {
		options.ClaimDuration = defaultClaimDuration
	}
	if options.OperationTimeout == 0 {
		options.OperationTimeout = defaultOperationTimeout
	}
	if options.PlannerTimeout == 0 {
		options.PlannerTimeout = defaultPlannerTimeout
	}
	if options.FinalizationTimeout == 0 {
		options.FinalizationTimeout = defaultFinalizationTimeout
	}
	if options.AbortTimeout == 0 {
		options.AbortTimeout = defaultAbortTimeout
	}
	if options.LeaseScanInterval == 0 {
		options.LeaseScanInterval = defaultLeaseScanInterval
	}
	if options.MetricsCleanupInterval == 0 {
		options.MetricsCleanupInterval = defaultMetricsCleanupInterval
	}
	if options.MetricsCleanupBatch == 0 {
		options.MetricsCleanupBatch = defaultMetricsCleanupBatch
	}
	if options.Clock == nil {
		options.Clock = realClock{}
	}
	if options.NewID == nil {
		options.NewID = schedulerID
	}
	if options.Logger == nil {
		options.Logger = slog.New(slog.NewTextHandler(io.Discard, nil))
	}
	options.TelemetrySecrets = append([]string(nil), options.TelemetrySecrets...)
}

func validateRuntimeSettings(settings contracts.RuntimeSettings) error {
	parsed, err := url.Parse(settings.ArtifactAPIURL)
	if err != nil || parsed.Host == "" || parsed.User != nil ||
		(parsed.Scheme != "http" && parsed.Scheme != "https") {
		return fmt.Errorf("Scheduler Artifact API URL is invalid")
	}
	if settings.RequestTimeoutSeconds <= 0 {
		return fmt.Errorf("Scheduler RuntimeSettings require a positive timeout")
	}
	return nil
}

// Wake requests an immediate claim attempt. It is edge-triggered and never
// blocks a public API handler.
func (s *Scheduler) Wake() {
	select {
	case s.wake <- struct{}{}:
	default:
	}
}

// Cancel interrupts an in-process Planner or lifecycle request after the
// public API has already made cancellation durable. A different Scheduler
// process observes the same state through claim reconciliation.
func (s *Scheduler) Cancel(runID string) {
	s.interruptRun(runID, ErrRunCancellationRequested)
	s.Wake()
}

func (s *Scheduler) interruptRun(runID string, cause error) {
	s.activeMu.Lock()
	cancel := s.active[runID]
	s.activeMu.Unlock()
	if cancel != nil {
		cancel(cause)
	}
}

// Run serves claims serially. One Scheduler instance therefore executes one
// Stage at a time; PostgreSQL claims still protect against another process.
func (s *Scheduler) Run(ctx context.Context) error {
	monitorContext, cancelMonitor := context.WithCancel(ctx)
	var monitor sync.WaitGroup
	monitor.Add(2)
	go func() {
		defer monitor.Done()
		s.monitorAllocationLosses(monitorContext)
	}()
	go func() {
		defer monitor.Done()
		s.monitorMetricsRetention(monitorContext)
	}()
	defer func() {
		cancelMonitor()
		monitor.Wait()
	}()
	for {
		worked, err := s.RunOnce(ctx)
		if ctx.Err() != nil {
			return nil
		}
		if err != nil && !errors.Is(err, ErrDeferred) {
			s.options.Logger.Error("Workflow Scheduler iteration failed", "error", err)
		}
		if worked && err == nil {
			continue
		}
		select {
		case <-ctx.Done():
			return nil
		case <-s.wake:
		case <-s.options.Clock.After(s.options.PollInterval):
		}
	}
}

// RunOnce claims and advances at most one WorkflowRun.
func (s *Scheduler) RunOnce(ctx context.Context) (bool, error) {
	s.pollAllocationLosses()
	released, terminalReleaseErr := s.recoverTerminalRelease(ctx)
	if terminalReleaseErr != nil {
		s.options.Logger.Warn("terminal allocation release recovery failed", "error", terminalReleaseErr)
	}
	claimID, err := s.options.NewID("claim_")
	if err != nil {
		return false, fmt.Errorf("generate Scheduler claim ID: %w", err)
	}
	claimContext, cancelClaim := context.WithTimeout(ctx, s.options.OperationTimeout)
	run, err := s.store.ClaimRunnableRun(claimContext, claimID, s.options.ClaimDuration)
	cancelClaim()
	if errors.Is(err, runstore.ErrNoWork) {
		return released && terminalReleaseErr == nil, nil
	}
	if err != nil {
		return false, err
	}

	executionContext, cancelExecution := context.WithCancelCause(ctx)
	s.activeMu.Lock()
	s.active[run.RunID] = cancelExecution
	s.activeMu.Unlock()
	defer func() {
		s.activeMu.Lock()
		if current := s.active[run.RunID]; current != nil {
			delete(s.active, run.RunID)
		}
		s.activeMu.Unlock()
	}()
	stopRenewal := make(chan struct{})
	var renewal sync.WaitGroup
	renewal.Add(1)
	go func() {
		defer renewal.Done()
		s.renewClaim(executionContext, cancelExecution, stopRenewal, run.RunID, claimID)
	}()

	err = s.executeRun(executionContext, run)
	if cause := context.Cause(executionContext); ctx.Err() == nil &&
		(errors.Is(cause, ErrRunCancellationRequested) || errors.Is(cause, ErrAllocationLeaseLost)) {
		// The interrupted context intentionally cannot be reused for cleanup.
		// Retain the same durable claim while entering the authoritative bounded path.
		current, loadErr := s.store.GetRun(ctx, run.RunID)
		if loadErr != nil {
			err = loadErr
		} else if current.State == runstore.RunCancelling ||
			(errors.Is(cause, ErrAllocationLeaseLost) && current.State == runstore.RunRunning) {
			err = s.executeRun(ctx, current)
		} else {
			err = nil
		}
	}
	close(stopRenewal)
	cancelExecution(nil)
	renewal.Wait()

	releaseContext, cancelRelease := context.WithTimeout(context.Background(), s.options.OperationTimeout)
	releaseErr := s.store.ReleaseRunClaim(releaseContext, run.RunID, claimID)
	cancelRelease()
	if releaseErr != nil && !errors.Is(releaseErr, runstore.ErrConflict) && err == nil {
		err = releaseErr
	}
	if cause := context.Cause(executionContext); cause != nil &&
		!errors.Is(cause, context.Canceled) && !errors.Is(cause, ErrRunCancellationRequested) &&
		!errors.Is(cause, ErrAllocationLeaseLost) && err == nil {
		err = cause
	}
	return true, err
}

func (s *Scheduler) monitorAllocationLosses(ctx context.Context) {
	for {
		s.pollAllocationLosses()
		select {
		case <-ctx.Done():
			return
		case <-s.options.Clock.After(s.options.LeaseScanInterval):
		}
	}
}

func (s *Scheduler) monitorMetricsRetention(ctx context.Context) {
	for {
		cleanupContext, cancel := context.WithTimeout(ctx, s.options.OperationTimeout)
		deleted, err := s.store.CleanupExpiredTelemetry(
			cleanupContext, s.options.Clock.Now(), s.options.MetricsCleanupBatch,
		)
		cancel()
		if err != nil && ctx.Err() == nil {
			s.options.Logger.Warn("telemetry retention cleanup failed", "error", err)
		} else if deleted > 0 {
			s.options.Logger.Info("expired telemetry removed", "rows", deleted)
		}
		select {
		case <-ctx.Done():
			return
		case <-s.options.Clock.After(s.options.MetricsCleanupInterval):
		}
	}
}

func (s *Scheduler) pollAllocationLosses() {
	for _, loss := range s.allocator.PollAllocationLosses() {
		s.options.Logger.Warn(
			"Runtime Agent allocation was lost",
			"run_id", loss.RunID,
			"stage_execution_id", loss.StageExecutionID,
			"allocation_id", loss.AllocationID,
			"reason", loss.Reason,
		)
		s.interruptRun(loss.RunID, &AllocationLeaseLossError{Loss: loss})
		s.Wake()
	}
}

func (s *Scheduler) renewClaim(
	ctx context.Context,
	cancel context.CancelCauseFunc,
	stop <-chan struct{},
	runID string,
	claimID string,
) {
	interval := s.options.ClaimDuration / 3
	if s.options.PollInterval < interval {
		interval = s.options.PollInterval
	}
	if interval <= 0 {
		interval = time.Millisecond
	}
	for {
		select {
		case <-ctx.Done():
			return
		case <-stop:
			return
		case <-s.options.Clock.After(interval):
		}
		renewContext, renewCancel := context.WithTimeout(ctx, s.options.OperationTimeout)
		err := s.store.RenewRunClaim(renewContext, runID, claimID, s.options.ClaimDuration)
		if err != nil {
			renewCancel()
			cancel(errors.Join(ErrClaimLost, err))
			return
		}
		current, err := s.store.GetRun(renewContext, runID)
		renewCancel()
		if err != nil {
			cancel(errors.Join(ErrClaimLost, err))
			return
		}
		if current.State == runstore.RunCancelling {
			cancel(ErrRunCancellationRequested)
			return
		}
	}
}

func (s *Scheduler) executeRun(ctx context.Context, run runstore.WorkflowRun) error {
	current, err := s.store.GetRun(ctx, run.RunID)
	if err != nil {
		return err
	}
	run = current
	if run.State == runstore.RunInitializing &&
		run.StateReason.Code == runstore.SkillInitializationPendingReason {
		if s.options.RunSkills == nil {
			return fmt.Errorf("Run Skill initializer is not configured")
		}
		run, err = s.options.RunSkills.InitializeRunSkills(ctx, run.RunID)
		if err != nil {
			return err
		}
	}
	if run.State == runstore.RunCancelling {
		return s.executeCancelling(ctx, run)
	}
	if run.State != runstore.RunRunning {
		return nil
	}
	workflow, err := decodeExecutableWorkflow(run)
	if err != nil {
		return s.failUnsupportedRun(ctx, run.RunID, err)
	}
	executions, err := s.store.ListStageExecutions(ctx, run.RunID)
	if err != nil {
		return err
	}
	var execution runstore.StageExecution
	if len(executions) == 0 {
		execution, err = s.createStageExecution(ctx, run, workflow)
		if err != nil {
			return err
		}
	} else {
		active := make([]runstore.StageExecution, 0, 1)
		for _, candidate := range executions {
			switch candidate.State {
			case runstore.StagePreparing, runstore.StageRunning, runstore.StageFinalizing, runstore.StageAborting:
				active = append(active, candidate)
			}
		}
		if len(active) != 1 {
			return s.failInvalidRunState(
				ctx,
				run.RunID,
				fmt.Errorf("running WorkflowRun has %d active StageExecutions", len(active)),
			)
		}
		execution = active[0]
		workflow, err = workflow.selectStage(execution.StageName)
		if err != nil {
			return s.failInvalidRunState(ctx, run.RunID, err)
		}
	}
	workflow, err = workflow.selectExecution(execution)
	if err != nil {
		return s.failInvalidRunState(ctx, run.RunID, err)
	}
	if err := validatePersistedExecution(execution, run, workflow); err != nil {
		return s.failInvalidRunState(ctx, run.RunID, err)
	}

	switch execution.State {
	case runstore.StagePreparing:
		if missing := missingRequiredContext(execution); missing != "" {
			return s.beginAbort(ctx, run, workflow, execution, nil, planner.Failure{
				Code:      "context_artifact_missing",
				Message:   "Required Stage context artifact is unavailable",
				Retryable: false,
			})
		}
		return s.prepareAndPlan(ctx, run, workflow, execution)
	case runstore.StageRunning:
		return s.prepareAndPlan(ctx, run, workflow, execution)
	case runstore.StageFinalizing:
		return s.resumeFinalizing(ctx, run, workflow, execution, nil)
	case runstore.StageAborting:
		return s.resumeAborting(ctx, run, workflow, execution, nil)
	case runstore.StageInterrupted, runstore.StageCancelled, runstore.StageSucceeded, runstore.StageFailed:
		return s.failInvalidRunState(ctx, run.RunID, fmt.Errorf("running WorkflowRun selected a terminal StageExecution"))
	default:
		return s.failInvalidRunState(ctx, run.RunID, fmt.Errorf("unknown StageExecution state %q", execution.State))
	}
}

func (s *Scheduler) executeCancelling(ctx context.Context, run runstore.WorkflowRun) error {
	executions, err := s.store.ListStageExecutions(ctx, run.RunID)
	if err != nil {
		return err
	}
	workflow, workflowErr := decodeExecutableWorkflow(run)
	if workflowErr != nil {
		s.options.Logger.Warn("cancelling Run has an unreadable Workflow snapshot", "run_id", run.RunID)
	}

	active := make([]runstore.StageExecution, 0, 1)
	for _, execution := range executions {
		switch execution.State {
		case runstore.StagePreparing, runstore.StageRunning, runstore.StageFinalizing, runstore.StageAborting:
			active = append(active, execution)
		}
	}
	if len(active) > 1 {
		return fmt.Errorf("cancelling MVP Run %q has multiple active StageExecutions", run.RunID)
	}
	if len(active) == 0 {
		for _, execution := range executions {
			_ = s.fenceRecordedAllocations(ctx, execution.StageExecutionID, nil)
			if workflowErr == nil {
				reservations := s.existingLiveReservations(ctx, run, workflow, execution)
				_ = s.releaseTerminal(execution.StageExecutionID, reservations)
			}
		}
		return s.finishCancelledRun(ctx, run.RunID)
	}

	execution := active[0]
	var reservations []controlplane.Reservation
	if workflowErr == nil {
		workflow, workflowErr = workflow.selectExecution(execution)
	}
	if workflowErr == nil {
		reservations = s.existingLiveReservations(ctx, run, workflow, execution)
	}
	switch execution.State {
	case runstore.StagePreparing, runstore.StageRunning:
		return s.beginCancellationAbort(ctx, run, workflow, execution, reservations)
	case runstore.StageAborting:
		return s.resumeAborting(ctx, run, workflow, execution, reservations)
	case runstore.StageFinalizing:
		return s.resumeFinalizing(ctx, run, workflow, execution, reservations)
	default:
		return fmt.Errorf("unhandled cancelling StageExecution state %q", execution.State)
	}
}

func (s *Scheduler) finishCancelledRun(ctx context.Context, runID string) error {
	operationContext, cancel := context.WithTimeout(context.WithoutCancel(ctx), s.options.OperationTimeout)
	defer cancel()
	_, err := s.store.TransitionRun(
		operationContext,
		runID,
		runstore.RunCancelling,
		runstore.RunCancelled,
		runstore.Reason{Code: runstore.CancellationUserRequested},
	)
	if errors.Is(err, runstore.ErrConflict) {
		current, loadErr := s.store.GetRun(operationContext, runID)
		if loadErr == nil && current.State == runstore.RunCancelled {
			return nil
		}
	}
	return err
}

func (s *Scheduler) failUnsupportedRun(ctx context.Context, runID string, cause error) error {
	operationContext, cancel := context.WithTimeout(ctx, s.options.OperationTimeout)
	defer cancel()
	_, err := s.store.TransitionRun(
		operationContext,
		runID,
		runstore.RunRunning,
		runstore.RunFailed,
		runstore.Reason{Code: "unsupported_workflow_shape"},
	)
	if err != nil {
		return errors.Join(cause, err)
	}
	return nil
}

func (s *Scheduler) failInvalidRunState(ctx context.Context, runID string, cause error) error {
	operationContext, cancel := context.WithTimeout(ctx, s.options.OperationTimeout)
	defer cancel()
	_, err := s.store.TransitionRun(
		operationContext,
		runID,
		runstore.RunRunning,
		runstore.RunFailed,
		runstore.Reason{Code: "scheduler_state_invalid"},
	)
	if err != nil {
		return errors.Join(cause, err)
	}
	return nil
}

func (s *Scheduler) createStageExecution(
	ctx context.Context,
	run runstore.WorkflowRun,
	workflow executableWorkflow,
) (runstore.StageExecution, error) {
	creation, err := s.buildStageCreation(
		ctx, run, workflow, 1, nil,
		stageExecutionConfiguration{variant: runstore.StageExecutionConfigBase},
	)
	if err != nil {
		return runstore.StageExecution{}, err
	}
	operationContext, cancel := context.WithTimeout(ctx, s.options.OperationTimeout)
	defer cancel()
	return s.persistence.CreateStageWithContext(operationContext, creation.Params, creation.ContextPins)
}

func (s *Scheduler) buildStageCreation(
	ctx context.Context,
	run runstore.WorkflowRun,
	workflow executableWorkflow,
	attempt int,
	previousExecutionID *string,
	configuration stageExecutionConfiguration,
) (NextStageCreation, error) {
	stageExecutionID, err := s.options.NewID("stage_execution_")
	if err != nil {
		return NextStageCreation{}, fmt.Errorf("generate StageExecution ID: %w", err)
	}
	contextSnapshot := runstore.StageContextSnapshot{
		Parameters: cloneParameters(run.Parameters),
		Artifacts:  make(map[string]runstore.PinnedContextArtifact, len(workflow.stage.Context.Artifacts)),
	}
	pins := make([]ContextPin, 0, len(workflow.stage.Context.Artifacts))
	names := make([]string, 0, len(workflow.stage.Context.Artifacts))
	for name := range workflow.stage.Context.Artifacts {
		names = append(names, name)
	}
	sort.Strings(names)
	for _, name := range names {
		declaration := workflow.stage.Context.Artifacts[name]
		current := contracts.ArtifactRef{Namespace: declaration.Namespace, Name: declaration.Name}
		resolved, resolveErr := s.artifacts.Resolve(ctx, run.RunID, current)
		if errors.Is(resolveErr, artifacts.ErrArtifactNotFound) {
			contextSnapshot.Artifacts[name] = runstore.PinnedContextArtifact{Required: declaration.Required}
			continue
		}
		if resolveErr != nil {
			return NextStageCreation{}, fmt.Errorf("resolve StageContext artifact %q: %w", name, resolveErr)
		}
		if resolved.Ref.Namespace != declaration.Namespace || resolved.Ref.Name != declaration.Name {
			return NextStageCreation{}, fmt.Errorf("ArtifactStore resolved StageContext artifact %q to another binding", name)
		}
		exact := cloneArtifactRef(resolved.Ref)
		contextSnapshot.Artifacts[name] = runstore.PinnedContextArtifact{
			Required: declaration.Required,
			Artifact: &exact,
		}
		pins = append(pins, ContextPin{Name: name, Ref: exact})
	}
	stage := workflow.stage
	if configuration.variant == "" {
		configuration.variant = runstore.StageExecutionConfigBase
	}
	switch configuration.variant {
	case runstore.StageExecutionConfigBase:
		if configuration.ordinal != nil || configuration.effective != nil {
			return NextStageCreation{}, fmt.Errorf("base Stage execution configuration has escalation data")
		}
	case runstore.StageExecutionConfigFailedEscalation,
		runstore.StageExecutionConfigInterruptedEscalation:
		if configuration.ordinal == nil || *configuration.ordinal <= 0 || configuration.effective == nil {
			return NextStageCreation{}, fmt.Errorf("escalated Stage execution configuration is incomplete")
		}
		stage.ExecutionConfig = *configuration.effective
	default:
		return NextStageCreation{}, fmt.Errorf("unknown Stage execution configuration %q", configuration.variant)
	}
	encodedStage, err := stageSnapshot(stage)
	if err != nil {
		return NextStageCreation{}, err
	}
	return NextStageCreation{Params: runstore.CreateStageExecutionParams{
		StageExecutionID: stageExecutionID,
		RunID:            run.RunID, StageName: workflow.stageName, Attempt: attempt,
		PreviousExecutionID:    previousExecutionID,
		ExecutionConfigVariant: configuration.variant,
		EscalationOrdinal:      configuration.ordinal,
		StageSpecSchemaVersion: contracts.APIVersion, StageSpecSnapshot: encodedStage,
		StageContextSchemaVersion: contracts.APIVersion, StageContext: contextSnapshot,
	}, ContextPins: pins}, nil
}

type stageExecutionConfiguration struct {
	variant   runstore.StageExecutionConfigVariant
	ordinal   *int
	effective *workflowconfig.ResolvedStageExecutionConfig
}

func missingRequiredContext(execution runstore.StageExecution) string {
	names := make([]string, 0, len(execution.StageContext.Artifacts))
	for name := range execution.StageContext.Artifacts {
		names = append(names, name)
	}
	sort.Strings(names)
	for _, name := range names {
		value := execution.StageContext.Artifacts[name]
		if value.Required && value.Artifact == nil {
			return name
		}
	}
	return ""
}

func (s *Scheduler) prepareAndPlan(
	ctx context.Context,
	run runstore.WorkflowRun,
	workflow executableWorkflow,
	execution runstore.StageExecution,
) error {
	stageDeadline := s.stageDeadline(execution)
	reservations, fresh, err := s.liveOrNewReservations(ctx, run, workflow, execution)
	if errors.Is(err, controlplane.ErrInsufficientCapacity) {
		if !s.options.Clock.Now().Before(stageDeadline) {
			return s.beginAbort(ctx, run, workflow, execution, nil, stageDeadlineFailure())
		}
		return ErrDeferred
	}
	if errors.Is(err, errControlPlaneStateLost) {
		return s.beginAbort(ctx, run, workflow, execution, nil, planner.Failure{
			Code: "control_plane_state_lost", Message: "Control Plane lost the active allocation set", Retryable: true,
		})
	}
	if errors.Is(err, errControlPlaneAllocationLost) {
		return s.beginAbort(context.WithoutCancel(ctx), run, workflow, execution, reservations, planner.Failure{
			Code: "control_lease_expired", Message: "Runtime Agent allocation control lease was lost", Retryable: true,
		})
	}
	if err != nil {
		return err
	}
	if !s.options.Clock.Now().Before(stageDeadline) {
		return s.beginAbort(ctx, run, workflow, execution, reservations, stageDeadlineFailure())
	}
	if cause := context.Cause(ctx); errors.Is(cause, ErrAllocationLeaseLost) {
		return cause
	}
	if fresh {
		if err := s.recordReservations(ctx, execution.StageExecutionID, reservations); err != nil {
			s.releaseUnprepared(reservations)
			return s.beginAbort(ctx, run, workflow, execution, nil, planner.Failure{
				Code: "allocation_record_failed", Message: "Stage allocation provenance could not be recorded", Retryable: true,
			})
		}
	}

	workerSettings, err := s.workerExecutionSettings(ctx, workflow.stage, reservations)
	if err != nil {
		return s.beginAbort(ctx, run, workflow, execution, reservations, planner.Failure{
			Code: "worker_execution_config_unavailable", Message: "Worker execution configuration is unavailable", Retryable: false,
		})
	}
	prepareContext, cancelPrepare := context.WithTimeout(ctx, s.options.OperationTimeout)
	handles, err := s.workers.PrepareAll(prepareContext, reservations, workerSettings)
	cancelPrepare()
	clearWorkerExecutionSettings(workerSettings)
	if err != nil {
		failure := infrastructureFailure("allocation_preparation_failed", "Worker allocation preparation failed", err)
		return s.beginAbort(ctx, run, workflow, execution, nil, failure)
	}
	if cause := context.Cause(ctx); errors.Is(cause, ErrAllocationLeaseLost) {
		return cause
	}
	if !s.options.Clock.Now().Before(stageDeadline) {
		return s.beginAbort(ctx, run, workflow, execution, reservations, stageDeadlineFailure())
	}

	modelAccess, err := s.plannerModelAccess(ctx, workflow.stage)
	if err != nil {
		return s.beginAbort(ctx, run, workflow, execution, reservations, planner.Failure{
			Code: "planner_execution_config_unavailable", Message: "Planner execution configuration is unavailable", Retryable: false,
		})
	}
	plannerRef := workflow.stage.Planner.PlannerID + "@" + workflow.stage.Planner.Version
	plannerTelemetry := s.newPlannerTelemetry(
		ctx, run, execution, reservations, plannerRef, modelAccess,
	)
	if plannerTelemetry != nil {
		defer plannerTelemetry.Close()
	}
	instrumentation := telemetry.NoopPlannerInstrumentation()
	if plannerTelemetry != nil {
		instrumentation = plannerTelemetry.Instrumentation()
	}
	invocation := planner.Invocation{
		StageExecutionID: execution.StageExecutionID,
		RunID:            run.RunID,
		Stage:            workflow.stage,
		Context:          plannerContext(execution.StageContext),
		Workers:          handles,
		ModelAccess:      modelAccess,
		Instrumentation:  instrumentation,
		Deadline:         stageDeadline,
	}
	invocationSpan := instrumentation.StartSpan(
		telemetry.PlannerSpanInvocation,
		telemetry.PlannerSpanAttributes{Operation: "planner.run"},
	)
	instance, err := s.planners.Create(plannerRef, invocation)
	if err != nil {
		invocationSpan.End("failed", telemetry.PlannerSpanAttributes{ErrorCode: "planner_initialization_failed"})
		s.flushPlannerTelemetry(ctx, plannerTelemetry, stageDeadline, execution.StageExecutionID)
		return s.beginAbort(ctx, run, workflow, execution, reservations, planner.Failure{
			Code: "planner_initialization_failed", Message: "Planner could not be initialized", Retryable: false,
		})
	}
	candidate, err := instance.Run(ctx)
	invocationOutcome := "succeeded"
	invocationAttributes := telemetry.PlannerSpanAttributes{}
	if err != nil {
		failure := planner.FailureFrom(err)
		invocationOutcome = "failed"
		invocationAttributes.ErrorCode = failure.Code
	}
	invocationSpan.End(invocationOutcome, invocationAttributes)
	exportResult := s.flushPlannerTelemetry(
		ctx, plannerTelemetry, stageDeadline, execution.StageExecutionID,
	)
	if cause := context.Cause(ctx); errors.Is(cause, ErrAllocationLeaseLost) {
		return cause
	}
	currentExecution, loadErr := s.store.GetStageExecution(ctx, execution.StageExecutionID)
	if loadErr != nil {
		return errors.Join(err, loadErr)
	}
	execution = currentExecution
	s.persistPlannerReport(execution, instance, exportResult)
	if err != nil {
		return s.beginAbort(ctx, run, workflow, execution, reservations, planner.FailureFrom(err))
	}
	if err := s.validateCandidate(ctx, run.RunID, workflow.stage, candidate); err != nil {
		return s.beginAbort(ctx, run, workflow, execution, reservations, planner.Failure{
			Code: "result_contract_violation", Message: "Planner candidate violates the Stage result contract", Retryable: false,
		})
	}
	if err := s.fenceRecordedAllocations(ctx, execution.StageExecutionID, reservations); err != nil {
		return s.beginAbort(ctx, run, workflow, execution, reservations, planner.Failure{
			Code: "allocation_fence_failed", Message: "Stage allocations could not be write-fenced", Retryable: true,
		})
	}
	finalizationID, err := s.options.NewID("finalization_")
	if err != nil {
		return err
	}
	deadline := s.options.Clock.Now().Add(s.options.FinalizationTimeout)
	transitionContext, cancelTransition := context.WithTimeout(ctx, s.options.OperationTimeout)
	err = s.persistence.EnterFinalizingWithResult(transitionContext, runstore.EnterFinalizingParams{
		StageExecutionID:    execution.StageExecutionID,
		ResultSchemaVersion: contracts.APIVersion,
		Candidate:           candidate,
		FinalizationID:      finalizationID,
		Deadline:            deadline,
		Reason:              runstore.Reason{Code: "planner_completed"},
	})
	cancelTransition()
	if err != nil {
		return err
	}
	execution.State = runstore.StageFinalizing
	execution.CandidateResultSchemaVersion = stringPointer(contracts.APIVersion)
	clonedCandidate := cloneStageResult(candidate)
	execution.CandidateResult = &clonedCandidate
	execution.FinalizationID = &finalizationID
	execution.FinalizationDeadline = &deadline
	return s.resumeFinalizing(ctx, run, workflow, execution, reservations)
}

func (s *Scheduler) stageDeadline(execution runstore.StageExecution) time.Time {
	startedAt := execution.CreatedAt
	if startedAt.IsZero() {
		// Legacy embeddings and focused in-memory stores created before the
		// durable timestamp contract receive one finite budget from observation.
		// PostgreSQL StageExecutions always use their immutable database time.
		startedAt = s.options.Clock.Now()
	}
	return startedAt.Add(s.options.PlannerTimeout)
}

func stageDeadlineFailure() planner.Failure {
	return planner.Failure{
		Code:      "stage_deadline_exceeded",
		Message:   "Stage execution deadline expired before Planner completion",
		Retryable: true,
	}
}

func (s *Scheduler) workerExecutionSettings(
	ctx context.Context,
	stage workflowconfig.ResolvedStage,
	reservationSets ...[]controlplane.Reservation,
) (map[string]contracts.WorkerExecutionSettingsV2, error) {
	reservations := make(map[string]controlplane.Reservation)
	if len(reservationSets) > 1 {
		return nil, fmt.Errorf("Worker execution settings accept at most one reservation set")
	}
	if len(reservationSets) == 1 {
		for _, reservation := range reservationSets[0] {
			reservations[reservation.Grant.LogicalAgentName] = reservation
		}
	}
	result := make(map[string]contracts.WorkerExecutionSettingsV2, len(stage.ExecutionConfig.Agents))
	for logicalName, selection := range stage.ExecutionConfig.Agents {
		resolved, err := fallbackResolvedWorkerConfig(selection)
		if reservation, ok := reservations[logicalName]; ok {
			if reservation.ResolvedRuntimeConfig != nil {
				resolved = reservation.ResolvedRuntimeConfig.Clone()
			}
		}
		if err != nil || resolved.Validate() != nil {
			return nil, fmt.Errorf("Worker %q has no complete Runtime configuration", logicalName)
		}
		runtimeSettings, err := s.materializeRuntimeSettings(ctx, resolved)
		if err != nil {
			return nil, err
		}
		result[logicalName] = contracts.WorkerExecutionSettingsV2{
			ModelPolicy: cloneModelPolicy(resolved.ModelPolicy), RuntimeSettings: runtimeSettings,
			ResolvedRuntimeConfigProvenance: resolved.Provenance,
		}
	}
	if len(reservations) != 0 && len(reservations) != len(result) {
		return nil, fmt.Errorf("reservation set differs from Worker execution settings")
	}
	return result, nil
}

func fallbackResolvedWorkerConfig(
	selection workflowconfig.ResolvedConsumerExecutionConfig,
) (runtimeconfig.ResolvedRuntimeConfig, error) {
	if selection.LLMGateway == nil {
		return runtimeconfig.ResolvedRuntimeConfig{}, fmt.Errorf("Worker has no complete LLM Gateway route")
	}
	gatewayRef := selection.LLMGateway.Ref
	provenance := contracts.ResolvedRuntimeConfigProvenanceV2{
		Default: contracts.RuntimeLabelBindingProvenanceV2{
			Label: "default", BindingRevision: 1,
			Config: contracts.RuntimeConfigRefV2{
				Name: runtimeconfig.BuiltInName, Version: runtimeconfig.BuiltInVersion,
				Digest: runtimeconfig.BuiltInDigest,
			},
		},
		RunLabels:        []contracts.RuntimeLabelBindingProvenanceV2{},
		AgentLabels:      []contracts.RuntimeLabelBindingProvenanceV2{},
		RuntimeAdapters:  []contracts.RuntimeAdapterRef{},
		LLMGatewayConfig: &gatewayRef, LLMCredential: cloneCredentialRef(selection.Credential),
		RuntimeCredentialRefs: []contracts.RuntimeCredentialRefV2{},
	}
	return runtimeconfig.ResolvedRuntimeConfig{
		ModelPolicy: cloneModelPolicy(selection.ModelPolicy), LLMGateway: *selection.LLMGateway,
		LLMCredential:           cloneCredentialRef(selection.Credential),
		RequiredRuntimeAdapters: []contracts.RuntimeAdapterRef{}, Provenance: provenance,
	}, nil
}

func (s *Scheduler) materializeRuntimeSettings(
	ctx context.Context,
	resolved runtimeconfig.ResolvedRuntimeConfig,
) (contracts.RuntimeSettingsV2, error) {
	result := contracts.RuntimeSettingsV2{
		LLMGatewayURL:         resolved.LLMGateway.URL,
		ArtifactAPIURL:        s.options.RuntimeSettings.ArtifactAPIURL,
		RequestTimeoutSeconds: s.options.RuntimeSettings.RequestTimeoutSeconds,
	}
	if resolved.LLMCredential != nil {
		if s.options.Credentials == nil {
			return contracts.RuntimeSettingsV2{}, fmt.Errorf("selected LLM credential is unavailable")
		}
		token, err := s.options.Credentials.ResolveLLMCredential(
			ctx, *resolved.LLMCredential, resolved.LLMGateway.Ref,
		)
		if err != nil || token.Reveal() == "" {
			return contracts.RuntimeSettingsV2{}, fmt.Errorf("selected LLM credential is unavailable")
		}
		result.LLMGatewayToken = &token
	}
	if resolved.WorkerTelemetry != nil {
		telemetry := &contracts.TelemetrySettingsV2{
			Adapter: contracts.RuntimeAdapterOTLPHTTP, Endpoint: resolved.WorkerTelemetry.Endpoint,
			Headers: map[string]contracts.SecretString{}, CaptureContent: resolved.WorkerTelemetry.CaptureContent,
			FlushTimeoutSeconds: resolved.WorkerTelemetry.FlushTimeoutSeconds,
		}
		if credentialID := resolved.WorkerTelemetry.Credential; credentialID != "" {
			if err := s.useRuntimeCredential(
				ctx, credentialID, []contracts.RuntimeCredentialKind{contracts.RuntimeCredentialOTLPHeaders},
				func(kind contracts.RuntimeCredentialKind, plaintext []byte) error {
					var material struct {
						Headers map[string]string `json:"headers"`
					}
					if kind != contracts.RuntimeCredentialOTLPHeaders || decodeRuntimeCredential(plaintext, &material) != nil {
						return errors.New("invalid OTLP credential material")
					}
					for name, value := range material.Headers {
						telemetry.Headers[name] = contracts.NewSecretString(value)
					}
					return nil
				},
			); err != nil {
				return contracts.RuntimeSettingsV2{}, fmt.Errorf("Worker telemetry credential is unavailable")
			}
		}
		result.Telemetry = telemetry
	}
	if resolved.HTTPProxy != nil {
		proxy := &contracts.HTTPProxySettingsV2{
			Adapter: contracts.RuntimeAdapterHTTPProxy, ProxyURL: resolved.HTTPProxy.ProxyURL,
			Targets: make([]contracts.HTTPProxyTarget, len(resolved.HTTPProxy.Targets)),
		}
		for index, target := range resolved.HTTPProxy.Targets {
			proxy.Targets[index] = contracts.HTTPProxyTarget(target)
		}
		if resolved.HTTPProxy.CABundlePEM != "" {
			bundle := resolved.HTTPProxy.CABundlePEM
			proxy.CABundlePEM = &bundle
		}
		if credentialID := resolved.HTTPProxy.Credential; credentialID != "" {
			if err := s.useRuntimeCredential(
				ctx, credentialID,
				[]contracts.RuntimeCredentialKind{
					contracts.RuntimeCredentialProxyBasic, contracts.RuntimeCredentialProxyBearer,
				},
				func(kind contracts.RuntimeCredentialKind, plaintext []byte) error {
					switch kind {
					case contracts.RuntimeCredentialProxyBasic:
						var material struct {
							Password string `json:"password"`
							Username string `json:"username"`
						}
						if decodeRuntimeCredential(plaintext, &material) != nil {
							return errors.New("invalid proxy basic credential material")
						}
						proxy.BasicAuth = &contracts.HTTPProxyBasicAuthV2{
							Username: contracts.NewSecretString(material.Username),
							Password: contracts.NewSecretString(material.Password),
						}
					case contracts.RuntimeCredentialProxyBearer:
						var material struct {
							Token string `json:"token"`
						}
						if decodeRuntimeCredential(plaintext, &material) != nil {
							return errors.New("invalid proxy bearer credential material")
						}
						token := contracts.NewSecretString(material.Token)
						proxy.BearerToken = &token
					default:
						return errors.New("invalid proxy credential kind")
					}
					return nil
				},
			); err != nil {
				return contracts.RuntimeSettingsV2{}, fmt.Errorf("Worker HTTP proxy credential is unavailable")
			}
		}
		result.HTTPProxy = proxy
	}
	if resolved.Caido != nil {
		timeout := resolved.Caido.RequestTimeoutSeconds
		if timeout == 0 || timeout > result.RequestTimeoutSeconds {
			timeout = result.RequestTimeoutSeconds
		}
		if timeout > 120 {
			timeout = 120
		}
		caido := &contracts.CaidoSettingsV2{
			Adapter: contracts.RuntimeAdapterCaidoGraphQL, Endpoint: resolved.Caido.Endpoint,
			RequestTimeoutSeconds: timeout,
		}
		if resolved.Caido.CABundlePEM != "" {
			bundle := resolved.Caido.CABundlePEM
			caido.CABundlePEM = &bundle
		}
		if credentialID := resolved.Caido.Credential; credentialID != "" {
			if err := s.useRuntimeCredential(
				ctx, credentialID, []contracts.RuntimeCredentialKind{contracts.RuntimeCredentialCaidoBearer},
				func(kind contracts.RuntimeCredentialKind, plaintext []byte) error {
					var material struct {
						Token string `json:"token"`
					}
					if kind != contracts.RuntimeCredentialCaidoBearer || decodeRuntimeCredential(plaintext, &material) != nil {
						return errors.New("invalid Caido credential material")
					}
					token := contracts.NewSecretString(material.Token)
					caido.BearerToken = &token
					return nil
				},
			); err != nil {
				return contracts.RuntimeSettingsV2{}, fmt.Errorf("Worker Caido credential is unavailable")
			}
		}
		result.Caido = caido
	}
	if err := result.Validate(); err != nil {
		return contracts.RuntimeSettingsV2{}, fmt.Errorf("materialized Runtime settings are invalid")
	}
	return result, nil
}

func (s *Scheduler) newPlannerTelemetry(
	ctx context.Context,
	run runstore.WorkflowRun,
	execution runstore.StageExecution,
	reservations []controlplane.Reservation,
	plannerRef string,
	modelAccess *planner.ModelAccess,
) telemetry.PlannerTelemetry {
	if s.options.PlannerTelemetry == nil || len(reservations) == 0 {
		return nil
	}
	selected := reservations[0].ResolvedRuntimeConfig
	for _, reservation := range reservations {
		if reservation.ResolvedRuntimeConfig != nil &&
			reservation.ResolvedRuntimeConfig.PlannerTelemetry != nil &&
			!plannerTelemetryOriginAllowed(reservation.ResolvedRuntimeConfig.Origins.PlannerTelemetry) {
			s.options.Logger.Warn(
				"Planner telemetry rejected a non-Run origin",
				"stage_execution_id", execution.StageExecutionID,
			)
			return nil
		}
	}
	for _, reservation := range reservations[1:] {
		if !samePlannerTelemetrySelection(selected, reservation.ResolvedRuntimeConfig) {
			s.options.Logger.Warn(
				"Planner telemetry selection differs across allocations",
				"stage_execution_id", execution.StageExecutionID,
			)
			return nil
		}
	}
	if selected == nil || selected.PlannerTelemetry == nil {
		return nil
	}
	configuration := *selected.PlannerTelemetry
	headers := make(map[string]contracts.SecretString)
	if configuration.Credential != "" {
		if selected.PlannerRuntimeCredential == nil ||
			selected.PlannerRuntimeCredential.CredentialID != configuration.Credential ||
			selected.PlannerRuntimeCredential.Kind != contracts.RuntimeCredentialOTLPHeaders {
			s.options.Logger.Warn(
				"Planner telemetry credential provenance is unavailable",
				"stage_execution_id", execution.StageExecutionID,
			)
			return nil
		}
		err := s.useRuntimeCredential(
			ctx, configuration.Credential,
			[]contracts.RuntimeCredentialKind{contracts.RuntimeCredentialOTLPHeaders},
			func(kind contracts.RuntimeCredentialKind, plaintext []byte) error {
				var material struct {
					Headers map[string]string `json:"headers"`
				}
				if kind != contracts.RuntimeCredentialOTLPHeaders || decodeRuntimeCredential(plaintext, &material) != nil {
					return errors.New("invalid OTLP credential material")
				}
				for name, value := range material.Headers {
					headers[name] = contracts.NewSecretString(value)
				}
				return nil
			},
		)
		if err != nil {
			for name := range headers {
				delete(headers, name)
			}
			s.options.Logger.Warn(
				"Planner telemetry credential is unavailable",
				"stage_execution_id", execution.StageExecutionID,
			)
			return nil
		}
	}
	resource := telemetry.PlannerResource{
		RunID: run.RunID, StageExecutionID: execution.StageExecutionID, PlannerRef: plannerRef,
		RuntimeCredentialID:  configuration.Credential,
		RuntimeConfigRefs:    plannerRuntimeConfigRefs(run.RuntimeConfig),
		RuntimeConfigDigests: plannerRuntimeConfigDigests(run.RuntimeConfig),
		RunLabels:            run.RuntimeConfig.ExplicitLabels(),
	}
	if modelAccess != nil {
		resource.ModelAlias = modelAccess.ModelPolicy.Model
		resource.ModelPolicyRef = modelAccess.ModelPolicy.Ref.PolicyID + "@" + modelAccess.ModelPolicy.Ref.Version
		resource.LLMGatewayRef = modelAccess.LLMGateway.Ref.GatewayID + "@" + modelAccess.LLMGateway.Ref.Version
		if modelAccess.Credential != nil {
			resource.LLMCredentialID = modelAccess.Credential.CredentialID
		}
	}
	created, err := s.options.PlannerTelemetry.Create(configuration.Adapter, telemetry.PlannerAdapterSettings{
		Endpoint: configuration.Endpoint, Headers: headers,
		FlushTimeout: time.Duration(configuration.FlushTimeoutSeconds) * time.Second,
		Resource:     resource,
	})
	for name := range headers {
		delete(headers, name)
	}
	if err != nil {
		s.options.Logger.Warn(
			"Planner telemetry adapter could not be created",
			"stage_execution_id", execution.StageExecutionID,
		)
		return nil
	}
	return created
}

func plannerTelemetryOriginAllowed(origin *runtimeconfig.RuntimeFieldOrigin) bool {
	return origin != nil && (origin.Layer == runtimeconfig.LayerDefault || origin.Layer == runtimeconfig.LayerRunLabels)
}

func samePlannerTelemetrySelection(
	left, right *runtimeconfig.ResolvedRuntimeConfig,
) bool {
	if left == nil || right == nil {
		return left == right
	}
	if (left.PlannerTelemetry == nil) != (right.PlannerTelemetry == nil) ||
		(left.PlannerRuntimeCredential == nil) != (right.PlannerRuntimeCredential == nil) {
		return false
	}
	if left.PlannerTelemetry != nil && *left.PlannerTelemetry != *right.PlannerTelemetry {
		return false
	}
	return left.PlannerRuntimeCredential == nil ||
		*left.PlannerRuntimeCredential == *right.PlannerRuntimeCredential
}

func plannerRuntimeConfigRefs(snapshot runtimeconfig.RunSnapshot) []string {
	result := make([]string, 0, len(snapshot.Labels)+1)
	result = append(result, snapshot.Default.Config.Name+"@"+snapshot.Default.Config.Version)
	for _, pinned := range snapshot.Labels {
		result = append(result, pinned.Config.Name+"@"+pinned.Config.Version)
	}
	return result
}

func plannerRuntimeConfigDigests(snapshot runtimeconfig.RunSnapshot) []string {
	result := make([]string, 0, len(snapshot.Labels)+1)
	result = append(result, snapshot.Default.Config.Digest)
	for _, pinned := range snapshot.Labels {
		result = append(result, pinned.Config.Digest)
	}
	return result
}

func (s *Scheduler) flushPlannerTelemetry(
	ctx context.Context,
	instance telemetry.PlannerTelemetry,
	stageDeadline time.Time,
	stageExecutionID string,
) *telemetry.PlannerExportResult {
	if instance == nil {
		return nil
	}
	bound := instance.FlushTimeout()
	if s.options.FinalizationTimeout < bound {
		bound = s.options.FinalizationTimeout
	}
	if remaining := stageDeadline.Sub(s.options.Clock.Now()); remaining < bound {
		bound = remaining
	}
	result := telemetry.PlannerExportResult{Attempted: true, ErrorCode: "flush_timeout"}
	if bound > 0 {
		flushContext, cancel := context.WithTimeout(ctx, bound)
		result = normalizePlannerExportResult(instance.Flush(flushContext))
		cancel()
	}
	if result.Attempted && !result.Succeeded {
		s.options.Logger.Warn(
			"Planner telemetry export failed",
			"stage_execution_id", stageExecutionID,
			"error_code", result.ErrorCode,
		)
	}
	return &result
}

func normalizePlannerExportResult(result telemetry.PlannerExportResult) telemetry.PlannerExportResult {
	if !result.Attempted {
		return telemetry.PlannerExportResult{}
	}
	if result.Succeeded {
		return telemetry.PlannerExportResult{Attempted: true, Succeeded: true}
	}
	switch result.ErrorCode {
	case "delivery_failed", "flush_timeout", "queue_overflow", "request_failed":
		return result
	default:
		return telemetry.PlannerExportResult{Attempted: true, ErrorCode: "request_failed"}
	}
}

func (s *Scheduler) useRuntimeCredential(
	ctx context.Context,
	credentialID string,
	allowed []contracts.RuntimeCredentialKind,
	consumer func(contracts.RuntimeCredentialKind, []byte) error,
) error {
	if s.options.RuntimeCredentials == nil {
		return errors.New("Runtime credential service is unavailable")
	}
	return s.options.RuntimeCredentials.UsePlaintext(ctx, credentialID, allowed, consumer)
}

func decodeRuntimeCredential(data []byte, target any) error {
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(target); err != nil {
		return err
	}
	var trailing any
	if err := decoder.Decode(&trailing); !errors.Is(err, io.EOF) {
		return errors.New("Runtime credential has trailing data")
	}
	return nil
}

func clearWorkerExecutionSettings(settings map[string]contracts.WorkerExecutionSettingsV2) {
	for name, value := range settings {
		value.RuntimeSettings.LLMGatewayToken = nil
		if value.RuntimeSettings.Telemetry != nil {
			value.RuntimeSettings.Telemetry.Headers = nil
		}
		if value.RuntimeSettings.HTTPProxy != nil {
			value.RuntimeSettings.HTTPProxy.BasicAuth = nil
			value.RuntimeSettings.HTTPProxy.BearerToken = nil
		}
		if value.RuntimeSettings.Caido != nil {
			value.RuntimeSettings.Caido.BearerToken = nil
		}
		settings[name] = value
	}
}

func (s *Scheduler) plannerModelAccess(
	ctx context.Context,
	stage workflowconfig.ResolvedStage,
) (*planner.ModelAccess, error) {
	if stage.ExecutionConfig.Planner == nil {
		return nil, nil
	}
	selection := *stage.ExecutionConfig.Planner
	if selection.LLMGateway == nil {
		return nil, fmt.Errorf("Planner has no complete LLM Gateway route")
	}
	token, err := s.resolveCredential(ctx, selection)
	if err != nil {
		return nil, err
	}
	result := &planner.ModelAccess{
		ModelPolicy: cloneModelPolicy(selection.ModelPolicy),
		LLMGateway:  *selection.LLMGateway,
		Token:       token,
	}
	if selection.Credential != nil {
		credential := *selection.Credential
		result.Credential = &credential
	}
	return result, nil
}

func (s *Scheduler) resolveCredential(
	ctx context.Context,
	selection workflowconfig.ResolvedConsumerExecutionConfig,
) (contracts.SecretString, error) {
	if selection.Credential == nil {
		return contracts.NewSecretString(""), nil
	}
	if selection.LLMGateway == nil {
		return contracts.SecretString{}, fmt.Errorf("selected LLM credential has no Gateway")
	}
	if s.options.Credentials == nil {
		return contracts.SecretString{}, fmt.Errorf("selected LLM credential is unavailable")
	}
	token, err := s.options.Credentials.ResolveLLMCredential(
		ctx, *selection.Credential, selection.LLMGateway.Ref,
	)
	if err != nil || token.Reveal() == "" {
		return contracts.SecretString{}, fmt.Errorf("selected LLM credential is unavailable")
	}
	return token, nil
}

var (
	errControlPlaneStateLost      = errors.New("Control Plane state for durable allocations is unavailable")
	errControlPlaneAllocationLost = errors.New("Runtime Agent allocation is irreversibly lost")
)

func (s *Scheduler) liveOrNewReservations(
	ctx context.Context,
	run runstore.WorkflowRun,
	workflow executableWorkflow,
	execution runstore.StageExecution,
) ([]controlplane.Reservation, bool, error) {
	requirements, err := bindingRequirements(workflow.stage, run.SkillSnapshot, execution.StageContext)
	if err != nil {
		return nil, false, err
	}
	recorded, err := s.store.ListStageAllocations(ctx, execution.StageExecutionID)
	if err != nil {
		return nil, false, err
	}
	if execution.State == runstore.StageRunning && len(recorded) == 0 {
		return nil, false, errControlPlaneStateLost
	}
	if len(recorded) > 0 {
		lost := false
		for _, allocation := range recorded {
			grant, grantErr := s.allocator.GetGrant(allocation.AllocationID)
			if grantErr != nil || grant.StageExecutionID != execution.StageExecutionID {
				return nil, false, errControlPlaneStateLost
			}
			lost = lost || grant.Lost
		}
		if lost {
			reservations, reserveErr := s.reserveAll(ctx, controlplane.ReservationRequest{
				RunID: run.RunID, StageExecutionID: execution.StageExecutionID,
				Bindings: requirements, RuntimeConfig: &run.RuntimeConfig,
			})
			if reserveErr != nil {
				return nil, false, errControlPlaneAllocationLost
			}
			return reservations, false, errControlPlaneAllocationLost
		}
	}
	reservations, err := s.reserveAll(ctx, controlplane.ReservationRequest{
		RunID: run.RunID, StageExecutionID: execution.StageExecutionID,
		Bindings: requirements, RuntimeConfig: &run.RuntimeConfig,
	})
	if err != nil {
		return nil, false, err
	}
	if err := verifyReservations(run, workflow, execution, recorded, reservations); err != nil {
		return nil, false, err
	}
	for _, reservation := range reservations {
		if reservation.Grant.Lost {
			return reservations, false, errControlPlaneAllocationLost
		}
	}
	return reservations, len(recorded) == 0, nil
}

func (s *Scheduler) recordReservations(
	ctx context.Context,
	stageExecutionID string,
	reservations []controlplane.Reservation,
) error {
	for _, reservation := range reservations {
		allocation := runstore.StageAllocation{
			AllocationID: reservation.Grant.AllocationID, StageExecutionID: stageExecutionID,
			LogicalAgentName: reservation.Grant.LogicalAgentName, Namespace: reservation.Grant.Namespace,
			AgentTemplateRef:          reservation.AgentTemplate.Ref,
			WorkerRuntimeRef:          reservation.AgentTemplate.Runtime,
			RuntimeAgentID:            reservation.Grant.RuntimeAgentID,
			RuntimeAgentInstanceID:    reservation.Grant.RuntimeInstanceID,
			RuntimeAgentLabelRevision: reservation.RuntimeAgentLabelRevision,
		}
		if reservation.ResolvedRuntimeConfig != nil {
			resolved := reservation.ResolvedRuntimeConfig
			allocation.RuntimeConfigurationSchemaVersion = runstore.AllocationRuntimeConfigurationSchemaVersion
			allocation.RuntimeConfiguration = &runstore.AllocationRuntimeConfiguration{
				ModelPolicy: resolved.ModelPolicy.Ref,
				Origins:     resolved.Origins,
				Provenance:  resolved.Provenance,
			}
		}
		operationContext, cancel := context.WithTimeout(ctx, s.options.OperationTimeout)
		err := s.store.RecordStageAllocation(operationContext, allocation)
		cancel()
		if err != nil {
			return err
		}
	}
	return nil
}

func (s *Scheduler) releaseUnprepared(reservations []controlplane.Reservation) {
	for _, reservation := range reservations {
		if err := s.allocator.Release(reservation.Grant.AllocationID); err != nil {
			s.options.Logger.Warn("release unprepared allocation failed", "allocation_id", reservation.Grant.AllocationID)
		}
	}
}

func verifyReservations(
	run runstore.WorkflowRun,
	workflow executableWorkflow,
	execution runstore.StageExecution,
	recorded []runstore.StageAllocation,
	reservations []controlplane.Reservation,
) error {
	expectedWorkspace, err := projectAllocationWorkspace(workflow.stage.Context.Workspace, execution.StageContext)
	if err != nil {
		return err
	}
	if len(reservations) != len(workflow.stage.Agents) {
		return fmt.Errorf("Control Plane returned an incomplete allocation set")
	}
	recordedByName := make(map[string]runstore.StageAllocation, len(recorded))
	for _, allocation := range recorded {
		recordedByName[allocation.LogicalAgentName] = allocation
	}
	seen := make(map[string]struct{}, len(reservations))
	for _, reservation := range reservations {
		grant := reservation.Grant
		binding, ok := workflow.stage.Agents[grant.LogicalAgentName]
		if !ok || grant.RunID != run.RunID || grant.StageExecutionID != execution.StageExecutionID ||
			grant.Namespace != binding.Namespace || reservation.AgentTemplate.Ref != binding.Template.Ref ||
			reservation.AgentTemplate.Runtime != binding.Template.Runtime || reservation.LeaseExpiresAt.IsZero() ||
			!reflect.DeepEqual(reservation.Workspace, expectedWorkspace) {
			return fmt.Errorf("Control Plane returned an allocation for different resolved inputs")
		}
		if reservation.ResolvedRuntimeConfig == nil {
			if !sameAllocationExecutionConfig(
				reservation.ExecutionConfig,
				allocationExecutionConfig(workflow.stage, grant.LogicalAgentName),
			) {
				return fmt.Errorf("Control Plane returned an allocation for different execution config")
			}
		} else {
			resolved := reservation.ResolvedRuntimeConfig
			selection := workflow.stage.ExecutionConfig.Agents[grant.LogicalAgentName]
			if resolved.Validate() != nil || resolved.ModelPolicy.Ref != selection.ModelPolicy.Ref ||
				reservation.RuntimeAgentLabelRevision == 0 || grant.RuntimeAgentID == "" ||
				!sameAllocationExecutionConfig(reservation.ExecutionConfig, controlplane.AllocationExecutionConfig{
					ModelPolicy: resolved.ModelPolicy.Ref, LLMGateway: resolved.LLMGateway.Ref,
					Credential: resolved.LLMCredential,
				}) {
				return fmt.Errorf("Control Plane returned invalid candidate Runtime provenance")
			}
		}
		if _, duplicate := seen[grant.LogicalAgentName]; duplicate {
			return fmt.Errorf("Control Plane returned duplicate logical Agent allocations")
		}
		seen[grant.LogicalAgentName] = struct{}{}
		if persisted, exists := recordedByName[grant.LogicalAgentName]; exists &&
			(persisted.AllocationID != grant.AllocationID ||
				persisted.RuntimeAgentInstanceID != grant.RuntimeInstanceID ||
				persisted.Namespace != grant.Namespace || persisted.AgentTemplateRef != binding.Template.Ref ||
				persisted.WorkerRuntimeRef != binding.Template.Runtime ||
				reservation.ResolvedRuntimeConfig != nil &&
					(persisted.RuntimeAgentID != grant.RuntimeAgentID ||
						persisted.RuntimeAgentLabelRevision != reservation.RuntimeAgentLabelRevision ||
						persisted.RuntimeConfigurationSchemaVersion != runstore.AllocationRuntimeConfigurationSchemaVersion ||
						!samePersistedRuntimeConfiguration(persisted.RuntimeConfiguration, reservation.ResolvedRuntimeConfig))) {
			return fmt.Errorf("live Control Plane allocation differs from durable provenance")
		}
	}
	if len(recorded) > 0 && len(recordedByName) != len(seen) {
		return fmt.Errorf("durable allocation set is incomplete")
	}
	return nil
}

func samePersistedRuntimeConfiguration(
	persisted *runstore.AllocationRuntimeConfiguration,
	resolved *runtimeconfig.ResolvedRuntimeConfig,
) bool {
	if persisted == nil || resolved == nil {
		return persisted == nil && resolved == nil
	}
	want := runstore.AllocationRuntimeConfiguration{
		ModelPolicy: resolved.ModelPolicy.Ref, Origins: resolved.Origins, Provenance: resolved.Provenance,
	}
	left, leftErr := json.Marshal(persisted)
	right, rightErr := json.Marshal(want)
	return leftErr == nil && rightErr == nil && string(left) == string(right)
}

func plannerContext(snapshot runstore.StageContextSnapshot) planner.StageContext {
	result := planner.StageContext{
		Parameters: cloneParameters(snapshot.Parameters),
		Artifacts:  make(map[string]*contracts.ArtifactRef, len(snapshot.Artifacts)),
	}
	for name, pinned := range snapshot.Artifacts {
		if pinned.Artifact != nil {
			ref := cloneArtifactRef(*pinned.Artifact)
			result.Artifacts[name] = &ref
		} else {
			result.Artifacts[name] = nil
		}
	}
	return result
}

func (s *Scheduler) validateCandidate(
	ctx context.Context,
	runID string,
	stage workflowconfig.ResolvedStage,
	result contracts.StageContentResult,
) error {
	if err := result.Validate(); err != nil {
		return err
	}
	encoded, err := json.Marshal(result)
	if err != nil || len(encoded) > maxCandidateBytes || len(result.Summary) > maxCandidateSummaryBytes ||
		len(result.Artifacts) > maxCandidateArtifacts {
		return fmt.Errorf("StageResult exceeds its bounded contract")
	}
	for name := range result.Artifacts {
		if _, declared := stage.Result.Artifacts[name]; !declared {
			return fmt.Errorf("undeclared Stage result artifact %q", name)
		}
	}
	if result.Outcome == contracts.StageSucceeded {
		for name, slot := range stage.Result.Artifacts {
			if _, present := result.Artifacts[name]; slot.Required && !present {
				return fmt.Errorf("required Stage result artifact %q is missing", name)
			}
		}
	}
	names := sortedArtifactNames(result.Artifacts)
	for _, name := range names {
		ref := result.Artifacts[name]
		if artifactpolicy.IsReservedMemoryBinding(ref.Namespace, ref.Name) {
			return fmt.Errorf("Stage result artifact %q identifies a reserved Memory binding", name)
		}
		resolved, err := s.artifacts.Resolve(ctx, runID, ref)
		if err != nil {
			return fmt.Errorf("verify Stage result artifact %q: %w", name, err)
		}
		if !sameExactRef(ref, resolved.Ref) ||
			!acceptsMediaType(stage.Result.Artifacts[name].MediaTypes, resolved.MediaType) {
			return fmt.Errorf("Stage result artifact %q differs from its declared exact version or media type", name)
		}
	}
	return nil
}

func (s *Scheduler) beginAbort(
	ctx context.Context,
	run runstore.WorkflowRun,
	workflow executableWorkflow,
	execution runstore.StageExecution,
	reservations []controlplane.Reservation,
	failure planner.Failure,
) error {
	return s.beginTermination(ctx, run, workflow, execution, reservations, runstore.StageTermination{
		Outcome: runstore.TerminationInterrupted,
		Code:    failure.Code, Message: failure.Message, Retryable: failure.Retryable,
	})
}

func (s *Scheduler) beginCancellationAbort(
	ctx context.Context,
	run runstore.WorkflowRun,
	workflow executableWorkflow,
	execution runstore.StageExecution,
	reservations []controlplane.Reservation,
) error {
	message := "WorkflowRun cancellation was requested"
	if run.Cancellation != nil && run.Cancellation.Reason != nil {
		message = *run.Cancellation.Reason
	}
	return s.beginTermination(ctx, run, workflow, execution, reservations, runstore.StageTermination{
		Outcome: runstore.TerminationCancelled,
		Code:    runstore.CancellationUserRequested, Message: message, Retryable: false,
	})
}

func (s *Scheduler) beginTermination(
	ctx context.Context,
	run runstore.WorkflowRun,
	workflow executableWorkflow,
	execution runstore.StageExecution,
	reservations []controlplane.Reservation,
	termination runstore.StageTermination,
) error {
	if execution.State != runstore.StagePreparing && execution.State != runstore.StageRunning {
		return fmt.Errorf("cannot abort StageExecution from %s", execution.State)
	}
	if err := s.fenceRecordedAllocations(ctx, execution.StageExecutionID, reservations); err != nil {
		return fmt.Errorf("write-fence Stage allocations: %w", err)
	}
	abortID, err := s.options.NewID("abort_")
	if err != nil {
		return err
	}
	deadline := s.options.Clock.Now().Add(s.options.AbortTimeout)
	phase := runstore.TerminationPreparing
	if execution.State == runstore.StageRunning {
		phase = runstore.TerminationRunning
	}
	termination.Phase = phase
	termination.OccurredAt = s.options.Clock.Now()
	transitionContext, cancelTransition := context.WithTimeout(ctx, s.options.OperationTimeout)
	err = s.persistence.EnterAbortingWithTermination(
		transitionContext,
		run.RunID,
		runstore.EnterAbortingParams{
			StageExecutionID:         execution.StageExecutionID,
			ExpectedState:            execution.State,
			TerminationSchemaVersion: contracts.APIVersion,
			Termination:              termination,
			AbortID:                  abortID,
			Deadline:                 deadline,
			Reason:                   runstore.Reason{Code: "stage_" + string(termination.Outcome)},
		},
	)
	cancelTransition()
	if err != nil {
		return err
	}
	execution.State = runstore.StageAborting
	execution.TerminationSchemaVersion = stringPointer(contracts.APIVersion)
	execution.Termination = &termination
	execution.AbortID = &abortID
	execution.AbortDeadline = &deadline
	return s.resumeAborting(ctx, run, workflow, execution, reservations)
}

func (s *Scheduler) fenceRecordedAllocations(
	ctx context.Context,
	stageExecutionID string,
	reservations []controlplane.Reservation,
) error {
	seen := make(map[string]struct{}, len(reservations))
	var failures []error
	for _, reservation := range reservations {
		allocationID := reservation.Grant.AllocationID
		seen[allocationID] = struct{}{}
		if err := s.allocator.SetWriteFence(allocationID); err != nil &&
			!errors.Is(err, controlplane.ErrAllocationNotFound) {
			failures = append(failures, err)
		}
	}
	allocations, err := s.store.ListStageAllocations(ctx, stageExecutionID)
	if err != nil {
		failures = append(failures, err)
	} else {
		for _, allocation := range allocations {
			if _, ok := seen[allocation.AllocationID]; ok {
				continue
			}
			if _, err := s.allocator.GetGrant(allocation.AllocationID); errors.Is(err, controlplane.ErrAllocationNotFound) {
				continue
			} else if err != nil {
				failures = append(failures, err)
				continue
			}
			if err := s.allocator.SetWriteFence(allocation.AllocationID); err != nil &&
				!errors.Is(err, controlplane.ErrAllocationNotFound) {
				failures = append(failures, err)
			}
		}
	}
	return errors.Join(failures...)
}

func (s *Scheduler) resumeAborting(
	ctx context.Context,
	run runstore.WorkflowRun,
	workflow executableWorkflow,
	execution runstore.StageExecution,
	reservations []controlplane.Reservation,
) error {
	if execution.Termination == nil || execution.AbortID == nil || execution.AbortDeadline == nil {
		return s.failInvalidRunState(ctx, run.RunID, fmt.Errorf("aborting StageExecution is incomplete"))
	}
	if reservations == nil && len(workflow.stage.Agents) > 0 {
		reservations = s.existingLiveReservations(ctx, run, workflow, execution)
	}
	_ = s.fenceRecordedAllocations(context.WithoutCancel(ctx), execution.StageExecutionID, reservations)
	var reports map[string]contracts.AllocationFinalReport
	if len(reservations) > 0 && execution.AbortDeadline.After(s.options.Clock.Now()) {
		abortContext, cancelAbort := context.WithDeadline(ctx, *execution.AbortDeadline)
		var err error
		reports, err = s.workers.AbortAll(
			abortContext,
			reservations,
			*execution.AbortID,
			contracts.TerminationError{
				Code:      execution.Termination.Code,
				Message:   execution.Termination.Message,
				Retryable: execution.Termination.Retryable,
			},
			*execution.AbortDeadline,
		)
		cancelAbort()
		if err != nil {
			s.options.Logger.Warn("bounded allocation abort was incomplete", "stage_execution_id", execution.StageExecutionID)
		}
	}
	s.persistReports(execution, reservations, reports)
	commitContext, cancelCommit := context.WithTimeout(context.WithoutCancel(ctx), s.options.OperationTimeout)
	current, err := s.store.GetRun(commitContext, run.RunID)
	if err != nil {
		cancelCommit()
		return err
	}
	if current.State == runstore.RunCancelling {
		err = s.persistence.CommitTerminationAndFinishRun(
			commitContext,
			run.RunID,
			execution.StageExecutionID,
			runstore.RunCancelling,
			runstore.RunCancelled,
			runstore.Reason{Code: runstore.CancellationUserRequested},
		)
		cancelCommit()
		if err != nil {
			return err
		}
		_ = s.releaseTerminal(execution.StageExecutionID, reservations)
		return nil
	}
	progression, err := s.buildProgression(
		commitContext,
		run,
		workflow,
		execution,
		workflow.stage.On.Interrupted,
		execution.Termination.Retryable,
		execution.Termination.Code,
		runstore.StageExecutionConfigInterruptedEscalation,
	)
	if err == nil {
		err = s.persistence.CommitTerminationProgression(commitContext, TerminationProgression{
			RunID: run.RunID, StageExecutionID: execution.StageExecutionID, Progression: progression,
		})
	}
	cancelCommit()
	if errors.Is(err, runstore.ErrConflict) {
		stateContext, cancelState := context.WithTimeout(context.WithoutCancel(ctx), s.options.OperationTimeout)
		latest, loadErr := s.store.GetRun(stateContext, run.RunID)
		cancelState()
		if loadErr == nil && latest.State == runstore.RunCancelling {
			return s.resumeAborting(ctx, latest, workflow, execution, reservations)
		}
	}
	if err != nil {
		return err
	}
	_ = s.releaseTerminal(execution.StageExecutionID, reservations)
	return nil
}

func (s *Scheduler) resumeFinalizing(
	ctx context.Context,
	run runstore.WorkflowRun,
	workflow executableWorkflow,
	execution runstore.StageExecution,
	reservations []controlplane.Reservation,
) error {
	if execution.CandidateResult == nil || execution.FinalizationID == nil ||
		execution.FinalizationDeadline == nil {
		return s.failInvalidRunState(ctx, run.RunID, fmt.Errorf("finalizing StageExecution is incomplete"))
	}
	if reservations == nil && len(workflow.stage.Agents) > 0 {
		reservations = s.existingLiveReservations(ctx, run, workflow, execution)
	}
	var reports map[string]contracts.AllocationFinalReport
	if len(reservations) > 0 && execution.FinalizationDeadline.After(s.options.Clock.Now()) {
		finalizeContext, cancelFinalize := context.WithDeadline(ctx, *execution.FinalizationDeadline)
		var err error
		reports, err = s.workers.FinalizeAll(
			finalizeContext,
			reservations,
			*execution.FinalizationID,
			*execution.FinalizationDeadline,
		)
		cancelFinalize()
		if err != nil {
			s.options.Logger.Warn("bounded allocation finalization was incomplete", "stage_execution_id", execution.StageExecutionID)
		}
	}
	s.persistReports(execution, reservations, reports)

	stateContext, cancelState := context.WithTimeout(context.WithoutCancel(ctx), s.options.OperationTimeout)
	current, err := s.store.GetRun(stateContext, run.RunID)
	cancelState()
	if err != nil {
		return err
	}
	if current.State == runstore.RunCancelling {
		return s.acceptFinalizingDuringCancellation(ctx, current, execution, reservations)
	}
	action := workflow.stage.On.Succeeded
	retryable := false
	reasonCode := "workflow_succeeded"
	escalationVariant := runstore.StageExecutionConfigBase
	if execution.CandidateResult.Outcome == contracts.StageFailed {
		action = workflow.stage.On.Failed
		retryable = execution.CandidateResult.Error != nil && execution.CandidateResult.Error.Retryable
		escalationVariant = runstore.StageExecutionConfigFailedEscalation
		reasonCode = "stage_failed"
		if execution.CandidateResult.Error != nil {
			reasonCode = execution.CandidateResult.Error.Code
		}
	}
	commitContext, cancelCommit := context.WithTimeout(context.WithoutCancel(ctx), s.options.OperationTimeout)
	progression, err := s.buildProgression(
		commitContext, run, workflow, execution, action, retryable, reasonCode, escalationVariant,
	)
	if err == nil {
		err = s.persistence.CommitResultProgression(commitContext, ResultProgression{
			RunID: run.RunID, StageExecutionID: execution.StageExecutionID,
			Result:          cloneStageResult(*execution.CandidateResult),
			WorkflowOutputs: cloneStringMap(workflow.stage.WorkflowOutputs),
			OutputContracts: cloneArtifactSlots(workflow.workflow.Outputs),
			Progression:     progression,
		})
	}
	cancelCommit()
	if errors.Is(err, runstore.ErrConflict) {
		stateContext, cancelState := context.WithTimeout(context.WithoutCancel(ctx), s.options.OperationTimeout)
		current, loadErr := s.store.GetRun(stateContext, run.RunID)
		cancelState()
		if loadErr == nil && current.State == runstore.RunCancelling {
			return s.acceptFinalizingDuringCancellation(ctx, current, execution, reservations)
		}
	}
	if err != nil {
		return err
	}
	_ = s.releaseTerminal(execution.StageExecutionID, reservations)
	return nil
}

func (s *Scheduler) buildProgression(
	ctx context.Context,
	run runstore.WorkflowRun,
	workflow executableWorkflow,
	execution runstore.StageExecution,
	action workflowconfig.TransitionAction,
	retryable bool,
	reasonCode string,
	escalationVariant runstore.StageExecutionConfigVariant,
) (StageProgression, error) {
	selected := action
	var nextEscalationOrdinal *int
	var exhaustedEscalationOrdinal *int
	if action.Kind == workflowconfig.TransitionRetry {
		if action.Retry == nil {
			return StageProgression{}, fmt.Errorf("retry Transition has no bounded policy")
		}
		if !retryable || execution.Attempt >= action.Retry.MaxAttempts {
			selected = action.Retry.Then
		}
	}
	if action.Kind == workflowconfig.TransitionEscalate {
		if action.Escalate == nil ||
			(escalationVariant != runstore.StageExecutionConfigFailedEscalation &&
				escalationVariant != runstore.StageExecutionConfigInterruptedEscalation) {
			return StageProgression{}, fmt.Errorf("escalate Transition has no bounded policy or outcome identity")
		}
		used, err := s.escalationAttempts(ctx, run.RunID, execution.StageName, escalationVariant)
		if err != nil {
			return StageProgression{}, err
		}
		if used >= action.Escalate.MaxAttempts {
			selected = action.Escalate.Then
			exhaustedEscalationOrdinal = intPointer(used)
		} else {
			nextEscalationOrdinal = intPointer(used + 1)
		}
	}
	decision := runstore.RecordStageTransitionDecisionParams{
		SourceExecutionID: execution.StageExecutionID,
		RunID:             run.RunID,
	}
	if exhaustedEscalationOrdinal != nil {
		decision.EscalationOrdinal = exhaustedEscalationOrdinal
		decision.EscalationExhausted = true
	}
	switch selected.Kind {
	case workflowconfig.TransitionRetry:
		targetWorkflow, err := workflow.selectStage(execution.StageName)
		if err != nil {
			return StageProgression{}, err
		}
		previous := execution.StageExecutionID
		creation, err := s.buildStageCreation(
			ctx, run, targetWorkflow, execution.Attempt+1, &previous,
			stageExecutionConfiguration{variant: runstore.StageExecutionConfigBase},
		)
		if err != nil {
			return StageProgression{}, err
		}
		targetStage, targetExecution := creation.Params.StageName, creation.Params.StageExecutionID
		decision.Action = runstore.StageTransitionRetry
		decision.TargetStageName = &targetStage
		decision.TargetExecutionID = &targetExecution
		return StageProgression{Decision: decision, NextStage: &creation}, nil
	case workflowconfig.TransitionEscalate:
		if action.Escalate == nil || nextEscalationOrdinal == nil {
			return StageProgression{}, fmt.Errorf("escalate Transition selection is incomplete")
		}
		targetWorkflow, err := workflow.selectStage(execution.StageName)
		if err != nil {
			return StageProgression{}, err
		}
		previous := execution.StageExecutionID
		effective := action.Escalate.ExecutionConfig.Effective
		creation, err := s.buildStageCreation(
			ctx, run, targetWorkflow, execution.Attempt+1, &previous,
			stageExecutionConfiguration{
				variant: escalationVariant, ordinal: nextEscalationOrdinal, effective: &effective,
			},
		)
		if err != nil {
			return StageProgression{}, err
		}
		targetStage, targetExecution := creation.Params.StageName, creation.Params.StageExecutionID
		decision.Action = runstore.StageTransitionEscalate
		decision.TargetStageName = &targetStage
		decision.TargetExecutionID = &targetExecution
		decision.EscalationOrdinal = nextEscalationOrdinal
		return StageProgression{Decision: decision, NextStage: &creation}, nil
	case workflowconfig.TransitionNext:
		targetWorkflow, err := workflow.selectStage(selected.NextStage)
		if err != nil {
			return StageProgression{}, err
		}
		creation, err := s.buildStageCreation(
			ctx, run, targetWorkflow, 1, nil,
			stageExecutionConfiguration{variant: runstore.StageExecutionConfigBase},
		)
		if err != nil {
			return StageProgression{}, err
		}
		targetStage, targetExecution := creation.Params.StageName, creation.Params.StageExecutionID
		decision.Action = runstore.StageTransitionNext
		decision.TargetStageName = &targetStage
		decision.TargetExecutionID = &targetExecution
		return StageProgression{Decision: decision, NextStage: &creation}, nil
	case workflowconfig.TransitionSucceed:
		decision.Action = runstore.StageTransitionSucceed
		return StageProgression{
			Decision: decision, TerminalRunState: runstore.RunSucceeded,
			RunReason: runstore.Reason{Code: "workflow_succeeded"},
		}, nil
	case workflowconfig.TransitionFail:
		if strings.TrimSpace(reasonCode) == "" {
			reasonCode = "stage_failed"
		}
		decision.Action = runstore.StageTransitionFail
		return StageProgression{
			Decision: decision, TerminalRunState: runstore.RunFailed,
			RunReason: runstore.Reason{Code: reasonCode},
		}, nil
	default:
		return StageProgression{}, fmt.Errorf("unknown Workflow Transition action %q", selected.Kind)
	}
}

func (s *Scheduler) escalationAttempts(
	ctx context.Context,
	runID string,
	stageName string,
	variant runstore.StageExecutionConfigVariant,
) (int, error) {
	executions, err := s.store.ListStageExecutions(ctx, runID)
	if err != nil {
		return 0, err
	}
	maxOrdinal := 0
	seen := make(map[int]struct{})
	for _, current := range executions {
		if current.StageName != stageName || current.ExecutionConfigVariant != variant {
			continue
		}
		if current.EscalationOrdinal == nil || *current.EscalationOrdinal <= 0 {
			return 0, fmt.Errorf("persisted escalation attempt has an invalid ordinal")
		}
		ordinal := *current.EscalationOrdinal
		if _, duplicate := seen[ordinal]; duplicate {
			return 0, fmt.Errorf("persisted escalation attempts have duplicate ordinal %d", ordinal)
		}
		seen[ordinal] = struct{}{}
		if ordinal > maxOrdinal {
			maxOrdinal = ordinal
		}
	}
	for ordinal := 1; ordinal <= maxOrdinal; ordinal++ {
		if _, present := seen[ordinal]; !present {
			return 0, fmt.Errorf("persisted escalation attempts have a non-contiguous ordinal")
		}
	}
	return maxOrdinal, nil
}

func (s *Scheduler) acceptFinalizingDuringCancellation(
	ctx context.Context,
	run runstore.WorkflowRun,
	execution runstore.StageExecution,
	reservations []controlplane.Reservation,
) error {
	commitContext, cancelCommit := context.WithTimeout(context.WithoutCancel(ctx), s.options.OperationTimeout)
	err := s.persistence.AcceptResultDuringCancellation(
		commitContext,
		run.RunID,
		execution.StageExecutionID,
		cloneStageResult(*execution.CandidateResult),
	)
	cancelCommit()
	if err != nil {
		return err
	}
	_ = s.releaseTerminal(execution.StageExecutionID, reservations)
	return nil
}

func (s *Scheduler) persistReports(
	execution runstore.StageExecution,
	reservations []controlplane.Reservation,
	reports map[string]contracts.AllocationFinalReport,
) {
	// A crash can resume directly in finalizing/aborting without recreating the
	// Planner instance. Ensure its durable session still has a report record;
	// an already persisted complete report wins over this placeholder.
	s.persistPlannerReport(execution, nil, nil)
	listContext, cancelList := context.WithTimeout(context.Background(), s.options.OperationTimeout)
	allocations, err := s.store.ListStageAllocations(listContext, execution.StageExecutionID)
	cancelList()
	if err != nil {
		s.options.Logger.Warn("list allocations for reports failed", "stage_execution_id", execution.StageExecutionID)
		return
	}
	if len(allocations) == 0 {
		return
	}
	byName := make(map[string]controlplane.Reservation, len(reservations))
	for _, reservation := range reservations {
		byName[reservation.Grant.LogicalAgentName] = reservation
	}
	finishedAt := s.options.Clock.Now().UTC().Round(0)
	startedAt := execution.CreatedAt.UTC().Round(0)
	if execution.PlannerStartedAt != nil {
		startedAt = execution.PlannerStartedAt.UTC().Round(0)
	}
	if startedAt.IsZero() || startedAt.After(finishedAt) {
		startedAt = finishedAt
	}
	for _, allocation := range allocations {
		report, ok := reports[allocation.LogicalAgentName]
		reservation, live := byName[allocation.LogicalAgentName]
		if !ok || report.AllocationID != allocation.AllocationID ||
			(live && reservation.Grant.AllocationID != allocation.AllocationID) {
			retryable := true
			report = contracts.AllocationFinalReport{
				ReportID:     "allocation-final-missing-" + allocation.AllocationID,
				AllocationID: allocation.AllocationID,
				StartedAt:    startedAt,
				FinishedAt:   finishedAt,
				Worker: contracts.ExecutionReport{
					ReportID:  "worker-missing-" + allocation.AllocationID,
					Complete:  false,
					Metrics:   contracts.ExecutionMetrics{Tools: map[string]contracts.ToolMetrics{}},
					ToolCalls: []contracts.ToolCallRecord{},
					Errors: []contracts.ExecutionError{{
						Code:      "allocation_report_unavailable",
						Message:   "Runtime Agent did not return an execution report before lifecycle completion",
						Retryable: &retryable,
					}},
				},
				Runtime: contracts.RuntimeReport{Complete: false},
			}
		}
		ctx, cancel := context.WithTimeout(context.Background(), s.options.OperationTimeout)
		err := s.store.RecordStageExecutionReport(ctx, runstore.RecordStageExecutionReportParams{
			StageExecutionID: execution.StageExecutionID, AllocationID: allocation.AllocationID,
			LogicalAgentName: allocation.LogicalAgentName, ReportSchemaVersion: contracts.APIVersion,
			Report: report, Secrets: s.telemetrySecrets(),
		})
		cancel()
		if err != nil {
			s.options.Logger.Warn(
				"execution report persistence failed",
				"stage_execution_id", execution.StageExecutionID,
				"logical_agent_name", allocation.LogicalAgentName,
			)
		}
	}
	s.rebuildStageMetrics(execution.StageExecutionID)
}

func (s *Scheduler) persistPlannerReport(
	execution runstore.StageExecution,
	instance planner.Planner,
	exportResult *telemetry.PlannerExportResult,
) {
	if execution.PlannerSessionID == nil || execution.PlannerInvocationID == nil {
		return
	}
	report := contracts.ExecutionReport{
		ReportID:  "planner-missing-" + *execution.PlannerSessionID,
		Complete:  false,
		Metrics:   contracts.ExecutionMetrics{Tools: map[string]contracts.ToolMetrics{}},
		ToolCalls: []contracts.ToolCallRecord{},
		Errors:    []contracts.ExecutionError{},
	}
	if provider, ok := instance.(planner.ReportProvider); ok {
		if provided, available := provider.ExecutionReport(); available {
			report = provided
		}
	}
	applyPlannerTelemetryResult(&report, exportResult)
	ctx, cancel := context.WithTimeout(context.Background(), s.options.OperationTimeout)
	startedAt := execution.CreatedAt.UTC().Round(0)
	if execution.PlannerStartedAt != nil {
		startedAt = execution.PlannerStartedAt.UTC().Round(0)
	}
	if startedAt.IsZero() {
		startedAt = s.options.Clock.Now().UTC().Round(0)
	}
	finishedAt := startedAt
	if report.Metrics.DurationMS != nil {
		finishedAt = startedAt.Add(time.Duration(*report.Metrics.DurationMS) * time.Millisecond)
	}
	err := s.store.RecordPlannerExecutionReport(ctx, runstore.RecordPlannerExecutionReportParams{
		StageExecutionID: execution.StageExecutionID,
		SessionID:        *execution.PlannerSessionID, InvocationID: *execution.PlannerInvocationID,
		StartedAt: startedAt, FinishedAt: finishedAt,
		ReportSchemaVersion: contracts.APIVersion, Report: report,
		Secrets: s.telemetrySecrets(),
	})
	cancel()
	if err != nil {
		s.options.Logger.Warn(
			"Planner report persistence failed",
			"stage_execution_id", execution.StageExecutionID,
		)
		return
	}
	s.rebuildStageMetrics(execution.StageExecutionID)
}

func applyPlannerTelemetryResult(
	report *contracts.ExecutionReport,
	result *telemetry.PlannerExportResult,
) {
	if report == nil || result == nil || !result.Attempted {
		return
	}
	if report.Metrics.Tools == nil {
		report.Metrics.Tools = make(map[string]contracts.ToolMetrics)
	}
	calls, succeeded, failed := int64(1), int64(0), int64(1)
	outcome := contracts.ToolCallFailed
	var executionError *contracts.ExecutionError
	if result.Succeeded {
		succeeded, failed = 1, 0
		outcome = contracts.ToolCallSucceeded
	} else {
		retryable := false
		executionError = &contracts.ExecutionError{
			Code: result.ErrorCode, Message: "Planner telemetry export failed", Retryable: &retryable,
		}
	}
	report.Metrics.Tools["telemetry.export"] = contracts.ToolMetrics{
		Calls: &calls, Succeeded: &succeeded, Failed: &failed,
	}
	if len(report.ToolCalls) >= 1000 {
		report.Truncated = true
		return
	}
	report.ToolCalls = append(report.ToolCalls, contracts.ToolCallRecord{
		CallID: "planner-telemetry-export", Tool: "telemetry.export",
		Arguments: map[string]any{}, Outcome: outcome, Error: executionError,
	})
}

func (s *Scheduler) rebuildStageMetrics(stageExecutionID string) {
	ctx, cancel := context.WithTimeout(context.Background(), s.options.OperationTimeout)
	err := s.store.RebuildStageMetrics(ctx, stageExecutionID, contracts.APIVersion)
	cancel()
	if err != nil {
		s.options.Logger.Warn(
			"StageMetrics persistence failed", "stage_execution_id", stageExecutionID,
		)
	}
}

func (s *Scheduler) telemetrySecrets() []string {
	result := make([]string, 0, len(s.options.TelemetrySecrets)+1)
	result = append(result, s.options.TelemetrySecrets...)
	result = append(result, s.options.RuntimeSettings.LLMGatewayToken.Reveal())
	return result
}

func (s *Scheduler) existingLiveReservations(
	ctx context.Context,
	run runstore.WorkflowRun,
	workflow executableWorkflow,
	execution runstore.StageExecution,
) []controlplane.Reservation {
	requirements, err := bindingRequirements(workflow.stage, run.SkillSnapshot, execution.StageContext)
	if err != nil {
		return nil
	}
	recorded, err := s.store.ListStageAllocations(ctx, execution.StageExecutionID)
	if err != nil || len(recorded) == 0 {
		return nil
	}
	for _, allocation := range recorded {
		if _, err := s.allocator.GetGrant(allocation.AllocationID); err != nil {
			return nil
		}
	}
	reservations, err := s.reserveAll(ctx, controlplane.ReservationRequest{
		RunID: run.RunID, StageExecutionID: execution.StageExecutionID,
		Bindings: requirements, RuntimeConfig: &run.RuntimeConfig,
	})
	if err != nil || verifyReservations(run, workflow, execution, recorded, reservations) != nil {
		return nil
	}
	return reservations
}

type contextAllocator interface {
	ReserveAllContext(context.Context, controlplane.ReservationRequest) ([]controlplane.Reservation, error)
}

func (s *Scheduler) reserveAll(
	ctx context.Context,
	request controlplane.ReservationRequest,
) ([]controlplane.Reservation, error) {
	if allocator, ok := s.allocator.(contextAllocator); ok {
		return allocator.ReserveAllContext(ctx, request)
	}
	return s.allocator.ReserveAll(request)
}

func (s *Scheduler) releaseTerminal(
	stageExecutionID string,
	reservations []controlplane.Reservation,
) error {
	if len(reservations) == 0 {
		return nil
	}
	var failures []error
	for _, reservation := range reservations {
		markContext, cancelMark := context.WithTimeout(context.Background(), s.options.OperationTimeout)
		err := s.store.MarkStageAllocationReleaseAttempt(
			markContext, reservation.Grant.AllocationID,
		)
		cancelMark()
		if err != nil {
			failures = append(failures, err)
		}
	}
	releaseContext, cancelRelease := context.WithTimeout(context.Background(), s.options.OperationTimeout)
	releaseErr := s.workers.ReleaseAll(releaseContext, reservations)
	cancelRelease()
	if releaseErr != nil {
		s.options.Logger.Warn(
			"terminal allocation release was incomplete",
			"stage_execution_id", stageExecutionID,
			"error", releaseErr,
		)
		failures = append(failures, releaseErr)
	}
	for _, reservation := range reservations {
		allocationID := reservation.Grant.AllocationID
		_, err := s.allocator.GetGrant(allocationID)
		if err == nil {
			continue
		}
		if !errors.Is(err, controlplane.ErrAllocationNotFound) {
			failures = append(failures, err)
			continue
		}
		markContext, cancelMark := context.WithTimeout(context.Background(), s.options.OperationTimeout)
		err = s.store.MarkStageAllocationReleased(markContext, allocationID)
		cancelMark()
		if err != nil {
			failures = append(failures, err)
		}
	}
	return errors.Join(failures...)
}

func (s *Scheduler) recoverTerminalRelease(ctx context.Context) (bool, error) {
	listContext, cancelList := context.WithTimeout(ctx, s.options.OperationTimeout)
	executions, err := s.store.ListTerminalStageExecutionsWithAllocations(listContext)
	cancelList()
	if err != nil {
		return false, err
	}
	for _, execution := range executions {
		allocationContext, cancelAllocations := context.WithTimeout(ctx, s.options.OperationTimeout)
		allocations, err := s.store.ListStageAllocations(
			allocationContext, execution.StageExecutionID,
		)
		cancelAllocations()
		if err != nil {
			return true, err
		}
		worked := false
		reservations := make([]controlplane.Reservation, 0, len(allocations))
		var failures []error
		for _, allocation := range allocations {
			if allocation.ReleaseCompletedAt != nil {
				continue
			}
			worked = true
			markContext, cancelMark := context.WithTimeout(ctx, s.options.OperationTimeout)
			markErr := s.store.MarkStageAllocationReleaseAttempt(markContext, allocation.AllocationID)
			cancelMark()
			if markErr != nil {
				failures = append(failures, markErr)
			}
			reservation, reservationErr := s.allocator.GetReservation(allocation.AllocationID)
			if errors.Is(reservationErr, controlplane.ErrAllocationNotFound) {
				markContext, cancelMark = context.WithTimeout(ctx, s.options.OperationTimeout)
				markErr = s.store.MarkStageAllocationReleased(markContext, allocation.AllocationID)
				cancelMark()
				if markErr != nil {
					failures = append(failures, markErr)
				}
				continue
			}
			if reservationErr != nil {
				failures = append(failures, reservationErr)
				continue
			}
			if err := verifyTerminalReleaseReservation(execution, allocation, reservation); err != nil {
				failures = append(failures, err)
				continue
			}
			reservations = append(reservations, reservation)
		}
		if len(reservations) != 0 {
			failures = append(failures, s.releaseTerminal(execution.StageExecutionID, reservations))
		}
		if worked {
			return true, errors.Join(failures...)
		}
	}
	return false, nil
}

func verifyTerminalReleaseReservation(
	execution runstore.StageExecution,
	allocation runstore.StageAllocation,
	reservation controlplane.Reservation,
) error {
	grant := reservation.Grant
	if grant.AllocationID != allocation.AllocationID ||
		grant.RunID != execution.RunID || grant.StageExecutionID != execution.StageExecutionID ||
		grant.LogicalAgentName != allocation.LogicalAgentName || grant.Namespace != allocation.Namespace ||
		grant.RuntimeInstanceID != allocation.RuntimeAgentInstanceID ||
		reservation.AgentTemplate.Ref != allocation.AgentTemplateRef ||
		reservation.AgentTemplate.Runtime != allocation.WorkerRuntimeRef {
		return fmt.Errorf("live allocation %q differs from durable terminal provenance", allocation.AllocationID)
	}
	return nil
}

func (s *Scheduler) finishRunFromTerminalTermination(
	ctx context.Context,
	run runstore.WorkflowRun,
	execution runstore.StageExecution,
) error {
	if execution.Termination == nil {
		return s.failInvalidRunState(ctx, run.RunID, fmt.Errorf("terminal interrupted Stage has no termination"))
	}
	operationContext, cancel := context.WithTimeout(ctx, s.options.OperationTimeout)
	defer cancel()
	_, err := s.store.TransitionRun(
		operationContext,
		run.RunID,
		runstore.RunRunning,
		runstore.RunFailed,
		runstore.Reason{Code: execution.Termination.Code},
	)
	return err
}

func infrastructureFailure(code, message string, cause error) planner.Failure {
	retryable := true
	var apiError *controlplane.RuntimeAPIError
	if errors.As(cause, &apiError) {
		retryable = apiError.Retryable
	}
	return planner.Failure{Code: code, Message: message, Retryable: retryable}
}

func cloneParameters(source map[string]string) map[string]string {
	result := make(map[string]string, len(source))
	for name, value := range source {
		result[name] = value
	}
	return result
}

func cloneStringMap(source map[string]string) map[string]string {
	result := make(map[string]string, len(source))
	for name, value := range source {
		result[name] = value
	}
	return result
}

func cloneArtifactSlots(
	source map[string]workflowconfig.ArtifactSlot,
) map[string]workflowconfig.ArtifactSlot {
	result := make(map[string]workflowconfig.ArtifactSlot, len(source))
	for name, slot := range source {
		slot.MediaTypes = append([]string(nil), slot.MediaTypes...)
		result[name] = slot
	}
	return result
}

func cloneArtifactRef(source contracts.ArtifactRef) contracts.ArtifactRef {
	result := source
	if source.Revision != nil {
		revision := *source.Revision
		result.Revision = &revision
	}
	return result
}

func cloneCredentialRef(source *contracts.LLMCredentialRef) *contracts.LLMCredentialRef {
	if source == nil {
		return nil
	}
	result := *source
	return &result
}

func cloneModelPolicy(source contracts.ResolvedModelPolicy) contracts.ResolvedModelPolicy {
	result := source
	if source.Temperature != nil {
		temperature := *source.Temperature
		result.Temperature = &temperature
	}
	return result
}

func cloneStageResult(source contracts.StageContentResult) contracts.StageContentResult {
	result := source
	result.Artifacts = make(map[string]contracts.ArtifactRef, len(source.Artifacts))
	for name, ref := range source.Artifacts {
		result.Artifacts[name] = cloneArtifactRef(ref)
	}
	if source.Error != nil {
		cloned := *source.Error
		result.Error = &cloned
	}
	return result
}

func sameExactRef(left, right contracts.ArtifactRef) bool {
	return left.Namespace == right.Namespace && left.Name == right.Name &&
		left.Revision != nil && right.Revision != nil && *left.Revision == *right.Revision
}

func stringPointer(value string) *string { return &value }
func intPointer(value int) *int          { return &value }

func schedulerID(prefix string) (string, error) {
	buffer := make([]byte, 16)
	if _, err := rand.Read(buffer); err != nil {
		return "", err
	}
	return prefix + hex.EncodeToString(buffer), nil
}
