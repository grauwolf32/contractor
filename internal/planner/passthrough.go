package planner

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/telemetry"
)

const (
	maxStageContentBytes   = 256 * 1024
	maxResultArtifacts     = 128
	maxResultSummaryBytes  = 64 * 1024
	completionWriteTimeout = time.Second
)

type PassthroughFactory struct {
	sessions  SessionService
	invoker   WorkerInvoker
	inspector ArtifactInspector
}

func NewPassthroughFactory(
	sessions SessionService,
	invoker WorkerInvoker,
	inspector ArtifactInspector,
) (*PassthroughFactory, error) {
	if sessions == nil || invoker == nil || inspector == nil {
		return nil, fmt.Errorf("PassthroughPlanner requires session, Worker, and artifact adapters")
	}
	return &PassthroughFactory{sessions: sessions, invoker: invoker, inspector: inspector}, nil
}

func (*PassthroughFactory) Ref() string { return PassthroughRef }

func (f *PassthroughFactory) Create(invocation Invocation) (Planner, error) {
	binding, handle, err := validateInvocation(invocation)
	if err != nil {
		return nil, err
	}
	request, err := stageRequest(invocation)
	if err != nil {
		return nil, err
	}
	return &passthroughPlanner{
		invocation: invocation,
		binding:    binding,
		handle:     handle,
		request:    request,
		sessions:   f.sessions,
		invoker:    f.invoker,
		inspector:  f.inspector,
	}, nil
}

type passthroughPlanner struct {
	invocation Invocation
	binding    string
	handle     contracts.WorkerHandle
	request    contracts.StageContentRequest
	sessions   SessionService
	invoker    WorkerInvoker
	inspector  ArtifactInspector
	reportMu   sync.RWMutex
	report     contracts.ExecutionReport
	hasReport  bool
}

func (p *passthroughPlanner) Run(
	ctx context.Context,
) (candidate contracts.StageContentResult, runErr error) {
	reportStarted := time.Now()
	var identity SessionIdentity
	var invoked bool
	var invokeDurationMS int64
	var invokeFailure *Failure
	defer func() {
		p.finishReport(reportStarted, identity, invoked, invokeDurationMS, invokeFailure, runErr)
	}()

	instrumentation := InvocationInstrumentation(p.invocation)
	sessionSpan := instrumentation.StartSpan(
		telemetry.PlannerSpanSession,
		telemetry.PlannerSpanAttributes{Operation: "session.begin"},
	)
	started, err := p.sessions.Begin(ctx, p.invocation.StageExecutionID)
	if err != nil {
		sessionSpan.End("unavailable", telemetry.PlannerSpanAttributes{ErrorCode: "planner_session_unavailable"})
		return contracts.StageContentResult{}, sessionError("start", err)
	}
	identity = started.Identity
	if started.Completion != nil {
		sessionSpan.End("recovered", telemetry.PlannerSpanAttributes{SessionID: identity.SessionID})
		return p.recoverCompletion(ctx, *started.Completion)
	}
	if !started.Invoke {
		sessionSpan.End("rejected", telemetry.PlannerSpanAttributes{
			SessionID: identity.SessionID, ErrorCode: "planner_session_invalid",
		})
		return contracts.StageContentResult{}, NewError(
			"planner_session_invalid", "Planner session did not grant invocation ownership", false, nil,
		)
	}
	sessionSpan.End("succeeded", telemetry.PlannerSpanAttributes{SessionID: identity.SessionID})

	facts := RequestFactsFor([]string{p.binding}, p.request)
	recordSpan := instrumentation.StartSpan(
		telemetry.PlannerSpanSession,
		telemetry.PlannerSpanAttributes{Operation: "session.record_request", SessionID: identity.SessionID},
	)
	if err := p.sessions.RecordRequest(ctx, started.Identity, facts); err != nil {
		recordSpan.End("unavailable", telemetry.PlannerSpanAttributes{ErrorCode: "planner_session_unavailable"})
		return contracts.StageContentResult{}, sessionError("record request", err)
	}
	recordSpan.End("succeeded", telemetry.PlannerSpanAttributes{})
	deadline := p.invocation.Deadline
	if !deadline.After(time.Now()) {
		return contracts.StageContentResult{}, p.fail(
			ctx,
			started.Identity,
			NewError(
				"worker_deadline_exceeded", "Worker invocation deadline expired", true,
				context.DeadlineExceeded,
			),
		)
	}

	invokeContext, cancel := context.WithDeadline(ctx, deadline)
	defer cancel()
	invoked = true
	invokeStarted := time.Now()
	workerSpan := instrumentation.StartSpan(
		telemetry.PlannerSpanWorker,
		telemetry.PlannerSpanAttributes{Operation: "a2a.invoke", WorkerName: p.binding},
	)
	result, err := p.invoker.Invoke(
		invokeContext, p.binding, cloneWorkerHandle(p.handle), cloneStageRequest(p.request),
	)
	invokeDurationMS = max(0, time.Since(invokeStarted).Milliseconds())
	if err != nil {
		failure := FailureFrom(err)
		invokeFailure = &failure
		workerSpan.End("failed", telemetry.PlannerSpanAttributes{ErrorCode: failure.Code})
		return contracts.StageContentResult{}, p.fail(
			ctx, started.Identity, NewErrorFromFailure(FailureFrom(err), err),
		)
	}
	if err := validateCandidate(
		invokeContext, p.invocation.RunID, p.invocation.Stage.Result.Artifacts, result, p.inspector,
	); err != nil {
		failure := FailureFrom(err)
		workerSpan.End("rejected", telemetry.PlannerSpanAttributes{ErrorCode: failure.Code})
		return contracts.StageContentResult{}, p.fail(ctx, started.Identity, err)
	}
	workerSpan.End("succeeded", telemetry.PlannerSpanAttributes{})
	completion := Completion{Result: pointerToResult(cloneStageResult(result))}
	if err := p.recordCompletion(ctx, started.Identity, completion); err != nil {
		return contracts.StageContentResult{}, sessionError("record completion", err)
	}
	return cloneStageResult(result), nil
}

func (p *passthroughPlanner) ExecutionReport() (contracts.ExecutionReport, bool) {
	p.reportMu.RLock()
	defer p.reportMu.RUnlock()
	return p.report, p.hasReport
}

func (p *passthroughPlanner) finishReport(
	startedAt time.Time,
	identity SessionIdentity,
	invoked bool,
	invokeDurationMS int64,
	invokeFailure *Failure,
	runErr error,
) {
	durationMS := max(0, time.Since(startedAt).Milliseconds())
	reportID := "planner-unavailable-" + p.invocation.StageExecutionID
	if identity.SessionID != "" {
		reportID = "planner-" + identity.SessionID
	}
	report := contracts.ExecutionReport{
		ReportID: reportID,
		Complete: true,
		Metrics: contracts.ExecutionMetrics{
			DurationMS: int64Pointer(durationMS),
			Tools:      map[string]contracts.ToolMetrics{},
		},
		ToolCalls: []contracts.ToolCallRecord{},
		Errors:    []contracts.ExecutionError{},
	}
	if invoked {
		calls, succeeded, failed := int64(1), int64(1), int64(0)
		outcome := contracts.ToolCallSucceeded
		var executionError *contracts.ExecutionError
		if invokeFailure != nil {
			succeeded, failed = 0, 1
			outcome = contracts.ToolCallFailed
			retryable := invokeFailure.Retryable
			executionError = &contracts.ExecutionError{
				Code: invokeFailure.Code, Message: invokeFailure.Message, Retryable: &retryable,
			}
		}
		report.Metrics.Tools["a2a.invoke"] = contracts.ToolMetrics{
			Calls: &calls, Succeeded: &succeeded, Failed: &failed,
		}
		report.ToolCalls = append(report.ToolCalls, contracts.ToolCallRecord{
			CallID: "a2a-" + reportID, Tool: "a2a.invoke",
			Arguments: map[string]any{"binding": p.binding}, Outcome: outcome,
			DurationMS: &invokeDurationMS, Error: executionError,
		})
	}
	if runErr != nil {
		failure := FailureFrom(runErr)
		retryable := failure.Retryable
		report.Errors = append(report.Errors, contracts.ExecutionError{
			Code: failure.Code, Message: failure.Message, Retryable: &retryable,
		})
	}
	p.reportMu.Lock()
	p.report = report
	p.hasReport = true
	p.reportMu.Unlock()
}

func int64Pointer(value int64) *int64 { return &value }

func (p *passthroughPlanner) recoverCompletion(
	ctx context.Context, completion Completion,
) (contracts.StageContentResult, error) {
	if err := validateCompletion(completion); err != nil {
		return contracts.StageContentResult{}, NewError(
			"planner_session_invalid", "Recorded Planner completion is invalid", false, err,
		)
	}
	if completion.Failure != nil {
		return contracts.StageContentResult{}, NewErrorFromFailure(*completion.Failure, nil)
	}
	result := cloneStageResult(*completion.Result)
	if err := validateCandidate(
		ctx, p.invocation.RunID, p.invocation.Stage.Result.Artifacts, result, p.inspector,
	); err != nil {
		return contracts.StageContentResult{}, err
	}
	return result, nil
}

func (p *passthroughPlanner) fail(
	ctx context.Context, identity SessionIdentity, plannerError *Error,
) *Error {
	failure := plannerError.Failure
	if err := p.recordCompletion(ctx, identity, Completion{Failure: &failure}); err != nil {
		return sessionError("record failure", errors.Join(plannerError, err))
	}
	return plannerError
}

func (p *passthroughPlanner) recordCompletion(
	ctx context.Context, identity SessionIdentity, completion Completion,
) error {
	recordContext, cancel := context.WithTimeout(context.WithoutCancel(ctx), completionWriteTimeout)
	defer cancel()
	return p.sessions.Complete(recordContext, identity, completion)
}

func NewErrorFromFailure(failure Failure, cause error) *Error {
	return NewError(failure.Code, failure.Message, failure.Retryable, cause)
}

func sessionError(operation string, cause error) *Error {
	if errors.Is(cause, ErrInvocationInProgress) {
		return NewError(
			"planner_invocation_in_progress",
			"Planner invocation is already in progress and cannot be resumed",
			true,
			cause,
		)
	}
	return NewError(
		"planner_session_unavailable",
		"Planner durable session is unavailable during "+operation,
		true,
		cause,
	)
}

func validateCompletion(completion Completion) error {
	if (completion.Result == nil) == (completion.Failure == nil) {
		return fmt.Errorf("Planner completion requires exactly one result or failure")
	}
	if completion.Result != nil {
		return completion.Result.Validate()
	}
	return validateFailure(*completion.Failure)
}

// RequestFactsFor reduces model-visible Stage input to a bounded durable audit
// record. Text and parameter values are represented only by digests or names.
func RequestFactsFor(bindings []string, request contracts.StageContentRequest) RequestFacts {
	parameterNames := make([]string, 0, len(request.Parameters))
	for name := range request.Parameters {
		parameterNames = append(parameterNames, name)
	}
	sort.Strings(parameterNames)
	artifacts := make(map[string]contracts.ArtifactRef, len(request.Artifacts))
	for name, ref := range request.Artifacts {
		artifacts[name] = cloneArtifactRef(ref)
	}
	return RequestFacts{
		Bindings:           append([]string(nil), bindings...),
		ObjectiveDigest:    textDigest(request.Objective),
		InstructionsDigest: textDigest(request.Instructions),
		ParameterNames:     parameterNames,
		Artifacts:          artifacts,
	}
}

func textDigest(value string) string {
	digest := sha256.Sum256([]byte(value))
	return "sha256:" + hex.EncodeToString(digest[:])
}

func stageRequest(invocation Invocation) (contracts.StageContentRequest, error) {
	parameters := make(map[string]string, len(invocation.Context.Parameters))
	for name, value := range invocation.Context.Parameters {
		parameters[name] = value
	}
	artifacts := make(map[string]contracts.ArtifactRef, len(invocation.Context.Artifacts))
	for name, ref := range invocation.Context.Artifacts {
		if ref != nil {
			artifacts[name] = cloneArtifactRef(*ref)
		}
	}
	request := contracts.StageContentRequest{
		APIVersion: contracts.APIVersion, Objective: invocation.Stage.Objective,
		Instructions: invocation.Stage.Instructions.Text, Parameters: parameters, Artifacts: artifacts,
	}
	if err := request.Validate(); err != nil {
		return contracts.StageContentRequest{}, fmt.Errorf("invalid StageContentRequest: %w", err)
	}
	encoded, err := json.Marshal(request)
	if err != nil || len(encoded) > maxStageContentBytes {
		return contracts.StageContentRequest{}, fmt.Errorf("StageContentRequest exceeds its bounded contract")
	}
	return request, nil
}

// BuildStageRequest creates the immutable Stage-level request used as the
// starting context by Planner implementations.
func BuildStageRequest(invocation Invocation) (contracts.StageContentRequest, error) {
	return stageRequest(invocation)
}

func validateInvocation(invocation Invocation) (string, contracts.WorkerHandle, error) {
	if strings.TrimSpace(invocation.StageExecutionID) == "" || strings.TrimSpace(invocation.RunID) == "" {
		return "", contracts.WorkerHandle{}, fmt.Errorf("StageExecution and Run IDs are required")
	}
	if invocation.Deadline.IsZero() {
		return "", contracts.WorkerHandle{}, fmt.Errorf("Planner deadline is required")
	}
	if strings.TrimSpace(invocation.Stage.Objective) == "" ||
		strings.TrimSpace(invocation.Stage.Instructions.Text) == "" {
		return "", contracts.WorkerHandle{}, fmt.Errorf("Stage objective and instructions are required")
	}
	if invocation.Stage.Planner.PlannerID+"@"+invocation.Stage.Planner.Version != PassthroughRef {
		return "", contracts.WorkerHandle{}, fmt.Errorf("passthrough@1 cannot execute a Stage for another PlannerFactory")
	}
	if len(invocation.Stage.Agents) != 1 || len(invocation.Workers) != 1 {
		return "", contracts.WorkerHandle{}, fmt.Errorf("passthrough@1 requires exactly one Agent and Worker")
	}
	var binding string
	var handle contracts.WorkerHandle
	for name, current := range invocation.Workers {
		binding, handle = name, current
	}
	resolved, ok := invocation.Stage.Agents[binding]
	if !ok {
		return "", contracts.WorkerHandle{}, fmt.Errorf("prepared Worker has no matching Stage Agent binding")
	}
	if strings.TrimSpace(handle.AllocationID) == "" || len(handle.AgentCard) == 0 ||
		handle.LeaseExpiresAt.IsZero() || handle.AgentTemplateRef != resolved.Template.Ref ||
		handle.WorkerRuntimeRef != resolved.Template.Runtime {
		return "", contracts.WorkerHandle{}, fmt.Errorf("prepared Worker does not match the Agent binding")
	}
	if len(invocation.Context.Artifacts) != len(invocation.Stage.Context.Artifacts) {
		return "", contracts.WorkerHandle{}, fmt.Errorf("StageContext does not match the Stage artifact contract")
	}
	for name, declared := range invocation.Stage.Context.Artifacts {
		ref, exists := invocation.Context.Artifacts[name]
		if !exists || declared.Required && ref == nil {
			return "", contracts.WorkerHandle{}, fmt.Errorf("StageContext artifact %q is unresolved", name)
		}
		if ref != nil {
			if err := ref.ValidateExact(); err != nil {
				return "", contracts.WorkerHandle{}, fmt.Errorf("StageContext artifact %q: %w", name, err)
			}
		}
	}
	return binding, cloneWorkerHandle(handle), nil
}

func cloneArtifactRef(ref contracts.ArtifactRef) contracts.ArtifactRef {
	result := ref
	if ref.Revision != nil {
		revision := *ref.Revision
		result.Revision = &revision
	}
	return result
}

// CloneArtifactRef returns a detached exact reference.
func CloneArtifactRef(ref contracts.ArtifactRef) contracts.ArtifactRef {
	return cloneArtifactRef(ref)
}

func cloneStageRequest(request contracts.StageContentRequest) contracts.StageContentRequest {
	result := request
	result.Parameters = make(map[string]string, len(request.Parameters))
	for name, value := range request.Parameters {
		result.Parameters[name] = value
	}
	result.Artifacts = make(map[string]contracts.ArtifactRef, len(request.Artifacts))
	for name, ref := range request.Artifacts {
		result.Artifacts[name] = cloneArtifactRef(ref)
	}
	return result
}

// CloneStageRequest detaches all mutable request maps.
func CloneStageRequest(request contracts.StageContentRequest) contracts.StageContentRequest {
	return cloneStageRequest(request)
}

func cloneStageResult(result contracts.StageContentResult) contracts.StageContentResult {
	cloned := result
	cloned.Artifacts = make(map[string]contracts.ArtifactRef, len(result.Artifacts))
	for name, ref := range result.Artifacts {
		cloned.Artifacts[name] = cloneArtifactRef(ref)
	}
	if result.Error != nil {
		errorCopy := *result.Error
		cloned.Error = &errorCopy
	}
	return cloned
}

// CloneStageResult detaches all mutable result values.
func CloneStageResult(result contracts.StageContentResult) contracts.StageContentResult {
	return cloneStageResult(result)
}

func pointerToResult(result contracts.StageContentResult) *contracts.StageContentResult {
	return &result
}

func cloneWorkerHandle(handle contracts.WorkerHandle) contracts.WorkerHandle {
	result := handle
	encoded, err := json.Marshal(handle.AgentCard)
	if err == nil {
		_ = json.Unmarshal(encoded, &result.AgentCard)
	}
	return result
}

// CloneWorkerHandle detaches mutable Agent Card data before crossing a Planner
// adapter boundary.
func CloneWorkerHandle(handle contracts.WorkerHandle) contracts.WorkerHandle {
	return cloneWorkerHandle(handle)
}
