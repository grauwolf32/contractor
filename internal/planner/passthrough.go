package planner

import (
	"context"
	"encoding/json"
	"fmt"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/grauwolf32/contractor/internal/contentdigest"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/telemetry"
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
	started, err := StartSession(ctx, p.sessions, instrumentation, p.invocation.StageExecutionID)
	identity = started.Identity
	if err != nil {
		return contracts.StageContentResult{}, err
	}
	if started.Completion != nil {
		return p.recoverCompletion(ctx, *started.Completion)
	}

	facts := RequestFactsFor([]string{p.binding}, p.request)
	if err := RecordSessionRequest(ctx, p.sessions, instrumentation, started.Identity, facts); err != nil {
		return contracts.StageContentResult{}, err
	}
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
	workerCompletion, err := p.invoker.Invoke(
		invokeContext, p.binding, cloneWorkerHandle(p.handle), cloneStageRequest(p.request),
	)
	telemetry.CapturePlannerInput(workerSpan, func() any { return p.request })
	if err == nil {
		telemetry.CapturePlannerOutput(workerSpan, func() any { return workerCompletion })
	}
	invokeDurationMS = max(0, time.Since(invokeStarted).Milliseconds())
	if err != nil {
		failure := FailureFrom(err)
		invokeFailure = &failure
		workerSpan.End("failed", telemetry.PlannerSpanAttributes{ErrorCode: failure.Code})
		return contracts.StageContentResult{}, p.fail(
			ctx, started.Identity, NewErrorFromFailure(FailureFrom(err), err),
		)
	}
	if validation := ValidateWorkerCompletion(workerCompletion, p.request.SubtaskID); validation != nil {
		failure := FailureFrom(validation)
		invokeFailure = &failure
		workerSpan.End("rejected", telemetry.PlannerSpanAttributes{ErrorCode: failure.Code})
		return contracts.StageContentResult{}, p.fail(ctx, started.Identity, validation)
	}
	result, err := StageCandidateFromWorkerCompletion(workerCompletion)
	if err != nil {
		validation := NewError(
			"invalid_worker_result", "Worker completion cannot be mapped to a Stage candidate", false, err,
		)
		failure := FailureFrom(validation)
		invokeFailure = &failure
		workerSpan.End("rejected", telemetry.PlannerSpanAttributes{ErrorCode: failure.Code})
		return contracts.StageContentResult{}, p.fail(ctx, started.Identity, validation)
	}
	if err := validateCandidate(
		invokeContext, p.invocation.RunID, p.invocation.Stage.Result.Artifacts, result, p.inspector,
	); err != nil {
		failure := FailureFrom(err)
		workerSpan.End("rejected", telemetry.PlannerSpanAttributes{ErrorCode: failure.Code})
		return contracts.StageContentResult{}, p.fail(ctx, started.Identity, err)
	}
	if workerCompletion.Failure != nil {
		failure := Failure{
			Code: workerCompletion.Failure.Code, Message: workerCompletion.Failure.Message,
			Retryable: workerCompletion.Failure.Retryable,
		}
		invokeFailure = &failure
		workerSpan.End("failed", telemetry.PlannerSpanAttributes{ErrorCode: failure.Code})
	} else {
		workerSpan.End("succeeded", telemetry.PlannerSpanAttributes{})
	}
	completion := Completion{Result: pointerToResult(result.Clone())}
	if err := CompleteSession(ctx, p.sessions, started.Identity, completion); err != nil {
		return contracts.StageContentResult{}, SessionError("record completion", err)
	}
	return result.Clone(), nil
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
			DurationMS:   int64Pointer(durationMS),
			ModelCalls:   int64Pointer(0),
			InputTokens:  int64Pointer(0),
			OutputTokens: int64Pointer(0),
			TotalTokens:  int64Pointer(0),
			Tools:        map[string]contracts.ToolMetrics{},
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
	result := completion.Result.Clone()
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
	if err := CompleteSession(ctx, p.sessions, identity, Completion{Failure: &failure}); err != nil {
		return CompletionWriteError(plannerError, err)
	}
	return plannerError
}

func NewErrorFromFailure(failure Failure, cause error) *Error {
	return NewError(failure.Code, failure.Message, failure.Retryable, cause)
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
	artifacts := contracts.CloneArtifactRefs(request.Artifacts)
	return RequestFacts{
		Bindings:           append([]string(nil), bindings...),
		ObjectiveDigest:    textDigest(request.Objective),
		InstructionsDigest: textDigest(request.Instructions),
		ParameterNames:     parameterNames,
		Artifacts:          artifacts,
	}
}

func textDigest(value string) string {
	return contentdigest.Bytes([]byte(value))
}

func stageRequest(invocation Invocation) (contracts.StageContentRequest, error) {
	parameters := make(map[string]string, len(invocation.Context.Parameters))
	for name, value := range invocation.Context.Parameters {
		parameters[name] = value
	}
	artifacts := make(map[string]contracts.ArtifactRef, len(invocation.Context.Artifacts))
	for name, ref := range invocation.Context.Artifacts {
		if ref != nil {
			artifacts[name] = ref.Clone()
		}
	}
	resultArtifacts := make(map[string]contracts.ArtifactRef)
	for name, slot := range invocation.Stage.Result.Artifacts {
		if slot.From != nil {
			resultArtifacts[name] = contracts.ArtifactRef{
				Namespace: slot.From.Namespace,
				Name:      slot.From.Name,
			}
		}
	}
	request := contracts.StageContentRequest{
		APIVersion: contracts.APIVersion, SubtaskID: "0", Objective: invocation.Stage.Objective,
		Instructions: invocation.Stage.Instructions.Text, Parameters: parameters, Artifacts: artifacts,
		ResultArtifacts: resultArtifacts, Deadline: &invocation.Deadline,
	}
	if err := request.Validate(); err != nil {
		return contracts.StageContentRequest{}, fmt.Errorf("invalid StageContentRequest: %w", err)
	}
	encoded, err := json.Marshal(request)
	if err != nil || len(encoded) > contracts.MaxStageRequestBytes {
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

func cloneStageRequest(request contracts.StageContentRequest) contracts.StageContentRequest {
	result := request
	if request.Deadline != nil {
		deadline := *request.Deadline
		result.Deadline = &deadline
	}
	result.Parameters = make(map[string]string, len(request.Parameters))
	for name, value := range request.Parameters {
		result.Parameters[name] = value
	}
	result.Artifacts = contracts.CloneArtifactRefs(request.Artifacts)
	result.ResultArtifacts = contracts.CloneArtifactRefs(request.ResultArtifacts)
	return result
}

// CloneStageRequest detaches all mutable request maps.
func CloneStageRequest(request contracts.StageContentRequest) contracts.StageContentRequest {
	return cloneStageRequest(request)
}

func pointerToResult(result contracts.StageContentResult) *contracts.StageContentResult {
	return &result
}

func cloneWorkerHandle(handle contracts.WorkerHandle) contracts.WorkerHandle {
	result := handle
	// Decoding into a non-nil map reuses it. Clear the destination before the
	// round trip so cloning neither mutates nor shares the original card.
	// A non-JSON card is unusable by A2A; omit it instead of retaining aliases.
	result.AgentCard = nil
	encoded, err := json.Marshal(handle.AgentCard)
	if err == nil {
		_ = json.Unmarshal(encoded, &result.AgentCard)
	}
	return result
}

// CloneWorkerHandle detaches mutable Agent Card data before crossing a Planner
// adapter boundary. Cards containing non-JSON values are omitted, so invalid
// in-process inputs cannot bypass the copy boundary by failing serialization.
func CloneWorkerHandle(handle contracts.WorkerHandle) contracts.WorkerHandle {
	return cloneWorkerHandle(handle)
}
