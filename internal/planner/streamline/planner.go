package streamline

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"sort"
	"strings"
	"sync"
	"time"

	workflowconfig "github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/planner"
	plannersession "github.com/grauwolf32/contractor/internal/planner/session"
	"github.com/grauwolf32/contractor/internal/telemetry"
	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/model"
	"google.golang.org/adk/runner"
	adksession "google.golang.org/adk/session"
	"google.golang.org/adk/tool"
	"google.golang.org/genai"
)

const completionWriteTimeout = time.Second

type streamlinePlanner struct {
	profile        plannerProfile
	invocation     planner.Invocation
	request        contracts.StageContentRequest
	workers        []workerBinding
	plan           *planner.PlannerPlanController
	resultContract map[string]workflowconfig.ArtifactSlot
	sessions       planner.PlanSessionService
	adkSessions    ADKSessionFactory
	invoker        planner.WorkerInvoker
	inspector      planner.ArtifactInspector
	model          model.LLM
	limits         Limits
	deadline       time.Time

	reportMu  sync.RWMutex
	report    contracts.ExecutionReport
	hasReport bool
}

func (p *streamlinePlanner) Run(
	ctx context.Context,
) (candidate contracts.StageContentResult, runErr error) {
	startedAt := time.Now()
	state := newExecutionState(p.limits)
	var identity planner.SessionIdentity
	defer func() {
		if runErr != nil {
			failure := planner.FailureFrom(runErr)
			state.setExternalFailure(planner.NewErrorFromFailure(failure, nil))
		}
		reportID := "planner-unavailable-" + p.invocation.StageExecutionID
		if identity.SessionID != "" {
			reportID = "planner-" + identity.SessionID
		}
		report := state.report(reportID, time.Since(startedAt))
		p.reportMu.Lock()
		p.report, p.hasReport = report, true
		p.reportMu.Unlock()
	}()

	instrumentation := planner.InvocationInstrumentation(p.invocation)
	sessionSpan := instrumentation.StartSpan(
		telemetry.PlannerSpanSession,
		telemetry.PlannerSpanAttributes{Operation: "session.begin"},
	)
	started, err := p.sessions.Begin(ctx, p.invocation.StageExecutionID)
	if err != nil {
		sessionSpan.End("unavailable", telemetry.PlannerSpanAttributes{ErrorCode: "planner_session_unavailable"})
		return contracts.StageContentResult{}, sessionFailure("start", err)
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
		return contracts.StageContentResult{}, planner.NewError(
			"planner_session_invalid", "Planner session did not grant invocation ownership", false, nil,
		)
	}
	sessionSpan.End("succeeded", telemetry.PlannerSpanAttributes{SessionID: identity.SessionID})
	bindings := make([]string, 0, len(p.workers))
	for _, worker := range p.workers {
		bindings = append(bindings, worker.logicalName)
	}
	recordSpan := instrumentation.StartSpan(
		telemetry.PlannerSpanSession,
		telemetry.PlannerSpanAttributes{Operation: "session.record_request", SessionID: identity.SessionID},
	)
	if err := p.sessions.RecordRequest(
		ctx, identity, planner.RequestFactsFor(bindings, p.request),
	); err != nil {
		recordSpan.End("unavailable", telemetry.PlannerSpanAttributes{ErrorCode: "planner_session_unavailable"})
		return contracts.StageContentResult{}, sessionFailure("record request", err)
	}
	recordSpan.End("succeeded", telemetry.PlannerSpanAttributes{})

	tools, allowed, err := p.buildTools(state, identity)
	if err != nil {
		return contracts.StageContentResult{}, p.fail(
			ctx, identity, state, planner.NewError(
				"planner_initialization_failed", "Planner tools could not be initialized", false, err,
			),
		)
	}
	allowedNames := make([]string, 0, len(allowed))
	for name := range allowed {
		allowedNames = append(allowedNames, name)
	}
	sort.Strings(allowedNames)
	adkSessions, err := p.adkSessions.NewADKSession(ctx, identity, plannersession.ADKOptions{
		AppName: p.profile.adkAppName, UserID: p.invocation.StageExecutionID, AllowedTools: allowedNames,
	})
	if err != nil {
		return contracts.StageContentResult{}, p.fail(ctx, identity, state, sessionFailure("create ADK session", err))
	}
	adkSessionSpan := instrumentation.StartSpan(
		telemetry.PlannerSpanSession,
		telemetry.PlannerSpanAttributes{Operation: "session.adk_create", SessionID: identity.SessionID},
	)
	if _, err := adkSessions.Create(ctx, &adksession.CreateRequest{
		AppName: p.profile.adkAppName, UserID: p.invocation.StageExecutionID,
		SessionID: identity.SessionID, State: map[string]any{"metrics": map[string]any{}},
	}); err != nil {
		adkSessionSpan.End("unavailable", telemetry.PlannerSpanAttributes{ErrorCode: "planner_session_unavailable"})
		return contracts.StageContentResult{}, p.fail(ctx, identity, state, sessionFailure("initialize ADK session", err))
	}
	adkSessionSpan.End("succeeded", telemetry.PlannerSpanAttributes{})

	root, err := p.newRootAgent(state, tools, allowed)
	if err != nil {
		return contracts.StageContentResult{}, p.fail(
			ctx, identity, state, planner.NewError(
				"planner_initialization_failed", "Planner agent could not be initialized", false, err,
			),
		)
	}
	adkRunner, err := runner.New(runner.Config{
		AppName: p.profile.adkAppName, Agent: root, SessionService: adkSessions,
	})
	if err != nil {
		return contracts.StageContentResult{}, p.fail(
			ctx, identity, state, planner.NewError(
				"planner_initialization_failed", "Planner runner could not be initialized", false, err,
			),
		)
	}

	deadline := p.invocation.Deadline
	wallDeadline := time.Now().Add(p.limits.MaxWallTime)
	if wallDeadline.Before(deadline) {
		deadline = wallDeadline
	}
	p.deadline = deadline
	runContext, cancel := context.WithDeadline(ctx, deadline)
	defer cancel()
	message, err := p.initialMessage()
	if err != nil {
		return contracts.StageContentResult{}, p.fail(
			ctx, identity, state, planner.NewError(
				"planner_context_invalid", "Planner context could not be encoded", false, err,
			),
		)
	}

	for {
		var iterationErr error
		for _, eventErr := range adkRunner.Run(
			runContext,
			p.invocation.StageExecutionID,
			identity.SessionID,
			message,
			agent.RunConfig{},
		) {
			if eventErr != nil {
				iterationErr = eventErr
				break
			}
		}
		if result, failure := state.terminal(); result != nil {
			completion := planner.Completion{Result: result}
			if err := p.recordCompletion(ctx, identity, completion); err != nil {
				return contracts.StageContentResult{}, sessionFailure("record completion", err)
			}
			return planner.CloneStageResult(*result), nil
		} else if failure != nil {
			return contracts.StageContentResult{}, p.fail(ctx, identity, state, failure)
		}
		if runContext.Err() != nil {
			failure := planner.NewError(
				"planner_deadline_exceeded", "Planner exhausted its wall-time limit", true,
				runContext.Err(),
			)
			if ctx.Err() != nil {
				failure = planner.NewError(
					"planner_cancelled", "Planner invocation was cancelled", true, ctx.Err(),
				)
			}
			return contracts.StageContentResult{}, p.fail(ctx, identity, state, failure)
		}
		if iterationErr != nil {
			var plannerError *planner.Error
			if errors.As(iterationErr, &plannerError) {
				return contracts.StageContentResult{}, p.fail(ctx, identity, state, plannerError)
			}
			return contracts.StageContentResult{}, p.fail(
				ctx, identity, state, planner.NewError(
					"planner_execution_failed", "Planner ADK execution failed", true, iterationErr,
				),
			)
		}
		if exhausted := state.exhaustedAfterTurn(); exhausted != nil {
			return contracts.StageContentResult{}, p.fail(ctx, identity, state, exhausted)
		}
		message = genai.NewContentFromText(
			"Continue planning. Call exactly one declared planning, Worker, or finish tool; a text-only answer does not complete the Stage.",
			genai.RoleUser,
		)
	}
}

func (p *streamlinePlanner) ExecutionReport() (contracts.ExecutionReport, bool) {
	p.reportMu.RLock()
	defer p.reportMu.RUnlock()
	return p.report, p.hasReport
}

func (p *streamlinePlanner) newRootAgent(
	state *executionState, tools []tool.Tool, allowed map[string]struct{},
) (agent.Agent, error) {
	temperature := float32(0)
	instrumentation := planner.InvocationInstrumentation(p.invocation)
	modelAlias := "configured-model"
	if p.invocation.ModelAccess != nil {
		modelAlias = p.invocation.ModelAccess.ModelPolicy.Model
	}
	var modelSpanMu sync.Mutex
	var modelSpan telemetry.PlannerSpan
	startModelSpan := func() {
		modelSpanMu.Lock()
		previous := modelSpan
		modelSpan = instrumentation.StartSpan(
			telemetry.PlannerSpanModel,
			telemetry.PlannerSpanAttributes{
				Operation: "model.generate", ModelAlias: modelAlias,
			},
		)
		modelSpanMu.Unlock()
		if previous != nil {
			previous.End("failed", telemetry.PlannerSpanAttributes{ErrorCode: "planner_model_overlap"})
		}
	}
	endModelSpan := func(outcome, code string) {
		modelSpanMu.Lock()
		current := modelSpan
		modelSpan = nil
		modelSpanMu.Unlock()
		if current != nil {
			current.End(outcome, telemetry.PlannerSpanAttributes{ErrorCode: code})
		}
	}
	return llmagent.New(llmagent.Config{
		Name: p.profile.agentName, Model: p.model,
		Description: p.profile.agentDescription,
		InstructionProvider: func(agent.ReadonlyContext) (string, error) {
			return p.systemInstruction(), nil
		},
		GenerateContentConfig: &genai.GenerateContentConfig{Temperature: &temperature},
		Tools:                 tools,
		BeforeModelCallbacks: []llmagent.BeforeModelCallback{
			func(agent.CallbackContext, *model.LLMRequest) (*model.LLMResponse, error) {
				if err := state.beforeModel(); err != nil {
					return nil, err
				}
				startModelSpan()
				return nil, nil
			},
		},
		AfterModelCallbacks: []llmagent.AfterModelCallback{
			func(_ agent.CallbackContext, response *model.LLMResponse, _ error) (*model.LLMResponse, error) {
				result, err := state.afterModel(response, allowed)
				if err != nil {
					failure := planner.FailureFrom(err)
					endModelSpan("failed", failure.Code)
				} else {
					endModelSpan("succeeded", "")
				}
				return result, err
			},
		},
		OnModelErrorCallbacks: []llmagent.OnModelErrorCallback{
			func(ctx agent.CallbackContext, _ *model.LLMRequest, providerErr error) (*model.LLMResponse, error) {
				if ctx.Err() != nil {
					endModelSpan("cancelled", "planner_cancelled")
					return nil, providerErr
				}
				failure := state.providerFailure()
				endModelSpan("failed", planner.FailureFrom(failure).Code)
				return nil, failure
			},
		},
	})
}

type promptWorker struct {
	Name        string `json:"name"`
	Tool        string `json:"tool"`
	Description string `json:"description"`
}

type promptContext struct {
	Objective       string                                 `json:"objective"`
	Parameters      map[string]string                      `json:"parameters"`
	Artifacts       map[string]*contracts.ArtifactRef      `json:"artifacts"`
	Workers         []promptWorker                         `json:"workers"`
	ResultArtifacts map[string]workflowconfig.ArtifactSlot `json:"resultArtifacts"`
}

func (p *streamlinePlanner) initialMessage() (*genai.Content, error) {
	workers := make([]promptWorker, 0, len(p.workers))
	for _, binding := range p.workers {
		workers = append(workers, promptWorker{
			Name: binding.logicalName, Tool: executeCurrentSubtaskToolName, Description: binding.description,
		})
	}
	payload, err := json.Marshal(promptContext{
		Objective:       p.request.Objective,
		Parameters:      p.request.Parameters,
		Artifacts:       clonePromptArtifacts(p.invocation.Context.Artifacts),
		Workers:         workers,
		ResultArtifacts: cloneResultContract(p.resultContract),
	})
	if err != nil || len(payload) > maxStagePayloadBytes {
		return nil, fmt.Errorf("Planner context exceeds its bounded contract")
	}
	return genai.NewContentFromText(string(payload), genai.RoleUser), nil
}

func clonePromptArtifacts(
	input map[string]*contracts.ArtifactRef,
) map[string]*contracts.ArtifactRef {
	result := make(map[string]*contracts.ArtifactRef, len(input))
	for name, ref := range input {
		if ref == nil {
			result[name] = nil
			continue
		}
		cloned := planner.CloneArtifactRef(*ref)
		result[name] = &cloned
	}
	return result
}

func (p *streamlinePlanner) systemInstruction() string {
	if p.profile.routesWorkers {
		return p.routerSystemInstruction()
	}
	var mappings []string
	for _, binding := range p.workers {
		mappings = append(mappings, fmt.Sprintf("- %s: call %s", binding.logicalName, executeCurrentSubtaskToolName))
	}
	return strings.Join([]string{
		"You are the root Planner for exactly one immutable Contractor Stage.",
		"The JSON user message contains the Stage objective, string parameters, exact input artifact references, fixed Workers, and result contract.",
		"The Stage objective is the immutable global task. Create bounded ordered work with add_subtask; objective and instructions are stored once and cannot be changed during dispatch. Use list_subtasks when you need the authoritative current ID and statuses.",
		"Use Workers sequentially. You cannot create Workers, allocate capacity, change the Workflow, or address Runtime Agent identities.",
		"execute_current_subtask accepts only the exact current subtask_id. Server supplies its stored objective and instructions plus every immutable Stage string parameter and exact artifact revision; you cannot select or rewrite that context.",
		"The Stage is not complete when you emit prose. You must call finish with a succeeded or failed candidate.",
		"A succeeded finish requires at least one succeeded subtask and no pending work. A failed finish is allowed whenever no Worker dispatch is active.",
		"finish reports only the semantic outcome; Workflow Scheduler validates the candidate and alone chooses every Workflow transition, including retry or configured escalation.",
		"Fixed Worker mapping:",
		strings.Join(mappings, "\n"),
		"Stage-specific operating guidance:",
		p.request.Instructions,
	}, "\n\n")
}

func (p *streamlinePlanner) routerSystemInstruction() string {
	agents := make([]string, 0, len(p.workers))
	for _, binding := range p.workers {
		agents = append(agents, fmt.Sprintf("- %s: %s", binding.logicalName, binding.description))
	}
	return strings.Join([]string{
		"You are the Router Planner for exactly one immutable Contractor Stage.",
		"The JSON user message contains the Stage objective, string parameters, exact input artifact references, fixed logical Workers, and result contract.",
		"The Stage objective is the immutable global task. Create bounded ordered work with add_subtask; objective and instructions are stored once and cannot be changed during dispatch. Use list_subtasks when you need the authoritative current ID and statuses.",
		"Execute Workers sequentially with execute_current_subtask(subtask_id, worker_name). Select worker_name only from Available agents. Server supplies the stored subtask plus every immutable Stage string parameter and exact artifact revision.",
		"Logical routing is your only selection responsibility. You cannot create Workers, allocate capacity, change the Workflow, observe physical Runtime Agent identities, or rewrite Worker context.",
		"The Stage is not complete when you emit prose. You must call finish with a succeeded or failed candidate.",
		"A succeeded finish requires at least one succeeded subtask and no pending work. A failed finish is allowed whenever no Worker dispatch is active.",
		"finish reports only the semantic outcome; Workflow Scheduler validates the candidate and alone chooses every Workflow transition, including retry or configured escalation.",
		"Stage-specific operating guidance:",
		p.request.Instructions,
		"Available agents:",
		strings.Join(agents, "\n"),
	}, "\n\n")
}

func (p *streamlinePlanner) recoverCompletion(
	ctx context.Context, completion planner.Completion,
) (contracts.StageContentResult, error) {
	if (completion.Result == nil) == (completion.Failure == nil) {
		return contracts.StageContentResult{}, planner.NewError(
			"planner_session_invalid", "Recorded Planner completion is invalid", false, nil,
		)
	}
	if completion.Failure != nil {
		return contracts.StageContentResult{}, planner.NewErrorFromFailure(*completion.Failure, nil)
	}
	result := planner.CloneStageResult(*completion.Result)
	if err := planner.ValidateCandidate(
		ctx, p.invocation.RunID, p.resultContract, result, p.inspector,
	); err != nil {
		return contracts.StageContentResult{}, err
	}
	return result, nil
}

func (p *streamlinePlanner) fail(
	ctx context.Context,
	identity planner.SessionIdentity,
	state *executionState,
	value *planner.Error,
) *planner.Error {
	value = state.setExternalFailure(value)
	failure := value.Failure
	if err := p.recordCompletion(ctx, identity, planner.Completion{Failure: &failure}); err != nil {
		return sessionFailure("record failure", errors.Join(value, err))
	}
	return value
}

func (p *streamlinePlanner) recordCompletion(
	ctx context.Context, identity planner.SessionIdentity, completion planner.Completion,
) error {
	recordContext, cancel := context.WithTimeout(context.WithoutCancel(ctx), completionWriteTimeout)
	defer cancel()
	return p.sessions.Complete(recordContext, identity, completion)
}

func sessionFailure(operation string, cause error) *planner.Error {
	if errors.Is(cause, planner.ErrInvocationInProgress) {
		return planner.NewError(
			"planner_invocation_in_progress",
			"Planner invocation is already in progress and cannot be resumed",
			true,
			cause,
		)
	}
	return planner.NewError(
		"planner_session_unavailable",
		"Planner durable session is unavailable during "+operation,
		true,
		cause,
	)
}

var _ planner.Planner = (*streamlinePlanner)(nil)
var _ planner.ReportProvider = (*streamlinePlanner)(nil)
