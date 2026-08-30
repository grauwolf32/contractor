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
	invocation     planner.Invocation
	request        contracts.StageContentRequest
	workers        []workerBinding
	resultContract map[string]workflowconfig.ArtifactSlot
	sessions       planner.SessionService
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

	started, err := p.sessions.Begin(ctx, p.invocation.StageExecutionID)
	if err != nil {
		return contracts.StageContentResult{}, sessionFailure("start", err)
	}
	identity = started.Identity
	if started.Completion != nil {
		return p.recoverCompletion(ctx, *started.Completion)
	}
	if !started.Invoke {
		return contracts.StageContentResult{}, planner.NewError(
			"planner_session_invalid", "Planner session did not grant invocation ownership", false, nil,
		)
	}
	bindings := make([]string, 0, len(p.workers))
	for _, worker := range p.workers {
		bindings = append(bindings, worker.logicalName)
	}
	if err := p.sessions.RecordRequest(
		ctx, identity, planner.RequestFactsFor(bindings, p.request),
	); err != nil {
		return contracts.StageContentResult{}, sessionFailure("record request", err)
	}

	tools, allowed, err := p.buildTools(state)
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
		AppName: adkAppName, UserID: p.invocation.StageExecutionID, AllowedTools: allowedNames,
	})
	if err != nil {
		return contracts.StageContentResult{}, p.fail(ctx, identity, state, sessionFailure("create ADK session", err))
	}
	if _, err := adkSessions.Create(ctx, &adksession.CreateRequest{
		AppName: adkAppName, UserID: p.invocation.StageExecutionID,
		SessionID: identity.SessionID, State: map[string]any{"metrics": map[string]any{}},
	}); err != nil {
		return contracts.StageContentResult{}, p.fail(ctx, identity, state, sessionFailure("initialize ADK session", err))
	}

	root, err := p.newRootAgent(state, tools, allowed)
	if err != nil {
		return contracts.StageContentResult{}, p.fail(
			ctx, identity, state, planner.NewError(
				"planner_initialization_failed", "Planner agent could not be initialized", false, err,
			),
		)
	}
	adkRunner, err := runner.New(runner.Config{
		AppName: adkAppName, Agent: root, SessionService: adkSessions,
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
			"Continue planning. You must call exactly one declared Worker or finish; a text-only answer does not complete the Stage.",
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
	return llmagent.New(llmagent.Config{
		Name: "streamline_planner", Model: p.model,
		Description: "Plans one immutable Contractor Stage using only its fixed prepared Workers.",
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
				return nil, nil
			},
		},
		AfterModelCallbacks: []llmagent.AfterModelCallback{
			func(_ agent.CallbackContext, response *model.LLMResponse, _ error) (*model.LLMResponse, error) {
				return state.afterModel(response, allowed)
			},
		},
		OnModelErrorCallbacks: []llmagent.OnModelErrorCallback{
			func(ctx agent.CallbackContext, _ *model.LLMRequest, providerErr error) (*model.LLMResponse, error) {
				if ctx.Err() != nil {
					return nil, providerErr
				}
				return nil, state.providerFailure()
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
			Name: binding.logicalName, Tool: binding.toolName, Description: binding.description,
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
	var mappings []string
	for _, binding := range p.workers {
		mappings = append(mappings, fmt.Sprintf("- %s: call %s", binding.logicalName, binding.toolName))
	}
	return strings.Join([]string{
		"You are the root Planner for exactly one immutable Contractor Stage.",
		"The JSON user message contains the Stage objective, string parameters, exact input artifact references, fixed Workers, and result contract.",
		"Use Workers sequentially. You cannot create Workers, allocate capacity, change the Workflow, or address Runtime Agent identities.",
		"A Worker call must have a focused non-empty objective and instructions. Select the needed string parameters and exact artifact revisions explicitly; preserve revisions returned by Workers.",
		"The Stage is not complete when you emit prose. You must call finish with a succeeded or failed candidate.",
		"finish reports only the semantic outcome; Workflow Scheduler validates the candidate and alone chooses every Workflow transition, including retry or configured escalation.",
		"Fixed Worker mapping:",
		strings.Join(mappings, "\n"),
		"Stage-specific operating guidance:",
		p.request.Instructions,
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
