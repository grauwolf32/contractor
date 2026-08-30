package streamline

import (
	"context"
	"fmt"
	"sort"
	"strings"
	"time"

	workflowconfig "github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/planner"
	plannersession "github.com/grauwolf32/contractor/internal/planner/session"
	"google.golang.org/adk/model"
	adksession "google.golang.org/adk/session"
)

type plannerProfile struct {
	ref              string
	adkAppName       string
	agentName        string
	agentDescription string
	routesWorkers    bool
}

var (
	streamlineProfile = plannerProfile{
		ref: Ref, adkAppName: "contractor_streamline", agentName: "streamline_planner",
		agentDescription: "Plans one immutable Contractor Stage using its sole prepared Worker.",
	}
	routerProfile = plannerProfile{
		ref: planner.RouterRef, adkAppName: "contractor_router", agentName: "router_planner",
		agentDescription: "Plans one immutable Contractor Stage and routes each subtask to one fixed logical Worker.",
		routesWorkers:    true,
	}
)

type ADKSessionFactory interface {
	NewADKSession(
		context.Context,
		planner.SessionIdentity,
		plannersession.ADKOptions,
	) (adksession.Service, error)
}

type Factory struct {
	profile      plannerProfile
	sessions     planner.SessionService
	adkSessions  ADKSessionFactory
	invoker      planner.WorkerInvoker
	inspector    planner.ArtifactInspector
	model        model.LLM
	modelFactory InvocationModelFactory
	limits       Limits
}

type InvocationModelFactory func(planner.ModelAccess) (model.LLM, error)

func NewFactory(
	sessions planner.SessionService,
	adkSessions ADKSessionFactory,
	invoker planner.WorkerInvoker,
	inspector planner.ArtifactInspector,
	llm model.LLM,
	limits Limits,
) (*Factory, error) {
	return newFactory(streamlineProfile, sessions, adkSessions, invoker, inspector, llm, nil, limits)
}

func NewConfiguredFactory(
	sessions planner.SessionService,
	adkSessions ADKSessionFactory,
	invoker planner.WorkerInvoker,
	inspector planner.ArtifactInspector,
	modelFactory InvocationModelFactory,
	limits Limits,
) (*Factory, error) {
	return newFactory(
		streamlineProfile, sessions, adkSessions, invoker, inspector, nil, modelFactory, limits,
	)
}

// NewRouterDelegate creates the shared model-backed engine configured for
// router@1. The public Router factory wraps this delegate so Scheduler still
// registers distinct framework-neutral PlannerFactory implementations.
func NewRouterDelegate(
	sessions planner.SessionService,
	adkSessions ADKSessionFactory,
	invoker planner.WorkerInvoker,
	inspector planner.ArtifactInspector,
	llm model.LLM,
	limits Limits,
) (*Factory, error) {
	return newFactory(routerProfile, sessions, adkSessions, invoker, inspector, llm, nil, limits)
}

func NewConfiguredRouterDelegate(
	sessions planner.SessionService,
	adkSessions ADKSessionFactory,
	invoker planner.WorkerInvoker,
	inspector planner.ArtifactInspector,
	modelFactory InvocationModelFactory,
	limits Limits,
) (*Factory, error) {
	return newFactory(
		routerProfile, sessions, adkSessions, invoker, inspector, nil, modelFactory, limits,
	)
}

func newFactory(
	profile plannerProfile,
	sessions planner.SessionService,
	adkSessions ADKSessionFactory,
	invoker planner.WorkerInvoker,
	inspector planner.ArtifactInspector,
	llm model.LLM,
	modelFactory InvocationModelFactory,
	limits Limits,
) (*Factory, error) {
	if sessions == nil || adkSessions == nil || invoker == nil || inspector == nil ||
		(llm == nil) == (modelFactory == nil) {
		return nil, fmt.Errorf("model-backed Planner dependencies are incomplete")
	}
	normalized, err := normalizeLimits(limits)
	if err != nil {
		return nil, err
	}
	return &Factory{
		profile: profile, sessions: sessions, adkSessions: adkSessions, invoker: invoker,
		inspector: inspector, model: llm, modelFactory: modelFactory, limits: normalized,
	}, nil
}

func (f *Factory) Ref() string { return f.profile.ref }

func (f *Factory) Create(invocation planner.Invocation) (planner.Planner, error) {
	bindings, err := validateInvocation(f.profile, invocation)
	if err != nil {
		return nil, err
	}
	request, err := planner.BuildStageRequest(invocation)
	if err != nil {
		return nil, err
	}
	plan, err := planner.NewPlannerPlanController(invocation.Stage.Objective)
	if err != nil {
		return nil, fmt.Errorf("initialize Planner plan: %w", err)
	}
	selectedModel := f.model
	selectedLimits := f.limits
	if f.modelFactory != nil {
		if invocation.ModelAccess == nil {
			return nil, fmt.Errorf("%s requires resolved Planner model access", f.profile.ref)
		}
		if err := validateModelAccess(*invocation.ModelAccess); err != nil {
			return nil, err
		}
		selectedModel, err = f.modelFactory(*invocation.ModelAccess)
		if err != nil {
			return nil, fmt.Errorf("configure %s model client: %w", f.profile.ref, err)
		}
		selectedLimits, err = limitsFromPolicy(invocation.ModelAccess.ModelPolicy, f.limits.MaxWallTime)
		if err != nil {
			return nil, err
		}
	}
	workers := make([]workerBinding, 0, len(bindings))
	for _, logicalName := range bindings {
		binding := invocation.Stage.Agents[logicalName]
		workers = append(workers, workerBinding{
			logicalName: logicalName,
			description: binding.Template.Description,
			handle:      planner.CloneWorkerHandle(invocation.Workers[logicalName]),
		})
	}
	return &streamlinePlanner{
		profile: f.profile, invocation: invocation, request: request, workers: workers, plan: plan,
		resultContract: cloneResultContract(invocation.Stage.Result.Artifacts),
		sessions:       f.sessions, adkSessions: f.adkSessions,
		invoker: f.invoker, inspector: f.inspector, model: selectedModel, limits: selectedLimits,
	}, nil
}

func validateModelAccess(access planner.ModelAccess) error {
	if err := access.ModelPolicy.ValidateForPlanner(); err != nil {
		return err
	}
	if err := access.LLMGateway.Validate(); err != nil {
		return err
	}
	if access.Credential != nil {
		if err := access.Credential.Validate(); err != nil {
			return err
		}
		if access.Token.Reveal() == "" {
			return fmt.Errorf("selected Planner credential resolved to an empty token")
		}
	}
	return nil
}

func limitsFromPolicy(policy contracts.ResolvedModelPolicy, wallTime time.Duration) (Limits, error) {
	return normalizeLimits(Limits{
		MaxModelCalls:  policy.MaxModelCalls,
		MaxTokens:      int64(policy.MaxTotalTokens),
		MaxWorkerCalls: policy.MaxWorkerCalls,
		MaxWallTime:    wallTime,
	})
}

type workerBinding struct {
	logicalName string
	description string
	handle      contracts.WorkerHandle
}

func validateInvocation(profile plannerProfile, invocation planner.Invocation) ([]string, error) {
	if strings.TrimSpace(invocation.StageExecutionID) == "" || strings.TrimSpace(invocation.RunID) == "" {
		return nil, fmt.Errorf("StageExecution and Run IDs are required")
	}
	if invocation.Deadline.IsZero() {
		return nil, fmt.Errorf("Planner deadline is required")
	}
	if strings.TrimSpace(invocation.Stage.Objective) == "" ||
		strings.TrimSpace(invocation.Stage.Instructions.Text) == "" {
		return nil, fmt.Errorf("Stage objective and instructions are required")
	}
	if invocation.Stage.Planner.PlannerID+"@"+invocation.Stage.Planner.Version != profile.ref {
		return nil, fmt.Errorf("%s cannot execute a Stage for another PlannerFactory", profile.ref)
	}
	if profile.routesWorkers {
		if len(invocation.Stage.Agents) == 0 || len(invocation.Workers) != len(invocation.Stage.Agents) {
			return nil, fmt.Errorf("router@1 requires one or more matched logical Agent bindings and prepared Workers")
		}
	} else if len(invocation.Stage.Agents) != 1 || len(invocation.Workers) != 1 {
		return nil, fmt.Errorf("streamline@1 requires exactly one logical Agent binding and prepared Worker")
	}
	bindings := make([]string, 0, len(invocation.Stage.Agents))
	for logicalName, resolved := range invocation.Stage.Agents {
		handle, ok := invocation.Workers[logicalName]
		if !ok || strings.TrimSpace(logicalName) == "" || strings.Contains(logicalName, "/") ||
			strings.TrimSpace(handle.AllocationID) == "" || len(handle.AgentCard) == 0 ||
			handle.LeaseExpiresAt.IsZero() || handle.AgentTemplateRef != resolved.Template.Ref ||
			handle.WorkerRuntimeRef != resolved.Template.Runtime {
			return nil, fmt.Errorf("prepared Worker does not match Agent binding %q", logicalName)
		}
		bindings = append(bindings, logicalName)
	}
	for logicalName := range invocation.Workers {
		if _, ok := invocation.Stage.Agents[logicalName]; !ok {
			return nil, fmt.Errorf("prepared Worker %q has no Agent binding", logicalName)
		}
	}
	if len(invocation.Context.Artifacts) != len(invocation.Stage.Context.Artifacts) {
		return nil, fmt.Errorf("StageContext does not match the Stage artifact contract")
	}
	for name, declared := range invocation.Stage.Context.Artifacts {
		ref, exists := invocation.Context.Artifacts[name]
		if !exists || declared.Required && ref == nil {
			return nil, fmt.Errorf("StageContext artifact %q is unresolved", name)
		}
		if ref != nil {
			if err := ref.ValidateExact(); err != nil {
				return nil, fmt.Errorf("StageContext artifact %q is not exact", name)
			}
		}
	}
	sort.Strings(bindings)
	return bindings, nil
}

func cloneResultContract(
	input map[string]workflowconfig.ArtifactSlot,
) map[string]workflowconfig.ArtifactSlot {
	result := make(map[string]workflowconfig.ArtifactSlot, len(input))
	for name, slot := range input {
		slot.MediaTypes = append([]string(nil), slot.MediaTypes...)
		result[name] = slot
	}
	return result
}

var _ planner.Factory = (*Factory)(nil)
