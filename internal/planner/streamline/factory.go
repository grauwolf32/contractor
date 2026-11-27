package streamline

import (
	"fmt"
	"slices"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contracts"
	plannermemory "github.com/grauwolf32/contractor/internal/memory"
	"github.com/grauwolf32/contractor/internal/planner"
	"github.com/grauwolf32/contractor/internal/planner/stateview"
)

type plannerProfile struct {
	ref           string
	agentName     string
	routesWorkers bool
}

var (
	streamlineProfile = plannerProfile{
		ref: resolved.StreamlinePlanner, agentName: "streamline_planner",
	}
	routerProfile = plannerProfile{
		ref: resolved.RouterPlanner, agentName: "router_planner",
		routesWorkers: true,
	}
)

type Factory struct {
	runtimeHosts *sync.Map
	profile      plannerProfile
	sessions     planner.PlanSessionService
	invoker      planner.WorkerInvoker
	inspector    planner.ArtifactInspector
	stateReader  planner.WorkerStateReader
	memoryStore  plannermemory.Store
	limits       Limits
}

func NewRuntimeFactoryWithMemory(
	sessions planner.PlanSessionService,
	invoker planner.WorkerInvoker,
	inspector planner.ArtifactInspector,
	stateReader planner.WorkerStateReader,
	memoryStore plannermemory.Store,
	limits Limits,
) (*Factory, error) {
	profile := streamlineProfile
	return newFactory(profile, sessions, invoker, inspector, stateReader, memoryStore, limits)
}

func newFactory(
	profile plannerProfile,
	sessions planner.PlanSessionService,
	invoker planner.WorkerInvoker,
	inspector planner.ArtifactInspector,
	stateReader planner.WorkerStateReader,
	memoryStore plannermemory.Store,
	limits Limits,
) (*Factory, error) {
	if sessions == nil || invoker == nil || inspector == nil || stateReader == nil {
		return nil, fmt.Errorf("model-backed Planner dependencies are incomplete")
	}
	normalized, err := normalizeLimits(limits)
	if err != nil {
		return nil, err
	}
	return &Factory{
		profile: profile, sessions: sessions, invoker: invoker, runtimeHosts: &sync.Map{},
		inspector: inspector, stateReader: stateReader, memoryStore: memoryStore,
		limits: normalized,
	}, nil
}

func (f *Factory) Ref() string { return f.profile.ref }

// CreateRouter uses the shared Runtime host registry with Router's logical tools.
func (f *Factory) CreateRouter(invocation planner.Invocation) (planner.Planner, error) {
	routed := *f
	routed.profile = routerProfile
	return routed.Create(invocation)
}

func (f *Factory) Create(invocation planner.Invocation) (planner.Planner, error) {
	bindings, err := validateInvocation(f.profile, invocation)
	if err != nil {
		return nil, err
	}
	if invocation.Context.Continuation != nil {
		if f.profile.routesWorkers || len(bindings) != 1 {
			return nil, fmt.Errorf("summary continuation requires the single-Worker Streamline Planner")
		}
		if err := invocation.Context.Continuation.Validate(); err != nil {
			return nil, fmt.Errorf("invalid summary continuation: %w", err)
		}
		if invocation.Context.Continuation.SummaryArtifact.Namespace != invocation.Stage.Agents[bindings[0]].Namespace {
			return nil, fmt.Errorf("summary continuation differs from the Worker Namespace")
		}
		continuation := invocation.Context.Continuation.Clone()
		invocation.Context.Continuation = &continuation
	}
	request, err := planner.BuildStageRequest(invocation)
	if err != nil {
		return nil, err
	}
	plan, err := planner.NewPlannerPlanController(invocation.Stage.Objective, invocation.Stage.MaxSubtasks)
	if err != nil {
		return nil, fmt.Errorf("initialize Planner plan: %w", err)
	}
	workers := make([]workerBinding, 0, len(bindings))
	stateBindings := make([]stateview.Binding, 0, len(bindings))
	namespaces := make(map[string]*plannermemory.Namespace)
	for _, logicalName := range bindings {
		binding := invocation.Stage.Agents[logicalName]
		memoryOperations, err := selectedMemoryOperations(binding.Template)
		if err != nil {
			return nil, fmt.Errorf("derive MemoryTools for logical Worker %q: %w", logicalName, err)
		}
		var memoryNamespace *plannermemory.Namespace
		if len(memoryOperations) > 0 {
			if f.memoryStore == nil {
				return nil, fmt.Errorf("%s requires the Planner Memory store", f.profile.ref)
			}
			memoryNamespace = namespaces[binding.Namespace]
			if memoryNamespace == nil {
				memoryNamespace, err = plannermemory.NewNamespace(f.memoryStore, plannermemory.Binding{
					RunID: invocation.RunID, StageExecutionID: invocation.StageExecutionID,
					Namespace: binding.Namespace,
				})
				if err != nil {
					return nil, fmt.Errorf("bind Planner Memory Namespace for logical Worker %q: %w", logicalName, err)
				}
				namespaces[binding.Namespace] = memoryNamespace
			}
		}
		workspaceState := invocation.Stage.Context.Workspace != nil &&
			selectedWorkspaceObservation(binding.Template)
		handle := invocation.Workers[logicalName].Clone()
		workers = append(workers, workerBinding{
			logicalName:    logicalName,
			description:    binding.Template.Description,
			handle:         handle,
			namespace:      binding.Namespace,
			memoryTools:    memoryOperations,
			memory:         memoryNamespace,
			workspaceState: workspaceState,
		})
		stateBindings = append(stateBindings, stateview.Binding{
			LogicalName: logicalName, Handle: handle, WorkspaceEligible: workspaceState,
		})
	}
	// Resolve the complete immutable model-visible surface before constructing
	// a provider client. An invalid or unavailable Memory binding must fail
	// Planner construction without even local model-factory side effects.
	if invocation.ModelAccess == nil {
		return nil, fmt.Errorf("%s requires resolved Planner model access", f.profile.ref)
	}
	if err := validateModelAccess(*invocation.ModelAccess); err != nil {
		return nil, err
	}
	selectedLimits, err := limitsFromPolicy(invocation.ModelAccess.ModelPolicy, f.limits.MaxWallTime)
	if err != nil {
		return nil, err
	}
	stateViews, err := stateview.New(f.stateReader, stateBindings, stateview.Options{})
	if err != nil {
		return nil, fmt.Errorf("initialize Planner Worker State projections: %w", err)
	}
	instance := &streamlinePlanner{
		profile: f.profile, invocation: invocation, request: request, workers: workers, plan: plan, runtimeHosts: f.runtimeHosts,
		resultContract: cloneResultContract(invocation.Stage.Result.Artifacts),
		sessions:       f.sessions,
		invoker:        f.invoker, inspector: f.inspector, stateViews: stateViews,
		limits: selectedLimits,
	}
	if invocation.Context.Continuation != nil {
		if _, err := instance.initialMessage(); err != nil {
			return nil, err
		}
	}
	compaction, err := newCompactionSettings(invocation.Stage, invocation.ModelAccess.ModelPolicy)
	if err != nil {
		return nil, fmt.Errorf("configure %s context compaction: %w", f.profile.ref, err)
	}
	if invocation.RuntimeExecutor == nil {
		return nil, fmt.Errorf("Runtime Planner requires a prepared Planner allocation")
	}
	instance.compactionSettings = compaction
	return instance, nil
}

func validateModelAccess(access planner.ModelAccess) error {
	if err := resolved.ValidatePlannerModelPolicy(access.ModelPolicy); err != nil {
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
	logicalName    string
	description    string
	handle         contracts.WorkerHandle
	namespace      string
	memoryTools    map[string]struct{}
	memory         *plannermemory.Namespace
	workspaceState bool
}

// selectedWorkspaceObservation reports whether a Worker's template selects a
// tool whose calls the Runtime records as workspace observations, which the
// registered Toolset descriptor classifies (TLS-R38).
func selectedWorkspaceObservation(template contracts.ResolvedAgentTemplate) bool {
	return slices.ContainsFunc(template.Toolsets, resolved.SelectsWorkspaceObservationTool)
}

func selectedMemoryOperations(template contracts.ResolvedAgentTemplate) (map[string]struct{}, error) {
	result := map[string]struct{}{}
	for _, selection := range template.Toolsets {
		selector := selection.Ref.ToolsetID + "@" + selection.Ref.Version
		if selector != plannermemory.ToolsetRef {
			continue
		}
		for _, operation := range selection.Tools {
			if !plannermemory.IsOperation(operation) {
				return nil, fmt.Errorf("unknown %s operation %q", plannermemory.ToolsetRef, operation)
			}
			if _, duplicate := result[operation]; duplicate {
				return nil, fmt.Errorf("duplicate %s operation %q", plannermemory.ToolsetRef, operation)
			}
			result[operation] = struct{}{}
		}
	}
	return result, nil
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
	if invocation.Stage.Planner.Selector() != profile.ref {
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
	for logicalName, agentBinding := range invocation.Stage.Agents {
		handle, ok := invocation.Workers[logicalName]
		if !ok || strings.TrimSpace(logicalName) == "" || strings.Contains(logicalName, "/") ||
			strings.TrimSpace(handle.AllocationID) == "" || len(handle.AgentCard) == 0 ||
			handle.LeaseExpiresAt.IsZero() || handle.AgentTemplateRef != agentBinding.Template.Ref ||
			handle.WorkerRuntimeRef != agentBinding.Template.Runtime {
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
	input map[string]resolved.ArtifactSlot,
) map[string]resolved.ArtifactSlot {
	result := make(map[string]resolved.ArtifactSlot, len(input))
	for name, slot := range input {
		slot.MediaTypes = append([]string(nil), slot.MediaTypes...)
		result[name] = slot
	}
	return result
}

var _ planner.Factory = (*Factory)(nil)
