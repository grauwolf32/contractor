package streamline

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"regexp"
	"sort"
	"strings"

	workflowconfig "github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/planner"
	plannersession "github.com/grauwolf32/contractor/internal/planner/session"
	"google.golang.org/adk/model"
	adksession "google.golang.org/adk/session"
)

const adkAppName = "contractor_streamline"

type ADKSessionFactory interface {
	NewADKSession(
		context.Context,
		planner.SessionIdentity,
		plannersession.ADKOptions,
	) (adksession.Service, error)
}

type Factory struct {
	sessions    planner.SessionService
	adkSessions ADKSessionFactory
	invoker     planner.WorkerInvoker
	inspector   planner.ArtifactInspector
	model       model.LLM
	limits      Limits
}

func NewFactory(
	sessions planner.SessionService,
	adkSessions ADKSessionFactory,
	invoker planner.WorkerInvoker,
	inspector planner.ArtifactInspector,
	llm model.LLM,
	limits Limits,
) (*Factory, error) {
	if sessions == nil || adkSessions == nil || invoker == nil || inspector == nil || llm == nil {
		return nil, fmt.Errorf("StreamlinePlanner dependencies are incomplete")
	}
	normalized, err := normalizeLimits(limits)
	if err != nil {
		return nil, err
	}
	return &Factory{
		sessions: sessions, adkSessions: adkSessions, invoker: invoker,
		inspector: inspector, model: llm, limits: normalized,
	}, nil
}

func (*Factory) Ref() string { return Ref }

func (f *Factory) Create(invocation planner.Invocation) (planner.Planner, error) {
	bindings, err := validateInvocation(invocation)
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
	workers := make([]workerBinding, 0, len(bindings))
	usedToolNames := make(map[string]struct{}, len(bindings)+3)
	usedToolNames[finishToolName] = struct{}{}
	usedToolNames[addSubtaskToolName] = struct{}{}
	usedToolNames[listSubtasksToolName] = struct{}{}
	for _, logicalName := range bindings {
		binding := invocation.Stage.Agents[logicalName]
		toolName := workerToolName(logicalName, usedToolNames)
		usedToolNames[toolName] = struct{}{}
		workers = append(workers, workerBinding{
			logicalName: logicalName,
			toolName:    toolName,
			description: binding.Template.Description,
			handle:      planner.CloneWorkerHandle(invocation.Workers[logicalName]),
		})
	}
	return &streamlinePlanner{
		invocation: invocation, request: request, workers: workers, plan: plan,
		resultContract: cloneResultContract(invocation.Stage.Result.Artifacts),
		sessions:       f.sessions, adkSessions: f.adkSessions,
		invoker: f.invoker, inspector: f.inspector, model: f.model, limits: f.limits,
	}, nil
}

type workerBinding struct {
	logicalName string
	toolName    string
	description string
	handle      contracts.WorkerHandle
}

func validateInvocation(invocation planner.Invocation) ([]string, error) {
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
	if invocation.Stage.Planner.PlannerID+"@"+invocation.Stage.Planner.Version != Ref {
		return nil, fmt.Errorf("streamline@1 cannot execute a Stage for another PlannerFactory")
	}
	if len(invocation.Stage.Agents) == 0 || len(invocation.Workers) != len(invocation.Stage.Agents) {
		return nil, fmt.Errorf("streamline@1 requires one prepared Worker per Agent binding")
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

var invalidToolCharacter = regexp.MustCompile(`[^a-z0-9_]`)

func workerToolName(binding string, used map[string]struct{}) string {
	base := strings.ToLower(binding)
	base = invalidToolCharacter.ReplaceAllString(base, "_")
	base = strings.Trim(base, "_")
	if base == "" || base[0] < 'a' || base[0] > 'z' {
		base = "agent_" + base
	}
	if len(base) > 48 {
		base = base[:48]
	}
	candidate := "worker_" + base
	if len(candidate) <= 64 {
		if _, exists := used[candidate]; !exists {
			return candidate
		}
	}
	digest := sha256.Sum256([]byte(binding))
	suffix := hex.EncodeToString(digest[:4])
	if len(base) > 47 {
		base = base[:47]
	}
	return "worker_" + base + "_" + suffix
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
