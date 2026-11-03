package scan

import (
	"encoding/json"
	"fmt"
	"io"
	"strings"

	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/planner"
)

type Factory struct {
	sessions  planner.ScanSessionService
	invoker   planner.WorkerInvoker
	artifacts ArtifactStore
	inspector planner.ArtifactInspector
}

func NewFactory(sessions planner.ScanSessionService, invoker planner.WorkerInvoker, artifacts ArtifactStore, inspector planner.ArtifactInspector) (*Factory, error) {
	if sessions == nil || invoker == nil || artifacts == nil || inspector == nil {
		return nil, fmt.Errorf("scan planner requires session, Worker and Artifact adapters")
	}
	return &Factory{sessions: sessions, invoker: invoker, artifacts: artifacts, inspector: inspector}, nil
}

func (*Factory) Ref() string { return resolved.ScanPlanPlanner }

func (f *Factory) Create(invocation planner.Invocation) (planner.Planner, error) {
	if invocation.Stage.Planner.Selector() != resolved.ScanPlanPlanner || invocation.ModelAccess != nil || strings.TrimSpace(invocation.SchedulerClaimID) == "" || invocation.StageExecutionID == "" || invocation.RunID == "" || invocation.Deadline.IsZero() {
		return nil, fmt.Errorf("scan planner requires a model-free Stage and durable Scheduler ownership")
	}
	if err := resolved.ValidateScanPlanStage(invocation.Stage); err != nil {
		return nil, err
	}
	if invocation.Stage.ScanPreparation != nil {
		if _, ok := f.sessions.(planner.AuditScanHistoryReader); !ok {
			return nil, fmt.Errorf("Audit scan requires durable attempt history")
		}
	}
	if len(invocation.Workers) != len(invocation.Stage.Agents) {
		return nil, fmt.Errorf("scan Worker set differs from the fixed Stage bindings")
	}
	workers := make(map[string]contracts.WorkerHandle, len(invocation.Workers))
	for name, binding := range invocation.Stage.Agents {
		handle, ok := invocation.Workers[name]
		if !ok || handle.AllocationID == "" || len(handle.AgentCard) == 0 || handle.LeaseExpiresAt.IsZero() || handle.AgentTemplateRef != binding.Template.Ref || handle.WorkerRuntimeRef != binding.Template.Runtime {
			return nil, fmt.Errorf("scan Worker does not match its pinned template")
		}
		workers[name] = handle.Clone()
	}
	if len(invocation.Context.Artifacts) != len(invocation.Stage.Context.Artifacts) {
		return nil, fmt.Errorf("scan Stage context artifact set differs")
	}
	contextArtifacts := make(map[string]*contracts.ArtifactRef, len(invocation.Context.Artifacts))
	for name, slot := range invocation.Stage.Context.Artifacts {
		ref, ok := invocation.Context.Artifacts[name]
		if !ok || slot.Required && ref == nil {
			return nil, fmt.Errorf("scan Stage artifact is unresolved")
		}
		if ref != nil {
			if ref.ValidateExact() != nil || ref.Namespace != slot.Namespace || ref.Name != slot.Name {
				return nil, fmt.Errorf("scan Stage artifact must match its exact declared binding")
			}
			value := ref.Clone()
			contextArtifacts[name] = &value
		} else {
			contextArtifacts[name] = nil
		}
	}
	// Detach policy, literal argument maps and source bindings from the caller.
	encoded, err := json.Marshal(invocation.Stage)
	if err != nil {
		return nil, fmt.Errorf("scan Stage cannot be copied")
	}
	var stage resolved.ResolvedStage
	if json.Unmarshal(encoded, &stage) != nil {
		return nil, fmt.Errorf("scan Stage cannot be copied")
	}
	invocation.Stage = stage
	invocation.Workers = workers
	invocation.Context = planner.StageContext{Parameters: map[string]string{}, Artifacts: contextArtifacts}
	return &execution{factory: f, invocation: invocation}, nil
}

type execution struct {
	factory    *Factory
	invocation planner.Invocation
	audit      *auditScanInputs
}

// Format redacts the execution as a whole for every verb (SEC-R88): fmt cannot call the model access token's own redaction through the unexported invocation field.
func (execution) Format(state fmt.State, _ rune) {
	_, _ = io.WriteString(state, "scan.execution([REDACTED])")
}

var _ planner.Factory = (*Factory)(nil)
