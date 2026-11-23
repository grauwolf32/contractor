package controlplane

import (
	"encoding/json"
	"fmt"
	"sort"
	"strings"

	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/reporting"
	"github.com/grauwolf32/contractor/internal/contracts/runlabels"
	"github.com/grauwolf32/contractor/internal/contracts/runtimesettings"
	"github.com/grauwolf32/contractor/internal/helpers/contentdigest"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
)

func validatePinnedPerformanceCollection(configuration PinnedReservationConfig) error {
	if err := configuration.PerformanceCollectionPolicy.ValidatePinned(); err != nil {
		return err
	}
	if configuration.PerformanceCollectionPolicy == reporting.PerformanceCollectionRequested {
		if configuration.PerformanceMetrics == nil {
			return ErrInvalidRequest
		}
		return configuration.PerformanceMetrics.Validate()
	}
	if configuration.PerformanceMetrics != nil {
		return ErrInvalidRequest
	}
	return nil
}

func validateCandidateEdges(bindings []BindingRequirement, edges []CandidateEdge) error {
	if edges == nil || len(edges) > len(bindings)*maximumOperationsItems {
		return fmt.Errorf("%w: candidate edge set is invalid", ErrInvalidRequest)
	}
	bindingsByName := make(map[string]struct{}, len(bindings))
	for _, binding := range bindings {
		bindingsByName[binding.LogicalAgentName] = struct{}{}
	}
	seen := make(map[candidateEdgeKey]struct{}, len(edges))
	for _, edge := range edges {
		if _, ok := bindingsByName[edge.LogicalAgentName]; !ok ||
			strings.TrimSpace(edge.RuntimeAgentInstanceID) == "" ||
			validateAuthenticatedPrincipal(AuthenticatedPrincipal{
				RuntimeAgentID: edge.RuntimeAgentID, Labels: []string{},
				LabelRevision: edge.RuntimeAgentLabelRevision,
			}) != nil {
			return fmt.Errorf("%w: candidate edge identity is invalid", ErrInvalidRequest)
		}
		key := candidateEdgeKey{edge.LogicalAgentName, edge.RuntimeAgentInstanceID}
		if _, duplicate := seen[key]; duplicate {
			return fmt.Errorf("%w: candidate edge is duplicated", ErrInvalidRequest)
		}
		seen[key] = struct{}{}
		previous := runtimesettings.RuntimeAdapterRef("")
		for _, adapter := range edge.RequiredRuntimeAdapters {
			if adapter.Validate() != nil || adapter <= previous {
				return fmt.Errorf("%w: candidate adapter requirements are invalid", ErrInvalidRequest)
			}
			previous = adapter
		}
	}
	return nil
}

func containsRuntimeAdapters(
	available []runtimesettings.RuntimeAdapterRef,
	required []runtimesettings.RuntimeAdapterRef,
) bool {
	availableIndex := 0
	for _, expected := range required {
		for availableIndex < len(available) && available[availableIndex] < expected {
			availableIndex++
		}
		if availableIndex == len(available) || available[availableIndex] != expected {
			return false
		}
	}
	return true
}

// validateBindingContextManagement checks the effective branch against the
// binding's immutable selection before any candidate is reserved (WRK-R184).
func validateBindingContextManagement(binding BindingRequirement) error {
	context := contracts.AllocationContext{
		Template:    binding.AgentTemplate,
		SessionMode: binding.WorkerSessionMode,
	}
	if binding.RuntimeSelection != nil {
		context.WorkerPolicy = binding.RuntimeSelection.ModelPolicy
		context.SummarizerModelPolicy = binding.RuntimeSelection.SummarizerModelPolicy
		context.CompactorModelPolicy = binding.RuntimeSelection.CompactorModelPolicy
	}
	return binding.ContextManagement.ValidateAllocation(context)
}

func normalizeReservationRequest(request ReservationRequest) (string, []BindingRequirement, error) {
	if request.ProjectID != "" && !contracts.ValidResourceID(request.ProjectID) {
		return "", nil, fmt.Errorf("%w: Project identity is invalid", ErrInvalidRequest)
	}
	if err := contracts.ValidatePlacementLabels(request.RequiredPlacementLabels); err != nil {
		return "", nil, fmt.Errorf("%w: required placement labels are invalid: %v", ErrInvalidRequest, err)
	}
	if strings.TrimSpace(request.RunID) == "" || strings.TrimSpace(request.StageExecutionID) == "" || len(request.Bindings) == 0 {
		return "", nil, fmt.Errorf("%w: run, StageExecution, and bindings are required", ErrInvalidRequest)
	}
	if request.RuntimeConfig != nil {
		if err := request.RuntimeConfig.Validate(); err != nil {
			return "", nil, fmt.Errorf("%w: Run RuntimeConfig snapshot is invalid", ErrInvalidRequest)
		}
	}
	metadataLabels, err := runlabels.NormalizeRunMetadataLabels(request.RunMetadataLabels)
	if err != nil {
		return "", nil, fmt.Errorf("%w: Run metadata labels are invalid", ErrInvalidRequest)
	}
	bindings := make([]BindingRequirement, len(request.Bindings))
	seen := make(map[string]struct{}, len(request.Bindings))
	for index, binding := range request.Bindings {
		// Operations shows the logical Agent name as a ConfigId (WFL-R84), so
		// placement never reserves a name it could not show.
		if !contracts.ValidName(binding.LogicalAgentName) || contracts.ValidateArtifactName(binding.Namespace) != nil {
			return "", nil, fmt.Errorf("%w: valid logical Agent name and Artifact namespace are required", ErrInvalidRequest)
		}
		if binding.Namespace == "inputs" || binding.Namespace == "outputs" {
			return "", nil, fmt.Errorf("%w: Agent binding cannot use a Run-reserved namespace", ErrInvalidRequest)
		}
		if err := resolved.ValidateWorkerCompletionSelection(binding.CompletionContract, binding.Namespace, binding.AgentTemplate, binding.ContextManagement); err != nil {
			return "", nil, fmt.Errorf("%w: %v", ErrInvalidRequest, err)
		}
		if err := binding.AgentTemplate.Validate(); err != nil {
			return "", nil, fmt.Errorf("%w: invalid AgentTemplate for %q: %v", ErrInvalidRequest, binding.LogicalAgentName, err)
		}
		if err := binding.WorkerSessionMode.Validate(); err != nil {
			return "", nil, fmt.Errorf("%w: invalid Worker session mode for %q", ErrInvalidRequest, binding.LogicalAgentName)
		}
		if err := validateBindingContextManagement(binding); err != nil {
			return "", nil, fmt.Errorf("%w: invalid context management for %q: %v", ErrInvalidRequest, binding.LogicalAgentName, err)
		}
		resolvedSkills := binding.ResolvedSkills
		if resolvedSkills == nil && len(binding.AgentTemplate.Skills) == 0 {
			resolvedSkills = []contracts.ResolvedSkill{}
		}
		if err := contracts.ValidateResolvedSkills(binding.AgentTemplate, resolvedSkills); err != nil {
			return "", nil, fmt.Errorf("%w: invalid resolved Skills for %q: %v", ErrInvalidRequest, binding.LogicalAgentName, err)
		}
		if err := binding.ExecutionConfig.Validate(); err != nil {
			return "", nil, fmt.Errorf("%w: invalid execution config for %q: %v", ErrInvalidRequest, binding.LogicalAgentName, err)
		}
		if binding.AgentTemplate.IsToolWorker() != (binding.ExecutionConfig == (AllocationExecutionConfig{})) {
			return "", nil, fmt.Errorf("%w: execution config does not match Worker runtime", ErrInvalidRequest)
		}
		if request.ModelFree && !binding.AgentTemplate.IsToolWorker() {
			return "", nil, fmt.Errorf("%w: a model-free Stage binds only tool Workers", ErrInvalidRequest)
		}
		if binding.Workspace != nil {
			if err := binding.Workspace.Validate(); err != nil {
				return "", nil, fmt.Errorf("%w: invalid workspace for %q: %v", ErrInvalidRequest, binding.LogicalAgentName, err)
			}
		}
		if (request.RuntimeConfig == nil) != (binding.RuntimeSelection == nil) {
			return "", nil, fmt.Errorf("%w: candidate Runtime selection is incomplete", ErrInvalidRequest)
		}
		if binding.RuntimeSelection != nil {
			if err := validateRuntimeSelection(*binding.RuntimeSelection, binding.AgentTemplate.IsToolWorker()); err != nil {
				return "", nil, fmt.Errorf("%w: invalid Runtime selection for %q", ErrInvalidRequest, binding.LogicalAgentName)
			}
		}
		if _, duplicate := seen[binding.LogicalAgentName]; duplicate {
			return "", nil, fmt.Errorf("%w: duplicate logical Agent name", ErrInvalidRequest)
		}
		seen[binding.LogicalAgentName] = struct{}{}
		bindings[index] = BindingRequirement{
			CompletionContract: contracts.CloneWorkerCompletionContract(binding.CompletionContract),
			LogicalAgentName:   binding.LogicalAgentName, Namespace: binding.Namespace,
			WorkerSessionMode: binding.WorkerSessionMode,
			ContextManagement: binding.ContextManagement.Clone(),
			AgentTemplate:     binding.AgentTemplate.Clone(),
			ResolvedSkills:    contracts.CloneResolvedSkills(resolvedSkills),
			ExecutionConfig:   cloneAllocationExecutionConfig(binding.ExecutionConfig),
			RuntimeSelection:  binding.RuntimeSelection, // Borrowed only for the synchronous fingerprint.
			Workspace:         contracts.CloneAllocationWorkspaceSpec(binding.Workspace),
		}
	}
	sort.Slice(bindings, func(i, j int) bool { return bindings[i].LogicalAgentName < bindings[j].LogicalAgentName })
	encoded, err := json.Marshal(struct {
		RequiredPlacementLabels []string
		RunID                   string
		StageExecutionID        string
		RunMetadataLabels       runlabels.RunMetadataLabels
		ProjectID               string
		Bindings                []BindingRequirement
		RuntimeConfig           *runtimeconfig.RunSnapshot
	}{
		RequiredPlacementLabels: append([]string{}, request.RequiredPlacementLabels...),
		RunID:                   request.RunID,
		StageExecutionID:        request.StageExecutionID,
		RunMetadataLabels:       metadataLabels,
		ProjectID:               request.ProjectID,
		Bindings:                bindings,
		RuntimeConfig:           reservationRuntimeSnapshot(request.RuntimeConfig),
	})
	if err != nil {
		return "", nil, fmt.Errorf("encode reservation request: %w", err)
	}
	return contentdigest.Bytes(encoded), bindings, nil
}

func validateAuthenticatedPrincipal(principal AuthenticatedPrincipal) error {
	if !contracts.ValidRuntimeAgentID(principal.RuntimeAgentID) || principal.LabelRevision == 0 ||
		runtimeconfig.ValidateAgentLabels(principal.Labels) != nil {
		return fmt.Errorf("%w: authenticated Runtime Agent principal is invalid", ErrInvalidRequest)
	}
	return nil
}

func validateRuntimeSelection(value resolved.ResolvedConsumerExecutionConfig, toolWorker bool) error {
	if toolWorker {
		if !value.ModelPolicy.IsZero() || value.SummarizerModelPolicy != nil || value.LLMGateway != nil || value.Credential != nil || value.Origins != (resolved.ExecutionConfigOrigins{}) {
			return ErrInvalidRequest
		}
		return nil
	}
	if err := value.ModelPolicy.Validate(); err != nil || strings.TrimSpace(value.Origins.ModelPolicy) == "" {
		return ErrInvalidRequest
	}
	if value.SummarizerModelPolicy != nil {
		if err := value.SummarizerModelPolicy.ValidateForWorkerSummarizer(); err != nil || strings.TrimSpace(value.Origins.SummarizerModelPolicy) == "" {
			return ErrInvalidRequest
		}
	}
	if value.LLMGateway != nil {
		if err := value.LLMGateway.Validate(); err != nil || strings.TrimSpace(value.Origins.LLMGateway) == "" {
			return ErrInvalidRequest
		}
	}
	if value.Credential != nil {
		if err := value.Credential.Validate(); err != nil || strings.TrimSpace(value.Origins.Credential) == "" {
			return ErrInvalidRequest
		}
	}
	return nil
}
