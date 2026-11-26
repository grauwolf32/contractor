package controlplane

import (
	"context"
	"errors"
	"fmt"
	"slices"

	"github.com/grauwolf32/contractor/internal/agentprincipal"
	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
)

// candidateEdges resolves every compatible binding and candidate pair outside
// the Registry lock, sharing the attempt's observations, and returns the edges
// with the Runtime configuration resolved for each. The returned resolutions
// may share memory with the attempt cache and must not be mutated.
func (a *PlacementAllocator) candidateEdges(
	ctx context.Context,
	request ReservationRequest,
	candidates []AgentSnapshot,
	attempt *placementAttemptCache,
) ([]CandidateEdge, map[string]runtimeconfig.ResolvedRuntimeConfig, error) {
	candidates = matchingLocationCandidates(request.RequiredPlacementLabels, candidates)
	capacity := 0
	for _, candidate := range candidates {
		capacity += candidate.Registration.AllocationSlots
	}
	if len(request.Bindings) > capacity {
		return []CandidateEdge{}, nil, nil
	}
	toolCapabilities := make([]toolCapabilityIndex, len(candidates))
	for index, candidate := range candidates {
		toolCapabilities[index] = indexToolCapabilities(candidate.Registration.SupportedToolsets)
	}
	optimistic := make(map[string]runtimeconfig.ResolvedRuntimeConfig)
	lookups := placementCredentialLookups{
		llm: a.llmCredentials, runtime: a.runtimeCredentials,
		llmCache: attempt.llmCredentials, runtimeCache: attempt.runtimeCredentials,
	}
	edges := make([]CandidateEdge, 0, len(request.Bindings)*len(candidates))
	for _, binding := range request.Bindings {
		toolAdapters, err := resolved.RequiredRuntimeAdaptersForTemplate(binding.AgentTemplate)
		if err != nil {
			return nil, nil, fmt.Errorf("%w: AgentTemplate infrastructure requirements are invalid", ErrInvalidRequest)
		}
		for candidateIndex, candidate := range candidates {
			if !isBindingCompatibleWithTools(candidate.Registration, binding, toolCapabilities[candidateIndex]) {
				continue
			}
			runtimeConfig, err := a.resolveAttemptCandidate(ctx, request, binding, candidate.Principal, lookups, attempt)
			if err != nil {
				if contextError := ctx.Err(); contextError != nil {
					return nil, nil, contextError
				}
				if candidateResolutionUnavailable(err) {
					continue
				}
				return nil, nil, err
			}
			if !containsRuntimeAdapters(candidate.Registration.SupportedRuntimeAdapters, runtimeConfig.RequiredRuntimeAdapters) || !containsRuntimeAdapters(candidate.Registration.SupportedRuntimeAdapters, request.additionalAdapters) {
				continue
			}
			if !containsRuntimeAdapters(runtimeConfig.RequiredRuntimeAdapters, toolAdapters) {
				continue
			}
			key := placementResolutionKey(binding.LogicalAgentName, candidate.Registration.InstanceID)
			optimistic[key] = runtimeConfig
			requiredAdapters := append(slices.Clone(runtimeConfig.RequiredRuntimeAdapters), request.additionalAdapters...)
			slices.Sort(requiredAdapters)
			requiredAdapters = slices.Compact(requiredAdapters)
			edges = append(edges, CandidateEdge{
				LogicalAgentName: binding.LogicalAgentName, RuntimeAgentID: candidate.Principal.RuntimeAgentID,
				RuntimeAgentInstanceID:    candidate.Registration.InstanceID,
				RuntimeAgentLabelRevision: candidate.Principal.LabelRevision,
				RequiredRuntimeAdapters:   requiredAdapters,
			})
		}
	}
	return edges, optimistic, nil
}

// matchingLocationCandidates filters immutable declarations before candidate
// configuration reads. A wrong-location principal cannot become eligible through
// a mutable RuntimeConfig-label refresh, so capacity retries never query it.
func matchingLocationCandidates(required []string, candidates []AgentSnapshot) []AgentSnapshot {
	if len(required) == 0 {
		return candidates
	}
	matching := make([]AgentSnapshot, 0, len(candidates))
	for _, candidate := range candidates {
		if contracts.PlacementLabelsMatch(required, candidate.Registration.PlacementLabels) {
			matching = append(matching, candidate)
		}
	}
	return matching
}

func (a *PlacementAllocator) refreshCandidatePrincipals(ctx context.Context, candidates []AgentSnapshot) (bool, error) {
	repository := agentprincipal.NewRepository(a.pool)
	seen := make(map[string]struct{}, len(candidates))
	changed := false
	for _, candidate := range candidates {
		id := candidate.Principal.RuntimeAgentID
		if _, duplicate := seen[id]; duplicate {
			continue
		}
		seen[id] = struct{}{}
		current, err := repository.Get(ctx, id)
		if errors.Is(err, runtimeconfig.ErrNotFound) {
			changed = a.registry.MarkPrincipalMissing(id) || changed
			continue
		}
		if err != nil {
			return false, err
		}
		principal := AuthenticatedPrincipal{
			RuntimeAgentID: id, Labels: current.Labels, LabelRevision: current.LabelRevision,
		}
		if equalAuthenticatedPrincipal(candidate.Principal, principal) {
			continue
		}
		if err := a.registry.ApplyPrincipalLabels(principal); err != nil {
			return false, err
		}
		changed = true
	}
	return changed, nil
}

// holdForWaitingModelStage holds the live Agents that a model Stage waiting for
// allocated capacity could use (CTL-R155). The hold is advisory: an error
// while resolving it holds nothing and never changes the Stage's own
// placement outcome, and the Stage's next failed attempt computes it again.
// Live candidates that the failed passes already resolved reuse the attempt's
// resolutions; only allocated or newly live candidates are resolved here.
func (a *PlacementAllocator) holdForWaitingModelStage(
	ctx context.Context,
	request ReservationRequest,
	attempt *placementAttemptCache,
) {
	if request.RuntimeConfig == nil {
		return
	}
	edges, _, err := a.candidateEdges(ctx, request, a.registry.liveCandidates(), attempt)
	if err != nil {
		return
	}
	_ = a.registry.holdForModelStage(request, edges)
}

func placementResolutionKey(logicalAgentName, instanceID string) string {
	return logicalAgentName + "\x00" + instanceID
}
