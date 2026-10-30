package scheduler

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"slices"

	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/controlplane"
	"github.com/grauwolf32/contractor/internal/gatewayrecovery"
	"github.com/grauwolf32/contractor/internal/runstore"
)

func reservationModelRoutes(ownerID string, reservation controlplane.Reservation) []gatewayrecovery.Route {
	runtimeConfig := reservation.ResolvedRuntimeConfig
	if runtimeConfig == nil || runtimeConfig.ModelFree {
		return nil
	}
	route := gatewayrecovery.Route{OwnerID: ownerID, GatewayDigest: runtimeConfig.LLMGateway.Ref.Digest, Model: runtimeConfig.ModelPolicy.Model}
	if runtimeConfig.LLMCredential != nil {
		route.CredentialID = runtimeConfig.LLMCredential.CredentialID
	}
	if proxy := runtimeConfig.HTTPProxy; proxy != nil && slices.Contains(proxy.Targets, "llm-gateway") {
		encoded, _ := json.Marshal(proxy)
		digest := sha256.Sum256(encoded)
		route.TransportDigest = hex.EncodeToString(digest[:])
	}
	routes := []gatewayrecovery.Route{route}
	// Auxiliary consumers share the Worker route except for their model alias.
	for _, auxiliary := range []*contracts.ResolvedModelPolicy{runtimeConfig.SummarizerModelPolicy, runtimeConfig.CompactorModelPolicy} {
		if auxiliary == nil || slices.ContainsFunc(routes, func(existing gatewayrecovery.Route) bool {
			return existing.Model == auxiliary.Model
		}) {
			continue
		}
		next := route
		next.Model = auxiliary.Model
		routes = append(routes, next)
	}
	return routes
}
func (s *Scheduler) admitModelRoutes(ctx context.Context, run runstore.WorkflowRun, stage resolved.ResolvedStage, reservations []controlplane.Reservation) (bool, error) {
	if s.options.GatewayRecovery == nil {
		return true, nil
	}
	var routes []gatewayrecovery.Route
	if route := plannerModelRoute(run.OwnerID, stage); route != nil {
		routes = append(routes, *route)
	}
	for _, reservation := range reservations {
		routes = append(routes, reservationModelRoutes(run.OwnerID, reservation)...)
	}
	return s.options.GatewayRecovery.Admit(ctx, run.RunID, routes)
}
func (s *Scheduler) bindModelRoutes(ctx context.Context, run runstore.WorkflowRun, reservations []controlplane.Reservation) error {
	if s.options.GatewayRecovery == nil {
		return nil
	}
	for _, reservation := range reservations {
		for _, route := range reservationModelRoutes(run.OwnerID, reservation) {
			if err := s.options.GatewayRecovery.Bind(ctx, reservation.Grant.AllocationID, run.RunID, route); err != nil {
				return err
			}
		}
	}
	return nil
}

func plannerModelRoute(ownerID string, stage resolved.ResolvedStage) *gatewayrecovery.Route {
	selection := stage.ExecutionConfig.Planner
	if selection == nil || selection.LLMGateway == nil {
		return nil
	}
	route := gatewayrecovery.Route{OwnerID: ownerID, GatewayDigest: selection.LLMGateway.Ref.Digest, Model: selection.ModelPolicy.Model}
	if selection.Credential != nil {
		route.CredentialID = selection.Credential.CredentialID
	}
	return &route
}
