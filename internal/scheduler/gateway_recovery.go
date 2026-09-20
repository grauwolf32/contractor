package scheduler

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"slices"

	workflowconfig "github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/controlplane"
	"github.com/grauwolf32/contractor/internal/gatewayrecovery"
	"github.com/grauwolf32/contractor/internal/runstore"
)

func reservationModelRoutes(ownerID string, reservation controlplane.Reservation) []gatewayrecovery.Route {
	resolved := reservation.ResolvedRuntimeConfig
	if resolved == nil || resolved.ModelFree {
		return nil
	}
	route := gatewayrecovery.Route{OwnerID: ownerID, GatewayDigest: resolved.LLMGateway.Ref.Digest, Model: resolved.ModelPolicy.Model}
	if resolved.LLMCredential != nil {
		route.CredentialID = resolved.LLMCredential.CredentialID
	}
	if proxy := resolved.HTTPProxy; proxy != nil && slices.Contains(proxy.Targets, "llm-gateway") {
		encoded, _ := json.Marshal(proxy)
		digest := sha256.Sum256(encoded)
		route.TransportDigest = hex.EncodeToString(digest[:])
	}
	routes := []gatewayrecovery.Route{route}
	if summarizer := reservation.AgentTemplate.Summarizer; summarizer != nil && summarizer.ModelPolicy.Model != route.Model {
		route.Model = summarizer.ModelPolicy.Model
		routes = append(routes, route)
	}
	return routes
}
func (s *Scheduler) admitModelRoutes(ctx context.Context, run runstore.WorkflowRun, stage workflowconfig.ResolvedStage, reservations []controlplane.Reservation) (bool, error) {
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

func plannerModelRoute(ownerID string, stage workflowconfig.ResolvedStage) *gatewayrecovery.Route {
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
