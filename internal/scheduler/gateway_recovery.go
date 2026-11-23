package scheduler

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"slices"
	"time"

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
func (s *Scheduler) admitModelRoutes(ctx context.Context, run runstore.WorkflowRun, stage resolved.ResolvedStage, stageID string, reservations []controlplane.Reservation) (bool, error) {
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
	allowed, err := s.options.GatewayRecovery.Admit(ctx, run.RunID, stageID, routes)
	if err != nil || allowed {
		return allowed, err
	}
	status, err := s.options.GatewayRecovery.ReadStatus(ctx, run.RunID, stageID)
	if err == nil && status != nil {
		err = s.persistence.ReconcileModelWait(ctx, run.RunID, stageID, status.Code)
	}
	return false, err
}
func (s *Scheduler) bindModelRoutes(ctx context.Context, run runstore.WorkflowRun, reservations []controlplane.Reservation) error {
	if s.options.GatewayRecovery == nil {
		return nil
	}
	for _, reservation := range reservations {
		for _, route := range reservationModelRoutes(run.OwnerID, reservation) {
			bindContext, cancel := context.WithTimeout(ctx, s.options.OperationTimeout)
			err := s.options.GatewayRecovery.Bind(bindContext, reservation.Grant.AllocationID, run.RunID, reservation.Grant.StageExecutionID, route)
			cancel()
			if err != nil {
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

// monitorModelWaits owns the durable projection during one live invocation.
// Updates coalesce on a per-Run broadcast. A failed/ambiguous write is retried
// from the latest memory snapshot; it never makes a model request repeat.
func (s *Scheduler) monitorModelWaits(ctx context.Context, run runstore.WorkflowRun, stageID string) func() {
	if s.options.GatewayRecovery == nil {
		return func() {}
	}
	monitorContext, cancel := context.WithCancel(ctx)
	done := make(chan struct{})
	go func() {
		defer close(done)
		last := ""
		dirty := false
		if run.State == runstore.RunWaiting {
			last = run.StateReason.Code
		}
		for {
			code, changed := s.options.GatewayRecovery.Waiting(run.RunID, stageID)
			var retry <-chan time.Time
			if code != last || dirty {
				operation, stop := context.WithTimeout(monitorContext, s.options.OperationTimeout)
				err := s.persistence.ReconcileModelWait(operation, run.RunID, stageID, code)
				stop()
				if err == nil {
					last = code
					dirty = false
				} else if monitorContext.Err() != nil {
					return
				} else {
					dirty = true
					if errors.Is(err, ErrClaimLost) || errors.Is(err, runstore.ErrConflict) {
						return
					}
					s.options.Logger.Warn("Run model wait needs reconciliation", "run_id", run.RunID, "error", err)
					retry = s.after(s.options.PollInterval)
				}
			}
			select {
			case <-monitorContext.Done():
				return
			case <-changed:
			case <-retry:
			}
		}
	}()
	return func() { cancel(); <-done }
}

func (s *Scheduler) endModelRoutes(runID, stageID string) {
	if s.options.GatewayRecovery != nil {
		s.options.GatewayRecovery.EndStage(runID, stageID)
	}
}
