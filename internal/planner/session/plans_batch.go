package session

import (
	"context"
	"fmt"
	"strings"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/planner"
	"github.com/grauwolf32/contractor/internal/runstore"
)

// LoadPlans returns projections keyed by StageExecution ID; absent plans are
// omitted, but missing sessions or mismatched invocation ownership still fail.
func (s *Service) LoadPlans(ctx context.Context, identities []planner.SessionIdentity) (map[string]planner.PlannerPlanProjection, error) {
	result := make(map[string]planner.PlannerPlanProjection, len(identities))
	if len(identities) == 0 {
		return result, nil
	}
	ids := make([]string, len(identities))
	for i, identity := range identities {
		if strings.TrimSpace(identity.SessionID) == "" || strings.TrimSpace(identity.StageExecutionID) == "" || strings.TrimSpace(identity.InvocationID) == "" {
			return nil, fmt.Errorf("complete session identity is required")
		}
		ids[i] = identity.SessionID
	}
	sessions, err := s.store.GetPlannerSessions(ctx, ids)
	if err != nil {
		return nil, fmt.Errorf("load Planner sessions: %w", err)
	}
	for _, identity := range identities {
		session, ok := sessions[identity.SessionID]
		if !ok {
			return nil, fmt.Errorf("load Planner session: %w", runstore.ErrNotFound)
		}
		if session.StageExecutionID != identity.StageExecutionID || session.InvocationID != identity.InvocationID || session.StateSchemaVersion != contracts.APIVersion {
			return nil, fmt.Errorf("Planner session identity differs")
		}
		state, err := decodeState(session.State)
		if err != nil {
			return nil, err
		}
		if state.Plan == nil {
			continue
		}
		projection := clonePlanProjection(*state.Plan)
		if err := projection.Validate(); err != nil {
			return nil, fmt.Errorf("persisted Planner plan is invalid: %w", err)
		}
		result[identity.StageExecutionID] = projection
	}
	return result, nil
}
