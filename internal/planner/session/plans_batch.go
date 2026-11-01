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
	expected := make(map[string]planner.SessionIdentity, len(identities))
	for i, identity := range identities {
		if strings.TrimSpace(identity.SessionID) == "" || strings.TrimSpace(identity.StageExecutionID) == "" || strings.TrimSpace(identity.InvocationID) == "" {
			return nil, fmt.Errorf("complete session identity is required")
		}
		ids[i] = identity.SessionID
		if previous, exists := expected[identity.SessionID]; exists && previous != identity {
			return nil, fmt.Errorf("Planner session identity differs")
		}
		expected[identity.SessionID] = identity
	}
	err := s.store.ReadPlannerSessions(ctx, ids, func(session runstore.PlannerSession) error {
		identity, ok := expected[session.SessionID]
		if !ok {
			return fmt.Errorf("unexpected Planner session identity")
		}
		if session.StageExecutionID != identity.StageExecutionID || session.InvocationID != identity.InvocationID || session.StateSchemaVersion != contracts.APIVersion {
			return fmt.Errorf("Planner session identity differs")
		}
		state, err := decodeState(session.State)
		if err != nil {
			return err
		}
		delete(expected, session.SessionID)
		if state.Plan == nil {
			return nil
		}
		projection := canonicalPlanProjection(*state.Plan)
		if err := projection.Validate(); err != nil {
			return fmt.Errorf("persisted Planner plan is invalid: %w", err)
		}
		result[identity.StageExecutionID] = projection
		return nil
	})
	if err != nil {
		return nil, fmt.Errorf("load Planner sessions: %w", err)
	}
	if len(expected) > 0 {
		return nil, fmt.Errorf("load Planner session: %w", runstore.ErrNotFound)
	}
	return result, nil
}
