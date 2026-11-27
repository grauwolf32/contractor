package session

import (
	"context"
	"errors"
	"fmt"

	"github.com/grauwolf32/contractor/internal/planner"
	"github.com/grauwolf32/contractor/internal/runstore"
)

const maxADKAppendAttempts = 4

func (s *Service) RecordActivity(
	ctx context.Context, identity planner.SessionIdentity, facts planner.Activity,
) error {
	if err := facts.Validate(); err != nil {
		return fmt.Errorf("%v: %w", err, runstore.ErrInvalid)
	}

	for attempt := 0; attempt < maxADKAppendAttempts; attempt++ {
		stored, state, err := s.load(ctx, identity)
		if err != nil {
			return err
		}
		if state.Status != statusRunning {
			return fmt.Errorf("Planner session is not running")
		}
		next := state
		next.NextSequence++
		next.ADKEventCount++
		encodedState, err := encodeState(next)
		if err != nil {
			return err
		}
		err = s.append(
			ctx, stored, state.NextSequence, encodedState,
			identity, planner.PlannerEventActivity,
			&plannerRunEventData{Activity: &facts},
		)
		if err == nil {
			return nil
		}
		if !errors.Is(err, runstore.ErrConflict) {
			return fmt.Errorf("append redacted ADK event: %w", err)
		}
	}
	return fmt.Errorf("append redacted ADK event: %w", runstore.ErrConflict)
}
