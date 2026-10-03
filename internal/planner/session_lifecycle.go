package planner

import (
	"context"
	"errors"

	"github.com/grauwolf32/contractor/internal/telemetry"
)

// StartSession begins or resumes the durable Planner session for one Stage
// execution inside a session.begin span. A recorded completion is returned in
// start.Completion for the caller to replay. A non-nil error means the caller
// must not invoke Workers; start.Identity is still set when the session exists.
func StartSession(
	ctx context.Context,
	sessions SessionService,
	instrumentation telemetry.PlannerInstrumentation,
	stageExecutionID string,
) (SessionStart, error) {
	span := instrumentation.StartSpan(
		telemetry.PlannerSpanSession,
		telemetry.PlannerSpanAttributes{Operation: "session.begin"},
	)
	start, err := sessions.Begin(ctx, stageExecutionID)
	if err != nil {
		span.End("unavailable", telemetry.PlannerSpanAttributes{ErrorCode: "planner_session_unavailable"})
		return SessionStart{}, SessionError("start", err)
	}
	if start.Completion != nil {
		span.End("recovered", telemetry.PlannerSpanAttributes{SessionID: start.Identity.SessionID})
		return start, nil
	}
	if !start.Invoke {
		span.End("rejected", telemetry.PlannerSpanAttributes{
			SessionID: start.Identity.SessionID, ErrorCode: "planner_session_invalid",
		})
		return start, NewError(
			"planner_session_invalid", "Planner session did not grant invocation ownership", false, nil,
		)
	}
	span.End("succeeded", telemetry.PlannerSpanAttributes{SessionID: start.Identity.SessionID})
	return start, nil
}

// RecordSessionRequest durably records bounded request facts inside a
// session.record_request span.
func RecordSessionRequest(
	ctx context.Context,
	sessions SessionService,
	instrumentation telemetry.PlannerInstrumentation,
	identity SessionIdentity,
	facts RequestFacts,
) error {
	span := instrumentation.StartSpan(
		telemetry.PlannerSpanSession,
		telemetry.PlannerSpanAttributes{Operation: "session.record_request", SessionID: identity.SessionID},
	)
	if err := sessions.RecordRequest(ctx, identity, facts); err != nil {
		span.End("unavailable", telemetry.PlannerSpanAttributes{ErrorCode: "planner_session_unavailable"})
		return SessionError("record request", err)
	}
	span.End("succeeded", telemetry.PlannerSpanAttributes{})
	return nil
}

// CompleteSession records a completion with a bounded write that outlives
// caller cancellation, so a cancelled invocation still settles its session.
func CompleteSession(
	ctx context.Context, sessions SessionService, identity SessionIdentity, completion Completion,
) error {
	recordContext, cancel := context.WithTimeout(context.WithoutCancel(ctx), completionWriteTimeout)
	defer cancel()
	return sessions.Complete(recordContext, identity, completion)
}

// SessionError classifies a durable session failure during operation.
func SessionError(operation string, cause error) *Error {
	if errors.Is(cause, ErrInvocationInProgress) {
		return NewError(
			"planner_invocation_in_progress",
			"Planner invocation is already in progress and cannot be resumed",
			true,
			cause,
		)
	}
	return NewError(
		"planner_session_unavailable",
		"Planner durable session is unavailable during "+operation,
		true,
		cause,
	)
}
