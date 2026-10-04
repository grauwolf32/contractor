package planner

import (
	"context"
	"errors"
	"time"

	workflowconfig "github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/telemetry"
)

// DefaultCompletionWriteTimeout covers the default two-second database acquire
// and lock waits plus a margin when no application settings are supplied.
const DefaultCompletionWriteTimeout = 5 * time.Second

const maxCompletionWriteAttempts = 3

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

// RecoverCompletion replays a completion recorded by an earlier invocation of
// the same Stage execution. A recorded failure ends this invocation with that
// failure; a recorded result is revalidated against the result contract.
func RecoverCompletion(
	ctx context.Context,
	runID string,
	contract map[string]workflowconfig.ArtifactSlot,
	completion Completion,
	inspector ArtifactInspector,
) (contracts.StageContentResult, error) {
	if err := validateCompletion(completion); err != nil {
		return contracts.StageContentResult{}, NewError(
			"planner_session_invalid", "Recorded Planner completion is invalid", false, err,
		)
	}
	if completion.Failure != nil {
		return contracts.StageContentResult{}, NewErrorFromFailure(*completion.Failure, nil)
	}
	result := completion.Result.Clone()
	if err := validateCandidate(ctx, runID, contract, result, inspector); err != nil {
		return contracts.StageContentResult{}, err
	}
	return result, nil
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
	return writeCompletion(ctx, sessions, func(recordContext context.Context) error {
		return sessions.Complete(recordContext, identity, completion)
	})
}

// CompleteScanSession records a scan completion with the same bounded write
// and transient retry as CompleteSession.
func CompleteScanSession(
	ctx context.Context, sessions ScanSessionService, identity ScanSessionIdentity, completion Completion,
) error {
	return writeCompletion(ctx, sessions, func(recordContext context.Context) error {
		return sessions.CompleteScan(recordContext, identity, completion)
	})
}

// writeCompletion retries a transient database failure within one completion
// budget. Both session writes are idempotent: repeating a committed
// completion succeeds, which a lost commit acknowledgement requires.
func writeCompletion(ctx context.Context, sessions any, write func(context.Context) error) error {
	budget := DefaultCompletionWriteTimeout
	if configured, ok := sessions.(interface{ CompletionWriteTimeout() time.Duration }); ok {
		if timeout := configured.CompletionWriteTimeout(); timeout > 0 {
			budget = timeout
		}
	}
	recordContext, cancel := context.WithTimeout(context.WithoutCancel(ctx), budget)
	defer cancel()
	var lastError error
	for attempt := range maxCompletionWriteAttempts {
		err := write(recordContext)
		if err == nil {
			return nil
		}
		lastError = err
		if attempt == maxCompletionWriteAttempts-1 || !persistencepostgres.IsTransientFailure(err) ||
			recordContext.Err() != nil {
			return err
		}
		timer := time.NewTimer(time.Duration(attempt+1) * 50 * time.Millisecond)
		select {
		case <-recordContext.Done():
			timer.Stop()
			return err
		case <-timer.C:
		}
	}
	return lastError
}

// CompletionWriteError reports an unwritten failure without changing the
// original Planner failure's retry policy.
func CompletionWriteError(original *Error, cause error) *Error {
	return NewError(
		"planner_session_unavailable",
		"Planner durable session is unavailable during record failure",
		original.Retryable,
		errors.Join(original, cause),
	)
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
