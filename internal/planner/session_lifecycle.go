package planner

import (
	"context"
	"errors"
	"time"

	"github.com/grauwolf32/contractor/internal/config/resolved"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/stageresult"
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
		sessionError := SessionError("start", err)
		EndSessionSpan(span, sessionError)
		return SessionStart{}, sessionError
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
	contract map[string]resolved.ArtifactSlot,
	completion Completion,
	inspector ArtifactInspector,
) (stageresult.StageContentResult, error) {
	if err := validateCompletion(completion); err != nil {
		return stageresult.StageContentResult{}, NewError(
			"planner_session_invalid", "Recorded Planner completion is invalid", false, err,
		)
	}
	if completion.Failure != nil {
		return stageresult.StageContentResult{}, NewErrorFromFailure(*completion.Failure, nil)
	}
	result := completion.Result.Clone()
	if err := validateCandidate(ctx, runID, contract, result, inspector); err != nil {
		return stageresult.StageContentResult{}, err
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
		sessionError := SessionError("record request", err)
		EndSessionSpan(span, sessionError)
		return sessionError
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

// CompletionWriteError reports an outcome that the Planner could not record
// in its session (PLN-R106): the session error of the write, which never
// makes a final outcome retryable. A non-retryable Planner failure or a
// failed candidate with a non-retryable error, a summarized Worker result
// included, keeps retryable false, so a Stage retry cannot invoke again a
// Worker whose failure was final.
func CompletionWriteError(completion Completion, cause error) *Error {
	written := SessionError("record completion", cause)
	if !completionRetryable(completion) {
		written.Retryable = false
	}
	return written
}

// completionRetryable reports whether another attempt may change the
// outcome a completion records: a successful candidate always, a Planner
// failure or failed candidate only when its error is retryable.
func completionRetryable(completion Completion) bool {
	switch {
	case completion.Failure != nil:
		return completion.Failure.Retryable
	case completion.Result != nil && completion.Result.Error != nil:
		return completion.Result.Error.Retryable
	default:
		return true
	}
}

// SessionError classifies a durable session failure during operation: a
// session that another invocation still runs is
// planner_invocation_in_progress (PLN-R107), a write the session store
// refused, such as a state over its stored size bound, is the non-retryable
// planner_session_rejected, and every other failure the retryable
// planner_session_unavailable (PLN-R106).
func SessionError(operation string, cause error) *Error {
	if errors.Is(cause, ErrInvocationInProgress) {
		return NewError(
			"planner_invocation_in_progress",
			"Planner invocation is already in progress and cannot be resumed",
			true,
			cause,
		)
	}
	if SessionWriteRejected(cause) {
		return NewError(
			"planner_session_rejected",
			"Planner durable session rejected the write during "+operation,
			false,
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

// EndSessionSpan ends a Planner session span with the code of the session
// error the Planner returns: unavailable for planner_session_unavailable and
// rejected for a session that refused the operation.
func EndSessionSpan(span telemetry.PlannerSpan, sessionError *Error) {
	outcome := "rejected"
	if sessionError.Code == "planner_session_unavailable" {
		outcome = "unavailable"
	}
	span.End(outcome, telemetry.PlannerSpanAttributes{ErrorCode: sessionError.Code})
}

// SessionWriteRejected reports a session write that the store refused as
// invalid, such as a value over a stored size bound: a deterministic
// rejection that repeating the same write cannot change (PLN-R106).
func SessionWriteRejected(err error) bool {
	return errors.Is(err, runstore.ErrInvalid)
}
