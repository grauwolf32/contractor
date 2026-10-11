package gatewayrecovery

import (
	"context"
	"errors"
	"fmt"
	"io"
	"math"
	"net"
	"net/http"
	"net/http/httptrace"
	"strconv"
	"sync/atomic"
	"time"

	"github.com/grauwolf32/contractor/internal/contracts/llmgateway"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/randomid"
)

// FailureError reports a classified model request failure. Its text names only
// the safe classification code, never provider response content.
type FailureError struct {
	Failure Failure
}

func (e *FailureError) Error() string {
	return "Planner Gateway request failed (" + e.Failure.Code + ")"
}

// Do retries the same serialized model request inside a planner invocation.
// Each attempt is bounded by the client's own timeout. Context cancellation
// remains authoritative over all waits and network calls. A non-retryable or
// abandoned failure ends the call as a *FailureError; cancellation of an
// outstanding attempt releases its grant and ends the call with the context's
// cause, never as a Gateway failure.
func (p *Participant) Do(request *http.Request, client *http.Client, maxResponseBytes int64) ([]byte, error) {
	for {
		requestID, err := newRequestID()
		if err != nil {
			return nil, err
		}
		if _, err := p.acquire(request.Context(), requestID); err != nil {
			return nil, err
		}
		attempt := request.Clone(request.Context())
		attempt.Body, err = request.GetBody()
		if err != nil {
			return nil, fmt.Errorf("copy model request")
		}
		body, failure, delay := send(attempt, client, maxResponseBytes, p.signatures)
		// A failure seen after the invocation ended is the cancellation, not a
		// Gateway outcome: the grant is released without blocking the route
		// and the cancellation itself is reported (SCH-R108).
		cancelled := failure != nil && request.Context().Err() != nil
		update := attemptUpdate(requestID, failure, delay, cancelled)
		err = p.deliver(request.Context(), update)
		if cancelled {
			return nil, context.Cause(request.Context())
		}
		if update.Action == "failed" {
			// The route must record the failure before the request is resent.
			if err != nil {
				return nil, err
			}
			if request.Context().Err() != nil {
				return nil, context.Cause(request.Context())
			}
			continue
		}
		// A reply or permanent rejection ends the call, and a transient failure
		// to record it must not discard it: the route and the Run's waits
		// reconcile on a later update, the probe lease expiry or the Stage end.
		if err != nil && (request.Context().Err() != nil || !persistencepostgres.IsTransientFailure(err)) {
			return nil, err
		}
		if failure != nil {
			return nil, &FailureError{Failure: *failure}
		}
		return body, nil
	}
}

// maxRecordAttempts bounds the bookkeeping of one observed outcome. Each
// attempt is itself bounded by the database acquire and lock budgets.
const maxRecordAttempts = 3

// terminalRedeliveryTimeout bounds the best-effort re-delivery of an update
// after the invocation ended, like the Runtime's
// RECOVERY_RELEASE_TIMEOUT_SECONDS; the end of the Stage relinquishes a probe
// whose update is lost.
const terminalRedeliveryTimeout = 5 * time.Second

// attemptUpdate is the update that reports one granted attempt (SCH-R108,
// SCH-R109, SCH-R112): succeeded for a model response, released for an
// attempt that ended without an observed outcome (cancelled, or abandoned by
// the Gateway after it received the request), failed with its code and retry
// hint for a retryable failure, which is resent, and finished for any other
// failure. The Runtime's _attempt_update applies the same rule;
// api/testdata/v1alpha1/gateway-recovery-participant-cases.json pins both.
func attemptUpdate(requestID string, failure *Failure, retryAfterSeconds float64, cancelled bool) Request {
	update := Request{RequestID: requestID}
	switch {
	case cancelled:
		update.Action = "released"
	case failure == nil:
		update.Action = "succeeded"
	case failure.Abandoned:
		update.Action = "released"
	case failure.Retryable:
		update.Action, update.Code, update.RetryAfterSeconds = "failed", failure.Code, retryAfterSeconds
	default:
		update.Action = "finished"
	}
	return update
}

// redelivery is the update re-delivered when the invocation ends before
// update is acknowledged: a terminal update itself, and released in place of
// failed, so an ending invocation relinquishes its probe without making its
// Run wait; the Server accepts only terminal updates from an ending Stage
// (SCH-R108).
func redelivery(update Request) Request {
	if update.Terminal() {
		return update
	}
	return Request{RequestID: update.RequestID, Action: "released"}
}

// deliver records the update of a granted attempt. When the invocation ends
// before the update is acknowledged, it re-delivers its redelivery on a
// detached, bounded context, so cancellation does not leave the probe held
// until its lease expires (SCH-R108).
func (p *Participant) deliver(ctx context.Context, update Request) error {
	err := p.record(ctx, update)
	if err == nil || ctx.Err() == nil {
		return err
	}
	detached, cancel := context.WithTimeout(context.WithoutCancel(ctx), terminalRedeliveryTimeout)
	defer cancel()
	return p.record(detached, redelivery(update))
}

// record reports an observed outcome and retries a transient database failure.
// Repeating an update for the same request ID is idempotent, including after
// an unacknowledged commit.
func (p *Participant) record(ctx context.Context, update Request) error {
	for attempt := 1; ; attempt++ {
		_, err := p.Update(ctx, update)
		if err == nil || attempt == maxRecordAttempts || ctx.Err() != nil ||
			!persistencepostgres.IsTransientFailure(err) {
			return err
		}
		timer := time.NewTimer(time.Duration(attempt) * 50 * time.Millisecond)
		select {
		case <-ctx.Done():
			timer.Stop()
			return err
		case <-timer.C:
		}
	}
}

// Send performs one model request without a recovery authority. A failure is
// classified exactly as Do classifies it and returned as a *FailureError.
func Send(
	request *http.Request, client *http.Client, maxResponseBytes int64, signatures llmgateway.GatewayFailureSignatures,
) ([]byte, error) {
	body, failure, _ := send(request, client, maxResponseBytes, signatures)
	if failure != nil {
		return nil, &FailureError{Failure: *failure}
	}
	return body, nil
}

// send performs one attempt and classifies its outcome. Whether the request
// was fully written separates an unreachable Gateway from one that received
// the request and is still working on it.
func send(
	request *http.Request, client *http.Client, limit int64, signatures llmgateway.GatewayFailureSignatures,
) ([]byte, *Failure, float64) {
	var delivered atomic.Bool
	trace := &httptrace.ClientTrace{WroteRequest: func(info httptrace.WroteRequestInfo) {
		if info.Err == nil {
			delivered.Store(true)
		}
	}}
	response, sendErr := client.Do(request.WithContext(httptrace.WithClientTrace(request.Context(), trace)))
	return readResponse(response, sendErr, delivered.Load(), limit, signatures)
}

func (p *Participant) acquire(ctx context.Context, requestID string) (Decision, error) {
	for {
		decision, err := p.Update(ctx, Request{RequestID: requestID, Action: "acquire"})
		if err != nil || decision.Allowed {
			return decision, err
		}
		timer := time.NewTimer(time.Duration(decision.RetryAfterSeconds * float64(time.Second)))
		select {
		case <-ctx.Done():
			timer.Stop()
			return Decision{}, ctx.Err()
		case <-timer.C:
		}
	}
}

func readResponse(
	response *http.Response, sendErr error, delivered bool, limit int64, signatures llmgateway.GatewayFailureSignatures,
) ([]byte, *Failure, float64) {
	if sendErr != nil {
		var netError net.Error
		if errors.Is(sendErr, context.DeadlineExceeded) || errors.As(sendErr, &netError) && netError.Timeout() {
			return nil, &Failure{Code: "gateway_timeout", Retryable: true, Abandoned: delivered}, 0
		}
		return nil, &Failure{Code: "gateway_unavailable", Retryable: true}, 0
	}
	defer response.Body.Close()
	body, err := io.ReadAll(io.LimitReader(response.Body, limit+1))
	if err != nil {
		return nil, &Failure{Code: "gateway_unavailable", Retryable: true}, 0
	}
	if len(body) > int(limit) {
		return nil, &Failure{Code: "invalid_gateway_response", Retryable: false}, 0
	}
	if response.StatusCode >= 200 && response.StatusCode < 300 {
		return body, nil, 0
	}
	failure := Classify(response.StatusCode, response.Header, body, signatures)
	if response.Header.Get("x-should-retry") == "false" {
		failure.Retryable = false
	}
	return nil, &failure, retryAfter(response.Header)
}

func retryAfter(header http.Header) float64 {
	if delay, err := strconv.ParseFloat(header.Get("retry-after-ms"), 64); err == nil && !math.IsNaN(delay) && !math.IsInf(delay, 0) {
		return max(0, delay/1000)
	}
	if delay, err := strconv.ParseFloat(header.Get("retry-after"), 64); err == nil && !math.IsNaN(delay) && !math.IsInf(delay, 0) {
		return max(0, delay)
	}
	if deadline, err := http.ParseTime(header.Get("retry-after")); err == nil {
		return max(0, time.Until(deadline).Seconds())
	}
	return 0
}

func newRequestID() (string, error) {
	// 128-bit random request identity, also used by Runtime's UUID request IDs.
	return randomid.New("")
}
