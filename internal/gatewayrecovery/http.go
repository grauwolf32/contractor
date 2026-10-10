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
// abandoned failure ends the call as a *FailureError.
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
		action := "succeeded"
		if failure != nil {
			switch {
			case failure.Abandoned:
				// No model result was observed: relinquish a probe without
				// reopening or further blocking the route.
				action = "released"
			case failure.Retryable:
				action = "failed"
			default:
				action = "finished"
			}
		}
		update := Request{RequestID: requestID, Action: action, RetryAfterSeconds: delay}
		if action == "failed" {
			update.Code = failure.Code
		}
		err = p.record(request.Context(), update)
		if action == "failed" {
			// The route must record the failure before the request is resent.
			if err != nil {
				return nil, err
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
