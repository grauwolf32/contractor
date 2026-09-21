package gatewayrecovery

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"math"
	"net"
	"net/http"
	"strconv"
	"time"

	"github.com/grauwolf32/contractor/internal/contracts"
)

// Do retries the same serialized model request inside a planner invocation.
// Context cancellation remains authoritative over all waits and network calls.
func (p *Participant) Do(request *http.Request, client *http.Client, maxResponseBytes int64) ([]byte, error) {
	for {
		requestID, err := newRequestID()
		if err != nil {
			return nil, err
		}
		decision, err := p.acquire(request.Context(), requestID)
		if err != nil {
			return nil, err
		}
		attemptContext, cancel := context.WithTimeout(request.Context(), time.Duration(decision.RequestTimeoutSeconds*float64(time.Second)))
		attempt := request.Clone(attemptContext)
		attempt.Body, err = request.GetBody()
		if err != nil {
			cancel()
			return nil, fmt.Errorf("copy model request")
		}
		response, sendErr := client.Do(attempt)
		body, failure, delay := readResponse(response, sendErr, maxResponseBytes, p.signatures)
		cancel()
		action := "succeeded"
		if failure != nil {
			action = "finished"
			if failure.Retryable {
				action = "failed"
			}
		}
		update := Request{RequestID: requestID, Action: action, RetryAfterSeconds: delay}
		if failure != nil {
			update.Code = failure.Code
		}
		if _, err := p.Update(request.Context(), update); err != nil {
			return nil, err
		}
		if failure == nil {
			return body, nil
		}
		if !failure.Retryable {
			return nil, fmt.Errorf("Planner Gateway request failed (%s)", failure.Code)
		}
	}
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

func readResponse(response *http.Response, sendErr error, limit int64, signatures contracts.GatewayFailureSignatures) ([]byte, *Failure, float64) {
	if sendErr != nil {
		var netError net.Error
		if errors.Is(sendErr, context.DeadlineExceeded) || errors.As(sendErr, &netError) && netError.Timeout() {
			return nil, &Failure{"gateway_timeout", true}, 0
		}
		return nil, &Failure{"gateway_unavailable", true}, 0
	}
	defer response.Body.Close()
	body, err := io.ReadAll(io.LimitReader(response.Body, limit+1))
	if err != nil {
		return nil, &Failure{"gateway_unavailable", true}, 0
	}
	if len(body) > int(limit) {
		return nil, &Failure{"invalid_gateway_response", false}, 0
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
	var value [16]byte
	if _, err := rand.Read(value[:]); err != nil {
		return "", err
	}
	return hex.EncodeToString(value[:]), nil
}
