package publicclient

import (
	"bytes"
	"context"
	"errors"
	"io"
	"math/rand/v2"
	"net"
	"net/http"
	"net/url"
	"strings"
	"syscall"
	"time"
)

const maximumRequestAttempts = 3

// RoundTrip retries only methods that can be replayed without creating a
// second resource. The original request and its Idempotency-Key are immutable
// across attempts; a non-empty body needs GetBody to rewind byte-identically.
func (t *checkedTransport) RoundTrip(request *http.Request) (*http.Response, error) {
	if request.URL == nil || request.URL.Scheme+"://"+request.URL.Host != t.origin ||
		!strings.HasPrefix(request.URL.EscapedPath(), "/v1/") {
		return nil, errors.New("public API request escaped the configured Server boundary")
	}
	limit := 1
	if replayableRequest(request) {
		limit = maximumRequestAttempts
	}
	for attempt := 0; attempt < limit; attempt++ {
		if attempt > 0 {
			if err := waitForRetry(request.Context(), attempt); err != nil {
				return nil, err
			}
		}
		sent, cancel, err := t.attemptRequest(request, attempt)
		if err != nil {
			return nil, err
		}
		response, err := t.roundTripOnce(sent, cancel)
		if err != nil {
			if sent.Body != nil {
				_ = sent.Body.Close()
			}
			if attempt+1 == limit || request.Context().Err() != nil || !IsTransient(err) {
				return nil, err
			}
			continue
		}
		if attempt+1 < limit && retryableResponse(response) {
			_ = response.Body.Close()
			continue
		}
		return response, nil
	}
	return nil, errors.New("public API retry loop exhausted")
}

func (t *checkedTransport) attemptRequest(original *http.Request, attempt int) (*http.Request, func(), error) {
	ctx := original.Context()
	cancel := func() {}
	if t.timeout > 0 {
		var stop context.CancelFunc
		ctx, stop = context.WithTimeout(ctx, t.timeout)
		cancel = stop
	}
	request := original.Clone(ctx)
	if attempt > 0 && original.Body != nil && original.Body != http.NoBody {
		body, err := original.GetBody()
		if err != nil {
			cancel()
			return nil, nil, err
		}
		request.Body = body
	}
	request.Header.Set("Authorization", "Bearer "+t.token)
	request.Header.Set("Accept", "application/json")
	if t.userAgent != "" {
		request.Header.Set("User-Agent", t.userAgent)
	}
	return request, cancel, nil
}

func replayableRequest(request *http.Request) bool {
	if request.Method != http.MethodGet && strings.TrimSpace(request.Header.Get("Idempotency-Key")) == "" {
		return false
	}
	return request.Body == nil || request.Body == http.NoBody || request.GetBody != nil
}

func waitForRetry(ctx context.Context, attempt int) error {
	delay := time.Duration(75*(1<<(attempt-1))+rand.IntN(25)) * time.Millisecond
	timer := time.NewTimer(delay)
	defer timer.Stop()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-timer.C:
		return nil
	}
}

func retryableResponse(response *http.Response) bool {
	if response.StatusCode == http.StatusUnauthorized || response.StatusCode == http.StatusNotFound {
		return false
	}
	if transientStatus(response.StatusCode) {
		return true
	}
	if response.StatusCode < http.StatusBadRequest {
		return false
	}
	data, err := io.ReadAll(response.Body)
	_ = response.Body.Close()
	response.Body = io.NopCloser(bytes.NewReader(data))
	if err != nil {
		return false
	}
	var apiError *APIError
	return errors.As(DecodeError(response.StatusCode, data), &apiError) && apiError.Retryable
}

func transientStatus(status int) bool {
	return status == http.StatusBadGateway || status == http.StatusServiceUnavailable ||
		status == http.StatusGatewayTimeout
}

// IsTransient reports whether another poll can recover from a public call
// error. A caller must still stop when its own context expires.
func IsTransient(err error) bool {
	var compatibility *CompatibilityError
	if errors.As(err, &compatibility) {
		return false
	}
	var intermediary *IntermediaryError
	if errors.As(err, &intermediary) {
		return transientStatus(intermediary.Status)
	}
	var apiError *APIError
	if errors.As(err, &apiError) {
		if apiError.Status == http.StatusUnauthorized || apiError.Status == http.StatusNotFound {
			return false
		}
		return apiError.Retryable || transientStatus(apiError.Status)
	}
	var urlError *url.Error
	if errors.As(err, &urlError) {
		err = urlError.Err
	}
	if errors.Is(err, context.DeadlineExceeded) || errors.Is(err, io.EOF) || errors.Is(err, io.ErrUnexpectedEOF) {
		return true
	}
	if errors.Is(err, syscall.ECONNRESET) || errors.Is(err, syscall.ECONNREFUSED) ||
		errors.Is(err, syscall.EPIPE) || errors.Is(err, syscall.ETIMEDOUT) {
		return true
	}
	var networkError net.Error
	return errors.As(err, &networkError) && (networkError.Timeout() || networkError.Temporary())
}

type cancelOnClose struct {
	io.ReadCloser
	cancel func()
}

func (body *cancelOnClose) Close() error {
	err := body.ReadCloser.Close()
	body.cancel()
	return err
}
