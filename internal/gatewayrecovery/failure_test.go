package gatewayrecovery

import (
	"encoding/json"
	"net/http"
	"testing"
	"time"
)

func TestClassifyGatewayFailure(t *testing.T) {
	tests := []struct {
		status    int
		message   any
		code      string
		retryable bool
	}{
		{400, "Model is unloaded.", "model_unavailable", true},
		{400, "Model unloaded by user or API request.", "model_unavailable", true},
		{400, map[string]any{"message": "litellm.BadRequestError: OpenAIException - Error code: 400 - {'error': 'Model is unloaded.'}. Received Model Group=worker-model"}, "model_unavailable", true},
		{400, "Invalid request mentions Model is unloaded.", "gateway_request_rejected", false},
		{400, "invalid messages", "gateway_request_rejected", false},
		{429, map[string]any{"code": "insufficient_quota"}, "insufficient_quota", false},
		{400, map[string]any{"code": "context_length_exceeded"}, "context_length_exceeded", false},
		{401, "invalid token", "gateway_access_denied", false},
		{429, "slow down", "gateway_rate_limited", true},
		{504, "upstream timed out", "gateway_timeout", true},
		{503, "unavailable", "gateway_unavailable", true},
	}
	for _, test := range tests {
		body, _ := json.Marshal(map[string]any{"error": test.message})
		got := Classify(test.status, http.Header{}, body)
		if got.Code != test.code || got.Retryable != test.retryable {
			t.Fatalf("HTTP %d: %+v", test.status, got)
		}
	}
}
func TestRecoveryDelayUsesConfiguredCap(t *testing.T) {
	policy := Policy{RequestTimeout: time.Minute, InitialDelay: 3 * time.Second, MaxDelay: 19 * time.Second, AutomaticWindow: time.Minute}
	for i, want := range []time.Duration{3, 6, 12, 19, 19} {
		if got := policy.failureDelay(int64(i)); got != want*time.Second {
			t.Fatalf("failure %d delay=%s", i, got)
		}
	}
	for _, raw := range []string{"NaN", "+Inf", "-Inf", "garbage", "-1"} {
		if got := retryAfter(http.Header{"Retry-After": []string{raw}}); got != 0 {
			t.Fatalf("invalid hint %q: %f", raw, got)
		}
	}
}
