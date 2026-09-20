package gatewayrecovery

import (
	"encoding/json"
	"net/http"
	"strings"
)

// Failure stores a safe classification, never provider response content.
type Failure struct {
	Code      string
	Retryable bool
}

const maximumClassificationBytes = 16 * 1024

func Classify(status int, header http.Header, body []byte) Failure {
	var response struct {
		Error json.RawMessage `json:"error"`
	}
	var detail struct {
		Code    string `json:"code"`
		Message string `json:"message"`
	}
	var message string
	if len(body) <= maximumClassificationBytes && json.Unmarshal(body, &response) == nil {
		if json.Unmarshal(response.Error, &detail) == nil {
			message = detail.Message
		} else {
			_ = json.Unmarshal(response.Error, &message)
		}
	}
	switch detail.Code {
	case "insufficient_quota", "budget_exceeded", "context_length_exceeded":
		return Failure{detail.Code, false}
	}
	if status == http.StatusBadRequest && isModelUnloaded(message) {
		return Failure{"model_unavailable", true}
	}
	switch status {
	case http.StatusUnauthorized, http.StatusForbidden:
		return Failure{"gateway_access_denied", false}
	case http.StatusTooManyRequests:
		return Failure{"gateway_rate_limited", true}
	case http.StatusRequestTimeout, http.StatusGatewayTimeout:
		return Failure{"gateway_timeout", true}
	case http.StatusConflict:
		return Failure{"gateway_unavailable", true}
	}
	if status >= 500 && status < 600 || header.Get("x-should-retry") == "true" {
		return Failure{"gateway_unavailable", true}
	}
	return Failure{"gateway_request_rejected", false}
}

func isModelUnloaded(message string) bool {
	// Exact LM Studio signatures from the 2026-09-20 incident, including LiteLLM's
	// observed wrapper. Arbitrary 400 messages never become availability errors.
	const prefix = "litellm.BadRequestError: OpenAIException - Error code: 400 - "
	for _, text := range []string{"Model is unloaded.", "Model unloaded by user or API request."} {
		if message == text {
			return true
		}
		wrapped := prefix + "{'error': '" + text + "'}"
		if message == wrapped || strings.HasPrefix(message, wrapped+". Received Model Group=") {
			return true
		}
	}
	return false
}
