package gatewayrecovery

import (
	"encoding/json"
	"net/http"
	"strings"

	"github.com/grauwolf32/contractor/internal/contracts"
)

// Failure stores a safe classification, never provider response content.
type Failure struct {
	Code      string
	Retryable bool
}

const maximumClassificationBytes = 16 * 1024

// LiteLLM wraps the upstream OpenAI-compatible error as a Python dict repr.
// Only this observed wrapper is recognized; the wrapped text itself still has
// to be declared as a signature.
const (
	litellmWrapperPrefix = "litellm.BadRequestError: OpenAIException - Error code: 400 - "
	litellmWrapperSuffix = ". Received Model Group="
)

// Classify applies the openai-compatible@1 status rules plus the Gateway's
// declared failure signatures. Signature text is matched exactly so arbitrary
// 4xx bodies can never become a transient availability failure.
func Classify(status int, header http.Header, body []byte, signatures contracts.GatewayFailureSignatures) Failure {
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
	for _, code := range signatures.PermanentCodes {
		if detail.Code == code {
			return Failure{code, false}
		}
	}
	if modelUnavailable(status, message, signatures) {
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

func modelUnavailable(status int, message string, signatures contracts.GatewayFailureSignatures) bool {
	for _, signature := range signatures.ModelUnavailable {
		if signature.Status != status {
			continue
		}
		if signature.MessageEquals != "" && message == signature.MessageEquals {
			return true
		}
		if signature.LiteLLMWrapped != "" && litellmWrapped(message, signature.LiteLLMWrapped) {
			return true
		}
	}
	return false
}

// litellmWrapped compares against Python's repr of {"error": text}, which is
// how LiteLLM renders the upstream body; single quotes unless the text itself
// contains one and no double quote.
func litellmWrapped(message, upstreamText string) bool {
	if !strings.HasPrefix(message, litellmWrapperPrefix) {
		return false
	}
	upstream := strings.TrimPrefix(message, litellmWrapperPrefix)
	wrapped := "{'error': " + pythonRepr(upstreamText) + "}"
	return upstream == wrapped || strings.HasPrefix(upstream, wrapped+litellmWrapperSuffix)
}

// pythonRepr mirrors CPython's str repr quoting for the printable text that
// signature validation admits: no control characters, so no escaping beyond
// quote selection and backslashes is needed.
func pythonRepr(text string) string {
	quote := "'"
	if strings.Contains(text, "'") && !strings.Contains(text, `"`) {
		quote = `"`
	}
	escaped := strings.ReplaceAll(text, `\`, `\\`)
	if quote == "'" {
		escaped = strings.ReplaceAll(escaped, "'", `\'`)
	}
	return quote + escaped + quote
}
